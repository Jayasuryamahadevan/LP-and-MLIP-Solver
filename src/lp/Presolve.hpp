#pragma once

#include "LpProblem.hpp"

#include <cstdint>
#include <vector>

namespace sihps {

enum class PresolveStatus { OK, INFEASIBLE, UNBOUNDED };

// Result of the presolve pass, carrying BOTH the reduced model and the
// information needed to invert the reduction exactly.
//
// docs/architecture/SYSTEM.md \S2.3 states the contract this type exists to
// honour: "a presolve reduction that cannot be exactly inverted must not be
// applied, full stop." Everything below is therefore recorded at the moment
// a reduction fires, not reconstructed afterwards.
struct PresolveResult {
    PresolveStatus status = PresolveStatus::OK;

    // Which reduction concluded INFEASIBLE/UNBOUNDED, and on which row or
    // column. A presolve that reports infeasibility without saying why is
    // effectively unfalsifiable -- and since a buggy reduction and a
    // genuinely infeasible model produce the identical status, this is the
    // only thing that distinguishes them when triaging.
    const char* reason = nullptr;
    std::int32_t reason_row = -1;
    std::int32_t reason_col = -1;
    int reason_pass = -1;

    LpProblem reduced;

    // reduced column k is original column kept_columns[k];
    // reduced row r is original row kept_rows[r].
    std::vector<std::int32_t> kept_columns;
    std::vector<std::int32_t> kept_rows;

    // For every original column removed by presolve, the value it was
    // fixed at. Indexed by ORIGINAL column; meaningful only where
    // column_removed is true AND column_is_doubleton is false -- a
    // doubleton-eliminated column's value is not a constant, see
    // doubleton_eliminations below.
    std::vector<double> fixed_value;
    std::vector<char> column_removed;

    // A doubleton-eliminated column: original column eliminated_col was
    // removed by expressing it as an EXACT affine function of target_col,
    //   x[eliminated_col] = intercept + slope * x[target_col]
    // Recorded in DISCOVERY order (the order presolve found them in,
    // across passes). target_col is guaranteed, by construction, to have
    // been ACTIVE at the moment this was recorded -- a doubleton row can
    // only choose an active column as its surviving variable -- so any
    // later event that removes target_col (an ordinary fix, or a further
    // doubleton) is necessarily recorded AFTER this one. Replaying this
    // list in REVERSE (see postsolve() below) therefore always resolves a
    // dependency before it's needed, with no separate topological sort:
    // this is exactly the "replay the log in reverse" contract
    // SYSTEM.md \S2.3 describes for postsolve.
    struct DoubletonElimination {
        std::int32_t eliminated_col = -1;
        std::int32_t target_col = -1;
        double intercept = 0.0;
        double slope = 0.0;
        // eliminated_col's own bounds AT THE MOMENT of elimination (already
        // reflecting every reduction applied so far, so the tightest
        // correct range known for it). postsolve() clamps the reconstructed
        // value into this range: the implied bound placed on target_col is
        // deliberately relaxed outward (this file's own kBoundRelax
        // policy, applied everywhere so no reduction ever over-tightens),
        // and that outward slack, carried through intercept/slope, can let
        // the reconstructed x[eliminated_col] land fractionally outside its
        // own true bound -- harmless in relative terms, but large enough in
        // absolute terms on a big-magnitude model (MEASURED on Netlib
        // `greenbea`/`greenbeb`: a bound violation of ~1.4e-6, just over
        // this project's 1e-6 final verification gate) to fail the
        // original-space check without the clamp.
        double eliminated_lower = -kInfinity;
        double eliminated_upper = kInfinity;
    };
    std::vector<DoubletonElimination> doubleton_eliminations;
    // column_removed[j] is still set for a doubleton-eliminated column (so
    // removed_cols() and any other existing caller of column_removed keeps
    // working unmodified), but fixed_value[j] is NOT meaningful for it --
    // this flag distinguishes the two removal kinds so postsolve() knows
    // which mechanism to use for column j.
    std::vector<char> column_is_doubleton;

    std::int32_t original_n_rows = 0;
    std::int32_t original_n_cols = 0;

    std::int32_t removed_rows() const {
        return original_n_rows - static_cast<std::int32_t>(kept_rows.size());
    }
    std::int32_t removed_cols() const {
        return original_n_cols - static_cast<std::int32_t>(kept_columns.size());
    }
};

// Reduces `problem` by repeated application of the reductions below, to a
// fixed point (or a bounded pass cap).
//
// ESTABLISHED METHODS. The reduction set is the classical one; the lineage
// is Andersen & Andersen, "Presolving in linear programming", Mathematical
// Programming 71, 1995, with the modern MIP-oriented survey in Achterberg,
// Bixby, Gu, Rothberg & Weninger, "Presolve Reductions in Mixed Integer
// Programming", INFORMS Journal on Computing 32(2), 2020 (both already
// cited in docs/research/SOTA.md \S1.1). Implemented here from the
// published descriptions of the transformations; no solver source was
// consulted or adapted.
//
// Reductions applied:
//   - empty row            : feasible iff 0 lies in the row's bounds
//   - empty column         : fix at whichever bound optimizes its cost
//   - fixed column         : l_j == u_j, substituted out into the row bounds
//   - singleton row        : a_ij x_j in [lo_i, hi_i] becomes a bound on x_j
//   - redundant row        : activity bounds already imply the row
//   - forcing row          : activity bound meets the row bound exactly, so
//                            every variable in it is pinned to one bound
//   - activity-based bound tightening (bound propagation)
//   - integer bound rounding (see `integer_columns` below)
//   - doubleton-equation substitution, SCOPED (see `enable_doubleton_
//     substitution` below) -- NOT the general case
//   - GCD-based row tightening (see `enable_gcd_tightening` below)
//
// Reduction deliberately NOT applied in its general form, and why: a
// doubleton row's eliminated variable, in general, can appear in OTHER
// active rows too, and folding its contribution into those rows requires
// mutating coefficients this file currently treats as immutable (`A`/
// `A_csc`, read once at entry) and can introduce fill-in (a nonzero in a
// row that previously had a structural zero there). That is a materially
// larger, more invasive change than anything else in this file, and this
// project's own recent history (docs/ROADMAP_STATUS.md items 4-5) shows
// mathematically-correct changes to adjacent, equally load-bearing code
// can still destabilize a specific Netlib instance (`pilot87`) via
// floating-point summation order, purely from ADDED computation -- not
// attempted here. What IS applied is the intersection of "doubleton row"
// and "free-column singleton": the eliminated variable additionally
// appears in NO other active row (checked via col_count), so no other
// row's coefficients ever change and every existing call site
// (row_activity, fix_column, drop_row, the propagation loop, the final
// triplet emission) needs zero modification -- see `enable_doubleton_
// substitution` below and doubleton_eliminations above for the exact
// contract. The general (fill-in-capable) case remains a candidate for a
// later milestone.
//
// NUMERICAL POLICY: every test below carries an explicit tolerance and is
// applied CONSERVATIVELY -- a row is dropped only when redundancy is
// unambiguous, and INFEASIBLE is returned only when the violation exceeds
// tolerance by a margin. A presolve that is aggressive at the tolerance
// boundary can turn a feasible model infeasible, which is a far worse
// failure than leaving a reduction on the table.
//
// `integer_columns`, empty by default: this LP-level presolve has no notion
// of integrality on its own -- deliberately, so an ordinary LP caller is
// never affected by it (see MilpProblem.hpp's own note that "the LP engine
// never sees this metadata" by default). A MILP caller that has ALREADY
// determined, from the original model, which columns are integer-restricted
// may opt in explicitly by passing a mask the same size as the problem's
// column count (nonzero = integer). When present, every bound this pass
// derives or is given for an integer column is additionally rounded inward
// to the nearest integer (`ceil` for a lower bound, `floor` for an upper
// bound) -- ESTABLISHED METHOD, unconditionally sound: any integer-feasible
// x_j respects the rounded bound exactly as it respected the original one.
// This requires no new postsolve machinery: it only narrows a box bound,
// never removes a column, so `postsolve()` below is already correct for it
// unmodified. A mask whose size is neither 0 nor the column count is
// treated as if no column were flagged integer, rather than indexed
// out of bounds -- this is a defensive guard against a caller bug, not a
// scenario this function's own contract expects to occur.
//
// `enable_doubleton_substitution`, false by default (this project's own
// standing rule: no optimization ships default-on without a measured KPI
// improvement -- see the GMI-cuts and RENS precedents). When true, an
// equality row with exactly two active nonzero coefficients, where one of
// the two variables appears in NO other active row, eliminates that
// variable by substitution -- see `PresolveResult::doubleton_eliminations`
// above for the exact contract and soundness argument.
//
// `enable_gcd_tightening`, false by default (same standing rule as
// `enable_doubleton_substitution`). ESTABLISHED METHOD (Achterberg, Bixby,
// Gu, Rothberg & Weninger 2020, \S3.4 "gcd" reduction). Requires
// `integer_columns` to be populated -- with no integer columns flagged, this
// reduction can never find a row where EVERY active variable qualifies, so
// it is a silent no-op for an ordinary LP caller, exactly like integer
// bound rounding above.
//
// For a row where every active coefficient is (numerically) integer AND
// every active column is integer-restricted, the row's activity `Ax` can
// only ever take a value that is a multiple of g = gcd(|a_j|). When g > 1,
// the row's finite bounds are tightened to the nearest reachable multiple
// of g (floor for the upper bound, ceil for the lower bound); if this
// pushes the tightened lower bound above the tightened upper bound, no
// multiple of g exists inside the original bounds and the row -- hence the
// whole model -- is INFEASIBLE, detected directly rather than left for the
// simplex to discover indirectly. UNCONDITIONALLY SOUND: no integer-
// feasible point is ever excluded, since every such point's activity was
// already a multiple of g before this reduction ran. A row mixing even one
// continuous column, or one non-integer coefficient, is left untouched in
// its entirety -- this is a scope boundary (the reduction's own soundness
// argument does not hold otherwise), not a missed case.
PresolveResult presolve(const LpProblem& problem, int max_passes = 20,
                         const std::vector<char>& integer_columns = {},
                         bool enable_doubleton_substitution = false,
                         bool enable_gcd_tightening = false);

// Expands a solution of the reduced problem back into ORIGINAL column
// space: kept columns take their solved value, removed columns take the
// value presolve fixed them at (or, for a doubleton-eliminated column,
// its recorded affine function of another column -- see
// PresolveResult::doubleton_eliminations). `reduced_x` must have size
// result.kept_columns.size().
std::vector<double> postsolve(const PresolveResult& result,
                               const std::vector<double>& reduced_x);

} // namespace sihps
