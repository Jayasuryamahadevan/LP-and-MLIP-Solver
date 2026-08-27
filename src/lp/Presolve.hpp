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
    // column_removed is true.
    std::vector<double> fixed_value;
    std::vector<char> column_removed;

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
//
// Reductions deliberately NOT applied, and why: doubleton-equation
// substitution and free-column-singleton substitution both change the
// sparsity pattern of A and require reconstructing an eliminated variable
// from a row equation during postsolve. They are effective but their
// postsolve is materially harder to get right, and SYSTEM.md \S2.3 makes
// exact invertibility a hard invariant rather than a target. They are
// candidates for a later milestone, once this set is validated.
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
PresolveResult presolve(const LpProblem& problem, int max_passes = 20,
                         const std::vector<char>& integer_columns = {});

// Expands a solution of the reduced problem back into ORIGINAL column
// space: kept columns take their solved value, removed columns take the
// value presolve fixed them at. `reduced_x` must have size
// result.kept_columns.size().
std::vector<double> postsolve(const PresolveResult& result,
                               const std::vector<double>& reduced_x);

} // namespace sihps
