#include "../test_framework.hpp"
#include "io/MpsReader.hpp"
#include "lp/LpProblem.hpp"
#include "lp/LpSolver.hpp"
#include "lp/Presolve.hpp"
#include "sparse/Triplet.hpp"

#include <cmath>
#include <string>

using sihps::CSRMatrix;
using sihps::kInfinity;
using sihps::LpProblem;
using sihps::LpSolverOptions;
using sihps::LpStatus;
using sihps::presolve;
using sihps::PresolveStatus;
using sihps::read_mps_file;
using sihps::solve_lp;
using sihps::Triplet;

namespace {

std::string netlib(const std::string& name) {
    return std::string(SIHPS_PROJECT_ROOT) + "/data/netlib_lp/feasible/" + name + ".mps";
}

// minimize -x - y  s.t.  x + 2y <= 4,  3x + y <= 6,  x,y >= 0.
// Hand-derived optimum (see tests/lp/test_simplex.cpp): (1.6, 1.2), -2.8.
LpProblem tiny_lp() {
    LpProblem p;
    std::vector<Triplet> t = {{0, 0, 1.0}, {0, 1, 2.0}, {1, 0, 3.0}, {1, 1, 1.0}};
    p.A = CSRMatrix::from_triplets(2, 2, t);
    p.obj = {-1.0, -1.0};
    p.rhs = {4.0, 6.0};
    p.row_types = {'L', 'L'};
    p.lower = {0.0, 0.0};
    p.upper = {kInfinity, kInfinity};
    sihps::apply_default_row_bounds(p);
    return p;
}

} // namespace

// A row whose single coefficient bounds one variable must become a bound,
// not survive as a row: x <= 3 written as a row, with y otherwise free to
// two-variable row.
SIHPS_TEST(presolve_converts_singleton_row_to_bound) {
    LpProblem p;
    std::vector<Triplet> t = {{0, 0, 1.0}, {1, 0, 1.0}, {1, 1, 1.0}};
    p.A = CSRMatrix::from_triplets(2, 2, t);
    p.obj = {1.0, 1.0};
    p.rhs = {3.0, 10.0};
    p.row_types = {'L', 'E'};
    p.lower = {0.0, 0.0};
    p.upper = {kInfinity, kInfinity};
    sihps::apply_default_row_bounds(p);

    auto r = presolve(p);
    SIHPS_ASSERT_TRUE(r.status == PresolveStatus::OK);
    SIHPS_ASSERT_TRUE(r.removed_rows() >= 1);
    // x's upper bound must have absorbed the singleton row. The comparison
    // allows Presolve.cpp's deliberate outward safety margin
    // (kBoundRelax * (1 + |bound|), so ~4e-9 at a bound of 3): presolve is
    // required to produce a RELAXATION of the true bound, never an
    // over-tightening, so a bound landing marginally above 3 is the
    // specified behaviour rather than an error.
    constexpr double kMargin = 1e-8;
    bool found = false;
    for (std::size_t k = 0; k < r.kept_columns.size(); ++k) {
        if (r.kept_columns[k] == 0) {
            SIHPS_ASSERT_TRUE(r.reduced.upper[k] <= 3.0 + kMargin);
            found = true;
        }
    }
    SIHPS_ASSERT_TRUE(found);
}

// A row that its variables' bounds already imply carries no information and
// must be dropped: x in [0,1], y in [0,1], row x + y <= 5.
SIHPS_TEST(presolve_drops_redundant_row) {
    LpProblem p;
    std::vector<Triplet> t = {{0, 0, 1.0}, {0, 1, 1.0}};
    p.A = CSRMatrix::from_triplets(1, 2, t);
    p.obj = {-1.0, -1.0};
    p.rhs = {5.0};
    p.row_types = {'L'};
    p.lower = {0.0, 0.0};
    p.upper = {1.0, 1.0};
    sihps::apply_default_row_bounds(p);

    auto r = presolve(p);
    SIHPS_ASSERT_TRUE(r.status == PresolveStatus::OK);
    SIHPS_ASSERT_EQ(r.removed_rows(), 1);
}

// A fixed column must be substituted out and its value recoverable.
SIHPS_TEST(presolve_removes_fixed_column_and_postsolve_restores_it) {
    LpProblem p;
    std::vector<Triplet> t = {{0, 0, 1.0}, {0, 1, 1.0}};
    p.A = CSRMatrix::from_triplets(1, 2, t);
    p.obj = {0.0, 1.0};
    p.rhs = {10.0};
    p.row_types = {'E'};
    p.lower = {4.0, 0.0};
    p.upper = {4.0, kInfinity}; // x fixed at 4
    sihps::apply_default_row_bounds(p);

    auto r = presolve(p);
    SIHPS_ASSERT_TRUE(r.status == PresolveStatus::OK);
    SIHPS_ASSERT_TRUE(r.column_removed[0] == 1);
    SIHPS_ASSERT_NEAR(r.fixed_value[0], 4.0, 1e-12);

    // Whatever the reduced problem reports for the surviving column,
    // postsolve must put x back at 4. Presolve here cascades: fixing x
    // leaves the row as the singleton y = 6, so y is fixed too and NO
    // column survives -- postsolve must still reconstruct both. The 1e-8
    // tolerance covers the outward safety margin documented above.
    std::vector<double> reduced_x(r.kept_columns.size(), 6.0);
    auto full = sihps::postsolve(r, reduced_x);
    SIHPS_ASSERT_EQ(static_cast<int>(full.size()), 2);
    SIHPS_ASSERT_NEAR(full[0], 4.0, 1e-8);
    SIHPS_ASSERT_NEAR(full[1], 6.0, 1e-8);
}

// Contradictory bounds must be caught by presolve rather than passed on.
SIHPS_TEST(presolve_detects_infeasible_bounds) {
    LpProblem p;
    std::vector<Triplet> t = {{0, 0, 1.0}, {1, 0, 1.0}};
    p.A = CSRMatrix::from_triplets(2, 1, t);
    p.obj = {0.0};
    p.rhs = {3.0, 5.0};
    p.row_types = {'L', 'G'}; // x <= 3 and x >= 5
    p.lower = {0.0};
    p.upper = {kInfinity};
    sihps::apply_default_row_bounds(p);

    auto r = presolve(p);
    SIHPS_ASSERT_TRUE(r.status == PresolveStatus::INFEASIBLE);
}

// An empty column with a cost that improves without limit is unbounded.
SIHPS_TEST(presolve_detects_unbounded_empty_column) {
    LpProblem p;
    std::vector<Triplet> t = {{0, 0, 1.0}};
    p.A = CSRMatrix::from_triplets(1, 2, t);
    p.obj = {0.0, -1.0}; // column 1 is in no row and improves forever
    p.rhs = {1.0};
    p.row_types = {'L'};
    p.lower = {0.0, 0.0};
    p.upper = {kInfinity, kInfinity};
    sihps::apply_default_row_bounds(p);

    auto r = presolve(p);
    SIHPS_ASSERT_TRUE(r.status == PresolveStatus::UNBOUNDED);
}

SIHPS_TEST(solve_lp_matches_hand_verified_optimum_with_presolve) {
    LpProblem p = tiny_lp();
    LpSolverOptions options;
    options.use_presolve = true;
    auto s = solve_lp(p, options);
    SIHPS_ASSERT_TRUE(s.status == LpStatus::OPTIMAL);
    SIHPS_ASSERT_NEAR(s.objective_value, -2.8, 1e-6);
    SIHPS_ASSERT_NEAR(s.x[0], 1.6, 1e-6);
    SIHPS_ASSERT_NEAR(s.x[1], 1.2, 1e-6);
    SIHPS_ASSERT_TRUE(s.primal_residual < 1e-6);
}

// The sharpest available check that the reductions are sound: presolve must
// not change the answer. Any reduction that is subtly wrong shows up here as
// a differing optimum on a real model, which no amount of synthetic unit
// testing would catch.
SIHPS_TEST(presolve_does_not_change_the_optimum_on_netlib_instances) {
    const char* names[] = {"afiro", "adlittle", "share2b", "blend", "sc205", "scagr7", "bore3d"};
    for (const char* name : names) {
        auto model = read_mps_file(netlib(name));
        LpProblem p = sihps::lp_problem_from_mps(model);

        LpSolverOptions with;
        with.use_presolve = true;
        LpSolverOptions without;
        without.use_presolve = false;

        auto a = solve_lp(p, with);
        auto b = solve_lp(p, without);

        SIHPS_ASSERT_TRUE(a.status == LpStatus::OPTIMAL);
        SIHPS_ASSERT_TRUE(b.status == LpStatus::OPTIMAL);
        const double scale = 1.0 + std::fabs(b.objective_value);
        SIHPS_ASSERT_TRUE(std::fabs(a.objective_value - b.objective_value) / scale < 1e-7);
        SIHPS_ASSERT_TRUE(a.primal_residual < 1e-6);
    }
}

// Postsolve must always return a vector in ORIGINAL column space, whatever
// presolve removed.
SIHPS_TEST(solve_lp_returns_original_space_solution_dimensions) {
    auto model = read_mps_file(netlib("afiro"));
    LpProblem p = sihps::lp_problem_from_mps(model);
    LpSolverOptions options;
    options.use_presolve = true;
    auto s = solve_lp(p, options);
    SIHPS_ASSERT_TRUE(s.status == LpStatus::OPTIMAL);
    SIHPS_ASSERT_EQ(static_cast<int>(s.x.size()), p.n_cols());
}

// --- integer bound rounding (MIP-specific presolve, opt-in via
// `integer_columns`) ---------------------------------------------------

// A derived upper bound on an integer column must round down to the
// nearest integer. Structured so the rounded value is directly observable:
// the singleton row `x <= 5.6` gets absorbed and dropped, x becomes an
// empty column, and a negative cost (minimize -x, i.e. maximize x) fixes
// it at its UPPER bound -- exposing exactly what presolve computed for
// that bound as `fixed_value`.
SIHPS_TEST(presolve_rounds_derived_upper_bound_for_integer_column) {
    LpProblem p;
    std::vector<Triplet> t = {{0, 0, 1.0}};
    p.A = CSRMatrix::from_triplets(1, 1, t);
    p.obj = {-1.0};
    p.rhs = {5.6};
    p.row_types = {'L'};
    p.lower = {0.0};
    p.upper = {kInfinity};
    sihps::apply_default_row_bounds(p);

    std::vector<char> integer_columns = {1};
    auto r = presolve(p, 20, integer_columns);
    SIHPS_ASSERT_TRUE(r.status == PresolveStatus::OK);
    SIHPS_ASSERT_TRUE(r.column_removed[0] == 1);
    SIHPS_ASSERT_NEAR(r.fixed_value[0], 5.0, 1e-9);
}

// The lower-bound analogue: `x >= 2.3`, cost minimizes x (fixes at the
// LOWER bound), rounded value must be ceil(2.3) = 3.
SIHPS_TEST(presolve_rounds_derived_lower_bound_for_integer_column) {
    LpProblem p;
    std::vector<Triplet> t = {{0, 0, 1.0}};
    p.A = CSRMatrix::from_triplets(1, 1, t);
    p.obj = {1.0};
    p.rhs = {2.3};
    p.row_types = {'G'};
    p.lower = {0.0};
    p.upper = {kInfinity};
    sihps::apply_default_row_bounds(p);

    std::vector<char> integer_columns = {1};
    auto r = presolve(p, 20, integer_columns);
    SIHPS_ASSERT_TRUE(r.status == PresolveStatus::OK);
    SIHPS_ASSERT_TRUE(r.column_removed[0] == 1);
    SIHPS_ASSERT_NEAR(r.fixed_value[0], 3.0, 1e-9);
}

// Backward compatibility: the default call signature (no `integer_columns`
// argument at all, matching every pre-existing call site) must leave the
// exact same model's derived bound UNROUNDED -- proves the new parameter
// is genuinely opt-in, not a behavior change by default.
SIHPS_TEST(presolve_integer_rounding_is_off_by_default) {
    LpProblem p;
    std::vector<Triplet> t = {{0, 0, 1.0}};
    p.A = CSRMatrix::from_triplets(1, 1, t);
    p.obj = {-1.0};
    p.rhs = {5.6};
    p.row_types = {'L'};
    p.lower = {0.0};
    p.upper = {kInfinity};
    sihps::apply_default_row_bounds(p);

    auto r = presolve(p); // no integer_columns argument
    SIHPS_ASSERT_TRUE(r.status == PresolveStatus::OK);
    SIHPS_ASSERT_TRUE(r.column_removed[0] == 1);
    // Tolerance wider than 1e-9: the value itself already carries
    // Presolve.cpp's own pre-existing kBoundRelax outward pad (~6.6e-9 at
    // this magnitude), unrelated to this test -- the point here is only
    // that it stays near 5.6, nowhere near the rounded 5.0.
    SIHPS_ASSERT_NEAR(r.fixed_value[0], 5.6, 1e-7);
}

// A bound that is genuinely EXACTLY integer in principle but computed via a
// real floating-point division that lands just below it must still round
// to the true integer, not one below -- 0.7 / 0.1 is a well-known IEEE754
// double artifact that lands at 6.999999999999999, not 7.0. This is real
// floating-point noise from an actual division in presolve's own
// arithmetic, not a hand-picked near-integer literal, and is exactly the
// case kBoundRelax's outward pad (applied BEFORE the integer floor/ceil,
// see tighten_upper's own comment) exists to protect against.
SIHPS_TEST(presolve_integer_rounding_survives_floating_point_noise_at_a_boundary) {
    LpProblem p;
    std::vector<Triplet> t = {{0, 0, 0.1}};
    p.A = CSRMatrix::from_triplets(1, 1, t);
    p.obj = {-1.0};
    p.rhs = {0.7};
    p.row_types = {'L'};
    p.lower = {0.0};
    p.upper = {kInfinity};
    sihps::apply_default_row_bounds(p);

    std::vector<char> integer_columns = {1};
    auto r = presolve(p, 20, integer_columns);
    SIHPS_ASSERT_TRUE(r.status == PresolveStatus::OK);
    SIHPS_ASSERT_TRUE(r.column_removed[0] == 1);
    SIHPS_ASSERT_NEAR(r.fixed_value[0], 7.0, 1e-9); // not 6.0
}

// Two-row cascade: rounding x's bound in pass 1 (row 0, singleton) must
// change what row 1's activity-based propagation derives for y in the
// SAME pass, since row-reduction and propagation both run before the pass
// loop checks for a fixed point. Row 0: x <= 7.5 (x integer) rounds x's
// upper to 7. Row 1: -x + y <= 2.3 then derives y's upper as
// (2.3 - (-7))/1 = 9.3 -- vs (2.3 - (-7.5))/1 = 9.8 had x's bound NOT been
// rounded first. Both expected values are deliberately non-integer, so if
// y (mask[1] == 0, NOT integer-restricted) were incorrectly rounded too,
// this test would catch it as a wrong (integer) result instead of 9.3.
SIHPS_TEST(presolve_integer_rounding_cascades_into_a_later_propagated_bound) {
    LpProblem p;
    std::vector<Triplet> t = {
        {0, 0, 1.0},   // row 0: x <= 7.5
        {1, 0, -1.0},  // row 1: -x + y <= 2.3
        {1, 1, 1.0},
    };
    p.A = CSRMatrix::from_triplets(2, 2, t);
    p.obj = {0.0, 0.0};
    p.rhs = {7.5, 2.3};
    p.row_types = {'L', 'L'};
    p.lower = {0.0, 0.0};
    p.upper = {kInfinity, kInfinity};
    sihps::apply_default_row_bounds(p);

    std::vector<char> integer_columns = {1, 0}; // x integer, y continuous
    auto r = presolve(p, 20, integer_columns);
    SIHPS_ASSERT_TRUE(r.status == PresolveStatus::OK);

    bool found_x = false;
    bool found_y = false;
    for (std::size_t k = 0; k < r.kept_columns.size(); ++k) {
        if (r.kept_columns[k] == 0) {
            SIHPS_ASSERT_NEAR(r.reduced.upper[k], 7.0, 1e-6);
            found_x = true;
        } else if (r.kept_columns[k] == 1) {
            SIHPS_ASSERT_NEAR(r.reduced.upper[k], 9.3, 1e-6);
            found_y = true;
        }
    }
    SIHPS_ASSERT_TRUE(found_x);
    SIHPS_ASSERT_TRUE(found_y);
}

// --- doubleton row substitution (opt-in via `enable_doubleton_
// substitution`; SCOPED to equality rows where the eliminated variable
// appears in NO other active row -- see Presolve.hpp) ------------------

// Clean elimination + implied bound + postsolve round-trip. x + 2y = 10
// (x isolated -- appears nowhere else), y + z <= 20. Hand-derived: x is
// eliminated (only isolated candidate), intercept = 10/1 = 10,
// slope = -2/1 = -2. x's own bounds [0, inf) imply, via slope < 0,
// y <= (0 - 10)/(-2) = 5 -- row1 (y + z <= 20) is untouched, since x never
// appears there (Increment 1's own scope boundary: no other row's
// coefficients ever change).
SIHPS_TEST(presolve_doubleton_eliminates_isolated_variable_and_tightens_the_survivor) {
    LpProblem p;
    std::vector<Triplet> t = {
        {0, 0, 1.0}, {0, 1, 2.0}, // row0 (E): x + 2y = 10
        {1, 1, 1.0}, {1, 2, 1.0}, // row1 (L): y + z <= 20
    };
    p.A = CSRMatrix::from_triplets(2, 3, t);
    p.obj = {0.0, 0.0, 0.0};
    p.rhs = {10.0, 20.0};
    p.row_types = {'E', 'L'};
    p.lower = {0.0, 0.0, 0.0};
    p.upper = {kInfinity, kInfinity, kInfinity};
    sihps::apply_default_row_bounds(p);

    auto r = presolve(p, 20, {}, /*enable_doubleton_substitution=*/true);
    SIHPS_ASSERT_TRUE(r.status == PresolveStatus::OK);
    SIHPS_ASSERT_TRUE(r.column_removed[0] == 1);
    SIHPS_ASSERT_TRUE(r.column_is_doubleton[0] == 1);
    SIHPS_ASSERT_EQ(static_cast<int>(r.doubleton_eliminations.size()), 1);
    SIHPS_ASSERT_EQ(r.doubleton_eliminations[0].eliminated_col, 0);
    SIHPS_ASSERT_EQ(r.doubleton_eliminations[0].target_col, 1);
    SIHPS_ASSERT_NEAR(r.doubleton_eliminations[0].intercept, 10.0, 1e-9);
    SIHPS_ASSERT_NEAR(r.doubleton_eliminations[0].slope, -2.0, 1e-9);

    bool found_y = false, found_z = false;
    for (std::size_t k = 0; k < r.kept_columns.size(); ++k) {
        if (r.kept_columns[k] == 1) {
            SIHPS_ASSERT_TRUE(r.reduced.upper[k] <= 5.0 + 1e-6);
            found_y = true;
        } else if (r.kept_columns[k] == 2) {
            found_z = true;
        }
    }
    SIHPS_ASSERT_TRUE(found_y);
    SIHPS_ASSERT_TRUE(found_z);
    // row1 must survive untouched (only x's isolation makes this safe;
    // x never appears in row1, so row1's own coefficients are unaffected).
    SIHPS_ASSERT_TRUE(r.removed_rows() == 1);

    std::vector<double> reduced_x(r.kept_columns.size());
    for (std::size_t k = 0; k < r.kept_columns.size(); ++k) {
        if (r.kept_columns[k] == 1) reduced_x[k] = 3.0; // y = 3
        else if (r.kept_columns[k] == 2) reduced_x[k] = 1.0; // z = 1
    }
    auto full = sihps::postsolve(r, reduced_x);
    SIHPS_ASSERT_NEAR(full[0], 4.0, 1e-9); // x = 10 - 2*3 = 4
    SIHPS_ASSERT_NEAR(full[1], 3.0, 1e-9);
}

// Numerical-safety guard: a near-zero coefficient on the isolated variable
// must decline the elimination entirely (row and column both left
// untouched), not substitute through a poorly-conditioned pivot.
SIHPS_TEST(presolve_doubleton_declines_a_near_zero_pivot) {
    LpProblem p;
    std::vector<Triplet> t = {
        {0, 0, 1e-13}, {0, 1, 1.0}, // row0 (E): 1e-13*x + y = 5
        {1, 1, 1.0}, {1, 2, 1.0},   // row1 (L): y + z <= 20
    };
    p.A = CSRMatrix::from_triplets(2, 3, t);
    p.obj = {0.0, 0.0, 0.0};
    p.rhs = {5.0, 20.0};
    p.row_types = {'E', 'L'};
    p.lower = {0.0, 0.0, 0.0};
    p.upper = {kInfinity, kInfinity, kInfinity};
    sihps::apply_default_row_bounds(p);

    auto r = presolve(p, 20, {}, /*enable_doubleton_substitution=*/true);
    SIHPS_ASSERT_TRUE(r.status == PresolveStatus::OK);
    SIHPS_ASSERT_TRUE(r.column_removed[0] == 0); // x untouched
    SIHPS_ASSERT_TRUE(r.column_is_doubleton[0] == 0);
    SIHPS_ASSERT_EQ(static_cast<int>(r.doubleton_eliminations.size()), 0);
    bool row0_kept = false;
    for (std::int32_t row : r.kept_rows) {
        if (row == 0) row0_kept = true;
    }
    SIHPS_ASSERT_TRUE(row0_kept);
}

// Interaction with an existing reduction two passes later, purely through
// shared col_lo/col_hi state, with no bespoke wiring between the two
// reductions: row0's doubleton tightens y's upper to 5 in pass 1; row2's
// EXISTING (unmodified) singleton-row absorption tightens y's lower to 5
// in the SAME pass; pass 2's EXISTING (unmodified) fixed-column check then
// fixes y = 5 via ordinary substitution (not a doubleton), which in turn
// reduces row1 to a singleton on z (upper -> 15, absorbed), and THAT in
// turn empties z's last active row -- z is then fixed at 0 by the
// EXISTING (unmodified) empty-column reduction (cost 0 -> lower bound).
// The whole point of this fixture: verify the doubleton triggers this
// entire pre-existing cascade correctly, not just its own immediate
// effect.
SIHPS_TEST(presolve_doubleton_enables_a_later_ordinary_fixed_column_reduction) {
    LpProblem p;
    std::vector<Triplet> t = {
        {0, 0, 1.0}, {0, 1, 2.0}, // row0 (E): x + 2y = 10
        {1, 1, 1.0}, {1, 2, 1.0}, // row1 (L): y + z <= 20
        {2, 1, 1.0},              // row2 (G): y >= 5
    };
    p.A = CSRMatrix::from_triplets(3, 3, t);
    p.obj = {0.0, 0.0, 0.0};
    p.rhs = {10.0, 20.0, 5.0};
    p.row_types = {'E', 'L', 'G'};
    p.lower = {0.0, 0.0, 0.0};
    p.upper = {kInfinity, kInfinity, kInfinity};
    sihps::apply_default_row_bounds(p);

    auto r = presolve(p, 20, {}, /*enable_doubleton_substitution=*/true);
    SIHPS_ASSERT_TRUE(r.status == PresolveStatus::OK);

    SIHPS_ASSERT_TRUE(r.column_removed[0] == 1); // x: doubleton
    SIHPS_ASSERT_TRUE(r.column_is_doubleton[0] == 1);
    SIHPS_ASSERT_TRUE(r.column_removed[1] == 1); // y: ordinary fix
    SIHPS_ASSERT_TRUE(r.column_is_doubleton[1] == 0);
    SIHPS_ASSERT_NEAR(r.fixed_value[1], 5.0, 1e-6);
    SIHPS_ASSERT_TRUE(r.column_removed[2] == 1); // z: cascades to empty-column fix
    SIHPS_ASSERT_TRUE(r.column_is_doubleton[2] == 0);
    SIHPS_ASSERT_NEAR(r.fixed_value[2], 0.0, 1e-6);
    SIHPS_ASSERT_EQ(static_cast<int>(r.kept_columns.size()), 0);

    // Postsolve must still recover the full, mutually consistent point:
    // x=0, y=5, z=0 satisfies every ORIGINAL equation directly.
    auto full = sihps::postsolve(r, {});
    SIHPS_ASSERT_NEAR(full[0], 0.0, 1e-6);
    SIHPS_ASSERT_NEAR(full[1], 5.0, 1e-6);
    SIHPS_ASSERT_NEAR(full[2], 0.0, 1e-6);
    SIHPS_ASSERT_NEAR(full[0] + 2.0 * full[1], 10.0, 1e-6);
    SIHPS_ASSERT_TRUE(full[1] + full[2] <= 20.0 + 1e-6);
    SIHPS_ASSERT_TRUE(full[1] >= 5.0 - 1e-6);
}

// Chained doubletons: eliminating x (via row0) drops row0, which lowers
// y's active-row count to 1 -- making y itself doubleton-eligible in
// row1, in the SAME pass. x + y = 10 (x isolated) eliminates x
// (intercept=10, slope=-1); 2y + z = 4 then eliminates y in favor of z
// (intercept=2, slope=-0.5) -- z, unlike y, is NOT isolated (it also
// appears in row2, a loose z + w <= 100 that keeps both z and w alive as
// genuinely free surviving columns, so this fixture actually exercises
// postsolve reconstructing a chain, rather than everything cascading to
// fixed constants the way the fixture above deliberately does). This is
// the direct proof that reverse-order replay in postsolve resolves the
// dependency chain correctly: y's value must be known before x's can be
// computed, and y's own recorded elimination depends on z.
SIHPS_TEST(presolve_doubleton_chains_through_postsolve_in_the_correct_order) {
    LpProblem p;
    std::vector<Triplet> t = {
        {0, 0, 1.0}, {0, 1, 1.0}, // row0 (E): x + y = 10
        {1, 1, 2.0}, {1, 2, 1.0}, // row1 (E): 2y + z = 4
        {2, 2, 1.0}, {2, 3, 1.0}, // row2 (L): z + w <= 100
    };
    p.A = CSRMatrix::from_triplets(3, 4, t);
    p.obj = {0.0, 0.0, 0.0, 0.0};
    p.rhs = {10.0, 4.0, 100.0};
    p.row_types = {'E', 'E', 'L'};
    p.lower = {0.0, 0.0, 0.0, 0.0};
    p.upper = {kInfinity, kInfinity, kInfinity, kInfinity};
    sihps::apply_default_row_bounds(p);

    auto r = presolve(p, 20, {}, /*enable_doubleton_substitution=*/true);
    SIHPS_ASSERT_TRUE(r.status == PresolveStatus::OK);
    SIHPS_ASSERT_TRUE(r.column_removed[0] == 1); // x
    SIHPS_ASSERT_TRUE(r.column_removed[1] == 1); // y
    SIHPS_ASSERT_TRUE(r.column_removed[2] == 0); // z survives
    SIHPS_ASSERT_TRUE(r.column_removed[3] == 0); // w survives
    SIHPS_ASSERT_TRUE(r.column_is_doubleton[0] == 1);
    SIHPS_ASSERT_TRUE(r.column_is_doubleton[1] == 1);
    SIHPS_ASSERT_EQ(static_cast<int>(r.doubleton_eliminations.size()), 2);
    SIHPS_ASSERT_EQ(r.doubleton_eliminations[0].eliminated_col, 0);
    SIHPS_ASSERT_EQ(r.doubleton_eliminations[0].target_col, 1);
    SIHPS_ASSERT_EQ(r.doubleton_eliminations[1].eliminated_col, 1);
    SIHPS_ASSERT_EQ(r.doubleton_eliminations[1].target_col, 2);
    SIHPS_ASSERT_EQ(static_cast<int>(r.kept_columns.size()), 2);

    std::vector<double> reduced_x(r.kept_columns.size());
    for (std::size_t k = 0; k < r.kept_columns.size(); ++k) {
        if (r.kept_columns[k] == 2) reduced_x[k] = 2.0; // z = 2
        else if (r.kept_columns[k] == 3) reduced_x[k] = 5.0; // w = 5 (unused by the chain)
    }
    auto full = sihps::postsolve(r, reduced_x);
    SIHPS_ASSERT_NEAR(full[2], 2.0, 1e-9); // z
    SIHPS_ASSERT_NEAR(full[1], 1.0, 1e-9); // y = 2 + (-0.5)*2 = 1
    SIHPS_ASSERT_NEAR(full[0], 9.0, 1e-9); // x = 10 + (-1)*1 = 9
    // Cross-check directly against the ORIGINAL equations, independent of
    // the postsolve formula itself.
    SIHPS_ASSERT_NEAR(full[0] + full[1], 10.0, 1e-9);
    SIHPS_ASSERT_NEAR(2.0 * full[1] + full[2], 4.0, 1e-9);
}

// Backward compatibility: the default call signature (no
// `enable_doubleton_substitution` argument at all) must leave a fixture
// that WOULD be eliminated if flagged completely untouched.
SIHPS_TEST(presolve_doubleton_substitution_is_off_by_default) {
    LpProblem p;
    std::vector<Triplet> t = {
        {0, 0, 1.0}, {0, 1, 2.0}, // row0 (E): x + 2y = 10
        {1, 1, 1.0}, {1, 2, 1.0}, // row1 (L): y + z <= 20
    };
    p.A = CSRMatrix::from_triplets(2, 3, t);
    p.obj = {0.0, 0.0, 0.0};
    p.rhs = {10.0, 20.0};
    p.row_types = {'E', 'L'};
    p.lower = {0.0, 0.0, 0.0};
    p.upper = {kInfinity, kInfinity, kInfinity};
    sihps::apply_default_row_bounds(p);

    auto r = presolve(p); // no enable_doubleton_substitution argument
    SIHPS_ASSERT_TRUE(r.status == PresolveStatus::OK);
    SIHPS_ASSERT_TRUE(r.column_removed[0] == 0);
    SIHPS_ASSERT_EQ(static_cast<int>(r.doubleton_eliminations.size()), 0);
}

// A real bug caught on Netlib `kb2` before this shipped, pinned down here
// so it can never silently regress: every hand-derived fixture above uses
// an all-zero objective, which never exercises objective-coefficient
// redistribution at all. When x_e = intercept + slope*x_s is substituted,
// x_e's cost contributes c_e*slope*x_s to the TRUE objective -- if that is
// not folded into x_s's own coefficient in the REDUCED problem, the
// simplex silently optimizes a DIFFERENT objective while still returning
// an apparently-valid feasible point. On `kb2` this produced a reduced
// problem whose trivial all-zero starting point looked optimal in 0
// iterations, `OPTIMAL` status, objective 0.0 against a true optimum of
// -1749.9 -- exactly the "confidently wrong, not obviously broken"
// failure this project's own verification gates exist to catch, caught
// here directly by the `validate_netlib` sweep this reduction's own
// KPI-gate measurement required.
//
// x + 2y = 10 (x isolated), y + z <= 20, minimize x - y. Hand-derived:
// x = 10 - 2y, so the true objective is (10-2y) - y = 10 - 3y, minimized
// by taking y to its largest feasible value. y's own implied bound from
// x >= 0 is y <= 5 (same derivation as the fixture above); z is free at
// zero cost, so any z in [0, 20-y] is equally optimal. True optimum:
// x=0, y=5, objective=-5, independent of z.
SIHPS_TEST(presolve_doubleton_redistributes_the_eliminated_variables_objective_coefficient) {
    LpProblem p;
    std::vector<Triplet> t = {
        {0, 0, 1.0}, {0, 1, 2.0}, // row0 (E): x + 2y = 10
        {1, 1, 1.0}, {1, 2, 1.0}, // row1 (L): y + z <= 20
    };
    p.A = CSRMatrix::from_triplets(2, 3, t);
    p.obj = {1.0, -1.0, 0.0}; // minimize x - y
    p.rhs = {10.0, 20.0};
    p.row_types = {'E', 'L'};
    p.lower = {0.0, 0.0, 0.0};
    p.upper = {kInfinity, kInfinity, kInfinity};
    sihps::apply_default_row_bounds(p);

    LpSolverOptions without;
    without.enable_doubleton_substitution = false;
    const auto baseline = solve_lp(p, without);

    LpSolverOptions with;
    with.enable_doubleton_substitution = true;
    const auto reduced = solve_lp(p, with);

    SIHPS_ASSERT_TRUE(baseline.status == LpStatus::OPTIMAL);
    SIHPS_ASSERT_TRUE(reduced.status == LpStatus::OPTIMAL);
    SIHPS_ASSERT_NEAR(baseline.objective_value, -5.0, 1e-6);
    SIHPS_ASSERT_NEAR(reduced.objective_value, -5.0, 1e-6);
    SIHPS_ASSERT_NEAR(reduced.x[0], 0.0, 1e-6);
    SIHPS_ASSERT_NEAR(reduced.x[1], 5.0, 1e-6);
}
