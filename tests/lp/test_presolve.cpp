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
