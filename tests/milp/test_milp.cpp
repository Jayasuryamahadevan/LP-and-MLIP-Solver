#include "test_framework.hpp"

#include "io/MpsReader.hpp"
#include "lp/LpProblem.hpp"
#include "milp/MilpProblem.hpp"
#include "milp/MilpSolver.hpp"
#include "sparse/Triplet.hpp"

#include <cmath>
#include <string>
#include <vector>

using sihps::CSRMatrix;
using sihps::LpProblem;
using sihps::MilpBranchingRule;
using sihps::MilpProblem;
using sihps::MilpStatus;
using sihps::MilpSolverOptions;
using sihps::Triplet;
using sihps::VariableType;

namespace {

MilpProblem binary_knapsack() {
    LpProblem lp;
    lp.A = CSRMatrix::from_triplets(1, 2, {Triplet{0, 0, 6.0}, Triplet{0, 1, 5.0}});
    lp.obj = {-10.0, -9.0};
    lp.rhs = {7.0};
    lp.row_types = {'L'};
    lp.lower = {0.0, 0.0};
    lp.upper = {1.0, 1.0};
    sihps::apply_default_row_bounds(lp);
    return MilpProblem{std::move(lp), {VariableType::BINARY, VariableType::BINARY}};
}

MilpProblem integer_cover() {
    LpProblem lp;
    lp.A = CSRMatrix::from_triplets(1, 1, {Triplet{0, 0, 2.0}});
    lp.obj = {1.0};
    lp.rhs = {3.0};
    lp.row_types = {'G'};
    lp.lower = {0.0};
    lp.upper = {10.0};
    sihps::apply_default_row_bounds(lp);
    return MilpProblem{std::move(lp), {VariableType::INTEGER}};
}

// Single row, single structural variable, upper-bound-active ('L') row:
// 2*x1 <= 7, x1 integer in [0,100], minimize -x1 (i.e. maximize x1). LP
// optimum x1=3.5 with the row's slack nonbasic AT ITS LOWER bound (0).
// Hand-derived Gomory mixed-integer cut (docs/architecture/MILP.md \S2.2):
// x1 <= 3, the exact rounded-down bound -- verified by direct
// substitution, not merely asserted.
MilpProblem single_row_l_type_integer() {
    LpProblem lp;
    lp.A = CSRMatrix::from_triplets(1, 1, {Triplet{0, 0, 2.0}});
    lp.obj = {-1.0};
    lp.rhs = {7.0};
    lp.row_types = {'L'};
    lp.lower = {0.0};
    lp.upper = {100.0};
    sihps::apply_default_row_bounds(lp);
    return MilpProblem{std::move(lp), {VariableType::INTEGER}};
}

// 2*x1 <= 6.01: LP optimum x1 = 3.005, f_r = 0.005 -- deliberately below
// kGmiMinFractionality (0.01, src/milp/MilpSolver.cpp's separate_gmi_cuts),
// the numerical-safety floor added after gen-ip054 (bench_miplib's
// 5-instance MIPLIB set) MEASURABLY produced a 3.7e7-magnitude cut
// coefficient from a row with f_r = 1.00491e-07 -- confirmed by direct
// diagnostic instrumentation to be the 1/f_r amplification, not a
// coding bug (docs/architecture/MILP.md \S2.2).
MilpProblem single_row_l_type_integer_near_integer_fraction() {
    LpProblem lp;
    lp.A = CSRMatrix::from_triplets(1, 1, {Triplet{0, 0, 2.0}});
    lp.obj = {-1.0};
    lp.rhs = {6.01};
    lp.row_types = {'L'};
    lp.lower = {0.0};
    lp.upper = {100.0};
    sihps::apply_default_row_bounds(lp);
    return MilpProblem{std::move(lp), {VariableType::INTEGER}};
}

MilpProblem impossible_integer_equality() {
    LpProblem lp;
    lp.A = CSRMatrix::from_triplets(1, 1, {Triplet{0, 0, 2.0}});
    lp.obj = {0.0};
    lp.rhs = {1.0};
    lp.row_types = {'E'};
    lp.lower = {0.0};
    lp.upper = {1.0};
    sihps::apply_default_row_bounds(lp);
    return MilpProblem{std::move(lp), {VariableType::INTEGER}};
}

MilpProblem mixed_nonnegative_packing_row() {
    LpProblem lp;
    lp.A = CSRMatrix::from_triplets(
        1, 3, {Triplet{0, 0, 6.0}, Triplet{0, 1, 5.0}, Triplet{0, 2, 1.0}});
    lp.obj = {-10.0, -9.0, 0.0};
    lp.rhs = {7.0};
    lp.row_types = {'L'};
    lp.lower = {0.0, 0.0, 0.0};
    lp.upper = {1.0, 1.0, 10.0};
    sihps::apply_default_row_bounds(lp);
    return MilpProblem{std::move(lp),
                       {VariableType::BINARY, VariableType::BINARY, VariableType::CONTINUOUS}};
}

} // namespace

SIHPS_TEST(milp_finds_integer_optimum_with_fractional_lp_root) {
    MilpSolverOptions options;
    options.use_rounding_heuristic = false;
    options.enable_root_cover_cuts = false;
    const auto result = sihps::solve_milp(binary_knapsack(), options);

    SIHPS_ASSERT_TRUE(result.status == MilpStatus::OPTIMAL);
    SIHPS_ASSERT_TRUE(result.has_incumbent);
    SIHPS_ASSERT_NEAR(result.objective_value, -10.0, 1e-8);
    SIHPS_ASSERT_NEAR(result.x[0], 1.0, 0.0);
    SIHPS_ASSERT_NEAR(result.x[1], 0.0, 0.0);
    SIHPS_ASSERT_TRUE(result.nodes_processed >= 3);
    SIHPS_ASSERT_TRUE(result.nodes_pruned >= 1);
    SIHPS_ASSERT_TRUE(result.strong_branching_probes >= 2);
    SIHPS_ASSERT_NEAR(result.relative_gap, 0.0, 0.0);
}

SIHPS_TEST(milp_handles_general_integer_variables) {
    const auto result = sihps::solve_milp(integer_cover());

    SIHPS_ASSERT_TRUE(result.status == MilpStatus::OPTIMAL);
    SIHPS_ASSERT_NEAR(result.objective_value, 2.0, 1e-8);
    SIHPS_ASSERT_NEAR(result.x[0], 2.0, 0.0);
}

// The concrete "did MIP-specific presolve actually help" fixture: reuses
// single_row_l_type_integer() (2*x <= 7, x integer in [0,100], minimize
// -x). Its plain LP root optimum is x=3.5, fractional -- B&B must branch
// (x<=3 vs x>=4; the latter contradicts the row and prunes immediately,
// but at least one branch is still required). With integer bound rounding
// on, presolve rounds the row's own derived upper bound down to
// floor(3.5) = 3 BEFORE the root relaxation is even solved, so the root LP
// optimum is already x=3 -- integer-feasible, certifying at the root with
// zero branching. Same true optimum either way; strictly fewer nodes with
// rounding on.
SIHPS_TEST(milp_integer_bound_rounding_certifies_a_single_row_root_without_branching) {
    MilpProblem problem = single_row_l_type_integer();

    MilpSolverOptions off;
    off.use_rounding_heuristic = false;
    off.use_diving_heuristic = false;
    off.use_local_improvement = false;
    off.enable_root_cover_cuts = false;
    off.enable_root_gmi_cuts = false;
    off.enable_integer_bound_rounding = false;
    const auto without = sihps::solve_milp(problem, off);

    MilpSolverOptions on = off;
    on.enable_integer_bound_rounding = true;
    const auto with = sihps::solve_milp(problem, on);

    SIHPS_ASSERT_TRUE(without.status == MilpStatus::OPTIMAL);
    SIHPS_ASSERT_TRUE(with.status == MilpStatus::OPTIMAL);
    SIHPS_ASSERT_NEAR(without.objective_value, -3.0, 1e-8);
    SIHPS_ASSERT_NEAR(with.objective_value, -3.0, 1e-8);
    SIHPS_ASSERT_TRUE(with.nodes_processed < without.nodes_processed);
}

SIHPS_TEST(milp_reports_maximization_objective_in_original_sense) {
    auto problem = binary_knapsack();
    problem.maximize = true;
    for (double& coefficient : problem.relaxation.obj) coefficient = -coefficient;

    const auto result = sihps::solve_milp(problem);
    SIHPS_ASSERT_TRUE(result.status == MilpStatus::OPTIMAL);
    SIHPS_ASSERT_NEAR(result.objective_value, 10.0, 1e-8);
    SIHPS_ASSERT_NEAR(result.best_bound, 10.0, 1e-8);
}

// With integer bound rounding OFF, this must be proved the general way --
// by branch and bound actually exhausting the tree -- so this path stays
// covered independently of the presolve-level shortcut below.
SIHPS_TEST(milp_proves_integer_infeasibility_by_exhausting_nodes) {
    MilpSolverOptions options;
    options.enable_integer_bound_rounding = false;
    const auto result = sihps::solve_milp(impossible_integer_equality(), options);

    SIHPS_ASSERT_TRUE(result.status == MilpStatus::INFEASIBLE);
    SIHPS_ASSERT_TRUE(!result.has_incumbent);
    SIHPS_ASSERT_TRUE(result.nodes_processed >= 3);
}

// 2*x = 1 with x integer has no integer solution at all: presolve's
// integer bound rounding (default ON) proves this directly from the
// singleton row alone -- tighten_upper rounds x's upper to floor(0.5) = 0
// and tighten_lower rounds its lower to ceil(0.5) = 1, crossing (0 < 1),
// which the very next pass's column_infeasible() check catches -- with
// ZERO B&B nodes required, a strictly stronger and more efficient result
// than exhausting the tree.
SIHPS_TEST(milp_integer_bound_rounding_proves_infeasibility_without_branching) {
    const auto result = sihps::solve_milp(impossible_integer_equality());

    SIHPS_ASSERT_TRUE(result.status == MilpStatus::INFEASIBLE);
    SIHPS_ASSERT_TRUE(!result.has_incumbent);
    SIHPS_ASSERT_TRUE(result.nodes_processed <= 1);
}

SIHPS_TEST(milp_node_limit_is_not_reported_as_optimal) {
    MilpSolverOptions options;
    options.node_limit = 2;
    options.use_rounding_heuristic = false;
    options.enable_root_cover_cuts = false;
    const auto result = sihps::solve_milp(binary_knapsack(), options);

    SIHPS_ASSERT_TRUE(result.status == MilpStatus::NODE_LIMIT);
    SIHPS_ASSERT_TRUE(result.status != MilpStatus::OPTIMAL);
}

SIHPS_TEST(milp_root_cover_cut_is_valid_and_improves_the_root_relaxation) {
    MilpSolverOptions options;
    options.use_rounding_heuristic = false;
    const auto result = sihps::solve_milp(binary_knapsack(), options);

    SIHPS_ASSERT_TRUE(result.status == MilpStatus::OPTIMAL);
    SIHPS_ASSERT_TRUE(result.root_cover_cuts >= 1);
    SIHPS_ASSERT_NEAR(result.objective_value, -10.0, 1e-8);
}

SIHPS_TEST(milp_mixed_nonnegative_packing_row_gets_valid_cover_cut) {
    MilpSolverOptions options;
    options.use_rounding_heuristic = false;
    const auto result = sihps::solve_milp(mixed_nonnegative_packing_row(), options);

    SIHPS_ASSERT_TRUE(result.status == MilpStatus::OPTIMAL);
    SIHPS_ASSERT_TRUE(result.root_cover_cuts >= 1);
    SIHPS_ASSERT_NEAR(result.objective_value, -10.0, 1e-8);
}

SIHPS_TEST(milp_unit_bounded_integer_variables_get_binary_cover_cuts) {
    auto problem = binary_knapsack();
    problem.variable_types = {VariableType::INTEGER, VariableType::INTEGER};
    MilpSolverOptions options;
    options.use_rounding_heuristic = false;
    const auto result = sihps::solve_milp(problem, options);

    SIHPS_ASSERT_TRUE(result.status == MilpStatus::OPTIMAL);
    SIHPS_ASSERT_TRUE(result.root_cover_cuts >= 1);
    SIHPS_ASSERT_NEAR(result.objective_value, -10.0, 1e-8);
}

// Hand-verified against the tableau identity directly (docs/architecture/
// MILP.md \S2.2): row 2x1<=7 gives LP optimum x1=3.5 with the row's slack
// nonbasic AT ITS LOWER bound. The GMI cut derived from that row is
// exactly x1<=3 -- so with cover cuts off and a two-node budget (one pop
// to generate the cut and requeue the root, one more to re-solve it and
// find it already integral -- solve_milp counts each requeue-and-repop
// of the same node as a separate node), the root relaxation must have
// become integral after exactly one cut for OPTIMAL to be reachable at
// all, which only happens if the cut fired and is exactly this strong.
SIHPS_TEST(gmi_cut_from_l_row_slack_at_lower_bound_certifies_root_in_one_node) {
    MilpSolverOptions options;
    options.use_rounding_heuristic = false;
    options.enable_root_cover_cuts = false;
    options.enable_root_gmi_cuts = true;
    options.node_limit = 2;
    const auto result = sihps::solve_milp(single_row_l_type_integer(), options);

    SIHPS_ASSERT_TRUE(result.status == MilpStatus::OPTIMAL);
    SIHPS_ASSERT_TRUE(result.root_gmi_cuts >= 1);
    SIHPS_ASSERT_NEAR(result.objective_value, -3.0, 1e-8);
}

// Negative control for the test above: same instance, GMI cuts off. The
// same two-node budget must NOT suffice: with no cut, node 1 (the root)
// branches -- it does not resolve to an integral point -- leaving two
// unprocessed children after node 2, so the search cannot complete.
// Isolates that the cut itself, not some other mechanism, produced the
// OPTIMAL result above.
SIHPS_TEST(gmi_cut_from_l_row_is_required_for_one_node_certification) {
    MilpSolverOptions options;
    options.use_rounding_heuristic = false;
    options.enable_root_cover_cuts = false;
    options.enable_root_gmi_cuts = false;
    // This fixture is ALSO the one integer bound rounding certifies at the
    // root on its own (milp_integer_bound_rounding_certifies_a_single_row_
    // root_without_branching, above) -- disabled here so this test isolates
    // GMI cuts specifically, not that other, independent mechanism.
    options.enable_integer_bound_rounding = false;
    options.node_limit = 2;
    const auto result = sihps::solve_milp(single_row_l_type_integer(), options);

    SIHPS_ASSERT_TRUE(result.status == MilpStatus::NODE_LIMIT);
}

// Same construction as the pair above, but exercising the OTHER sign
// branch: integer_cover()'s row is 'G' (2x>=3), so at the LP optimum
// x=1.5 the row's slack is nonbasic AT ITS UPPER bound (0), not its
// lower -- this is the case an earlier, buggy version of the cut
// generator got backwards (it derived x<=1, which is invalid: it would
// cut off the true optimum x=2). Hand-verified: the correct cut is
// x>=2.
SIHPS_TEST(gmi_cut_from_g_row_slack_at_upper_bound_certifies_root_in_one_node) {
    MilpSolverOptions options;
    options.use_rounding_heuristic = false;
    options.enable_root_cover_cuts = false;
    options.enable_root_gmi_cuts = true;
    options.node_limit = 2;
    const auto result = sihps::solve_milp(integer_cover(), options);

    SIHPS_ASSERT_TRUE(result.status == MilpStatus::OPTIMAL);
    SIHPS_ASSERT_TRUE(result.root_gmi_cuts >= 1);
    SIHPS_ASSERT_NEAR(result.objective_value, 2.0, 1e-8);
}

SIHPS_TEST(gmi_cut_from_g_row_is_required_for_one_node_certification) {
    MilpSolverOptions options;
    options.use_rounding_heuristic = false;
    options.enable_root_cover_cuts = false;
    options.enable_root_gmi_cuts = false;
    // See gmi_cut_from_l_row_is_required_for_one_node_certification's own
    // comment: isolates GMI cuts from integer bound rounding, which this
    // fixture's own row (2*x >= 3) is also tight enough for on its own.
    options.enable_integer_bound_rounding = false;
    options.node_limit = 2;
    const auto result = sihps::solve_milp(integer_cover(), options);

    SIHPS_ASSERT_TRUE(result.status == MilpStatus::NODE_LIMIT);
}

// Precise, deterministic test of the numerical-safety fractionality
// floor: a row whose basic integer variable is only barely fractional
// (f_r = 0.005, below kGmiMinFractionality's 0.01) must NOT produce a
// GMI cut -- MEASURED to be exactly the mechanism that made gen-ip054
// numerically fail before this filter existed. The solver must still
// reach the correct answer via ordinary branching once the (rejected)
// row is left uncut.
SIHPS_TEST(gmi_cut_rejects_row_with_near_integer_basic_fraction) {
    MilpSolverOptions options;
    options.use_rounding_heuristic = false;
    options.enable_root_cover_cuts = false;
    options.enable_root_gmi_cuts = true;
    const auto result =
        sihps::solve_milp(single_row_l_type_integer_near_integer_fraction(), options);

    SIHPS_ASSERT_TRUE(result.status == MilpStatus::OPTIMAL);
    SIHPS_ASSERT_TRUE(result.root_gmi_cuts == 0);
    SIHPS_ASSERT_NEAR(result.objective_value, -3.0, 1e-8);
}

SIHPS_TEST(milp_maps_unbounded_pure_continuous_relaxation_to_unbounded) {
    LpProblem lp;
    lp.A = CSRMatrix::from_triplets(0, 1, {});
    lp.obj = {-1.0};
    lp.rhs = {};
    lp.row_types = {};
    lp.lower = {0.0};
    lp.upper = {sihps::kInfinity};
    MilpProblem problem{std::move(lp), {VariableType::CONTINUOUS}};
    const auto result = sihps::solve_milp(problem);
    SIHPS_ASSERT_TRUE(result.status == MilpStatus::UNBOUNDED);
}

SIHPS_TEST(mps_integer_markers_and_binary_bounds_are_preserved) {
    const auto model =
        sihps::read_mps_file(std::string(SIHPS_PROJECT_ROOT) + "/tests/data/tiny_milp.mps");
    SIHPS_ASSERT_EQ(model.col_types.size(), static_cast<std::size_t>(2));
    SIHPS_ASSERT_TRUE(model.col_types[0] == VariableType::BINARY);
    SIHPS_ASSERT_TRUE(model.col_types[1] == VariableType::INTEGER);

    const auto problem = sihps::milp_problem_from_mps(model);
    const auto result = sihps::solve_milp(problem);
    SIHPS_ASSERT_TRUE(result.status == MilpStatus::OPTIMAL);
    SIHPS_ASSERT_NEAR(result.objective_value, -10.0, 1e-8);
}

SIHPS_TEST(mps_maximize_sense_is_converted_and_reported_correctly) {
    const auto model = sihps::read_mps_file(std::string(SIHPS_PROJECT_ROOT) +
                                             "/tests/data/tiny_milp_max.mps");
    SIHPS_ASSERT_TRUE(model.objective_sense == sihps::ObjectiveSense::MAXIMIZE);
    const auto problem = sihps::milp_problem_from_mps(model);
    const auto result = sihps::solve_milp(problem);
    SIHPS_ASSERT_TRUE(result.status == MilpStatus::OPTIMAL);
    SIHPS_ASSERT_NEAR(result.objective_value, 10.0, 1e-8);
}

// --- Warm-started dual simplex for node relaxations
// (docs/architecture/LP.md \S1/\S2, MilpSolverOptions::
// warm_start_node_relaxations) -----------------------------------------
//
// Cover cuts disabled and rounding disabled here, matching
// milp_finds_integer_optimum_with_fractional_lp_root's recipe exactly:
// with the cut applied, binary_knapsack's root relaxation becomes
// integral immediately and no node ever reaches depth >= 2, which is
// where the warm path (seeded from a depth-1 node's exported basis)
// would actually be exercised.

SIHPS_TEST(milp_warm_start_matches_cold_start_on_binary_knapsack) {
    MilpSolverOptions options;
    options.use_rounding_heuristic = false;
    options.enable_root_cover_cuts = false;
    const auto cold = sihps::solve_milp(binary_knapsack(), options);

    options.warm_start_node_relaxations = true;
    const auto warm = sihps::solve_milp(binary_knapsack(), options);

    SIHPS_ASSERT_TRUE(cold.status == MilpStatus::OPTIMAL);
    SIHPS_ASSERT_TRUE(warm.status == MilpStatus::OPTIMAL);
    SIHPS_ASSERT_TRUE(warm.has_incumbent);
    SIHPS_ASSERT_NEAR(warm.objective_value, cold.objective_value, 1e-8);
    SIHPS_ASSERT_NEAR(warm.objective_value, -10.0, 1e-8);
    SIHPS_ASSERT_NEAR(warm.x[0], cold.x[0], 0.0);
    SIHPS_ASSERT_NEAR(warm.x[1], cold.x[1], 0.0);
    // Proves the path was actually exercised, not silently skipped.
    SIHPS_ASSERT_TRUE(warm.warm_started_relaxations > 0);
}

SIHPS_TEST(milp_warm_start_matches_cold_start_on_mixed_nonnegative_packing_row) {
    MilpSolverOptions options;
    options.use_rounding_heuristic = false;
    const auto cold = sihps::solve_milp(mixed_nonnegative_packing_row(), options);

    options.warm_start_node_relaxations = true;
    const auto warm = sihps::solve_milp(mixed_nonnegative_packing_row(), options);

    SIHPS_ASSERT_TRUE(cold.status == MilpStatus::OPTIMAL);
    SIHPS_ASSERT_TRUE(warm.status == MilpStatus::OPTIMAL);
    SIHPS_ASSERT_NEAR(warm.objective_value, cold.objective_value, 1e-8);
    SIHPS_ASSERT_NEAR(warm.objective_value, -10.0, 1e-8);
    SIHPS_ASSERT_TRUE(warm.root_cover_cuts >= 1);
}

SIHPS_TEST(milp_warm_start_reproduces_tiny_milp_mps_result) {
    const auto model =
        sihps::read_mps_file(std::string(SIHPS_PROJECT_ROOT) + "/tests/data/tiny_milp.mps");
    const auto problem = sihps::milp_problem_from_mps(model);
    MilpSolverOptions options;
    options.warm_start_node_relaxations = true;
    const auto result = sihps::solve_milp(problem, options);
    SIHPS_ASSERT_TRUE(result.status == MilpStatus::OPTIMAL);
    SIHPS_ASSERT_NEAR(result.objective_value, -10.0, 1e-8);
}

// A nonzero fallback count on a small, well-scaled instance would be a
// signal worth investigating, not an accepted steady state -- this pins
// the expectation that it stays at zero here.
SIHPS_TEST(milp_warm_start_fallback_counter_is_zero_or_explained) {
    MilpSolverOptions options;
    options.use_rounding_heuristic = false;
    options.enable_root_cover_cuts = false;
    options.warm_start_node_relaxations = true;
    const auto result = sihps::solve_milp(binary_knapsack(), options);

    SIHPS_ASSERT_TRUE(result.status == MilpStatus::OPTIMAL);
    SIHPS_ASSERT_TRUE(result.warm_start_verification_fallbacks == 0);
}
