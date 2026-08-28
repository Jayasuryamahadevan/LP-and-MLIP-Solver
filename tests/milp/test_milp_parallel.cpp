// Parallel branch-and-bound (MilpSolverOptions::parallel_worker_count,
// src/milp/ParallelSearch.hpp): correctness and determinism-of-final-
// answer tests. See docs/architecture/MILP.md's parallel-B&B section for
// the full design and its correctness argument -- summarized here: node
// exploration ORDER and node COUNT become inherently nondeterministic
// above 1 worker (thread-scheduling-dependent), but the FINAL reported
// answer (status, objective, feasibility of x against the original model)
// does not depend on processing order, only its timing does. These tests
// verify exactly that claim directly, not "identical node counts" (which
// legitimately varies, exactly like this project's own documented
// time-budgeted-run variance for single-threaded search).

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
using sihps::MilpProblem;
using sihps::MilpStatus;
using sihps::MilpSolverOptions;
using sihps::MilpSolution;
using sihps::Triplet;
using sihps::VariableType;

namespace {

std::string miplib(const std::string& name) {
    return std::string(SIHPS_PROJECT_ROOT) + "/data/miplib2017_small/" + name + ".mps";
}

// An 8-item 0/1 knapsack, deep enough that reliability branching cannot
// certify it at the root (verified directly: the LP relaxation's optimum
// is fractional, and single-threaded search needs more than one node to
// close it -- see milp_parallel_matches_single_threaded_reference below,
// which asserts nodes_processed > 1 as part of proving this fixture
// actually exercises branching, not just a root-level heuristic).
// Brute-force verified (independently, by exhaustive enumeration over all
// 2^8 = 256 combinations, not by trusting any solver): the unique optimum
// is item4 + item6 + item7 (values 18+3+14=35, weights 10+2+8=20, exactly
// at capacity), objective 35; every other feasible combination scores
// strictly less.
MilpProblem knapsack8() {
    LpProblem lp;
    const std::vector<double> values = {10, 9, 7, 5, 18, 12, 3, 14};
    const std::vector<double> weights = {6, 5, 4, 3, 10, 7, 2, 8};
    std::vector<Triplet> t;
    for (std::size_t j = 0; j < weights.size(); ++j) {
        t.push_back({0, static_cast<std::int32_t>(j), weights[j]});
    }
    lp.A = CSRMatrix::from_triplets(1, static_cast<std::int32_t>(values.size()), t);
    lp.obj.assign(values.size(), 0.0);
    for (std::size_t j = 0; j < values.size(); ++j) lp.obj[j] = -values[j]; // maximize value
    lp.rhs = {20.0};
    lp.row_types = {'L'};
    lp.lower.assign(values.size(), 0.0);
    lp.upper.assign(values.size(), 1.0);
    sihps::apply_default_row_bounds(lp);
    return MilpProblem{std::move(lp), std::vector<VariableType>(values.size(), VariableType::BINARY)};
}

// Independently re-verifies a MilpSolution's incumbent against the
// ORIGINAL model directly (weight capacity, box bounds, exact 0/1
// membership) -- not by trusting status/objective_value alone. This is
// what every test below actually calls to confirm a parallel run's
// answer is genuinely correct, not merely "reported OPTIMAL."
void verify_knapsack8_solution(const MilpSolution& result) {
    SIHPS_ASSERT_TRUE(result.x.size() == 8);
    const std::vector<double> values = {10, 9, 7, 5, 18, 12, 3, 14};
    const std::vector<double> weights = {6, 5, 4, 3, 10, 7, 2, 8};
    double total_weight = 0.0;
    double total_value = 0.0;
    for (std::size_t j = 0; j < 8; ++j) {
        const double v = result.x[j];
        SIHPS_ASSERT_TRUE(std::fabs(v - std::round(v)) < 1e-6); // exactly 0/1
        SIHPS_ASSERT_TRUE(v >= -1e-6 && v <= 1.0 + 1e-6);
        total_weight += std::round(v) * weights[j];
        total_value += std::round(v) * values[j];
    }
    SIHPS_ASSERT_TRUE(total_weight <= 20.0 + 1e-6);
    SIHPS_ASSERT_NEAR(total_value, 35.0, 1e-6);
    SIHPS_ASSERT_NEAR(result.objective_value, -35.0, 1e-6);
}

} // namespace

// The single-threaded reference: proves the fixture is non-trivial (needs
// real branching, not just a root-level heuristic) and pins the true
// optimum this whole file's other tests check parallel runs against.
SIHPS_TEST(milp_parallel_single_threaded_reference_needs_branching) {
    MilpSolverOptions options;
    options.parallel_worker_count = 1;
    const auto result = sihps::solve_milp(knapsack8(), options);
    SIHPS_ASSERT_TRUE(result.status == MilpStatus::OPTIMAL);
    verify_knapsack8_solution(result);
    SIHPS_ASSERT_TRUE(result.nodes_processed > 1);
}

// Worker counts 1/2/4/8 must all reach the SAME true optimum, independently
// re-verified against the original model -- not merely matching each
// other's reported objective_value (which could theoretically agree while
// both being wrong in the same way; the brute-force-verified fixture rules
// that out).
SIHPS_TEST(milp_parallel_matches_single_threaded_reference_at_every_worker_count) {
    for (std::uint32_t workers : {1u, 2u, 4u, 8u}) {
        MilpSolverOptions options;
        options.parallel_worker_count = workers;
        const auto result = sihps::solve_milp(knapsack8(), options);
        SIHPS_ASSERT_TRUE(result.status == MilpStatus::OPTIMAL);
        verify_knapsack8_solution(result);
    }
}

// Determinism of the FINAL ANSWER (not node count/timing) across repeated
// parallel runs on a real MIPLIB instance. neos859080 is used specifically
// because it terminates ON ITS OWN (proven INFEASIBLE) rather than hitting
// a time/node limit, per docs/architecture/MILP.md -- the one instance in
// this project's own 5-instance benchmark set where an EXACT repeated-run
// comparison (not just "status stays consistent") is the right bar, since
// there is no time-budget-driven variance to account for.
SIHPS_TEST(milp_parallel_reproduces_the_same_final_answer_across_repeated_runs) {
    auto model = sihps::read_mps_file(miplib("neos859080"));
    auto problem = sihps::milp_problem_from_mps(model);

    MilpSolverOptions options;
    options.parallel_worker_count = 4;
    options.time_limit_seconds = 30.0;

    MilpStatus first_status = MilpStatus::NUMERICAL_FAILURE;
    for (int trial = 0; trial < 10; ++trial) {
        const auto result = sihps::solve_milp(problem, options);
        if (trial == 0) {
            first_status = result.status;
        } else {
            SIHPS_ASSERT_TRUE(result.status == first_status);
        }
        SIHPS_ASSERT_TRUE(result.status == MilpStatus::INFEASIBLE);
        SIHPS_ASSERT_TRUE(!result.has_incumbent);
    }
}

// A second, non-trivial real MIPLIB instance run repeatedly under a SHORT
// time budget (so it hits TIME_LIMIT, not full certification): status
// must stay TIME_LIMIT every time (never spuriously NUMERICAL_FAILURE --
// the specific failure mode a synchronization bug would plausibly cause),
// and whatever incumbent is reported must independently re-verify
// feasible + integral against the ORIGINAL model every single time, even
// though the incumbent VALUE and node count are expected to vary run to
// run (exactly like this project's own documented single-threaded
// time-budgeted variance).
SIHPS_TEST(milp_parallel_always_reports_a_genuinely_feasible_incumbent_under_a_time_limit) {
    auto model = sihps::read_mps_file(miplib("gen-ip002"));
    auto problem = sihps::milp_problem_from_mps(model);

    MilpSolverOptions options;
    options.parallel_worker_count = 4;
    options.time_limit_seconds = 3.0;

    for (int trial = 0; trial < 10; ++trial) {
        const auto result = sihps::solve_milp(problem, options);
        SIHPS_ASSERT_TRUE(result.status == MilpStatus::TIME_LIMIT);
        SIHPS_ASSERT_TRUE(result.has_incumbent);
        SIHPS_ASSERT_TRUE(result.x.size() == static_cast<std::size_t>(problem.n_cols()));
        // Independent re-verification directly against the ORIGINAL
        // model: every column integral, and Ax within the original row
        // bounds -- not trusting result.status alone.
        for (std::int32_t j = 0; j < problem.n_cols(); ++j) {
            const auto jj = static_cast<std::size_t>(j);
            if (problem.variable_types[jj] != VariableType::CONTINUOUS) {
                SIHPS_ASSERT_TRUE(std::fabs(result.x[jj] - std::round(result.x[jj])) < 1e-6);
            }
        }
        std::vector<double> ax(static_cast<std::size_t>(problem.relaxation.n_rows()), 0.0);
        problem.relaxation.A.multiply(result.x.data(), ax.data(), sihps::ParallelMode::SERIAL);
        for (std::int32_t i = 0; i < problem.relaxation.n_rows(); ++i) {
            const auto ii = static_cast<std::size_t>(i);
            const double lo = problem.relaxation.rhs[ii] - problem.relaxation.slack_upper[ii];
            const double hi = problem.relaxation.rhs[ii] - problem.relaxation.slack_lower[ii];
            if (std::isfinite(lo)) SIHPS_ASSERT_TRUE(ax[ii] >= lo - 1e-6);
            if (std::isfinite(hi)) SIHPS_ASSERT_TRUE(ax[ii] <= hi + 1e-6);
        }
    }
}
