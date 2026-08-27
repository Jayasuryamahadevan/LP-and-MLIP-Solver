// Phase 1 correctness hardening (prompt.md validation LEVEL 3;
// docs/ROADMAP_STATUS.md's own KNOWN LIMITATION: "no false
// INFEASIBLE"/"no false UNBOUNDED" claims were previously only asserted
// against the Netlib set -- small, historical, and not adversarial by
// construction). Every instance here has a TRUE status known by
// construction (see adversarial_lp_generator.hpp's file header for the
// argument), and every accepted OPTIMAL result is INDEPENDENTLY
// re-verified in this file -- recomputing feasibility and the objective
// from x directly, not by trusting LpSolution's own residual fields
// (those are checked too, as a second, non-independent signal).

#include "test_framework.hpp"
#include "adversarial_lp_generator.hpp"

#include "lp/LpSolver.hpp"

#include <cmath>
#include <cstdio>
#include <random>

using namespace sihps;
using namespace sihps_test;

namespace {

// Independent of LpSolution::primal_residual/dual_residual: recomputes
// column-bound feasibility, row-bound feasibility (via A*x), and the
// objective value directly from the returned x, and compares against
// what the solver reported. A bug that made the solver's OWN residual
// computation agree with a wrong x would not be caught by checking
// primal_residual alone -- this is.
bool independently_verify_optimal(const LpProblem& p, const LpSolution& result, double feas_tol,
                                   double obj_rel_tol) {
    if (result.status != LpStatus::OPTIMAL) return false;
    if (result.x.size() != static_cast<std::size_t>(p.n_cols())) return false;
    for (double v : result.x) {
        if (!std::isfinite(v)) return false;
    }

    for (std::int32_t j = 0; j < p.n_cols(); ++j) {
        const auto jj = static_cast<std::size_t>(j);
        const double x = result.x[jj];
        if (x < p.lower[jj] - feas_tol * (1.0 + std::fabs(p.lower[jj]))) return false;
        if (x > p.upper[jj] + feas_tol * (1.0 + std::fabs(p.upper[jj]))) return false;
    }

    std::vector<double> ax(static_cast<std::size_t>(p.n_rows()), 0.0);
    if (p.n_rows() > 0) p.A.multiply(result.x.data(), ax.data());
    for (std::int32_t i = 0; i < p.n_rows(); ++i) {
        const auto ii = static_cast<std::size_t>(i);
        const double lo = p.rhs[ii] - p.slack_upper[ii];
        const double hi = p.rhs[ii] - p.slack_lower[ii];
        if (std::isfinite(lo) && ax[ii] < lo - feas_tol * (1.0 + std::fabs(lo))) return false;
        if (std::isfinite(hi) && ax[ii] > hi + feas_tol * (1.0 + std::fabs(hi))) return false;
    }

    double recomputed_obj = 0.0;
    for (std::int32_t j = 0; j < p.n_cols(); ++j) {
        recomputed_obj += p.obj[static_cast<std::size_t>(j)] * result.x[static_cast<std::size_t>(j)];
    }
    const double denom = 1.0 + std::fabs(recomputed_obj);
    if (std::fabs(recomputed_obj - result.objective_value) > obj_rel_tol * denom) return false;

    return true;
}

struct Tally {
    int total = 0;
    int correct_status = 0;
    int independently_verified = 0; // meaningful only for the OPTIMAL-expected categories
};

} // namespace

SIHPS_TEST(adversarial_feasible_bounded_lps_are_never_falsely_infeasible_or_unbounded) {
    Tally t;
    for (int trial = 0; trial < 80; ++trial) {
        std::mt19937 rng(2000u + static_cast<unsigned>(trial));
        const int n_rows = 3 + (trial % 15);
        const int n_cols = 3 + ((trial * 7) % 20);
        const auto gen = generate_feasible_bounded(rng, n_rows, n_cols);
        const LpSolution result = solve_lp(gen.problem);
        ++t.total;
        // The feasible region is closed, bounded, and nonempty by
        // construction (file header): INFEASIBLE or UNBOUNDED here is
        // unconditionally wrong, regardless of numerical difficulty.
        SIHPS_ASSERT_TRUE(result.status != LpStatus::INFEASIBLE);
        SIHPS_ASSERT_TRUE(result.status != LpStatus::UNBOUNDED);
        SIHPS_ASSERT_TRUE(result.status == LpStatus::OPTIMAL);
        if (result.status == LpStatus::OPTIMAL) {
            ++t.correct_status;
            SIHPS_ASSERT_TRUE(result.primal_residual < 1e-6);
            SIHPS_ASSERT_TRUE(independently_verify_optimal(gen.problem, result, 1e-6, 1e-6));
            ++t.independently_verified;
        }
    }
    SIHPS_ASSERT_TRUE(t.total == 80);
    SIHPS_ASSERT_TRUE(t.independently_verified == 80);
}

SIHPS_TEST(adversarial_ill_conditioned_lps_are_never_falsely_infeasible_or_unbounded) {
    Tally t;
    for (int trial = 0; trial < 60; ++trial) {
        std::mt19937 rng(3000u + static_cast<unsigned>(trial));
        const int n_rows = 3 + (trial % 12);
        const int n_cols = 3 + ((trial * 5) % 15);
        const auto gen = generate_ill_conditioned(rng, n_rows, n_cols);
        const LpSolution result = solve_lp(gen.problem);
        ++t.total;
        SIHPS_ASSERT_TRUE(result.status != LpStatus::INFEASIBLE);
        SIHPS_ASSERT_TRUE(result.status != LpStatus::UNBOUNDED);
        if (result.status == LpStatus::OPTIMAL) {
            ++t.correct_status;
            // Looser tolerance than the well-scaled category: a
            // 12-order-of-magnitude coefficient spread makes tight
            // residuals genuinely hard even for correct code, not only
            // for buggy code -- ESTABLISHED numerical-linear-algebra
            // fact (conditioning bounds achievable accuracy), not a
            // relaxation of the underlying correctness requirement.
            if (independently_verify_optimal(gen.problem, result, 1e-4, 1e-4)) {
                ++t.independently_verified;
            }
        }
    }
    // Status must never be wrong. Optimality quality is allowed a small
    // number of misses on deliberately pathological conditioning --
    // reported below rather than asserted to 100%, but still checked, so
    // a real regression shows a falling count rather than nothing.
    SIHPS_ASSERT_TRUE(t.correct_status == t.total);
    std::printf("  ill_conditioned: %d/%d independently verified within relaxed tolerance\n",
                t.independently_verified, t.total);
    SIHPS_ASSERT_TRUE(t.independently_verified >= (t.total * 9) / 10); // at least 90%
}

SIHPS_TEST(adversarial_degenerate_lps_are_never_falsely_infeasible_or_unbounded) {
    Tally t;
    for (int trial = 0; trial < 80; ++trial) {
        std::mt19937 rng(4000u + static_cast<unsigned>(trial));
        const int n_rows = 4 + (trial % 15);
        const int n_cols = 3 + ((trial * 7) % 20);
        const auto gen = generate_degenerate(rng, n_rows, n_cols);
        const LpSolution result = solve_lp(gen.problem);
        ++t.total;
        SIHPS_ASSERT_TRUE(result.status != LpStatus::INFEASIBLE);
        SIHPS_ASSERT_TRUE(result.status != LpStatus::UNBOUNDED);
        SIHPS_ASSERT_TRUE(result.status == LpStatus::OPTIMAL);
        if (result.status == LpStatus::OPTIMAL) {
            ++t.correct_status;
            SIHPS_ASSERT_TRUE(independently_verify_optimal(gen.problem, result, 1e-6, 1e-6));
            ++t.independently_verified;
        }
    }
    SIHPS_ASSERT_TRUE(t.independently_verified == 80);
}

SIHPS_TEST(adversarial_infeasible_bound_contradictions_are_always_detected) {
    int total = 0, correct = 0;
    for (int trial = 0; trial < 60; ++trial) {
        std::mt19937 rng(5000u + static_cast<unsigned>(trial));
        const int n_cols = 2 + (trial % 10);
        const auto gen = generate_infeasible_bounds(rng, n_cols);
        const LpSolution result = solve_lp(gen.problem);
        ++total;
        SIHPS_ASSERT_TRUE(result.status != LpStatus::OPTIMAL);
        SIHPS_ASSERT_TRUE(result.status != LpStatus::UNBOUNDED);
        SIHPS_ASSERT_TRUE(result.status == LpStatus::INFEASIBLE);
        if (result.status == LpStatus::INFEASIBLE) ++correct;
    }
    SIHPS_ASSERT_TRUE(correct == total);
}

SIHPS_TEST(adversarial_infeasible_row_contradictions_are_always_detected) {
    int total = 0, correct = 0;
    for (int trial = 0; trial < 60; ++trial) {
        std::mt19937 rng(6000u + static_cast<unsigned>(trial));
        const int n_cols = 2 + (trial % 10);
        const auto gen = generate_infeasible_rows(rng, n_cols);
        const LpSolution result = solve_lp(gen.problem);
        ++total;
        SIHPS_ASSERT_TRUE(result.status != LpStatus::OPTIMAL);
        SIHPS_ASSERT_TRUE(result.status != LpStatus::UNBOUNDED);
        SIHPS_ASSERT_TRUE(result.status == LpStatus::INFEASIBLE);
        if (result.status == LpStatus::INFEASIBLE) ++correct;
    }
    SIHPS_ASSERT_TRUE(correct == total);
}

SIHPS_TEST(adversarial_unbounded_directions_are_always_detected) {
    int total = 0, correct = 0;
    for (int trial = 0; trial < 60; ++trial) {
        std::mt19937 rng(7000u + static_cast<unsigned>(trial));
        const int n_rows = 2 + (trial % 10);
        const int n_cols_other = 2 + ((trial * 3) % 10);
        const auto gen = generate_unbounded(rng, n_rows, n_cols_other);
        const LpSolution result = solve_lp(gen.problem);
        ++total;
        SIHPS_ASSERT_TRUE(result.status != LpStatus::OPTIMAL);
        SIHPS_ASSERT_TRUE(result.status != LpStatus::INFEASIBLE);
        SIHPS_ASSERT_TRUE(result.status == LpStatus::UNBOUNDED);
        if (result.status == LpStatus::UNBOUNDED) ++correct;
    }
    SIHPS_ASSERT_TRUE(correct == total);
}
