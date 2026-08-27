#include "LpSolver.hpp"

#include "Scaling.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace sihps {
namespace {

// Acceptance threshold for the original-space primal check. Matches the
// tolerance Simplex certifies its own result against, so composing the
// pipeline does not silently loosen what "feasible" means.
constexpr double kFinalPrimalTol = 1e-6;

double vector_inf_norm(const std::vector<double>& v) {
    double best = 0.0;
    for (double x : v) best = std::max(best, std::fabs(x));
    return best;
}

// Row and column feasibility of `x` against the ORIGINAL model.
double original_space_primal_residual(const LpProblem& problem, const std::vector<double>& x,
                                      ParallelMode parallel_mode) {
    const std::int32_t m = problem.n_rows();
    const std::int32_t n = problem.n_cols();

    std::vector<double> ax(static_cast<std::size_t>(m), 0.0);
    if (n > 0 && m > 0) problem.A.multiply(x.data(), ax.data(), parallel_mode);

    double row_violation = 0.0;
    for (std::int32_t i = 0; i < m; ++i) {
        const auto ii = static_cast<std::size_t>(i);
        const double lo = problem.rhs[ii] - problem.slack_upper[ii];
        const double hi = problem.rhs[ii] - problem.slack_lower[ii];
        if (std::isfinite(lo)) row_violation = std::max(row_violation, lo - ax[ii]);
        if (std::isfinite(hi)) row_violation = std::max(row_violation, ax[ii] - hi);
    }
    double bound_violation = 0.0;
    for (std::int32_t j = 0; j < n; ++j) {
        const auto jj = static_cast<std::size_t>(j);
        bound_violation = std::max(bound_violation, problem.lower[jj] - x[jj]);
        bound_violation = std::max(bound_violation, x[jj] - problem.upper[jj]);
    }
    row_violation = std::max(0.0, row_violation);
    bound_violation = std::max(0.0, bound_violation);

    const double rhs_norm = vector_inf_norm(problem.rhs);
    return std::max(row_violation / (1.0 + rhs_norm), bound_violation);
}

// Runs GPU PDLP on `target` and returns its primal point in that problem's
// own column space, or false if it did not converge.
//
// SCALING IS PART OF THE ALGORITHM HERE, not a tidiness measure. A
// first-order method's convergence rate depends directly on the
// conditioning of A, so the matrix is equilibrated with Ruiz and then
// preconditioned with Pock-Chambolle (Scaling.hpp explains why both, and
// why in that order). Skipping either turns models that converge in
// thousands of iterations into models that do not converge at all.
bool run_first_order(const LpProblem& target, const PdlpParams& params, std::vector<double>& x_out,
                     PdlpStats& stats_out) {
    const ScaleFactors ruiz = compute_ruiz_scaling(target.A);
    const CSRMatrix a_ruiz = apply_ruiz_scaling(target.A, ruiz);
    const ScaleFactors pc = compute_pock_chambolle_scaling(a_ruiz);
    const ScaleFactors scale = compose_scaling(ruiz, pc);
    const CSRMatrix a_scaled = apply_ruiz_scaling(target.A, scale);

    const auto n = static_cast<std::size_t>(target.n_cols());
    const auto m = static_cast<std::size_t>(target.n_rows());

    // x = C x', so costs multiply by the column scale and bounds divide by
    // it; the objective VALUE is invariant under this change of variable.
    std::vector<double> cost(n), lower(n), upper(n);
    for (std::size_t j = 0; j < n; ++j) {
        const double c = scale.col_scale[j];
        cost[j] = target.obj[j] * c;
        lower[j] = target.lower[j] / c;
        upper[j] = target.upper[j] / c;
    }

    // A x + s = rhs with slack_lower <= s <= slack_upper is exactly
    // rhs - slack_upper <= A x <= rhs - slack_lower.
    std::vector<double> row_lower(m), row_upper(m);
    for (std::size_t i = 0; i < m; ++i) {
        const double r = scale.row_scale[i];
        row_lower[i] = r * (target.rhs[i] - target.slack_upper[i]);
        row_upper[i] = r * (target.rhs[i] - target.slack_lower[i]);
    }

    GpuPdlp pdlp(a_scaled, cost, lower, upper, row_lower, row_upper);
    std::vector<double> x_scaled, y_scaled;
    stats_out = pdlp.solve(params, x_scaled, y_scaled);

    x_out.assign(n, 0.0);
    for (std::size_t j = 0; j < n; ++j) x_out[j] = x_scaled[j] * scale.col_scale[j];
    return stats_out.converged;
}

// The original-space feasibility gate of NUMERICS.md 6, factored out so
// that LpMethod::HYBRID can apply it to a candidate BEFORE committing to
// it. Returns the residual; the caller compares against kFinalPrimalTol.
//
// This exists because of a measured defect. When the first-order path leads
// and converges on its own KKT test, that is a claim about the SCALED,
// PRESOLVED problem. It can still fail the original-space gate -- `dfl001`
// with a fixed step size converged to a relative objective error of 4.2e-07
// and was then rejected here. In the first version of the structure-aware
// hybrid that rejection was terminal: the simplex was never tried, and a
// model the simplex solves comfortably was reported NUMERICAL_FAILURE.
//
// Convergence and verification are different questions, and a fallback
// keyed only on the first is not a fallback.
double candidate_primal_residual(const LpProblem& problem, const PresolveResult& reduction,
                                  bool use_presolve, const std::vector<double>& reduced_x,
                                  std::int32_t n, ParallelMode parallel_mode) {
    std::vector<double> x = use_presolve ? postsolve(reduction, reduced_x) : reduced_x;
    if (static_cast<std::int32_t>(x.size()) != n) {
        x.resize(static_cast<std::size_t>(n), 0.0);
    }
    return original_space_primal_residual(problem, x, parallel_mode);
}

} // namespace

LpSolution solve_lp(const LpProblem& problem, const LpSolverOptions& options) {
    LpSolution solution;
    const std::int32_t n = problem.n_cols();

    PresolveResult reduction;
    const LpProblem* target = &problem;

    if (options.use_presolve) {
        const auto t0 = std::chrono::steady_clock::now();
        reduction = presolve(problem, 20, options.integer_columns);
        solution.presolve_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        if (reduction.status == PresolveStatus::INFEASIBLE) {
            solution.status = LpStatus::INFEASIBLE;
            return solution;
        }
        if (reduction.status == PresolveStatus::UNBOUNDED) {
            solution.status = LpStatus::UNBOUNDED;
            return solution;
        }
        solution.presolve_removed_rows = reduction.removed_rows();
        solution.presolve_removed_cols = reduction.removed_cols();
        target = &reduction.reduced;
    }

    const auto t1 = std::chrono::steady_clock::now();

    std::vector<double> reduced_x;
    if (target->n_cols() == 0) {
        // Presolve fixed every variable. There is nothing left to optimize,
        // but the result still has to clear the same verification gate as
        // any other -- it is not exempt just because no simplex ran.
        solution.status = LpStatus::OPTIMAL;
    } else if (options.method == LpMethod::FIRST_ORDER) {
        solution.used_first_order = true;
        PdlpStats stats;
        bool ok = false;
        try {
            ok = run_first_order(*target, options.pdlp, reduced_x, stats);
        } catch (const std::exception&) {
            ok = false;
        }
        solution.pdlp = stats;
        solution.iterations = stats.iterations;
        // Converging on PDLP's own relative KKT triple is a claim about the
        // scaled reduced problem. It is NOT the claim this function makes,
        // which is about the ORIGINAL model -- so the status is provisional
        // here and only becomes OPTIMAL after the same original-space gate
        // every other result passes through, below.
        solution.status = ok ? LpStatus::OPTIMAL : LpStatus::ITERATION_LIMIT;
    } else if (options.method == LpMethod::HYBRID) {
        // ---------------------------------------------------------------
        // HYBRID: structure-aware lead, with the other method as fallback
        // ---------------------------------------------------------------
        // Two methods that fail on disjoint sets of models, and a cheap
        // feature that says which is likely to win. Whichever leads runs
        // under a budget; if it does not return a usable answer the other
        // one gets the instance. A wrong prediction therefore costs one
        // budget, never a wrong answer.
        //
        // The predictor is row count after presolve, and it was measured
        // rather than assumed -- see LpSolverOptions for the data and for
        // why nonzero count is the wrong feature.
        //
        // THIS IS THE THIRD DESIGN. The first raced the two solvers
        // concurrently and lost to CPU contention (92/93 -> 91/93, `woodw`
        // 0.174 s -> 5.6 s). The second always led with the simplex, which
        // is correct on Netlib and badly wrong at scale: it picks the 18x
        // slower path on `ken-11` by construction, and costs 419.9 s across
        // the Kennington set where choosing per instance costs 56.0 s.
        const bool first_order_leads =
            target->n_rows() >= options.hybrid_first_order_row_threshold;

        PdlpStats pdlp_stats;
        std::vector<double> pdlp_x;
        bool pdlp_ok = false;

        // Runs the first-order solver under its budget. Kept as a lambda so
        // the two orderings below differ only in sequence, not in behaviour.
        auto try_first_order = [&]() {
            PdlpParams fo = options.pdlp;
            fo.eps_optimal = options.hybrid_first_order_eps;
            if (options.hybrid_first_order_budget_seconds > 0.0) {
                fo.time_limit_seconds = options.hybrid_first_order_budget_seconds;
            }
            try {
                pdlp_ok = run_first_order(*target, fo, pdlp_x, pdlp_stats);
            } catch (const std::exception&) {
                pdlp_ok = false;
            }
            // Converging is not the same as being acceptable. PDLP's KKT
            // test is a statement about the scaled, presolved problem; the
            // caller's problem is what actually has to be satisfied. Check
            // it HERE, while the simplex is still available as a fallback,
            // rather than after the point of no return.
            if (pdlp_ok) {
                const double residual = candidate_primal_residual(
                    problem, reduction, options.use_presolve, pdlp_x, n, options.parallel_mode);
                if (!(residual <= kFinalPrimalTol)) pdlp_ok = false;
            }
            return pdlp_ok;
        };

        LpResult lp;
        bool simplex_ran = false;
        auto try_simplex = [&]() {
            Simplex simplex(*target, options.backend, options.use_ruiz_scaling,
                             options.pricing_rule, options.algorithm, options.parallel_mode);
            simplex.set_time_budget(options.hybrid_simplex_budget_seconds);
            lp = simplex.solve();
            simplex_ran = true;
            // INFEASIBLE and UNBOUNDED are proofs. Handing either to an
            // approximate method would trade a certificate for a guess.
            return lp.status == LpStatus::OPTIMAL || lp.status == LpStatus::INFEASIBLE ||
                   lp.status == LpStatus::UNBOUNDED;
        };

        bool solved_by_first_order = false;
        if (first_order_leads) {
            solved_by_first_order = try_first_order();
            if (!solved_by_first_order) try_simplex();
        } else {
            if (!try_simplex()) solved_by_first_order = try_first_order();
        }

        if (solved_by_first_order) {
            solution.used_first_order = true;
            solution.first_order_fallback_used = !first_order_leads;
            solution.pdlp = pdlp_stats;
            solution.iterations = pdlp_stats.iterations;
            // Provisional. The original-space gate below is the authority
            // on whether this is acceptable, exactly as for a simplex
            // result -- nothing is trusted because a solver stopped.
            solution.status = LpStatus::OPTIMAL;
            reduced_x = std::move(pdlp_x);
            if (simplex_ran) solution.refactorizations = lp.refactorizations;
        } else if (simplex_ran) {
            solution.status = lp.status;
            solution.iterations =
                lp.phase1_iterations + lp.phase2_iterations + lp.dual_iterations;
            solution.used_dual_simplex = lp.used_dual_simplex;
            solution.dual_iterations = lp.dual_iterations;
            solution.refactorizations = lp.refactorizations;
            solution.pricing_seconds = lp.pricing_seconds;
            solution.simplex_profile = lp.profile;
            solution.dual_residual = lp.dual_residual;
            solution.dual_residual_is_reduced_space = options.use_presolve;
            solution.pdlp = pdlp_stats;
            reduced_x = std::move(lp.x);
        } else {
            // First-order led, failed, and the simplex was never reached.
            // Only possible if the simplex constructor threw, so report the
            // first-order attempt honestly rather than inventing a status.
            solution.pdlp = pdlp_stats;
            solution.status = LpStatus::NUMERICAL_FAILURE;
        }
    } else {
        Simplex simplex(*target, options.backend, options.use_ruiz_scaling, options.pricing_rule,
                         options.algorithm, options.parallel_mode);
        simplex.set_time_budget(options.simplex_time_budget_seconds);
        LpResult lp = simplex.solve();
        solution.status = lp.status;
        solution.iterations = lp.phase1_iterations + lp.phase2_iterations + lp.dual_iterations;
        solution.used_dual_simplex = lp.used_dual_simplex;
        solution.dual_iterations = lp.dual_iterations;
        solution.refactorizations = lp.refactorizations;
        solution.pricing_seconds = lp.pricing_seconds;
        solution.simplex_profile = lp.profile;
        solution.dual_residual = lp.dual_residual;
        solution.dual_residual_is_reduced_space = options.use_presolve;
        reduced_x = std::move(lp.x);

    }
    solution.solve_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();

    if (solution.status != LpStatus::OPTIMAL) {
        return solution;
    }

    solution.x = options.use_presolve ? postsolve(reduction, reduced_x) : reduced_x;
    if (static_cast<std::int32_t>(solution.x.size()) != n) {
        solution.x.resize(static_cast<std::size_t>(n), 0.0);
    }

    // Objective from the ORIGINAL cost vector and the reconstructed full
    // solution. Computing it this way means presolve needs no separate
    // objective-offset bookkeeping for the columns it fixed: their
    // contribution is already present in x.
    solution.objective_value = 0.0;
    for (std::int32_t j = 0; j < n; ++j) {
        solution.objective_value +=
            problem.obj[static_cast<std::size_t>(j)] * solution.x[static_cast<std::size_t>(j)];
    }

    // The hard invariant (docs/architecture/NUMERICS.md \S6) applied to the
    // whole pipeline: OPTIMAL survives only if the reconstructed solution is
    // feasible for the problem the CALLER posed. A presolve reduction that
    // was not exactly invertible fails here rather than being reported as a
    // clean optimum.
    solution.primal_residual =
        original_space_primal_residual(problem, solution.x, options.parallel_mode);
    if (!(solution.primal_residual <= kFinalPrimalTol)) {
        solution.status = LpStatus::NUMERICAL_FAILURE;
    }
    return solution;
}

} // namespace sihps
