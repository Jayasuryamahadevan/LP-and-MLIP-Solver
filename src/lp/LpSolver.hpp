#pragma once

#include "../cuda/GpuPdlp.hpp"
#include "LpProblem.hpp"
#include "Presolve.hpp"
#include "Simplex.hpp"

#include <cstdint>
#include <vector>

namespace sihps {

// Which family of algorithm solves the LP.
//
// SIMPLEX is the default and returns an exact vertex solution certified by
// the residual gate in NUMERICS.md 6.
//
// FIRST_ORDER runs GPU PDLP (src/cuda/GpuPdlp.hpp). It is the ONLY place in
// this engine where the GPU beats the CPU, and the reason is structural
// rather than a matter of kernel quality: a simplex iteration needs a host
// decision (which column enters) and therefore a device synchronize every
// iteration, while PDHG needs none and can queue hundreds of iterations per
// sync. docs/architecture/CPU_GPU.md 4 records the measurement that killed
// GPU simplex; 6 records what this path does instead.
//
// The trade is real and is not hidden: a first-order method converges
// quickly to moderate accuracy and slowly to high accuracy, and it returns
// an interior-ish point rather than a vertex. Both methods are held to the
// SAME original-space verification gate, so a FIRST_ORDER result is never
// reported OPTIMAL on the strength of its own iteration count.
// SIMPLEX and FIRST_ORDER pin the algorithm. HYBRID runs the simplex and
// falls back to the first-order solver only when the simplex fails to
// return a verified optimum.
//
// The fallback exists because the two methods fail on disjoint sets of
// models, and neither failure is predictable in advance
// (docs/architecture/PDLP.md 5). On the Netlib feasible set at a
// 20,000-row cap the simplex solves 92 of 93 and abandons `dfl001` at its
// iteration limit after 792 s; the first-order path solves `dfl001` to an
// objective error of 1.4e-07 in about 2 s. Conversely the first-order path
// stalls on six models the simplex disposes of in under 5 s. Running the
// simplex first and falling back costs nothing on the 92 it already solves
// and converts the remaining failure into a pass.
enum class LpMethod { SIMPLEX, FIRST_ORDER, HYBRID };

struct LpSolverOptions {
    bool use_presolve = true;
    LpMethod method = LpMethod::SIMPLEX;
    PdlpParams pdlp;

    // HYBRID only: how long the simplex may run before the first-order
    // solver is given the instance. Every instance in the Netlib feasible
    // set that the simplex solves at all solves within 17 s measured
    // single-process, so 30 s leaves ample headroom while bounding what a
    // genuine stall costs -- `dfl001` runs 784 s to its iteration limit
    // without it.
    double hybrid_simplex_budget_seconds = 30.0;

    // Applies to LpMethod::SIMPLEX. Zero (the default) means unlimited,
    // which is what the validation sweeps use -- a budget that silently
    // truncates a solve would turn a slow instance into a false
    // ITERATION_LIMIT and corrupt the pass rate. Callers that must bound
    // wall time (the cross-method validator, where a single Kennington
    // model can otherwise run for hours) set it explicitly.
    double simplex_time_budget_seconds = 0.0;

    // HYBRID only: tolerance for the first-order fallback. Looser than the
    // FIRST_ORDER default because a first-order method's last digit is
    // disproportionately expensive; the original-space verification gate,
    // not this number, decides whether a result is accepted.
    double hybrid_first_order_eps = 1e-7;

    // HYBRID only: at or above this many rows (AFTER presolve) the
    // first-order solver leads and the simplex becomes the fallback; below
    // it, the order is reversed.
    //
    // MEASURED, not guessed. Across the 21 Kennington/QAP models, sorted by
    // row count, the first-order path wins every instance from 4,350 rows
    // upward and the simplex wins most below 3,000. Row count is the right
    // feature and nonzero count is not: `osa-07` has 143,694 nonzeros and
    // the simplex wins 2.2x, while `ken-11` has 49,058 and the first-order
    // path wins 18x. That matches the mechanics -- a simplex iteration
    // solves against an m x m basis and its iteration count grows with m,
    // whereas a PDHG iteration costs the same 57-111 us almost regardless
    // of size.
    //
    // Evaluated against an oracle that picks the better method per
    // instance (56.0 s): leading with the simplex everywhere costs 419.9 s,
    // this rule costs 67.8 s. A threshold of 4,000 costs 117.9 s because
    // `qap12` (3,192 rows) is solved only by the first-order path and would
    // fall on the wrong side.
    std::int32_t hybrid_first_order_row_threshold = 3000;

    // HYBRID only: wall-clock budget for the first-order solver when it
    // leads. Every first-order win in the measured set finishes inside
    // 12 s, so this bounds a misprediction at roughly one simplex budget
    // rather than at the 500,000-iteration default.
    double hybrid_first_order_budget_seconds = 30.0;
    bool use_ruiz_scaling = true;
    PricingBackend backend = PricingBackend::CPU;
    PricingRule pricing_rule = PricingRule::DEVEX;
    // Controls deterministic CPU inner loops. Use SERIAL when the caller
    // parallelizes independent domains/subproblems; use PARALLEL when this
    // call owns the machine and should parallelize even below the automatic
    // size gate. AUTO is safe for ordinary single-domain solves.
    ParallelMode parallel_mode = ParallelMode::AUTO;

    // AUTO resolves to the primal two-phase path on a cold start -- every
    // solve today -- per docs/architecture/LP.md \S2's decision table and
    // the cold-start measurement in \S2.1. PRIMAL and DUAL pin the choice,
    // which is what benchmarks/bench_lp_algorithm.cpp needs to compare them,
    // and what the future warm-started MILP caller will need to request the
    // dual path deliberately.
    LpAlgorithm algorithm = LpAlgorithm::AUTO;

    // Threaded straight through to presolve()'s own `integer_columns`
    // parameter (see Presolve.hpp). Empty by default, so an ordinary LP
    // caller is completely unaffected -- this exists for MilpSolver's node
    // relaxations, which DO know, from the original model, which columns
    // are integer-restricted, and opt in explicitly rather than this engine
    // inferring it (see MilpProblem.hpp's own note on why the LP engine
    // does not see integrality metadata by default).
    std::vector<char> integer_columns;

    // Threaded straight through to presolve()'s own
    // `enable_doubleton_substitution` parameter (see Presolve.hpp). False
    // by default pending a KPI-gate benchmark, matching this project's own
    // standing rule for every optimization shipped this session.
    bool enable_doubleton_substitution = false;

    // Threaded straight through to presolve()'s own `enable_gcd_tightening`
    // parameter (see Presolve.hpp). False by default pending a KPI-gate
    // benchmark, same rule as `enable_doubleton_substitution` above. Has no
    // effect unless `integer_columns` is also populated.
    bool enable_gcd_tightening = false;
};

struct LpSolution {
    LpStatus status = LpStatus::NUMERICAL_FAILURE;
    double objective_value = 0.0;
    std::vector<double> x; // ORIGINAL column space, always

    int iterations = 0;
    int refactorizations = 0;
    double presolve_seconds = 0.0;
    double solve_seconds = 0.0;

    // Cumulative time inside pricing alone -- reduced-cost computation plus
    // the entering-variable search, plus the Devex pivot row and weight
    // update when that rule is active. Reported separately because it is
    // the ONLY stage that differs between PricingBackend::CPU and ::GPU, so
    // it is what an honest backend comparison has to isolate
    // (benchmarks/bench_pricing_backend.cpp): a speedup measured on
    // solve_seconds alone would be diluted by the factorization and ratio
    // test, which are identical on both backends.
    double pricing_seconds = 0.0;

    // Which algorithm produced the reported result, and how many dual
    // iterations were spent (nonzero even on a fallback, since that work
    // was really performed -- see Simplex::solve).
    bool used_dual_simplex = false;
    int dual_iterations = 0;

    std::int32_t presolve_removed_rows = 0;
    std::int32_t presolve_removed_cols = 0;

    // Populated only when options.method == FIRST_ORDER. `host_syncs` in
    // particular is worth reading: it is the quantity the GPU path exists
    // to keep small, and a regression there would show up as a slowdown
    // with no other symptom.
    bool used_first_order = false;
    // True when HYBRID actually had to fall back -- i.e. the simplex did
    // not produce a verified optimum. Reported so a benchmark can tell a
    // free fallback from one that was exercised.
    bool first_order_fallback_used = false;
    PdlpStats pdlp;

    // Per-stage simplex cost breakdown (SIMPLEX method only).
    SimplexProfile simplex_profile;

    // Primal residual is measured against the ORIGINAL model -- row bounds
    // and column bounds as the caller stated them -- so it certifies the
    // whole chain (presolve, scaling, simplex, postsolve) rather than any
    // one stage's internal view of itself.
    double primal_residual = 0.0;

    // Dual residual is measured on whatever problem the simplex actually
    // solved. With presolve enabled that is the REDUCED problem: this
    // engine does not yet reconstruct duals through presolve, so dual
    // feasibility is certified for the reduction, and the argument that
    // optimality carries back to the original model rests on the
    // correctness of the reductions rather than on a second numerical
    // check. Stated here rather than left implicit -- see
    // docs/architecture/NUMERICS.md \S6.
    double dual_residual = 0.0;
    bool dual_residual_is_reduced_space = false;
};

// Full LP pipeline: presolve -> scale -> simplex -> postsolve -> verify in
// original space (docs/architecture/SYSTEM.md \S1). This is the entry point
// callers should use; Simplex on its own solves whatever problem it is
// handed and knows nothing about presolve.
LpSolution solve_lp(const LpProblem& problem, const LpSolverOptions& options = {});

} // namespace sihps
