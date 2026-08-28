#pragma once

#include "MilpProblem.hpp"
#include "../lp/LpSolver.hpp"

#include <cstdint>
#include <vector>

namespace sihps {

enum class MilpStatus {
    OPTIMAL,
    INFEASIBLE,
    UNBOUNDED,
    UNBOUNDED_RELAXATION,
    NODE_LIMIT,
    TIME_LIMIT,
    NUMERICAL_FAILURE
};

enum class MilpBranchingRule { MOST_FRACTIONAL, PSEUDOCOST, RELIABILITY };

struct MilpSolverOptions {
    // MILP bounds must be certified LP optima. The solver therefore forces
    // SIMPLEX for relaxations even if a caller's general LP preference is
    // HYBRID or FIRST_ORDER; approximate first-order points are never used
    // as proof bounds.
    LpSolverOptions lp_options;

    std::uint64_t node_limit = 0; // zero means unlimited
    double time_limit_seconds = 0.0; // zero means unlimited
    double integrality_tolerance = 1e-7;
    double feasibility_tolerance = 1e-6;
    double objective_tolerance = 1e-8;
    bool use_rounding_heuristic = true;
    // Deterministic LP diving only proposes incumbents; it never changes
    // node pruning or optimality certification. It is limited so a failed
    // dive cannot consume an unbounded fraction of the B&B budget.
    bool use_diving_heuristic = true;
    std::uint32_t diving_max_depth = 32;
    std::uint32_t diving_max_lp_relaxations = 64;
    bool use_local_improvement = true;
    std::uint32_t local_improvement_passes = 3;
    std::uint32_t local_improvement_max_trials = 128;
    // RENS (Berthold 2014): root-only, one restricted LP solve -- fixes
    // every already-integral relaxation column, bounds every fractional
    // one to {floor, ceil}, and checks whether the single re-solve is
    // integer-feasible outright. Distinct from both heuristics above (see
    // src/milp/MilpSolver.cpp's attempt_rens for the exact difference).
    // Default false pending a KPI-gate benchmark on bench_miplib, matching
    // this project's own rule (enable_root_gmi_cuts, and
    // enable_integer_bound_rounding before its own KPI gate cleared): a
    // heuristic ships off by default until MEASURED to help.
    bool use_rens_heuristic = false;
    MilpBranchingRule branching_rule = MilpBranchingRule::RELIABILITY;
    std::uint32_t reliability_threshold = 2;
    std::uint32_t strong_branching_candidates = 4;
    bool enable_root_cover_cuts = true;
    std::uint32_t max_root_cover_cuts = 64;
    double cut_violation_tolerance = 1e-7;

    // Gomory mixed-integer cuts (Gomory 1960; Wolsey, "Integer
    // Programming", 1998, Ch.5; Marchand & Wolsey, "Aggregation and mixed
    // integer rounding to solve MIPs", Math. Programming 91(1), 2001;
    // docs/architecture/MILP.md \S2.2), separated once at the root from the
    // final simplex tableau. Unlike cover cuts (binary knapsack rows
    // only), GMI cuts apply to any row with a fractional integer-
    // restricted basic variable, including general-integer variables and
    // non-knapsack-shaped rows -- the two families are complementary, not
    // redundant. Default false pending a KPI-gate benchmark on
    // bench_miplib, mirroring warm_start_node_relaxations above.
    bool enable_root_gmi_cuts = false;
    std::uint32_t max_root_gmi_cuts = 64;

    // MIP-specific presolve: rounds any bound presolve derives for an
    // integer-restricted column inward to the nearest integer (docs/
    // architecture/MILP.md \S1.4a; src/lp/Presolve.{hpp,cpp}'s
    // `integer_columns` parameter). Deliberately NOT called "probing" --
    // that word is already used below for reliability branching's own
    // strong-branching LP probes, a different mechanism. Unconditionally
    // sound (never excludes an integer-feasible point) and needs no new
    // postsolve machinery, unlike most presolve reductions.
    //
    // MEASURED, default true (docs/architecture/MILP.md \S1.4a):
    // bench_miplib on the 5-instance set, single process, nothing else
    // running, 60s budget each, flag off vs. on: every instance's node
    // count moved the same direction (down) or stayed flat, none regressed
    // -- `neos859080` (the one instance that terminates on its own rather
    // than the time limit, hence the only one where node count is
    // meaningful to compare exactly) went from 331 to 95 nodes (-71%) and
    // 0.861s to 0.614s wall-clock, reproduced bit-identically on a repeat
    // run. The 4 time-limited instances (gen-ip002/gen-ip054/markshare2/
    // pk1) show the same run-to-run node-count variance already documented
    // elsewhere in this project's history for time-budgeted runs (not a
    // determinism concern), but every measured run had FEWER nodes with
    // rounding on than off, never more. A `validate_netlib` sweep (90/90,
    // identical iteration count to the pre-change baseline) confirms zero
    // effect on plain LP callers, which never populate `integer_columns`.
    bool enable_integer_bound_rounding = true;

    // GCD-based row tightening (docs/architecture/MILP.md \S1.4b;
    // src/lp/Presolve.{hpp,cpp}'s `enable_gcd_tightening` parameter): for a
    // row where every active coefficient is integer and every active
    // column is integer-restricted, tightens the row's bounds to the
    // nearest reachable multiple of gcd(|a_j|), detecting a row-level
    // infeasibility directly when no such multiple exists. Built against
    // this project's own stated next roadmap item (docs/ROADMAP_STATUS.md,
    // "coefficient tightening / GCD-based reductions"), aimed at
    // `markshare2`-class rows: large integer-valued coefficients over
    // 0-1/general-integer columns. Unconditionally sound, like
    // `enable_integer_bound_rounding` above, and shares that flag's
    // `integer_columns` mask (populated whenever either flag is set --
    // see below). Default false pending a KPI-gate benchmark on
    // bench_miplib, matching this project's own rule for every new
    // reduction.
    bool enable_gcd_tightening = false;

    // Warm-started dual simplex for node relaxations (docs/architecture/
    // LP.md \S1/\S2, MILP.md's stated prerequisite): a non-root node
    // solves its LP relaxation by seating its parent's terminal basis and
    // repairing primal feasibility, instead of a full cold solve_lp call.
    // Default false so every existing caller/test/benchmark is unaffected
    // until a KPI comparison (CLAUDE_OPUS_SOLVER_ROADMAP.md's own rule)
    // shows a net win on the MIPLIB benchmark; only then does a follow-up
    // change flip this default. Root and its direct children never warm
    // start (the root solve goes through solve_lp's presolve, which a
    // warm basis is not guaranteed to remain valid under).
    bool warm_start_node_relaxations = false;

    // Caps how many parent bases are retained at once for warm starting.
    //
    // Every OPEN node carries a shared_ptr to its parent's terminal basis
    // (SearchNode::parent_basis), so live-basis memory scales with FRONTIER
    // SIZE -- not with tree depth, and not with the number of nodes already
    // processed. On an instance whose LP bound prunes almost nothing the
    // frontier grows without limit, and so did this: MEASURED on markshare2
    // (dual bound pinned at 0, so essentially nothing is ever pruned) at
    // ~150 MB/s, reaching 13.5 GB in 90 s at 8 workers, against ~10 MB/s for
    // the same run with warm starting off. An earlier 16-worker run reached
    // 17 GB RSS and had to be killed.
    //
    // Once this many bases are live, no NEW one is retained and those nodes
    // take the cold-solve path instead -- already the tested behaviour
    // whenever a parent basis is absent (see
    // milp_warm_start_fallback_counter_is_zero_or_explained). So this
    // degrades a heuristic speedup under memory pressure; it never changes
    // which nodes are pruned, the final status, or the reported objective.
    // Zero means unlimited (the pre-cap behaviour).
    std::uint64_t max_live_warm_start_bases = 100000;

    // Opt-in exact meet-in-the-middle path for "binary system with unit
    // slacks" models -- the market-split / multi-knapsack shape
    // (src/milp/ExactBinarySplit.hpp has the full structural contract and
    // the argument for why it exists). Off by default and gated on
    // STRUCTURE, never on an instance name: a model that is not exactly
    // that shape, or that would exceed the memory budget below, silently
    // falls through to ordinary branch-and-bound.
    //
    // It exists because LP-based B&B is provably the wrong tool for this
    // shape: the relaxation attains slack 0 fractionally, so the dual bound
    // sits at 0 and never moves. MEASURED on markshare2: bound exactly
    // 0.00000000 after 8.68M nodes. The enumeration is complete, so its
    // answer is a proof rather than a best effort -- but it is exponential
    // in n/2 and therefore never enabled implicitly.
    bool enable_exact_binary_split = false;
    std::uint64_t exact_binary_split_memory_bytes = 6ull << 30;
    std::uint32_t exact_binary_split_threads = 0; // 0 = hardware_concurrency/2

    // Parallel branch-and-bound (src/milp/ParallelSearch.hpp,
    // docs/architecture/MILP.md's parallel-B&B section): the root node is
    // always processed serially (its cuts/RENS/local-improvement are
    // root-only heuristics regardless of this setting); every node after
    // that is pulled from a shared best-bound queue by a fixed pool of
    // worker threads, each solving its own node's LP relaxation with
    // LpSolverOptions::parallel_mode forced to SERIAL (avoiding
    // oversubscription against this outer, node-level parallelism -- see
    // ParallelMode's own doc comment, which already anticipates exactly
    // this scenario: "use SERIAL when the caller parallelizes independent
    // domains/subproblems"). Tree control (node selection, branching,
    // incumbent management, pruning) stays entirely CPU-resident, matching
    // this project's own architecture rule; no GPU pricing backend is
    // permitted for any worker when this is > 1 (GPU pricing under
    // concurrent host threads has not been verified safe and is out of
    // scope for this increment).
    //
    // 1 (default) reproduces today's single-threaded search exactly --
    // this project's own standing rule that a new lever ships off by
    // default until MEASURED to help (enable_root_gmi_cuts,
    // use_rens_heuristic, warm_start_node_relaxations above all follow the
    // same pattern). 0 means "auto": min(8, hardware_concurrency).
    //
    // MEASURED (docs/architecture/MILP.md \S6.5) on the 5-instance MIPLIB
    // set, single process: worker counts 1/4/8/16 all showed positive,
    // no-regression scaling on this 16-logical-core machine, with the
    // practically significant result at exactly 8 workers -- pk1 goes
    // from TIME_LIMIT (obj 44) to CERTIFIED OPTIMAL (obj 11, exact), not
    // just faster. 16 workers kept helping (pk1 certified even faster,
    // 28.3s vs 45.2s) but with clearly diminishing marginal throughput
    // gains over 8 (roughly 20-45% further gain vs. 8's own ~1.6-2.9x
    // gain over 4), consistent with 16 worker threads leaving zero
    // headroom for OS/other threads on a 16-logical-core box. `auto`'s
    // cap is set to 8, not 16, specifically so it stays a reasonable
    // default across machines smaller than this one, rather than
    // asserting "use every core" -- an explicit worker count remains the
    // right choice on a box known to have more headroom.
    //
    // Node exploration ORDER and node COUNT become inherently
    // nondeterministic above 1 (thread-scheduling-dependent) -- a real,
    // stated tradeoff, not a bug: the FINAL reported answer (status,
    // objective, feasibility of x against the original model) does not
    // depend on processing order, only its timing does (full argument in
    // docs/architecture/MILP.md).
    std::uint32_t parallel_worker_count = 1;
};

struct MilpSolution {
    MilpStatus status = MilpStatus::NUMERICAL_FAILURE;
    bool has_incumbent = false;
    std::vector<double> x; // original variable space; empty without incumbent
    double objective_value = 0.0;
    double best_bound = 0.0;
    double relative_gap = 0.0;

    std::uint64_t nodes_processed = 0;
    std::uint64_t nodes_pruned = 0;
    std::uint64_t lp_relaxations = 0;
    std::uint64_t strong_branching_probes = 0;
    std::uint64_t root_cover_cuts = 0;
    std::uint64_t cover_cuts = 0;
    std::uint64_t root_gmi_cuts = 0;
    std::uint64_t incumbent_updates = 0;
    std::uint64_t diving_heuristic_lp_relaxations = 0;
    std::uint64_t local_improvement_lp_relaxations = 0;
    std::uint64_t rens_heuristic_lp_relaxations = 0;

    // Populated only when warm_start_node_relaxations is true: how many
    // non-root node relaxations actually took the warm path, and how many
    // of those had to fall back to a cold solve after a verification
    // failure. A nonzero fallback count on a small, well-scaled instance
    // is a signal worth investigating, not an accepted steady state.
    std::uint64_t warm_started_relaxations = 0;
    std::uint64_t warm_start_verification_fallbacks = 0;
};

MilpSolution solve_milp(const MilpProblem& problem,
                        const MilpSolverOptions& options = {});

} // namespace sihps
