#pragma once

#include "../cuda/GpuPricer.hpp"
#include "../sparse/CSCMatrix.hpp"
#include "BasisFactorization.hpp"
#include "LpProblem.hpp"
#include "Scaling.hpp"
#include "../parallel/Parallel.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace sihps {

enum class LpStatus { OPTIMAL, INFEASIBLE, UNBOUNDED, ITERATION_LIMIT, NUMERICAL_FAILURE };

// Which computational backend performs full pricing (recomputing reduced
// costs for every nonbasic column each iteration) -- the one piece of the
// algorithm this project is deliberately A/B testing per H5
// (docs/research/SOTA.md \S5): is GPU-accelerated pricing actually faster
// than CPU pricing on this workload, or does the literature's skepticism
// about GPU-simplex hold up under our own measurement? Every other part
// of the algorithm (pivoting, ratio test, basis update) is identical
// between the two backends -- this isolates the one variable being
// tested.
enum class PricingBackend { CPU, GPU };

// Entering-variable selection rule.
//
// DANTZIG picks the largest reduced-cost violation. It is the textbook
// rule and is cheap, but it ignores how far the step can actually travel,
// so on degenerate problems it stalls: it repeatedly selects a steep-
// looking column whose ratio test then permits (almost) no movement.
//
// DEVEX (Harris, "Pivot selection methods of the Devex LP code",
// Mathematical Programming 5, 1973 -- ESTABLISHED METHOD, docs/research/
// SOTA.md \S1.4.2) instead maximizes d_j^2 / w_j, where w_j is a
// dynamically-maintained reference weight approximating the squared norm
// of the edge direction. It is a cheap approximation to exact steepest-
// edge pricing (Goldfarb & Reid 1977): it costs one extra pricing pass per
// iteration (to obtain the pivot row) but is expected to cut iteration
// counts substantially on the degenerate models this project targets.
//
// Both are retained so the choice can be MEASURED rather than asserted --
// see benchmarks/bench_pricing_rule.cpp.
enum class PricingRule { DANTZIG, DEVEX };

// Which simplex variant drives the solve.
//
// PRIMAL maintains primal feasibility and works toward dual feasibility;
// it needs a phase 1 to find a feasible starting point.
//
// DUAL maintains DUAL feasibility and works toward primal feasibility. Its
// starting point is free whenever the objective's signs allow every
// nonbasic variable to rest at a bound that gives its reduced cost the
// correct sign -- with an all-slack basis the duals are zero, so the
// reduced costs ARE the costs, and the test is simply whether each
// variable has a finite bound on the side its cost points to. When that
// holds, the entire phase 1 is skipped.
//
// AUTO resolves to PRIMAL on a cold start (still every LpSolver::solve_lp
// call today; MILP node relaxations are the one caller that can reach the
// warm branch, and only when explicitly opted in -- MilpSolverOptions::
// warm_start_node_relaxations). That is docs/architecture/LP.md \S2's own
// decision table, not a preference: it selects dual simplex when a PARENT
// BASIS is available and dual-feasible for the child, and primal on a
// cold start. The parent-basis branch lives in Simplex::solve() itself now
// (set_warm_start_basis/seat_basis below), ahead of both cold paths.
// Cold-start measurement backs the AUTO-resolves-to-primal default up
// rather than merely permitting it -- LP.md \S2.1 records dual costing
// 2.71x the iterations and 3.22x the wall-clock of primal across the
// Netlib instances that enter it on a cold start.
//
// DUAL selects the dual path explicitly (for that measurement, and for the
// warm-start caller). It still falls back to the primal path when the
// model admits no dual-feasible start -- a cost pointing toward an
// infinite bound, or a free variable with nonzero cost -- or when the dual
// path's terminal basis fails verification (NUMERICS.md \S5).
//
// Dual simplex is also the algorithm the MILP engine needs for exactly
// this reason: after a branching bound change, the parent's optimal basis
// stays dual-feasible while primal feasibility is disturbed, which is
// exactly dual simplex's entry condition (docs/architecture/LP.md \S1).
enum class LpAlgorithm { PRIMAL, DUAL, AUTO };

// Per-stage cost breakdown of the simplex iteration.
//
// `pricing_seconds` alone lumps together four different operations with
// four different fixes, which is not enough to act on. MEASURED on
// stocfor3 (16,675 rows): Devex costs 17.3 ms per iteration against
// Dantzig's 0.55 ms -- 31x -- while the algorithm predicts roughly 2-3x
// (Devex adds one BTRAN and one O(nnz) pass to an iteration that already
// pays for both). A gap that size between predicted and measured cost is
// the signature of a defect, not a trade, and finding it needs the stages
// separated.
//
// Cost: about nine steady_clock reads per iteration against an iteration
// measuring hundreds of microseconds at minimum. Always compiled in, so
// what is measured is what ships.
struct SimplexProfile {
    double refactor_seconds = 0.0;      // periodic refactorization + value re-derivation
    double duals_seconds = 0.0;         // BTRAN for y = B^-T c_B
    double price_seconds = 0.0;         // reduced costs + entering-variable scan
    double ftran_seconds = 0.0;         // B^-1 A_enter
    double ratio_seconds = 0.0;         // both Harris passes
    double rho_btran_seconds = 0.0;     // BTRAN for the pivot row (Devex only)
    double rho_assemble_seconds = 0.0;  // A^T (B^-T e_r) assembly (Devex only)
    double devex_seconds = 0.0;         // reference-weight update (Devex only)
    double update_seconds = 0.0;        // eta push / basis update
    long iterations = 0;

    double total() const {
        return refactor_seconds + duals_seconds + price_seconds + ftran_seconds + ratio_seconds +
               rho_btran_seconds + rho_assemble_seconds + devex_seconds + update_seconds;
    }
};

struct LpResult {
    LpStatus status = LpStatus::NUMERICAL_FAILURE;
    double objective_value = 0.0;
    std::vector<double> x;      // structural variables, size n_cols
    std::vector<double> y_dual; // row duals, size n_rows
    int phase1_iterations = 0;
    int phase2_iterations = 0;
    int dual_iterations = 0;
    bool used_dual_simplex = false;
    double primal_residual = 0.0;
    double dual_residual = 0.0;
    double pricing_seconds = 0.0; // cumulative time in the pricing step alone
    int refactorizations = 0;
    SimplexProfile profile;

    // Set when a warm-start basis was supplied via set_warm_start_basis,
    // regardless of outcome; used_warm_start narrows that to "and it is
    // what produced the REPORTED result" (same distinction LpResult
    // already draws for used_dual_simplex vs dual_iterations on a
    // fallback -- the attempt happened either way, but only one path's
    // work is what solve() actually returned).
    bool warm_start_attempted = false;
    bool used_warm_start = false;
};

// Two-phase bounded-variable revised primal simplex (docs/architecture/LP.md).
//
// v1 scope, explicitly:
//   - Sparse LU basis factorization with product-form (eta) updates and
//     periodic refactorization (BasisFactorization.hpp). Storage and
//     per-pivot solve cost are proportional to the nonzeros in the factors
//     rather than to m^2. LP.md \S4 specifies Forrest-Tomlin updates
//     instead of PFI; that difference, and why it is deferred rather than
//     overlooked, is documented in BasisFactorization.hpp.
//   - Devex entering-variable selection (Harris 1973) is the default;
//     Dantzig is retained behind PricingRule so the two can be compared
//     empirically. The Harris two-pass ratio test is also implemented --
//     together these are the two pieces that actually matter for
//     degeneracy/stability, per SOTA.md \S1.4.2.
//   - Ruiz equilibration (Scaling.hpp, docs/architecture/NUMERICS.md \S2)
//     runs once at construction and is used for every internal pivot;
//     every quantity returned to the caller (x, y_dual, residuals) is
//     unscaled back to ORIGINAL model units before verification, per
//     NUMERICS.md \S2's unscaling-before-reporting requirement -- see
//     finalize_result. use_ruiz_scaling exists to allow disabling it (e.g.
//     to isolate whether a regression is scaling-related), not to signal
//     that the scaled path is experimental.
//   - Free variables (both bounds infinite) ARE supported, via the
//     AT_ZERO nonbasic status below. Once a free variable becomes basic
//     it never leaves, since its own bounds can never limit a ratio test.
//
// A result is only ever reported OPTIMAL after the residual checks in
// solve() pass, in original (unscaled) units --
// docs/architecture/NUMERICS.md \S6's hard invariant.
class Simplex {
public:
    // AT_ZERO is the resting state of a nonbasic FREE variable (both
    // bounds infinite, e.g. an MPS 'FR' column): it has no bound to sit
    // at, so it rests at 0 and is eligible to enter in EITHER direction
    // depending on the sign of its reduced cost.
    //
    // Public (moved from private) so a caller can hold and pass a Basis
    // (below) without reaching into the engine's other internals -- the
    // MILP warm-start caller needs exactly this and nothing else.
    enum class VarStatus : std::uint8_t { AT_LOWER, AT_UPPER, AT_ZERO, BASIC };

    // An exportable basis identity: which augmented column is basic in
    // each row, and which bound every other column currently rests at.
    // Deliberately excludes the LU factorization itself -- seat_basis
    // always refactorizes from basic_columns anyway, since a factorization
    // valid for the PARENT's bounds is not valid for the CHILD's, so
    // exporting it would add a second representation with nothing to keep
    // it in sync for.
    //
    // art_sign is included even though every artificial this path seats is
    // frozen at [0,0] and therefore never MOVES: run_primal_two_phase can
    // still leave one BASIC at exactly 0 going into phase 2 (a legitimate
    // degenerate leftover, not an error), and a basic artificial's column
    // sign is exactly art_sign_[row] -- refactorize() would build the
    // wrong basis matrix column for that row without it. A fresh Simplex
    // otherwise defaults every sign to +1, which is only right by
    // coincidence.
    struct Basis {
        std::vector<std::int32_t> basic_columns;  // size n_rows: basis_[i]
        std::vector<VarStatus> nonbasic_status;   // size n_total; BASIC entries unused
        std::vector<double> art_sign;             // size n_rows
        std::int32_t n_struct = 0, n_slack = 0, n_art = 0;
    };

    // Cooperative cancellation for LpMethod::HYBRID. The pointee must
    // outlive the solve. Null (the default) means the solve never aborts.
    void set_cancel_flag(const std::atomic<bool>* flag) { cancel_ = flag; }

    // Wall-clock budget for LpMethod::HYBRID. Zero (the default) means no
    // budget, which is what every other caller gets. On expiry the solve
    // returns ITERATION_LIMIT -- it has not proved anything, so it must not
    // claim to have.
    void set_time_budget(double seconds) { time_budget_seconds_ = seconds; }

    // Warm start for the MILP B&B node loop (docs/architecture/LP.md \S1,
    // \S2; MILP.md's stated prerequisite). `parent` must outlive this
    // call's solve() -- same lifetime contract as set_cancel_flag. A basis
    // whose shape (n_struct/n_slack/n_art) does not match this instance is
    // ignored by solve() rather than misapplied; see LpResult's
    // warm_start_attempted/used_warm_start for how to tell attempted from
    // used.
    void set_warm_start_basis(const Basis* parent) { warm_start_basis_ = parent; }

    // Exports the CURRENT basis_/status_ as a Basis. Meaningful once solve()
    // has returned OPTIMAL or INFEASIBLE (finalize_result has already
    // refactorized and re-derived basic values against the terminal basis
    // by then, so what this reads back is the verified basis, not a
    // drifted intermediate one).
    Basis export_basis() const;

    // Tableau access for cutting-plane separation (Gomory mixed-integer
    // cuts, docs/architecture/MILP.md \S2.2). Meaningful once solve() has
    // returned OPTIMAL, under the same terminal-basis guarantee as
    // export_basis() above. Column indices span [0, n_total()): structural
    // columns first ([0, n_struct())), then slacks ([n_struct(),
    // n_struct()+n_slack())), then phase-1 artificials.
    std::int32_t n_total() const { return n_total_; }
    std::int32_t n_struct() const { return n_struct_; }
    std::int32_t n_slack() const { return n_slack_; }

    // Row index (0..n_rows-1) in which `var` is currently basic, or -1 if
    // `var` is nonbasic.
    std::int32_t basic_row_of(std::int32_t var) const {
        return basic_row_of_[static_cast<std::size_t>(var)];
    }

    // Which bound (if any) `var` currently rests at. BASIC entries are
    // meaningless (the variable has no resting bound).
    VarStatus status_of(std::int32_t var) const {
        return status_[static_cast<std::size_t>(var)];
    }

    // Public wrapper around the private compute_tableau_row used
    // internally for Devex pricing: rho[j] = (e_row^T B^-1) A_j for every
    // augmented column j. `row` indexes basis rows, i.e. it is the value
    // returned by basic_row_of for the row's basic variable.
    std::vector<double> tableau_row(std::int32_t row) const {
        std::vector<double> rho;
        compute_tableau_row(row, rho);
        return rho;
    }

    explicit Simplex(const LpProblem& problem, PricingBackend backend = PricingBackend::CPU,
                      bool use_ruiz_scaling = true,
                      PricingRule pricing_rule = PricingRule::DEVEX,
                      LpAlgorithm algorithm = LpAlgorithm::AUTO,
                      ParallelMode parallel_mode = ParallelMode::AUTO,
                      const ScaleFactors* precomputed_scale = nullptr);

    LpResult solve();

private:
    void setup_phase1();
    LpStatus run_primal_simplex(bool phase1, int& iterations);

    // The complete primal path: crash basis, phase 1, then phase 2. Fills
    // result's iteration counts and returns the status to be VERIFIED --
    // it does not call finalize_result itself, because solve() may run this
    // as a fallback after the dual path and only one verification of one
    // terminal basis may be reported.
    LpStatus run_primal_two_phase(LpResult& result);

    // Installs an all-slack basis and parks every nonbasic structural at
    // the bound that gives its reduced cost the dual-feasible sign. With
    // that basis the duals are zero, so reduced cost == cost and the test
    // is purely structural. Returns false when no such assignment exists,
    // in which case the caller must use the primal path -- there is no
    // dual phase 1 in v1, and pretending otherwise would mean starting the
    // dual iteration from a point where its invariant does not hold.
    bool setup_dual_feasible_start();

    // Bounded-variable dual simplex. Maintains dual feasibility and drives
    // out primal infeasibility; on termination the basis is both primal and
    // dual feasible, hence optimal. Reports INFEASIBLE when the dual ratio
    // test finds no eligible entering column (dual unbounded).
    LpStatus run_dual_simplex(int& iterations);
    void price(const std::vector<double>& y, std::vector<double>& reduced_costs);
    void price_cpu(const std::vector<double>& y, std::vector<double>& reduced_costs) const;
    void compute_duals(std::vector<double>& y) const;

    // Entering-variable search over every augmented column: eligibility
    // (which sign of reduced cost improves the objective, given the bound
    // the variable rests at) plus the pricing score, reduced to a single
    // winner. Returns false when no column is eligible, which is the
    // optimality condition.
    //
    // Both backends implement the SAME rule; they differ only in where the
    // scan runs. On CPU it is a host loop over reduced costs computed by
    // price_cpu; on GPU the reduced costs are never materialized on the
    // host at all and the argmax is a device reduction, so one candidate
    // crosses PCIe instead of n_total doubles (GpuPricer.hpp).
    bool select_entering(std::int32_t& enter, double& enter_dq);

    // Uploads the quantities the GPU pricer caches across iterations
    // (costs, bounds, artificial signs) after any phase boundary that
    // changes them. A no-op on the CPU backend. Getting this wrong would
    // mean the GPU prices a stale objective, so every site that writes
    // cost_/lower_/upper_ outside the iteration loop calls it.
    void sync_gpu_phase();

    // Restarts the Devex reference framework on whichever backend holds
    // the weights. The CPU keeps them in devex_weight_; the GPU keeps them
    // in device memory and never ships them across the bus.
    void reset_devex_weights();

    // Row `row` of B^-1, i.e. (B^-T e_row)^T -- one BTRAN. This is the
    // only part of the pivot row the GPU path needs on the host; the
    // n_total-wide row itself is formed on the device.
    void compute_binv_row(std::int32_t row, std::vector<double>& out) const;

    // Computes the pivot row of the simplex tableau for basis row `row`:
    // rho_j = (e_row^T B^-1) A_j over EVERY augmented column j. Devex needs
    // this to update its reference weights. With a factorized basis,
    // e_row^T B^-1 is obtained as one BTRAN solve against the unit vector
    // e_row -- the dense-inverse version could read the row directly, so
    // this is the one place where the factorized representation costs
    // strictly more work than storing B^-1 outright. It is bounded by a
    // single triangular solve pair, against the O(m^2) per-pivot cost the
    // dense inverse imposed everywhere else.
    void compute_tableau_row(std::int32_t row, std::vector<double>& rho) const;

    // Devex weight update after a basis change (Harris 1973). `pivot` is
    // the entering column's tableau entry in the leaving row.
    void devex_update(std::int32_t entering, std::int32_t leaving_var, double pivot,
                       const std::vector<double>& rho);

    SparseColumn column(std::int32_t var) const;

    // Same as column(), but reuses the caller's buffer instead of
    // returning a fresh one -- the iteration loop calls this every pivot
    // and prompt.md \S3.1 forbids allocating there.
    void column_into(std::int32_t var, SparseColumn& out) const;

    // Multiplier converting a scaled quantity for `var` into original model
    // units. Termination tests that must agree with finalize_result's
    // original-space verification apply it; the iteration arithmetic itself
    // stays entirely in scaled space.
    double unscale_factor(std::int32_t var) const;

    // dir = B^-1 A_var, i.e. how each basic variable must move per unit of
    // `var` entering. (This is FTRAN; an earlier revision misnamed it
    // btran_column.)
    void ftran_column(const SparseColumn& col, std::vector<double>& dir) const;

    // Refactorizes from the current basis, repairing it first if the
    // factorization reports dependent columns (NUMERICS.md \S5). Throws
    // only when repair itself is impossible.
    void refactorize();

    // Replaces the variable at basis position `pos` with the artificial
    // column of `unmatched_row`, moving the displaced variable to a bound.
    // Returns false if that artificial is somehow already basic, which
    // would mean the repair cannot restore nonsingularity.
    bool repair_basis_position(std::int32_t pos, std::int32_t unmatched_row);

    // Recomputes x_B = B^-1 (rhs - N x_N) from the CURRENT basis and the
    // current nonbasic values, discarding whatever the incrementally-
    // maintained value_ entries had drifted to. Incremental updates
    // accumulate round-off over thousands of pivots (worse here than with
    // the sparse-LU-plus-Forrest-Tomlin scheme docs/architecture/LP.md
    // targets, because a dense product-form inverse is updated in full
    // every pivot), so the maintained values are periodically replaced by
    // a value derived directly from the basis.
    void recompute_basic_values();

    void apply_pivot_update(std::int32_t leaving_row, const std::vector<double>& dir);
    void finalize_result(LpStatus status, LpResult& result);

    // Seats `parent` into basis_/basic_row_of_/status_/value_, freezing
    // artificial bounds first (this path is phase-2-only -- it never runs
    // phase 1), reinstalls phase-2 costs, resets Devex weights, and
    // refactorizes. Returns false -- never throws past this function -- on
    // a shape mismatch, a refactorization failure, or the one structural
    // edge case it refuses rather than guesses at: a nonbasic AT_ZERO
    // (free) column that the CALLER's (possibly tightened) bounds have
    // just made finite. The caller falls back to the cold path either way.
    bool seat_basis(const Basis& parent);

    const LpProblem& problem_;
    PricingBackend backend_;
    PricingRule pricing_rule_;
    LpAlgorithm algorithm_;
    ParallelMode parallel_mode_;

    // Devex reference weights, one per augmented column. Reset to all-ones
    // whenever the reference framework is restarted (see devex_update):
    // the weights are approximations whose error compounds, so Harris's
    // scheme periodically discards them rather than letting them drift.
    std::vector<double> devex_weight_;

    std::int32_t n_rows_ = 0, n_struct_ = 0, n_slack_ = 0, n_art_ = 0, n_total_ = 0;

    // Ruiz scale factors (identity if use_ruiz_scaling is false, or if Ruiz
    // equilibration failed to converge -- Scaling.hpp's documented
    // recovery). Every internal array below (A_csc_, A_scaled_, rhs_scaled_,
    // lower_/upper_/cost_) is expressed in SCALED units; problem_ itself is
    // never mutated and remains the source of truth for unscaling in
    // finalize_result.
    ScaleFactors scale_;
    CSRMatrix A_scaled_;     // R * problem_.A * C, feeds setup_phase1/recompute_basic_values
    std::vector<double> rhs_scaled_; // R .* problem_.rhs
    CSCMatrix A_csc_;        // CSC of A_scaled_ -- what pricing/pivoting actually touch

    std::vector<double> lower_, upper_, cost_;
    std::vector<double> value_;
    std::vector<VarStatus> status_;
    std::vector<std::int32_t> basis_;        // size n_rows_: variable index basic in row i
    std::vector<std::int32_t> basic_row_of_; // size n_total_: row index if basic, else -1
    std::vector<double> art_sign_;           // sign_i used by row i's artificial column

    // Sparse LU of the basis plus its product-form update file. This
    // replaces the dense n_rows_ x n_rows_ explicit inverse an earlier
    // revision carried: that was O(m^2) memory and per-pivot work with
    // O(m^3) refactorization, which put every Netlib instance above ~1300
    // rows -- and every refinery-scale model -- permanently out of reach.
    BasisFactorization factor_;

    // Solve scratch, sized once at construction so the iteration loop does
    // not allocate (prompt.md \S3.1). Every one of these is filled with
    // assign()/clear() rather than being reconstructed, so after the
    // constructor sizes them no further heap traffic occurs: assign() on a
    // vector already at the required size reuses its buffer.
    mutable std::vector<double> solve_work_;
    std::vector<double> y_work_;        // duals, n_rows_
    std::vector<double> rc_work_;       // reduced costs, n_total_
    std::vector<double> dir_work_;      // B^-1 A_enter, n_rows_
    std::vector<double> rho_work_;      // pivot row, n_total_ (CPU Devex / dual)
    std::vector<double> binv_row_work_; // row of B^-1, n_rows_ (GPU Devex)
    SparseColumn col_work_;             // entering column, sparse
    // recompute_basic_values runs at every refactorization, which is
    // inside the iteration loop, so its working vectors are members too.
    mutable std::vector<double> residual_work_, x_struct_work_, ax_work_;

    int max_iterations_ = 0;
    // Polled in both simplex loops alongside the iteration limit. Set by
    // LpMethod::HYBRID so the loser of the race stops promptly instead of
    // running to completion on hardware the winner no longer needs.
    const std::atomic<bool>* cancel_ = nullptr;
    double time_budget_seconds_ = 0.0;
    // Set via set_warm_start_basis; consulted once at the top of solve().
    // Not owned -- same lifetime contract as cancel_.
    const Basis* warm_start_basis_ = nullptr;
    std::chrono::steady_clock::time_point start_time_{};
    // Polling a clock every iteration would show up in the profile on
    // models whose iterations cost microseconds, so the budget is checked
    // on a stride. The stride is a power of two and the check is a mask.
    static constexpr int kTimeCheckStride = 256;
    bool out_of_time(int iterations) const;

    // Refactorization cadence, measured in accumulated eta vectors rather
    // than raw iterations: PFI cost and drift both scale with the eta file,
    // not with iterations that were bound flips (which push no eta).
    int refactorize_every_ = 100;
    double pricing_seconds_ = 0.0;
    int refactorizations_ = 0;
    // mutable: compute_tableau_row is const but must charge its own time.
    mutable SimplexProfile profile_;

    // GPU pricing resources: device-resident reduced costs, Devex weights,
    // bounds and reduction scratch, plus the cuSPARSE A^T operator. Built
    // only when backend_ == GPU, and allocated entirely inside its own
    // constructor so nothing is allocated on the device once solving
    // starts (prompt.md \S3.1).
    std::unique_ptr<GpuPricer> gpu_pricer_;
};

} // namespace sihps
