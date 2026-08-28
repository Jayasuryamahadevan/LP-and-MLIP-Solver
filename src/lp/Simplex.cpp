#include "Simplex.hpp"

#include "../parallel/Parallel.hpp"

#include "../sparse/Convert.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace sihps {

namespace {
// Adds its lifetime to `sink`. Used to break the iteration into stages
// (SimplexProfile) without four lines of chrono boilerplate per stage.
class ScopedTimer {
public:
    explicit ScopedTimer(double& sink) : sink_(sink), start_(std::chrono::steady_clock::now()) {}
    ~ScopedTimer() {
        sink_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
    }
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    double& sink_;
    std::chrono::steady_clock::time_point start_;
};
} // namespace
namespace {

// v1 tolerances. Deliberately looser than docs/architecture/NUMERICS.md's
// eventual 1e-8 relative / 1e-7 absolute targets: Ruiz scaling (Scaling.hpp)
// mitigates ill-conditioning from raw model coefficients, but a dense-B_inv
// product-form basis still accumulates more round-off over many pivots
// than the sparse LU with periodic refactorization NUMERICS.md specifies,
// and pricing is still Dantzig-rule rather than Devex. The final
// verification step still enforces a real bound -- it does not accept any
// result unconditionally.
constexpr double kOptTol = 1e-7;
constexpr double kPivotTol = 1e-9;
constexpr double kFinalTol = 1e-6;

// Genuine floating-point cancellation noise on a row scales with that row's
// own component-wise term magnitude (|A|*|x|), but by an amount tied to
// machine precision, not to `kFinalTol` (1e-6) -- MEASURED directly on
// Netlib `shell`/`perold`: their legitimate solver rounding noise, across
// dozens of affected rows each, lands at a strikingly consistent ratio of
// ~5e-10 relative to the row's own |A|*|x| activity (never above 7e-10 in
// either instance). A genuine infeasibility, by contrast, is not bounded by
// machine precision at all -- MEASURED on MIPLIB `ej` (a deliberately
// adversarial "numerics" instance, per its own MIPLIB tag): an incumbent
// that violates its single equality row by a real 20 units, against a row
// activity of ~2.05e7, is a ratio of ~9.8e-7, nearly 2000x above the noise
// ceiling above. This sits in between on a log scale (~20x above the
// measured noise ceiling, ~100x below the measured violation ratio) --
// comfortable margin either way. Used ADDITIVELY with `kFinalTol`, not as a
// replacement multiplier: near-zero-activity rows still get exactly
// `kFinalTol`'s own original, unweakened absolute allowance.
constexpr double kActivityNoiseRatio = 1e-8;

// Harris ratio-test bound expansion (docs/research/SOTA.md \S1.4.2). Pass 1
// relaxes every basic variable's bound by this much so pass 2 has a set of
// near-tied rows to choose the largest pivot from.
//
// The expansion is not free: nothing in this implementation ever removes
// the infeasibility it permits, so a basic variable may finish the solve up
// to kHarrisExpand outside its bound, and the terminal basis is then not
// quite primal feasible. EXPERIMENTAL RESULT (Netlib feasible set, 89
// instances up to 2600 rows, both settings measured end to end):
//
//   expansion   pass rate   total iters   total sec   pilot87 rel.err
//   1e-7        88 / 89       195,966       34.72      1.07e-06  FAIL
//   1e-9        89 / 89       181,418       34.15      2.16e-07  PASS
//
// pilot87 at 1e-7 terminated with a structural 1.83e-07 below its lower
// bound -- inside the kFinalTol gate, so it was reported OPTIMAL, but the
// objective error it produced exceeded the 1e-6 agreement threshold the
// Netlib validation applies. Tightening to 1e-9 fixed that instance and
// cost nothing: iteration count FELL 7.4% across the set and no instance
// regressed by more than 25%. The classical stability argument for a wide
// expansion (Gill, Murray, Saunders & Wright, "A practical anti-cycling
// procedure for linearly constrained optimization", Mathematical
// Programming 45, 1989) is therefore not what governs at this scale -- the
// two-pass structure itself supplies the pivot choice, and the extra width
// mainly bought accumulated infeasibility. Kept as a named constant rather
// than folded into kPivotTol because the two mean different things and the
// EXPAND literature's shift-and-reset machinery remains the fallback if a
// future instance stalls at this width.
constexpr double kHarrisExpand = 1e-9;

// Primal-infeasibility threshold for dual simplex's leaving-row selection:
// a basic variable further outside its bounds than this is a candidate to
// leave, and the dual terminates OPTIMAL when none is. Deliberately far
// tighter than kFinalTol, for the reason the Harris note above documents:
// terminating at the verification gate's own width leaves nothing between
// "the algorithm stopped" and "the result failed verification".
constexpr double kDualPrimalFeasTol = 1e-9;

// Dual-feasibility expansion for the dual Harris ratio test's first pass:
// the amount by which a reduced cost may go negative before the step is
// refused. It plays exactly the role kHarrisExpand plays in the primal
// test -- it widens the candidate set so the second pass has a choice of
// pivots -- and is set to the same width for the same measured reason.
constexpr double kDualFeasTol = 1e-9;

// Phase 1 is considered to have driven the artificial sum to zero when it
// falls below this. Scaled by the problem's rhs magnitude in solve(),
// since an absolute threshold is meaningless on a model whose rhs entries
// are ~1e7 (as several Netlib instances are).
constexpr double kPhase1RelTol = 1e-9;

double vector_inf_norm(const std::vector<double>& v) {
    double m = 0.0;
    for (double x : v) m = std::max(m, std::fabs(x));
    return m;
}

} // namespace

Simplex::Simplex(const LpProblem& problem, PricingBackend backend, bool use_ruiz_scaling,
                  PricingRule pricing_rule, LpAlgorithm algorithm, ParallelMode parallel_mode,
                  const ScaleFactors* precomputed_scale)
    : problem_(problem),
      backend_(backend),
      pricing_rule_(pricing_rule),
      algorithm_(algorithm),
      parallel_mode_(parallel_mode) {
    n_rows_ = problem_.n_rows();
    n_struct_ = problem_.n_cols();
    n_slack_ = n_rows_;
    n_art_ = n_rows_;
    n_total_ = n_struct_ + n_slack_ + n_art_;

    // A caller re-solving the SAME matrix across many calls (the MILP B&B
    // node loop, where only bounds change between siblings) supplies its
    // own once-computed factors here rather than paying Ruiz's iterative
    // equilibration loop again on every node -- that cost is exactly what
    // warm-starting exists to avoid. Every other caller passes nullptr and
    // this is byte-for-byte the same construction as before this parameter
    // existed.
    if (use_ruiz_scaling && precomputed_scale != nullptr) {
        scale_ = *precomputed_scale;
    } else {
        scale_ = use_ruiz_scaling ? compute_ruiz_scaling(problem_.A)
                                  : ScaleFactors::identity(n_rows_, n_struct_);
    }
    A_scaled_ = apply_ruiz_scaling(problem_.A, scale_);
    A_csc_ = csr_to_csc(A_scaled_);

    rhs_scaled_.resize(static_cast<std::size_t>(n_rows_));
    for (std::int32_t i = 0; i < n_rows_; ++i) {
        rhs_scaled_[static_cast<std::size_t>(i)] =
            scale_.row_scale[static_cast<std::size_t>(i)] * problem_.rhs[static_cast<std::size_t>(i)];
    }

    lower_.resize(static_cast<std::size_t>(n_total_));
    upper_.resize(static_cast<std::size_t>(n_total_));
    cost_.assign(static_cast<std::size_t>(n_total_), 0.0);
    value_.assign(static_cast<std::size_t>(n_total_), 0.0);
    status_.assign(static_cast<std::size_t>(n_total_), VarStatus::AT_LOWER);
    basis_.resize(static_cast<std::size_t>(n_rows_));
    basic_row_of_.assign(static_cast<std::size_t>(n_total_), -1);
    art_sign_.assign(static_cast<std::size_t>(n_rows_), 1.0);
    devex_weight_.assign(static_cast<std::size_t>(n_total_), 1.0);
    solve_work_.assign(static_cast<std::size_t>(n_rows_), 0.0);

    // Iteration scratch, sized here and never resized again: every use
    // below is assign()/clear() at exactly these sizes, which reuses the
    // buffer rather than reallocating (prompt.md \S3.1 -- no malloc/new
    // after the solve begins). Sizing them at construction is what makes
    // that guarantee hold for the whole solve rather than for most of it.
    y_work_.assign(static_cast<std::size_t>(n_rows_), 0.0);
    rc_work_.assign(static_cast<std::size_t>(n_total_), 0.0);
    dir_work_.assign(static_cast<std::size_t>(n_rows_), 0.0);
    rho_work_.assign(static_cast<std::size_t>(n_total_), 0.0);
    binv_row_work_.assign(static_cast<std::size_t>(n_rows_), 0.0);
    residual_work_.assign(static_cast<std::size_t>(n_rows_), 0.0);
    x_struct_work_.assign(static_cast<std::size_t>(n_struct_), 0.0);
    ax_work_.assign(static_cast<std::size_t>(n_rows_), 0.0);
    // The entering column can be no denser than the densest column of A,
    // and slack/artificial columns are singletons. Reserving the worst
    // case once means column_into() never grows it mid-solve.
    std::int32_t densest_col = 1;
    for (std::int32_t j = 0; j < n_struct_; ++j) {
        densest_col = std::max(densest_col, A_csc_.col_ptr()[j + 1] - A_csc_.col_ptr()[j]);
    }
    col_work_.reserve(static_cast<std::size_t>(densest_col));

    // Bounds below are expressed in SCALED units: x = C x' means
    // x' in [l/C, u/C]; the augmented equation A'x' + s' = rhs' (with
    // s' = R.*s, per docs/architecture/NUMERICS.md \S2's unscaling
    // convention run in reverse) means s' in [R*slack_lower, R*slack_upper].
    // Dividing/multiplying by a finite positive scale factor preserves
    // +-infinity bounds exactly, so unbounded variables/rows stay unbounded.
    for (std::int32_t j = 0; j < n_struct_; ++j) {
        const auto jj = static_cast<std::size_t>(j);
        const double c = scale_.col_scale[jj];
        lower_[jj] = problem_.lower[jj] / c;
        upper_[jj] = problem_.upper[jj] / c;
    }
    for (std::int32_t i = 0; i < n_rows_; ++i) {
        const auto ii = static_cast<std::size_t>(i);
        const double r = scale_.row_scale[ii];
        std::int32_t s = n_struct_ + i;
        lower_[static_cast<std::size_t>(s)] = r * problem_.slack_lower[ii];
        upper_[static_cast<std::size_t>(s)] = r * problem_.slack_upper[ii];
        std::int32_t a = n_struct_ + n_slack_ + i;
        lower_[static_cast<std::size_t>(a)] = 0.0;
        upper_[static_cast<std::size_t>(a)] = kInfinity;
    }

    max_iterations_ = std::max(5000, 30 * (n_rows_ + n_struct_));
    start_time_ = std::chrono::steady_clock::now();

    if (backend_ == PricingBackend::GPU) {
        gpu_pricer_ = std::make_unique<GpuPricer>(A_csc_, n_rows_, n_struct_, n_slack_, n_art_);
        sync_gpu_phase();
    }
}

// The device kernels index status bytes directly, so the two enumerations
// are one representation with two spellings. A silent renumbering on
// either side would not fail to compile and would not crash -- it would
// quietly price a different LP on the GPU than on the CPU, which is the
// hardest possible failure to notice. Pinned here rather than trusted.
void Simplex::sync_gpu_phase() {
    static_assert(sizeof(VarStatus) == sizeof(std::uint8_t),
                   "VarStatus must be byte-sized to alias the device status array");
    static_assert(static_cast<std::uint8_t>(VarStatus::AT_LOWER) == gpu::kAtLower,
                   "device status encoding drifted from VarStatus");
    static_assert(static_cast<std::uint8_t>(VarStatus::AT_UPPER) == gpu::kAtUpper,
                   "device status encoding drifted from VarStatus");
    static_assert(static_cast<std::uint8_t>(VarStatus::AT_ZERO) == gpu::kAtZero,
                   "device status encoding drifted from VarStatus");
    static_assert(static_cast<std::uint8_t>(VarStatus::BASIC) == gpu::kBasic,
                   "device status encoding drifted from VarStatus");
    if (!gpu_pricer_) return;
    gpu_pricer_->sync_phase(cost_.data(), lower_.data(), upper_.data(), art_sign_.data());
}

void Simplex::reset_devex_weights() {
    std::fill(devex_weight_.begin(), devex_weight_.end(), 1.0);
    if (gpu_pricer_) gpu_pricer_->reset_devex_weights();
}

void Simplex::setup_phase1() {
    // Step 0: restore the artificial columns to [0, +inf) and the Devex
    // reference framework to its initial state. Both matter only when this
    // is a RESTART -- solve() may have already attempted the dual path,
    // which freezes every artificial at [0,0] (they are unused there) and
    // leaves the basis and factorization describing a different basis.
    // Phase 1 needs the artificials free to absorb row residuals, so this
    // function defines the whole state it depends on rather than assuming
    // it is running first. Anything left implicit here would surface as a
    // phase 1 that cannot move on exactly the instances where the dual path
    // failed -- i.e. only on the hardest models.
    for (std::int32_t i = 0; i < n_art_; ++i) {
        const auto a = static_cast<std::size_t>(n_struct_ + n_slack_ + i);
        lower_[a] = 0.0;
        upper_[a] = kInfinity;
    }
    reset_devex_weights();

    // Step 1: park every structural and slack variable at a bound (or at
    // zero if it is free).
    for (std::int32_t j = 0; j < n_struct_; ++j) {
        const auto jj = static_cast<std::size_t>(j);
        if (std::isfinite(lower_[jj])) {
            status_[jj] = VarStatus::AT_LOWER;
            value_[jj] = lower_[jj];
        } else if (std::isfinite(upper_[jj])) {
            status_[jj] = VarStatus::AT_UPPER;
            value_[jj] = upper_[jj];
        } else {
            status_[jj] = VarStatus::AT_ZERO;
            value_[jj] = 0.0;
        }
    }
    // Step 2: CRASH BASIS (ESTABLISHED METHOD: Bixby, "Implementing the
    // Simplex Method: The Initial Basis", ORSA Journal on Computing 4(4),
    // 1992).
    //
    // The naive start puts an artificial in EVERY row, so phase 1 must
    // drive out one artificial per row -- at least m pivots before phase 2
    // can even begin, and in practice far more. On dfl001 (m = 6071) Bixby
    // reports 465,810 phase-1 iterations arising from exactly this.
    //
    // But an inequality row's slack already has room to absorb its own
    // residual: an 'L' row's slack lives in [0, +inf), a 'G' row's in
    // (-inf, 0]. Wherever the required residual falls inside the slack's
    // bounds, that row starts FEASIBLE with the slack basic and needs no
    // artificial at all. Only rows whose slack cannot cover the residual
    // (equalities, and inequalities violated at the starting point) get one.
    //
    // The resulting basis is still one unit column per row -- slack columns
    // are e_i, artificial columns are +-e_i -- so B remains diagonal and the
    // factorization stays trivial. This is a strict improvement with no
    // structural cost, and it is entirely general: nothing about it is
    // tuned to any particular model.
    std::vector<double> x_struct(static_cast<std::size_t>(n_struct_), 0.0);
    for (std::int32_t j = 0; j < n_struct_; ++j) {
        x_struct[static_cast<std::size_t>(j)] = value_[static_cast<std::size_t>(j)];
    }
    std::vector<double> ax(static_cast<std::size_t>(n_rows_), 0.0);
    if (n_struct_ > 0) {
        A_scaled_.multiply(x_struct.data(), ax.data());
    }

    for (std::int32_t i = 0; i < n_rows_; ++i) {
        const auto ii = static_cast<std::size_t>(i);
        const auto s = static_cast<std::size_t>(n_struct_ + i);
        const auto a = static_cast<std::size_t>(n_struct_ + n_slack_ + i);

        // What the slack plus artificial together must supply for this row.
        const double target = rhs_scaled_[ii] - ax[ii];

        // The slack must not only COVER the residual, it must have room to
        // move. An equality row's slack is fixed at [0,0]; crashing that
        // into the basis seats a variable that can never change value, so
        // every ratio test through it forces a zero step and the basis is
        // degenerate from iteration one. Such rows keep the artificial,
        // which at least has [0, +inf) to move in.
        const bool slack_can_move = (upper_[s] - lower_[s]) > 1e-12;

        if (slack_can_move && target >= lower_[s] && target <= upper_[s]) {
            // The slack covers it: row starts feasible, no artificial.
            basis_[ii] = n_struct_ + i;
            basic_row_of_[s] = i;
            status_[s] = VarStatus::BASIC;
            value_[s] = target;

            art_sign_[ii] = 1.0;
            status_[a] = VarStatus::AT_LOWER;
            value_[a] = 0.0;
            basic_row_of_[a] = -1;
            // Freeze this artificial at zero for the whole of phase 1. It
            // starts NONBASIC (unlike the all-artificial start, where every
            // artificial was basic), so without this it remains eligible to
            // ENTER the basis and reintroduce infeasibility into a row that
            // already starts feasible. Freezing is sound: the ratio test
            // keeps the basic slack inside its own bounds, so this row stays
            // satisfied for the whole of phase 1 and the artificial is never
            // needed. The entering loop skips fixed columns by bound width.
            lower_[a] = 0.0;
            upper_[a] = 0.0;
        } else {
            // Park the slack at whichever bound is closest to `target`, and
            // let the artificial absorb only the shortfall -- strictly less
            // artificial infeasibility than parking it at zero.
            const double parked = (target < lower_[s]) ? lower_[s] : upper_[s];
            status_[s] = (target < lower_[s]) ? VarStatus::AT_LOWER : VarStatus::AT_UPPER;
            value_[s] = parked;
            basic_row_of_[s] = -1;

            const double residual = target - parked;
            art_sign_[ii] = (residual >= 0.0) ? 1.0 : -1.0;
            basis_[ii] = n_struct_ + n_slack_ + i;
            basic_row_of_[a] = i;
            status_[a] = VarStatus::BASIC;
            value_[a] = std::fabs(residual);
        }
    }

    // B is diagonal (each basic column is e_i or +-e_i). Factorized rather
    // than special-cased, so the factorization is exercised from the very
    // first iteration instead of only after the first refactorization -- a
    // diagonal basis is the one case where a latent bug in the permutation
    // bookkeeping would otherwise stay hidden.
    refactorize();

    for (std::int32_t j = 0; j < n_struct_ + n_slack_; ++j) {
        cost_[static_cast<std::size_t>(j)] = 0.0;
    }
    for (std::int32_t i = 0; i < n_art_; ++i) {
        cost_[static_cast<std::size_t>(n_struct_ + n_slack_ + i)] = 1.0;
    }

    // Phase 1 installs its own objective and re-opens the artificial
    // bounds; the GPU caches both across iterations and must be told.
    sync_gpu_phase();
}

bool Simplex::setup_dual_feasible_start() {
    // Phase-2 costs from the outset: dual simplex never runs a phase 1, so
    // the objective it works against is the real one immediately.
    for (std::int32_t j = 0; j < n_struct_; ++j) {
        const auto jj = static_cast<std::size_t>(j);
        cost_[jj] = problem_.obj[jj] * scale_.col_scale[jj];
    }
    for (std::int32_t i = 0; i < n_slack_ + n_art_; ++i) {
        cost_[static_cast<std::size_t>(n_struct_ + i)] = 0.0;
    }
    sync_gpu_phase();

    // All-slack basis: B = I, since every slack column is e_i. The duals
    // are therefore y = B^-T c_B = 0, and every structural's reduced cost
    // equals its own cost -- which is what makes the feasibility test below
    // a purely structural check rather than a numerical one.
    for (std::int32_t i = 0; i < n_rows_; ++i) {
        const auto s = static_cast<std::size_t>(n_struct_ + i);
        basis_[static_cast<std::size_t>(i)] = n_struct_ + i;
        basic_row_of_[s] = i;
        status_[s] = VarStatus::BASIC;

        // Artificials are unused on this path; freeze them out entirely so
        // they can never enter and reintroduce infeasibility.
        const auto a = static_cast<std::size_t>(n_struct_ + n_slack_ + i);
        art_sign_[static_cast<std::size_t>(i)] = 1.0;
        lower_[a] = 0.0;
        upper_[a] = 0.0;
        value_[a] = 0.0;
        status_[a] = VarStatus::AT_LOWER;
        basic_row_of_[a] = -1;
    }

    for (std::int32_t j = 0; j < n_struct_; ++j) {
        const auto jj = static_cast<std::size_t>(j);
        const double c = cost_[jj];
        basic_row_of_[jj] = -1;

        if (c > 0.0) {
            // Reduced cost must be >= 0 at a lower bound: needs a finite one.
            if (!std::isfinite(lower_[jj])) return false;
            status_[jj] = VarStatus::AT_LOWER;
            value_[jj] = lower_[jj];
        } else if (c < 0.0) {
            if (!std::isfinite(upper_[jj])) return false;
            status_[jj] = VarStatus::AT_UPPER;
            value_[jj] = upper_[jj];
        } else {
            // Zero cost is dual-feasible at either bound, or at zero if the
            // variable is free.
            if (std::isfinite(lower_[jj])) {
                status_[jj] = VarStatus::AT_LOWER;
                value_[jj] = lower_[jj];
            } else if (std::isfinite(upper_[jj])) {
                status_[jj] = VarStatus::AT_UPPER;
                value_[jj] = upper_[jj];
            } else {
                status_[jj] = VarStatus::AT_ZERO;
                value_[jj] = 0.0;
            }
        }
    }

    refactorize();
    recompute_basic_values();
    return true;
}

Simplex::Basis Simplex::export_basis() const {
    Basis basis;
    basis.n_struct = n_struct_;
    basis.n_slack = n_slack_;
    basis.n_art = n_art_;
    basis.basic_columns = basis_;
    basis.nonbasic_status = status_; // BASIC-marked entries are present but unused by seat_basis
    basis.art_sign = art_sign_;
    return basis;
}

// See Simplex.hpp's Basis/seat_basis doc comments for the shape this seats
// and why. This is the phase-2-only counterpart of setup_dual_feasible_start
// above: instead of deriving a dual-feasible all-slack basis from the
// costs, it installs a CALLER-SUPPLIED basis outright and lets the caller
// (solve()) run dual simplex from there to repair whatever primal
// infeasibility the bound change introduced. Dual feasibility itself needs
// no check here: reduced costs depend only on cost_ and basis_, and
// neither changes between a parent and a bound-only child, so a
// dual-feasible parent basis is dual-feasible for the child by
// construction -- the textbook argument this whole feature rests on.
bool Simplex::seat_basis(const Basis& parent) {
    if (parent.n_struct != n_struct_ || parent.n_slack != n_slack_ || parent.n_art != n_art_ ||
        parent.basic_columns.size() != static_cast<std::size_t>(n_rows_) ||
        parent.nonbasic_status.size() != static_cast<std::size_t>(n_total_) ||
        parent.art_sign.size() != static_cast<std::size_t>(n_rows_)) {
        return false;
    }

    // Freeze every artificial to [0,0] FIRST -- this path never runs phase
    // 1, so none may be free to move. Restoring art_sign_ here (rather
    // than leaving the fresh instance's all-+1 default) matters exactly
    // when a basic artificial is among parent.basic_columns below: its
    // column sign is what refactorize() builds the basis matrix from, and
    // a wrong sign would silently factorize the wrong matrix rather than
    // fail loudly.
    for (std::int32_t i = 0; i < n_art_; ++i) {
        const auto a = static_cast<std::size_t>(n_struct_ + n_slack_ + i);
        lower_[a] = 0.0;
        upper_[a] = 0.0;
    }
    art_sign_ = parent.art_sign;

    std::fill(basic_row_of_.begin(), basic_row_of_.end(), -1);
    for (std::int32_t i = 0; i < n_rows_; ++i) {
        const auto ii = static_cast<std::size_t>(i);
        const std::int32_t col = parent.basic_columns[ii];
        if (col < 0 || col >= n_total_) return false;
        basis_[ii] = col;
        basic_row_of_[static_cast<std::size_t>(col)] = i;
        status_[static_cast<std::size_t>(col)] = VarStatus::BASIC;
    }

    for (std::int32_t v = 0; v < n_total_; ++v) {
        const auto vv = static_cast<std::size_t>(v);
        if (basic_row_of_[vv] >= 0) continue; // seated as basic above
        const VarStatus st = parent.nonbasic_status[vv];
        switch (st) {
            case VarStatus::AT_LOWER:
                if (!std::isfinite(lower_[vv])) return false;
                value_[vv] = lower_[vv];
                break;
            case VarStatus::AT_UPPER:
                if (!std::isfinite(upper_[vv])) return false;
                value_[vv] = upper_[vv];
                break;
            case VarStatus::AT_ZERO:
                // The parent rested this column free. Refuse rather than
                // guess if THIS instance's bounds (tightened relative to
                // the parent's) have since given it a finite side --
                // seating it at 0 could silently rest it off the new
                // bound. In practice this can only be the branch variable
                // itself, and only if it was declared free, which
                // validate_milp_problem does not forbid but which is rare.
                if (std::isfinite(lower_[vv]) || std::isfinite(upper_[vv])) return false;
                value_[vv] = 0.0;
                break;
            case VarStatus::BASIC:
                // parent.nonbasic_status disagreed with parent.basic_columns
                // about this column -- an inconsistent Basis, not a valid
                // one to seat.
                return false;
        }
        status_[vv] = st;
    }

    for (std::int32_t j = 0; j < n_struct_; ++j) {
        const auto jj = static_cast<std::size_t>(j);
        cost_[jj] = problem_.obj[jj] * scale_.col_scale[jj];
    }
    for (std::int32_t i = 0; i < n_slack_ + n_art_; ++i) {
        cost_[static_cast<std::size_t>(n_struct_ + i)] = 0.0;
    }
    sync_gpu_phase();
    // The parent's Devex weights approximate edge norms with respect to a
    // basis this instance is about to move away from immediately (the
    // first dual pivot changes it) -- restarting is cheap and avoids
    // carrying forward an approximation with no argument for relevance.
    reset_devex_weights();

    try {
        refactorize();
    } catch (const std::exception&) {
        return false;
    }
    recompute_basic_values();
    return true;
}

bool Simplex::out_of_time(int iterations) const {
    if (time_budget_seconds_ <= 0.0) return false;
    if ((iterations % kTimeCheckStride) != 0) return false;
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time_).count();
    return elapsed >= time_budget_seconds_;
}

LpStatus Simplex::run_dual_simplex(int& iterations) {
    iterations = 0;
    const auto m = static_cast<std::size_t>(n_rows_);

    std::vector<double> y, rc, rho, dir;

    // Can this nonbasic variable move in the direction that repairs the
    // selected row's infeasibility? alpha is already sigma-folded (see the
    // ratio test below): at a LOWER bound the variable can only increase,
    // so it helps only when alpha > 0; at an UPPER bound only decrease, so
    // only when alpha < 0; a FREE nonbasic may move either way. Shared by
    // both ratio-test passes so the two can never disagree about which
    // columns are candidates.
    const auto eligible = [](VarStatus st, double alpha) {
        switch (st) {
            case VarStatus::AT_LOWER: return alpha > 0.0;
            case VarStatus::AT_UPPER: return alpha < 0.0;
            default: return true;
        }
    };

    // Whether value_ was derived from the current basis (rather than
    // carried forward by incremental pivot updates) since the last pivot.
    // Both of this loop's terminal conclusions are checked against FRESH
    // values, never drifted ones -- see the two uses below.
    bool values_fresh = false;

    while (true) {
        if (iterations >= max_iterations_) return LpStatus::ITERATION_LIMIT;
        if (out_of_time(iterations)) return LpStatus::ITERATION_LIMIT;
        if (factor_.eta_count() >= refactorize_every_) {
            refactorize();
            recompute_basic_values();
            values_fresh = true;
        }

        // --- Row selection: the most primal-infeasible basic variable,
        // MEASURED IN ORIGINAL UNITS. sigma records WHICH way it is out:
        // +1 means above its upper bound (must come down), -1 means below
        // its lower bound (must go up).
        //
        // The whole iteration runs in Ruiz-scaled space, but the threshold
        // that decides "primal feasible, therefore optimal" must be applied
        // in the units finalize_result verifies against, or the two
        // disagree: a violation of 1e-9 on a scaled variable is
        // col_scale[j] times that once unscaled, and on a model like
        // grow15 (optimum ~1e8) that is enough to terminate inside the
        // scaled tolerance yet fail the original-space gate -- MEASURED:
        // the dual reached its own termination test, was downgraded to
        // NUMERICAL_FAILURE by verification, and the AUTO fallback then
        // resolved the model with the primal path. Scaling the comparison
        // costs one multiply per basic variable per iteration.
        std::int32_t leaving_row = -1;
        double worst = kDualPrimalFeasTol;
        double sigma = 0.0;
        double delta = 0.0;
        for (std::size_t i = 0; i < m; ++i) {
            const auto bv = static_cast<std::size_t>(basis_[i]);
            const double v = value_[bv];
            const double below = lower_[bv] - v;
            const double above = v - upper_[bv];
            const double to_original = unscale_factor(basis_[i]);
            if (below * to_original > worst) {
                worst = below * to_original;
                leaving_row = static_cast<std::int32_t>(i);
                sigma = -1.0;
                delta = below;
            }
            if (above * to_original > worst) {
                worst = above * to_original;
                leaving_row = static_cast<std::int32_t>(i);
                sigma = 1.0;
                delta = above;
            }
        }
        // Primal feasible, and dual feasibility is the loop invariant --
        // together that is optimality. But `value_` has been carried by
        // incremental pivot updates since the last refactorization, and
        // that accumulated drift is exactly what a termination test must
        // not be decided on: MEASURED on grow15, the drifted values passed
        // this test while the values re-derived from the same basis were
        // 1.2e-05 infeasible in original units -- enough to fail
        // finalize_result's verification and send a correct model down the
        // AUTO fallback. Re-derive and re-scan before concluding; if the
        // fresh values are still feasible, the basis genuinely is.
        if (leaving_row == -1) {
            if (values_fresh) return LpStatus::OPTIMAL;
            refactorize();
            recompute_basic_values();
            values_fresh = true;
            continue;
        }

        compute_tableau_row(leaving_row, rho);
        compute_duals(y);
        auto t0 = std::chrono::steady_clock::now();
        price(y, rc);
        pricing_seconds_ +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        // --- Dual ratio test: Harris two-pass (ESTABLISHED METHOD -- the
        // dual counterpart of the primal test above; Koberstein, "The Dual
        // Simplex Method: Techniques for a Fast and Stable Implementation",
        // PhD thesis, Paderborn, 2005, \S3.3, following Harris 1973 and the
        // EXPAND framework of Gill, Murray, Saunders & Wright 1989).
        //
        // Folding sigma into the row (alpha' = sigma * alpha) makes the
        // eligibility test independent of which bound was violated: a
        // nonbasic at its LOWER bound can only increase, so it helps only
        // when alpha' > 0; one at its UPPER bound can only decrease, so it
        // helps only when alpha' < 0. Both give a non-negative true ratio
        // |d_j| / |alpha'_j|, which is the dual step this pivot would cost.
        //
        // WHY TWO PASSES, and this is not a refinement -- a single-pass
        // min-ratio rule makes this algorithm unusable. The primal step is
        // t = delta / alpha_enter, so a pivot barely above kPivotTol turns
        // a small bound violation into an enormous one, which becomes the
        // next iteration's leaving row, and the infeasibility compounds.
        // MEASURED, before this test was two-pass: on Netlib grow15 the
        // single-pass rule drove the worst basic infeasibility to 1.4e+12
        // over 5219 iterations, lost dual feasibility entirely (max dual
        // infeasibility 7.0 against a 1e-9 tolerance), and then reported
        // INFEASIBLE on a feasible model -- one of 10 such false
        // infeasibility claims across the 89-instance set. Pass 1 relaxes
        // each ratio by a dual-feasibility tolerance to define the set of
        // acceptable steps; pass 2 takes the LARGEST PIVOT in that set,
        // which is the whole numerical point of the construction.
        double theta_max = std::numeric_limits<double>::infinity();
        for (std::int32_t v = 0; v < n_total_; ++v) {
            const auto vv = static_cast<std::size_t>(v);
            if (status_[vv] == VarStatus::BASIC) continue;
            if (upper_[vv] - lower_[vv] < 1e-12) continue; // fixed: cannot move

            const double alpha = sigma * rho[vv];
            const double alpha_mag = std::fabs(alpha);
            if (alpha_mag <= kPivotTol) continue;
            if (!eligible(status_[vv], alpha)) continue;

            // |rc| rather than rc/alpha: for an eligible column the two
            // agree in exact arithmetic, but rc may carry a small
            // wrong-signed round-off, and a NEGATIVE ratio admitted here
            // would drive the dual step backwards and destroy the dual
            // feasibility this algorithm's termination argument rests on.
            theta_max = std::min(theta_max,
                                  (std::fabs(rc[vv]) + kDualFeasTol) / alpha_mag);
        }

        std::int32_t enter = -1;
        double best_alpha_mag = 0.0;
        for (std::int32_t v = 0; v < n_total_; ++v) {
            const auto vv = static_cast<std::size_t>(v);
            if (status_[vv] == VarStatus::BASIC) continue;
            if (upper_[vv] - lower_[vv] < 1e-12) continue;

            const double alpha = sigma * rho[vv];
            const double alpha_mag = std::fabs(alpha);
            if (alpha_mag <= kPivotTol) continue;
            if (!eligible(status_[vv], alpha)) continue;

            const double ratio_true = std::fabs(rc[vv]) / alpha_mag;
            if (ratio_true <= theta_max && alpha_mag > best_alpha_mag) {
                best_alpha_mag = alpha_mag;
                enter = v;
            }
        }

        // No column can absorb the dual step: the dual is unbounded, so the
        // primal is infeasible. Held to the same standard as the OPTIMAL
        // conclusion above -- an infeasibility claim decided on a drifted
        // leaving row would be a wrong answer, not a slow one.
        if (enter == -1) {
            if (!values_fresh) {
                refactorize();
                recompute_basic_values();
                values_fresh = true;
                continue;
            }
            return LpStatus::INFEASIBLE;
        }

        const auto ee = static_cast<std::size_t>(enter);
        const double alpha_enter = sigma * rho[ee];

        column_into(enter, col_work_);
        ftran_column(col_work_, dir);

        // Primal step for the entering variable, derived from requiring the
        // leaving basic variable to land exactly on the bound it violated.
        const double t = delta / alpha_enter;

        for (std::size_t i = 0; i < m; ++i) {
            value_[static_cast<std::size_t>(basis_[i])] -= dir[i] * t;
        }
        value_[ee] += t;

        const auto lv = static_cast<std::size_t>(basis_[static_cast<std::size_t>(leaving_row)]);
        status_[lv] = (sigma > 0.0) ? VarStatus::AT_UPPER : VarStatus::AT_LOWER;
        value_[lv] = (sigma > 0.0) ? upper_[lv] : lower_[lv];
        basic_row_of_[lv] = -1;

        status_[ee] = VarStatus::BASIC;
        basic_row_of_[ee] = leaving_row;
        basis_[static_cast<std::size_t>(leaving_row)] = enter;

        apply_pivot_update(leaving_row, dir);
        values_fresh = false;
        ++iterations;
    }
}

// Multiplier taking a SCALED quantity for `var` back to original model
// units, from the conventions the constructor establishes: structurals are
// scaled as x' = x / C_j, slacks as s' = R_i s (docs/architecture/
// NUMERICS.md \S2). Artificials exist only in the scaled augmented system
// and have no original-space counterpart, so they carry a factor of 1.
double Simplex::unscale_factor(std::int32_t var) const {
    if (var < n_struct_) {
        return scale_.col_scale[static_cast<std::size_t>(var)];
    }
    if (var < n_struct_ + n_slack_) {
        return 1.0 / scale_.row_scale[static_cast<std::size_t>(var - n_struct_)];
    }
    return 1.0;
}

void Simplex::column_into(std::int32_t var, SparseColumn& out) const {
    out.clear();
    if (var < n_struct_) {
        const std::int32_t begin = A_csc_.col_ptr()[var];
        const std::int32_t end = A_csc_.col_ptr()[var + 1];
        for (std::int32_t k = begin; k < end; ++k) {
            out.emplace_back(A_csc_.row_idx()[k], A_csc_.values()[k]);
        }
    } else if (var < n_struct_ + n_slack_) {
        out.emplace_back(var - n_struct_, 1.0);
    } else {
        const std::int32_t row = var - n_struct_ - n_slack_;
        out.emplace_back(row, art_sign_[static_cast<std::size_t>(row)]);
    }
}

SparseColumn Simplex::column(std::int32_t var) const {
    SparseColumn col;
    if (var < n_struct_) {
        std::int32_t begin = A_csc_.col_ptr()[var];
        std::int32_t end = A_csc_.col_ptr()[var + 1];
        col.reserve(static_cast<std::size_t>(end - begin));
        for (std::int32_t k = begin; k < end; ++k) {
            col.emplace_back(A_csc_.row_idx()[k], A_csc_.values()[k]);
        }
    } else if (var < n_struct_ + n_slack_) {
        col.emplace_back(var - n_struct_, 1.0);
    } else {
        std::int32_t row = var - n_struct_ - n_slack_;
        col.emplace_back(row, art_sign_[static_cast<std::size_t>(row)]);
    }
    return col;
}

void Simplex::ftran_column(const SparseColumn& col, std::vector<double>& dir) const {
    dir.assign(static_cast<std::size_t>(n_rows_), 0.0);
    for (const auto& [row, val] : col) {
        dir[static_cast<std::size_t>(row)] = val;
    }
    factor_.ftran(dir); // in: matrix-row indexed; out: basis-position indexed
}

void Simplex::compute_duals(std::vector<double>& y) const {
    // y = B^-T c_B: c_B is indexed by basis position, y by matrix row --
    // exactly BasisFactorization::btran's contract.
    y.assign(static_cast<std::size_t>(n_rows_), 0.0);
    for (std::int32_t r = 0; r < n_rows_; ++r) {
        y[static_cast<std::size_t>(r)] =
            cost_[static_cast<std::size_t>(basis_[static_cast<std::size_t>(r)])];
    }
    factor_.btran(y);
}

void Simplex::compute_binv_row(std::int32_t row, std::vector<double>& out) const {
    // Row `row` of B^-1 is (B^-T e_row)^T, so one BTRAN against the unit
    // vector recovers what a stored dense inverse would have read directly.
    out.assign(static_cast<std::size_t>(n_rows_), 0.0);
    out[static_cast<std::size_t>(row)] = 1.0;
    factor_.btran(out);
}

void Simplex::compute_tableau_row(std::int32_t row, std::vector<double>& rho) const {
    {
        ScopedTimer t(profile_.rho_btran_seconds);
        compute_binv_row(row, solve_work_);
    }
    ScopedTimer t_assemble(profile_.rho_assemble_seconds);
    const std::vector<double>& binv_row = solve_work_;

    rho.assign(static_cast<std::size_t>(n_total_), 0.0);

    // Same shape, same guarantee, as price_cpu above: one writer per
    // output, ascending accumulation.
    SIHPS_OMP(omp parallel for schedule(static) if(parallel_mode_ == ParallelMode::PARALLEL ||
                                                   (parallel_mode_ == ParallelMode::AUTO &&
                                                    A_csc_.nnz() >= kParallelNnzThreshold)))
    for (std::int32_t j = 0; j < n_struct_; ++j) {
        double dot = 0.0;
        const std::int32_t begin = A_csc_.col_ptr()[j];
        const std::int32_t end = A_csc_.col_ptr()[j + 1];
        for (std::int32_t k = begin; k < end; ++k) {
            dot += A_csc_.values()[k] * binv_row[static_cast<std::size_t>(A_csc_.row_idx()[k])];
        }
        rho[static_cast<std::size_t>(j)] = dot;
    }
    for (std::int32_t i = 0; i < n_slack_; ++i) {
        rho[static_cast<std::size_t>(n_struct_ + i)] = binv_row[static_cast<std::size_t>(i)];
    }
    for (std::int32_t i = 0; i < n_art_; ++i) {
        rho[static_cast<std::size_t>(n_struct_ + n_slack_ + i)] =
            art_sign_[static_cast<std::size_t>(i)] * binv_row[static_cast<std::size_t>(i)];
    }
}

void Simplex::devex_update(std::int32_t entering, std::int32_t leaving_var, double pivot,
                            const std::vector<double>& rho) {
    // Harris (1973). For every nonbasic j, the exact steepest-edge weight
    // update is bounded below by (rho_j / pivot)^2 * w_entering; Devex
    // takes that bound as the new weight whenever it exceeds the current
    // one, which keeps weights cheap to maintain while remaining a valid
    // underestimate of the true edge norm.
    const auto ee = static_cast<std::size_t>(entering);
    const double w_enter = devex_weight_[ee];
    const double pivot_sq = pivot * pivot;
    if (!(pivot_sq > 0.0)) return; // degenerate pivot; leave weights untouched

    const double ratio_scale = w_enter / pivot_sq;

    double max_weight = 1.0;
    for (std::int32_t j = 0; j < n_total_; ++j) {
        const auto jj = static_cast<std::size_t>(j);
        if (j == entering) continue;
        if (status_[jj] == VarStatus::BASIC) continue;
        const double r = rho[jj];
        if (r == 0.0) {
            max_weight = std::max(max_weight, devex_weight_[jj]);
            continue;
        }
        const double candidate = r * r * ratio_scale;
        if (candidate > devex_weight_[jj]) devex_weight_[jj] = candidate;
        max_weight = std::max(max_weight, devex_weight_[jj]);
    }

    // The variable that just left the basis becomes nonbasic and needs a
    // weight of its own.
    const auto lv = static_cast<std::size_t>(leaving_var);
    devex_weight_[lv] = std::max(1.0, ratio_scale);
    max_weight = std::max(max_weight, devex_weight_[lv]);

    // Restart the reference framework once weights have grown large: they
    // are approximations, and beyond a few orders of magnitude the
    // accumulated error makes them worse than starting over.
    if (max_weight > 1e8) {
        std::fill(devex_weight_.begin(), devex_weight_.end(), 1.0);
    }
}

void Simplex::price_cpu(const std::vector<double>& y, std::vector<double>& rc) const {
    // Column-parallel: rc[j] is written by exactly one iteration and its
    // dot product is accumulated in ascending k, so the result is
    // bit-identical to the serial loop and to any other thread count
    // (parallel/Parallel.hpp). Gated on nnz because on a small model the
    // fork/barrier costs more than the entire pass.
    SIHPS_OMP(omp parallel for schedule(static) if(parallel_mode_ == ParallelMode::PARALLEL ||
                                                   (parallel_mode_ == ParallelMode::AUTO &&
                                                    A_csc_.nnz() >= kParallelNnzThreshold)))
    for (std::int32_t j = 0; j < n_struct_; ++j) {
        double dot = 0.0;
        std::int32_t begin = A_csc_.col_ptr()[j];
        std::int32_t end = A_csc_.col_ptr()[j + 1];
        for (std::int32_t k = begin; k < end; ++k) {
            dot += A_csc_.values()[k] * y[static_cast<std::size_t>(A_csc_.row_idx()[k])];
        }
        rc[static_cast<std::size_t>(j)] = cost_[static_cast<std::size_t>(j)] - dot;
    }
    for (std::int32_t i = 0; i < n_slack_; ++i) {
        std::int32_t s = n_struct_ + i;
        rc[static_cast<std::size_t>(s)] =
            cost_[static_cast<std::size_t>(s)] - y[static_cast<std::size_t>(i)];
    }
    for (std::int32_t i = 0; i < n_art_; ++i) {
        std::int32_t a = n_struct_ + n_slack_ + i;
        rc[static_cast<std::size_t>(a)] = cost_[static_cast<std::size_t>(a)] -
                                          art_sign_[static_cast<std::size_t>(i)] *
                                              y[static_cast<std::size_t>(i)];
    }
}

void Simplex::price(const std::vector<double>& y, std::vector<double>& rc) {
    rc.assign(static_cast<std::size_t>(n_total_), 0.0);
    if (backend_ == PricingBackend::CPU) {
        price_cpu(y, rc);
    } else {
        // The full reduced-cost vector on the host. Only two callers need
        // it -- the dual ratio test and the final dual-residual check --
        // and neither is the primal iteration path, which uses
        // select_entering() and never materializes this vector at all.
        gpu_pricer_->price_to_host(y.data(), rc.data());
    }
}

bool Simplex::select_entering(std::int32_t& enter, double& enter_dq) {
    enter = -1;
    enter_dq = 0.0;

    if (backend_ == PricingBackend::GPU) {
        const gpu::PricingCandidate cand = gpu_pricer_->price_and_select(
            y_work_.data(), reinterpret_cast<const std::uint8_t*>(status_.data()),
            pricing_rule_ == PricingRule::DEVEX, kOptTol);
        if (cand.index < 0) return false;
        enter = cand.index;
        enter_dq = static_cast<double>(cand.direction);
        return true;
    }

    price_cpu(y_work_, rc_work_);

    // Eligibility is identical for both pricing rules; only the SCORE
    // differs:
    //   DANTZIG: |d_j|            -- steepest apparent descent
    //   DEVEX:   d_j^2 / w_j      -- descent per unit edge length
    double best_score = 0.0;
    for (std::int32_t v = 0; v < n_total_; ++v) {
        const auto vv = static_cast<std::size_t>(v);
        if (status_[vv] == VarStatus::BASIC) continue;
        if (upper_[vv] - lower_[vv] < 1e-12) continue; // fixed: can never move
        const double d = rc_work_[vv];

        double dq = 0.0;
        if (status_[vv] == VarStatus::AT_LOWER) {
            if (d < -kOptTol) dq = 1.0;
        } else if (status_[vv] == VarStatus::AT_UPPER) {
            if (d > kOptTol) dq = -1.0;
        } else { // AT_ZERO: a free variable may move either way
            if (std::fabs(d) > kOptTol) dq = (d < 0.0) ? 1.0 : -1.0;
        }
        if (dq == 0.0) continue;

        const double score = (pricing_rule_ == PricingRule::DEVEX)
                                  ? (d * d) / devex_weight_[vv]
                                  : std::fabs(d);
        // Strictly greater, scanning ascending, so ties resolve to the
        // LOWEST index -- the same tie-break the device reduction's
        // comparator applies, so the two backends pick the same column
        // whenever their scores agree.
        if (score > best_score) {
            best_score = score;
            enter = v;
            enter_dq = dq;
        }
    }
    return enter != -1;
}

bool Simplex::repair_basis_position(std::int32_t pos, std::int32_t unmatched_row) {
    const std::int32_t replacement = n_struct_ + n_slack_ + unmatched_row;
    const auto rr = static_cast<std::size_t>(replacement);
    if (basic_row_of_[rr] >= 0) return false; // already basic elsewhere; cannot repair this way

    const std::int32_t displaced = basis_[static_cast<std::size_t>(pos)];
    const auto dd = static_cast<std::size_t>(displaced);
    basic_row_of_[dd] = -1;
    if (std::isfinite(lower_[dd])) {
        status_[dd] = VarStatus::AT_LOWER;
        value_[dd] = lower_[dd];
    } else if (std::isfinite(upper_[dd])) {
        status_[dd] = VarStatus::AT_UPPER;
        value_[dd] = upper_[dd];
    } else {
        status_[dd] = VarStatus::AT_ZERO;
        value_[dd] = 0.0;
    }

    basis_[static_cast<std::size_t>(pos)] = replacement;
    basic_row_of_[rr] = pos;
    status_[rr] = VarStatus::BASIC;
    return true;
}

void Simplex::refactorize() {
    std::vector<SparseColumn> columns(static_cast<std::size_t>(n_rows_));
    for (std::int32_t r = 0; r < n_rows_; ++r) {
        columns[static_cast<std::size_t>(r)] = column(basis_[static_cast<std::size_t>(r)]);
    }

    auto result = factor_.factorize(n_rows_, columns);
    if (!result.ok) {
        throw std::runtime_error("Simplex: basis factorization failed");
    }

    // Basis repair (docs/architecture/NUMERICS.md \S5): a dependent basis
    // column is a legitimate runtime event, not a programming error, so it
    // is repaired by substituting the unmatched row's artificial and
    // refactorizing -- once. A second failure means repair is not
    // converging, which is reported rather than retried indefinitely.
    if (!result.singular.empty()) {
        for (const auto& [pos, unmatched_row] : result.singular) {
            if (!repair_basis_position(pos, unmatched_row)) {
                throw std::runtime_error("Simplex: basis repair failed (logical already basic)");
            }
        }
        for (std::int32_t r = 0; r < n_rows_; ++r) {
            columns[static_cast<std::size_t>(r)] = column(basis_[static_cast<std::size_t>(r)]);
        }
        result = factor_.factorize(n_rows_, columns);
        if (!result.ok || !result.singular.empty()) {
            throw std::runtime_error("Simplex: basis still singular after repair");
        }
    }
    ++refactorizations_;
}

void Simplex::recompute_basic_values() {
    const auto m = static_cast<std::size_t>(n_rows_);

    // Member scratch, sized at construction: this runs at every
    // refactorization, which is inside the iteration loop, and prompt.md
    // \S3.1 draws its line at the solve rather than at the pivot.
    std::vector<double>& residual = residual_work_;
    std::copy(rhs_scaled_.begin(), rhs_scaled_.end(), residual.begin());

    std::vector<double>& x_struct = x_struct_work_;
    for (std::int32_t j = 0; j < n_struct_; ++j) {
        const auto jj = static_cast<std::size_t>(j);
        x_struct[jj] = (status_[jj] != VarStatus::BASIC) ? value_[jj] : 0.0;
    }
    if (n_struct_ > 0) {
        A_scaled_.multiply(x_struct.data(), ax_work_.data()); // multiply() overwrites ax
        for (std::size_t i = 0; i < m; ++i) residual[i] -= ax_work_[i];
    }

    for (std::int32_t i = 0; i < n_rows_; ++i) {
        const auto ii = static_cast<std::size_t>(i);
        const auto s = static_cast<std::size_t>(n_struct_ + i);
        if (status_[s] != VarStatus::BASIC) residual[ii] -= value_[s];
        const auto a = static_cast<std::size_t>(n_struct_ + n_slack_ + i);
        if (status_[a] != VarStatus::BASIC) residual[ii] -= art_sign_[ii] * value_[a];
    }

    // x_B = B^-1 residual: residual is matrix-row indexed going in, basis-
    // position indexed coming out.
    factor_.ftran(residual);
    for (std::size_t r = 0; r < m; ++r) {
        value_[static_cast<std::size_t>(basis_[r])] = residual[r];
    }
}

void Simplex::apply_pivot_update(std::int32_t leaving_row, const std::vector<double>& dir) {
    // Records the basis change as one eta vector. A refused update (pivot
    // too small to invert safely) is recovered by discarding the eta file
    // and factorizing the new basis outright, rather than continuing
    // against a representation known to be unsound.
    if (!factor_.update(leaving_row, dir)) {
        refactorize();
        // refactorize() may have REPAIRED the basis, in which case the
        // incrementally-maintained basic values no longer correspond to it.
        // Rederiving them is cheap on this rare path and avoids carrying a
        // silent inconsistency forward.
        recompute_basic_values();
    }
}

LpStatus Simplex::run_primal_simplex(bool phase1, int& iterations) {
    iterations = 0;
    const auto m = static_cast<std::size_t>(n_rows_);

    while (true) {
        if (iterations >= max_iterations_) return LpStatus::ITERATION_LIMIT;
        if (out_of_time(iterations)) return LpStatus::ITERATION_LIMIT;
        // Refactorize on accumulated eta count, not iteration count: PFI
        // solve cost and numerical drift both track the eta file, and bound
        // flips advance the iteration counter without pushing an eta.
        if (factor_.eta_count() >= refactorize_every_) {
            ScopedTimer t(profile_.refactor_seconds);
            refactorize();
            recompute_basic_values();
        }

        {
            ScopedTimer t(profile_.duals_seconds);
            compute_duals(y_work_);
        }

        // Pricing AND the entering-variable search are timed together,
        // because on the GPU backend they are one fused operation: the
        // reduced costs are never brought back to the host, so there is no
        // boundary between "compute d" and "argmax over d" to time
        // separately. Timing them as one keeps the CPU/GPU comparison in
        // benchmarks/bench_pricing_backend.cpp honest -- it would flatter
        // the GPU to charge the CPU for a scan the GPU also performs but
        // has folded into its kernel.
        std::int32_t enter = -1;
        double enter_dq = 0.0;
        const auto t0 = std::chrono::steady_clock::now();
        const bool found = select_entering(enter, enter_dq);
        const double price_elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        pricing_seconds_ += price_elapsed;
        profile_.price_seconds += price_elapsed;

        if (!found) return LpStatus::OPTIMAL;

        std::vector<double>& dir = dir_work_;
        {
            ScopedTimer t(profile_.ftran_seconds);
            column_into(enter, col_work_);
            ftran_column(col_work_, dir);
        }
        const auto t_ratio = std::chrono::steady_clock::now();

        // Harris two-pass ratio test (docs/research/SOTA.md \S1.4.2).
        // Pass 1 uses bounds relaxed by kHarrisExpand to determine the
        // largest step worth considering.
        double theta1 = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < m; ++i) {
            const double a = enter_dq * dir[i];
            if (std::fabs(a) <= kPivotTol) continue;
            const auto bv = static_cast<std::size_t>(basis_[i]);
            const double bval = value_[bv];
            const double limit =
                (a > 0.0) ? (lower_[bv] - kHarrisExpand) : (upper_[bv] + kHarrisExpand);
            if (!std::isfinite(limit)) continue;
            double ratio = (bval - limit) / a;
            if (ratio < 0.0) ratio = 0.0;
            theta1 = std::min(theta1, ratio);
        }

        profile_.ratio_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t_ratio).count();

        const auto ee = static_cast<std::size_t>(enter);
        const double own_limit = upper_[ee] - lower_[ee];

        if (own_limit <= theta1) {
            // Bound flip: the entering variable reaches its opposite bound
            // before any basic variable blocks. No basis change.
            if (!std::isfinite(own_limit)) {
                return phase1 ? LpStatus::NUMERICAL_FAILURE : LpStatus::UNBOUNDED;
            }
            for (std::size_t i = 0; i < m; ++i) {
                value_[static_cast<std::size_t>(basis_[i])] -= own_limit * enter_dq * dir[i];
            }
            status_[ee] = (enter_dq > 0.0) ? VarStatus::AT_UPPER : VarStatus::AT_LOWER;
            value_[ee] = (enter_dq > 0.0) ? upper_[ee] : lower_[ee];
            ++iterations;
            continue;
        }

        if (!std::isfinite(theta1)) {
            return phase1 ? LpStatus::NUMERICAL_FAILURE : LpStatus::UNBOUNDED;
        }

        // Pass 2: among rows whose TRUE-bound ratio is within theta1, take
        // the largest-magnitude pivot -- this is the numerical-stability
        // payoff of the two-pass structure.
        std::int32_t leaving_row = -1;
        double best_pivot_mag = -1.0;
        double theta_final = 0.0;
        for (std::size_t i = 0; i < m; ++i) {
            const double a = enter_dq * dir[i];
            if (std::fabs(a) <= kPivotTol) continue;
            const auto bv = static_cast<std::size_t>(basis_[i]);
            const double true_limit = (a > 0.0) ? lower_[bv] : upper_[bv];
            if (!std::isfinite(true_limit)) continue;
            double ratio_true = (value_[bv] - true_limit) / a;
            if (ratio_true < 0.0) ratio_true = 0.0;
            if (ratio_true <= theta1 && std::fabs(a) > best_pivot_mag) {
                best_pivot_mag = std::fabs(a);
                leaving_row = static_cast<std::int32_t>(i);
                theta_final = ratio_true;
            }
        }
        if (leaving_row == -1) {
            return LpStatus::NUMERICAL_FAILURE;
        }

        // Devex needs the pivot ROW, and it must be taken from the basis as
        // it stands BEFORE this pivot is applied (apply_pivot_update below
        // pushes an eta that changes what the factorization represents).
        const auto t_rho = std::chrono::steady_clock::now();
        if (pricing_rule_ == PricingRule::DEVEX) {
            if (backend_ == PricingBackend::GPU) {
                // Only the BTRAN result is needed on the host: the pivot
                // row itself is A^T times this vector, which the device
                // forms with the same cuSPARSE SpMV and assembly kernel
                // pricing uses. The n_total-wide row never crosses PCIe.
                compute_binv_row(leaving_row, binv_row_work_);
            } else {
                compute_tableau_row(leaving_row, rho_work_);
            }
        }
        pricing_seconds_ +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t_rho).count();
        // rho_btran/rho_assemble are charged inside compute_tableau_row so
        // the BTRAN and the A^T pass can be told apart -- which is the
        // whole point of this breakdown.

        for (std::size_t i = 0; i < m; ++i) {
            value_[static_cast<std::size_t>(basis_[i])] -= theta_final * enter_dq * dir[i];
        }

        const auto lv = static_cast<std::size_t>(basis_[static_cast<std::size_t>(leaving_row)]);
        const double a_leaving = enter_dq * dir[static_cast<std::size_t>(leaving_row)];
        status_[lv] = (a_leaving > 0.0) ? VarStatus::AT_LOWER : VarStatus::AT_UPPER;
        value_[lv] = (a_leaving > 0.0) ? lower_[lv] : upper_[lv];
        basic_row_of_[lv] = -1;

        value_[ee] += enter_dq * theta_final;
        status_[ee] = VarStatus::BASIC;
        basic_row_of_[ee] = leaving_row;
        basis_[static_cast<std::size_t>(leaving_row)] = enter;

        if (pricing_rule_ == PricingRule::DEVEX) {
            const auto t_dx = std::chrono::steady_clock::now();
            if (backend_ == PricingBackend::GPU) {
                // Weights stay on the device across iterations; this call
                // queues work and returns without synchronizing. `status_`
                // is read AFTER the pivot bookkeeping above, matching what
                // the CPU path sees.
                gpu_pricer_->devex_update(binv_row_work_.data(), enter,
                                           static_cast<std::int32_t>(lv),
                                           dir[static_cast<std::size_t>(leaving_row)]);
            } else {
                devex_update(enter, static_cast<std::int32_t>(lv),
                              dir[static_cast<std::size_t>(leaving_row)], rho_work_);
            }
            const double dx_elapsed =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t_dx).count();
            pricing_seconds_ += dx_elapsed;
            profile_.devex_seconds += dx_elapsed;
        }

        {
            ScopedTimer t(profile_.update_seconds);
            apply_pivot_update(leaving_row, dir);
        }
        ++iterations;
        ++profile_.iterations;
    }
}

void Simplex::finalize_result(LpStatus status, LpResult& result) {
    result.status = status;
    result.pricing_seconds = pricing_seconds_;
    result.refactorizations = refactorizations_;
    result.profile = profile_;

    if (status != LpStatus::OPTIMAL) {
        return;
    }

    // Verify the BASIS, not the drifted iterate: rebuild B^-1 from scratch
    // and re-derive the basic values from it, so the residuals below test
    // whether this basis genuinely solves the problem rather than whether
    // the incrementally-maintained numbers happen to look self-consistent
    // (docs/architecture/NUMERICS.md \S6).
    try {
        refactorize();
        recompute_basic_values();
    } catch (const std::exception&) {
        result.status = LpStatus::NUMERICAL_FAILURE;
        return;
    }

    // Unscale x = C .* x' (docs/architecture/NUMERICS.md \S2) before any
    // downstream quantity is derived from it, so the objective, residuals,
    // and returned solution are all reported in ORIGINAL model units.
    result.x.assign(static_cast<std::size_t>(n_struct_), 0.0);
    for (std::int32_t j = 0; j < n_struct_; ++j) {
        const auto jj = static_cast<std::size_t>(j);
        result.x[jj] = value_[jj] * scale_.col_scale[jj];
    }
    result.objective_value = 0.0;
    for (std::int32_t j = 0; j < n_struct_; ++j) {
        result.objective_value +=
            problem_.obj[static_cast<std::size_t>(j)] * result.x[static_cast<std::size_t>(j)];
    }

    // y' (scaled dual) is what price() must be called with to get
    // consistent scaled reduced costs; unscale into the result only after.
    std::vector<double> y_scaled;
    compute_duals(y_scaled);
    std::vector<double> rc_scaled;
    price(y_scaled, rc_scaled);

    result.y_dual.assign(static_cast<std::size_t>(n_rows_), 0.0);
    for (std::int32_t i = 0; i < n_rows_; ++i) {
        const auto ii = static_cast<std::size_t>(i);
        result.y_dual[ii] = y_scaled[ii] * scale_.row_scale[ii]; // y = R .* y'
    }

    // Primal residual, in ORIGINAL units, as direct row-bound feasibility:
    // Ax (original A, unscaled x) must lie in
    // [rhs - slack_upper, rhs - slack_lower] (LpProblem.cpp's Ax + s = rhs
    // convention). This is the caller-facing feasibility statement and
    // needs no reference to this engine's internally-scaled slack/
    // artificial representation.
    std::vector<double> ax(static_cast<std::size_t>(n_rows_), 0.0);
    if (n_struct_ > 0) {
        problem_.A.multiply(result.x.data(), ax.data());
    }
    // PER-ROW violation with a component-wise noise budget (|A|*|x|,
    // Higham's standard scaled-residual denominator) subtracted out, not
    // one violation normalized by the model-wide max |rhs|
    // (LpSolver.cpp's `original_space_primal_residual` and
    // MilpSolver.cpp's `feasible_point` had the identical bug: a single
    // large-RHS row can inflate the effective absolute tolerance for every
    // unrelated small-RHS row in the same model -- MEASURED on MIPLIB
    // `flugplinf`). See `kActivityNoiseRatio`'s own comment above for why
    // the noise budget is additive and tied to machine-precision-scale
    // noise, not to `kFinalTol` itself (MEASURED on Netlib `shell`/
    // `perold` and MIPLIB `ej`).
    const auto* row_ptr = problem_.A.row_ptr();
    const auto* col_idx = problem_.A.col_idx();
    const auto* values = problem_.A.values();
    double row_violation = 0.0;
    for (std::int32_t i = 0; i < n_rows_; ++i) {
        const auto ii = static_cast<std::size_t>(i);
        double abs_activity = std::fabs(problem_.rhs[ii]);
        for (std::int32_t k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
            abs_activity +=
                std::fabs(values[k]) * std::fabs(result.x[static_cast<std::size_t>(col_idx[k])]);
        }
        const double noise_budget = kActivityNoiseRatio * abs_activity;
        const double lo = problem_.rhs[ii] - problem_.slack_upper[ii];
        const double hi = problem_.rhs[ii] - problem_.slack_lower[ii];
        row_violation = std::max(row_violation, std::max(0.0, (lo - ax[ii]) - noise_budget));
        row_violation = std::max(row_violation, std::max(0.0, (ax[ii] - hi) - noise_budget));
    }
    double bound_violation = 0.0;
    for (std::int32_t j = 0; j < n_struct_; ++j) {
        const auto jj = static_cast<std::size_t>(j);
        bound_violation =
            std::max(bound_violation, std::max(0.0, problem_.lower[jj] - result.x[jj]));
        bound_violation =
            std::max(bound_violation, std::max(0.0, result.x[jj] - problem_.upper[jj]));
    }
    result.primal_residual = std::max(row_violation, bound_violation);

    // Dual residual, in ORIGINAL units: rc_orig_j = rc'_j / C_j for
    // structural columns, rc_orig_i = R_i * rc'_i for slack columns
    // (derived from c - A^T y under A' = R A C, c' = C c, y = R y' --
    // Scaling.hpp). Both factors are strictly positive, so sign (hence the
    // AT_LOWER/AT_UPPER feasibility direction) is unaffected; only the
    // magnitude compared against the tolerance changes. Artificials are
    // always frozen to [0,0] by this point (solve(), post-phase-1), so the
    // fixed-variable skip below excludes them without needing a separate
    // range check.
    double dual_violation = 0.0;
    for (std::int32_t v = 0; v < n_total_; ++v) {
        const auto vv = static_cast<std::size_t>(v);
        if (status_[vv] == VarStatus::BASIC) continue;
        if (upper_[vv] - lower_[vv] < 1e-12) continue; // fixed/frozen (incl. artificials)

        const double d = (v < n_struct_)
                              ? rc_scaled[vv] / scale_.col_scale[vv]
                              : rc_scaled[vv] * scale_.row_scale[vv - static_cast<std::size_t>(n_struct_)];

        if (status_[vv] == VarStatus::AT_LOWER) {
            dual_violation = std::max(dual_violation, std::max(0.0, -d));
        } else if (status_[vv] == VarStatus::AT_UPPER) {
            dual_violation = std::max(dual_violation, std::max(0.0, d));
        } else { // AT_ZERO: a free nonbasic variable must be dual-neutral
            dual_violation = std::max(dual_violation, std::fabs(d));
        }
    }
    const double obj_norm = vector_inf_norm(problem_.obj);
    result.dual_residual = dual_violation / (1.0 + obj_norm);

    if (result.primal_residual > kFinalTol || result.dual_residual > kFinalTol) {
        result.status = LpStatus::NUMERICAL_FAILURE;
    }
}

LpStatus Simplex::run_primal_two_phase(LpResult& result) {
    setup_phase1();
    int p1_iters = 0;
    LpStatus s1 = run_primal_simplex(true, p1_iters);
    result.phase1_iterations = p1_iters;

    if (s1 == LpStatus::ITERATION_LIMIT || s1 == LpStatus::NUMERICAL_FAILURE) {
        return s1;
    }

    double phase1_obj = 0.0;
    for (std::int32_t i = 0; i < n_rows_; ++i) {
        const std::int32_t bv = basis_[static_cast<std::size_t>(i)];
        if (bv >= n_struct_ + n_slack_) {
            phase1_obj += std::fabs(value_[static_cast<std::size_t>(bv)]);
        }
    }
    const double phase1_threshold =
        kPhase1RelTol * (1.0 + vector_inf_norm(rhs_scaled_)) + kFinalTol;
    if (phase1_obj > phase1_threshold) {
        return LpStatus::INFEASIBLE;
    }

    // Drive any artificial still basic at ~0 to exactly 0 and freeze
    // every artificial at [0,0] so phase 2 can never reintroduce one.
    for (std::int32_t i = 0; i < n_rows_; ++i) {
        const auto a = static_cast<std::size_t>(n_struct_ + n_slack_ + i);
        if (status_[a] == VarStatus::BASIC) {
            value_[a] = 0.0;
        }
        lower_[a] = 0.0;
        upper_[a] = 0.0;
    }
    for (std::int32_t j = 0; j < n_struct_; ++j) {
        const auto jj = static_cast<std::size_t>(j);
        // c' = C .* c (docs/architecture/SYSTEM.md \S2.4).
        cost_[jj] = problem_.obj[jj] * scale_.col_scale[jj];
    }
    for (std::int32_t i = 0; i < n_slack_; ++i) {
        cost_[static_cast<std::size_t>(n_struct_ + i)] = 0.0;
    }
    for (std::int32_t i = 0; i < n_art_; ++i) {
        cost_[static_cast<std::size_t>(n_struct_ + n_slack_ + i)] = 0.0;
    }

    // Phase 2's objective and its frozen artificial bounds are both new
    // to the device-side cache.
    sync_gpu_phase();

    // Devex weights approximate edge norms with respect to a specific
    // objective. Phase 2 installs an entirely different one, so the
    // phase-1 reference framework carries no useful information and is
    // restarted rather than inherited.
    reset_devex_weights();

    int p2_iters = 0;
    LpStatus s2 = run_primal_simplex(false, p2_iters);
    result.phase2_iterations = p2_iters;
    return s2;
}

LpResult Simplex::solve() {
    LpResult result;
    try {
        // ALGORITHM SELECTION (docs/architecture/LP.md \S2's decision
        // table; NUMERICS.md \S5's fallback chain, now warm -> dual ->
        // primal -> reported failure).
        //
        // Absent a warm-started parent basis (the block just below), AUTO
        // resolves to PRIMAL on a cold start, and that is the
        // architecture's own rule rather than a preference: LP.md \S2
        // selects dual simplex when a PARENT BASIS is available and
        // dual-feasible for the child -- the B&B re-solve shape this
        // engine exists to serve -- and primal simplex on a cold start.
        //
        // EXPERIMENTAL RESULT supporting that cold-start rule (Netlib,
        // 89 instances up to 2600 rows, same tolerances, dual vs primal):
        // total time +153%, total iterations +65.5%. The spread is what
        // matters more than the aggregate -- d6cube 5.15s -> 0.22s (31,033
        // -> 1,309 iterations) against pilotnov 0.30s -> 4.33s and pilot87
        // 11.9s -> 65.3s. Cold-start dual is not uniformly worse; it is
        // unpredictable, which on a cold start buys nothing, since there is
        // no warm basis whose dual feasibility would make it the cheap
        // option. DUAL remains selectable for that measurement and for the
        // warm-start path below.
        //
        // WARM START (docs/architecture/LP.md \S1/\S2's long-stated missing
        // piece; MILP.md's prerequisite -- every B&B node is a warm child
        // solve). A caller-supplied parent basis gets first refusal, ahead
        // of both the cold paths below: if it seats and the dual-simplex
        // repair that follows clears the SAME original-space verification
        // gate every other path is held to, that is the reported result.
        // LpAlgorithm::PRIMAL opts out entirely -- that value's existing
        // meaning is "pin the primal path," and honoring it here means not
        // silently overriding an explicit caller choice.
        if (warm_start_basis_ != nullptr && algorithm_ != LpAlgorithm::PRIMAL &&
            warm_start_basis_->n_struct == n_struct_ && warm_start_basis_->n_slack == n_slack_ &&
            warm_start_basis_->n_art == n_art_) {
            result.warm_start_attempted = true;
            if (seat_basis(*warm_start_basis_)) {
                int d_iters = 0;
                const LpStatus sd = run_dual_simplex(d_iters);
                result.dual_iterations = d_iters;
                result.used_dual_simplex = true;
                result.used_warm_start = true;

                if (sd == LpStatus::OPTIMAL || sd == LpStatus::INFEASIBLE) {
                    finalize_result(sd, result);
                    if (result.status != LpStatus::NUMERICAL_FAILURE) {
                        return result;
                    }
                }

                // The warm path did not produce a verified result -- fall
                // back exactly as the existing cold-dual branch below
                // does. The flags that say WHICH path produced the
                // REPORTED result flip back so a benchmark cannot
                // misattribute it; dual_iterations is left as-is because
                // that work genuinely happened.
                result.used_dual_simplex = false;
                result.used_warm_start = false;
            }
        }

        if (algorithm_ == LpAlgorithm::DUAL && setup_dual_feasible_start()) {
            int d_iters = 0;
            const LpStatus sd = run_dual_simplex(d_iters);
            result.dual_iterations = d_iters;
            result.used_dual_simplex = true;

            if (sd == LpStatus::OPTIMAL || sd == LpStatus::INFEASIBLE) {
                finalize_result(sd, result);
                // finalize_result downgrades to NUMERICAL_FAILURE when the
                // terminal basis fails verification in original units. That
                // is precisely the condition the fallback chain exists for,
                // so under AUTO it is a reason to try the other algorithm --
                // not a result to report.
                if (result.status != LpStatus::NUMERICAL_FAILURE) {
                    return result;
                }
            }

            // The dual path did not produce a verified result. Fall back
            // to the primal path (NUMERICS.md \S5) rather than reporting a
            // failure the other algorithm may not share.
            //
            // dual_iterations is deliberately NOT cleared: that work was
            // performed, and a benchmark that hides it would understate the
            // cost of the fallback. used_dual_simplex tracks which
            // algorithm produced the REPORTED result, so it flips back to
            // false here.
            result.used_dual_simplex = false;
        }

        const LpStatus s = run_primal_two_phase(result);
        finalize_result(s, result);
        return result;
    } catch (const std::exception&) {
        result.status = LpStatus::NUMERICAL_FAILURE;
        return result;
    }
}

} // namespace sihps
