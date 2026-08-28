#include "Presolve.hpp"

#include "../sparse/CSCMatrix.hpp"
#include "../sparse/Convert.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>

namespace sihps {
namespace {

// Presolve tolerances point in OPPOSITE directions and must not share a
// constant. The two failure modes are not symmetric:
//
//   - Declaring INFEASIBLE wrongly destroys a solvable model. So the
//     infeasibility test is GENEROUS: a larger tolerance fires it less
//     often, which is the safe direction.
//
//   - Firing a reduction wrongly (dropping a row that is still binding,
//     pinning a variable that still has slack) silently changes the
//     problem. So reduction tests are CONSERVATIVE: a smaller tolerance
//     fires them less often, which is the safe direction here.
//
// Using one constant for both -- and worse, scaling both by total activity
// magnitude -- is what made Netlib bore3d fail: the forcing-row test fired
// on rows with real slack, and eight passes later an emptied row no longer
// contained zero.
constexpr double kInfeasTol = 1e-7;
constexpr double kReductionTol = 1e-9;

// A bound must improve by at least this much RELATIVE to its own magnitude
// before propagation accepts it. An absolute threshold is not enough: on a
// bound of magnitude 1e6, repeated propagation keeps "improving" it by
// amounts that are pure floating-point noise, and each such step eats a
// sliver of the true feasible region. Left unchecked that drift eventually
// crosses l > u and reports a perfectly feasible model as INFEASIBLE --
// observed directly on Netlib bore3d, which converged by pass 5 and then
// went infeasible at pass 9 under an absolute threshold.
constexpr double kBoundImproveTol = 1e-7;

// Every accepted bound is relaxed outward by this much (relative) before
// being stored, so that rounding in the propagation arithmetic can never
// cut into the genuine feasible region. Presolve must only ever produce a
// RELAXATION of the true bound, never an over-tightening.
constexpr double kBoundRelax = 1e-9;

// |u - l| below this treats a column as fixed.
constexpr double kFixTol = 1e-11;

// GCD row tightening (see Presolve.hpp's own comment on
// `enable_gcd_tightening`) reads a row coefficient as "integer" only within
// this relative tolerance -- tight enough that a coefficient carrying real
// fractional structure (not floating-point noise around a whole number)
// never gets misclassified, since misclassifying even one coefficient in a
// row breaks this reduction's soundness argument entirely (Ax is a
// multiple of gcd(a_j) only when EVERY term genuinely is).
constexpr double kGcdCoeffIntTol = 1e-9;
// Coefficients larger than this are declined rather than rounded to an
// int64_t gcd input -- a defensive guard against overflow in the
// std::llround conversion, not a scenario expected on any real model this
// project has measured against.
constexpr double kGcdMaxCoeffMagnitude = 1.0e15;

// Doubleton substitution divides by the eliminated variable's own row
// coefficient (both intercept and slope carry 1/a_e). A tiny a_e amplifies
// any noise in the row's midpoint / the surviving variable's coefficient
// into the substitution -- and unlike a bound derivation that affects one
// column reversibly (the simplex can still re-derive a tighter bound
// later), a doubleton elimination reads permanently into
// column_is_doubleton/postsolve with no further chance to re-verify it.
// Gated more conservatively than tighten_lower/upper's own derived-limit
// checks for exactly that reason.
constexpr double kDoubletonPivotFloor = 1e-9;      // absolute floor on |a_e|
constexpr double kDoubletonRelativeFloor = 1e-8;   // |a_e| >= this * |a_s|

// A SEPARATE numerical guard from the pivot floor above, for a different
// failure mode -- MEASURED, not theoretical, on Netlib `greenbea`/
// `greenbeb`. postsolve() clamps the reconstructed x[eliminated_col]
// exactly into its own recorded bounds (see DoubletonElimination's own
// comment), which is correct and necessary -- but that clamp can leave the
// DROPPED row's own equation violated by up to roughly
// kBoundRelax*(1+|eliminated_col's bound magnitude|), because
// tighten_lower/upper's own outward relaxation on the IMPLIED bound placed
// on the surviving variable carries straight through the affine
// reconstruction with no compensating slack. On a model whose row RHS
// values are small (so original_space_primal_residual's
// `/(1+rhs_norm)` scaling offers little protection -- exactly greenbea's
// case) that leak can exceed the project's fixed final-verification gate
// (kFinalPrimalTol, LpSolver.cpp) even though it is minuscule in relative
// terms. kBoundRelax and kFinalPrimalTol are both fixed, shared project
// constants (used far beyond this one reduction) that this file should
// not weaken just to make doubleton fire more often -- so the bound is
// enforced the other way: decline elimination outright when the
// eliminated column's own bound magnitude is large enough that the worst-
// case leak could plausibly threaten the gate, with a real safety margin
// rather than cutting it exactly at the line.
constexpr double kDoubletonMaxEliminatedBoundMagnitude = 99.0;

// Per-row min/max activity of A x, tracking infinite contributions
// separately so that bound propagation stays valid when a variable is
// unbounded: with exactly one infinite contributor, the row still bounds
// THAT variable even though the activity bound itself is infinite.
struct Activity {
    double finite_min = 0.0;
    double finite_max = 0.0;
    std::int32_t n_min_inf = 0;
    std::int32_t n_max_inf = 0;

    double min_activity() const {
        return n_min_inf > 0 ? -kInfinity : finite_min;
    }
    double max_activity() const {
        return n_max_inf > 0 ? kInfinity : finite_max;
    }
};

} // namespace

PresolveResult presolve(const LpProblem& problem, int max_passes,
                         const std::vector<char>& integer_columns,
                         bool enable_doubleton_substitution,
                         bool enable_gcd_tightening) {
    PresolveResult result;
    const std::int32_t m = problem.n_rows();
    const std::int32_t n = problem.n_cols();
    result.original_n_rows = m;
    result.original_n_cols = n;

    // See Presolve.hpp's own comment on `integer_columns`: bounds-checked
    // rather than assumed to be sized `n`, since a mismatched mask is a
    // caller bug this function should not turn into out-of-bounds access.
    const auto is_integer_column = [&](std::size_t jj) {
        return jj < integer_columns.size() && integer_columns[jj] != 0;
    };

    const CSRMatrix& A = problem.A;
    const CSCMatrix A_csc = csr_to_csc(A);

    // Work in the symmetric form  row_lo <= (A x)_i <= row_hi, which is
    // what every reduction below actually reasons about. LpProblem stores
    // A x + s = rhs with s in [slack_lower, slack_upper]; since
    // s = rhs - A x, that is exactly A x in [rhs - slack_upper,
    // rhs - slack_lower].
    std::vector<double> row_lo(static_cast<std::size_t>(m));
    std::vector<double> row_hi(static_cast<std::size_t>(m));
    for (std::int32_t i = 0; i < m; ++i) {
        const auto ii = static_cast<std::size_t>(i);
        row_lo[ii] = problem.rhs[ii] - problem.slack_upper[ii];
        row_hi[ii] = problem.rhs[ii] - problem.slack_lower[ii];
    }

    std::vector<double> col_lo = problem.lower;
    std::vector<double> col_hi = problem.upper;

    // A doubleton elimination x_e = intercept + slope*x_s changes what the
    // SURVIVING column's own cost coefficient must be for the REDUCED
    // problem's simplex to optimize the same true objective: the eliminated
    // term c_e*x_e contributes c_e*intercept (a constant, dropped -- the
    // final objective_value is recomputed from the full reconstructed x in
    // LpSolver.cpp regardless) PLUS c_e*slope*x_s, which must be ADDED onto
    // x_s's own coefficient, or the simplex silently optimizes a DIFFERENT
    // objective than the true one while still returning a feasible point --
    // exactly the failure mode this comment exists to prevent a future
    // editor from reintroducing. Read (empty-column's cost sign) and
    // written (doubleton elimination) from here on instead of
    // `problem.obj` directly, so a chain of eliminations (see
    // tests/lp/test_presolve.cpp's chained-doubleton fixture) accumulates
    // correctly: each elimination uses the ELIMINATED column's CURRENT
    // (possibly already-corrected, if it was itself an earlier
    // elimination's target) coefficient, not its original one.
    std::vector<double> working_obj = problem.obj;

    // Round the ORIGINAL bounds of any integer-restricted column inward
    // before the pass loop even starts, covering a fractional bound given
    // directly in the model (the common case -- a bound derived later by
    // propagation or a singleton row -- goes through tighten_lower/
    // tighten_upper below instead, which apply the same rounding).
    for (std::int32_t j = 0; j < n; ++j) {
        const auto jj = static_cast<std::size_t>(j);
        if (!is_integer_column(jj)) continue;
        if (std::isfinite(col_lo[jj])) col_lo[jj] = std::ceil(col_lo[jj]);
        if (std::isfinite(col_hi[jj])) col_hi[jj] = std::floor(col_hi[jj]);
    }

    std::vector<char> row_active(static_cast<std::size_t>(m), 1);
    std::vector<char> col_active(static_cast<std::size_t>(n), 1);

    result.fixed_value.assign(static_cast<std::size_t>(n), 0.0);
    result.column_removed.assign(static_cast<std::size_t>(n), 0);
    result.column_is_doubleton.assign(static_cast<std::size_t>(n), 0);

    std::vector<std::int32_t> row_count(static_cast<std::size_t>(m), 0);
    std::vector<std::int32_t> col_count(static_cast<std::size_t>(n), 0);
    for (std::int32_t i = 0; i < m; ++i) {
        for (std::int32_t k = A.row_ptr()[i]; k < A.row_ptr()[i + 1]; ++k) {
            if (A.values()[k] == 0.0) continue;
            ++row_count[static_cast<std::size_t>(i)];
            ++col_count[static_cast<std::size_t>(A.col_idx()[k])];
        }
    }

    // Fixing a column substitutes its value into every row it touches and
    // decrements those rows' active counts; dropping a row does the mirror
    // image for its columns. Both are recorded immediately so postsolve
    // never has to infer what happened.
    const auto fix_column = [&](std::int32_t j, double value) {
        const auto jj = static_cast<std::size_t>(j);
        if (!col_active[jj]) return;
        col_active[jj] = 0;
        result.column_removed[jj] = 1;
        result.fixed_value[jj] = value;
        for (std::int32_t k = A_csc.col_ptr()[j]; k < A_csc.col_ptr()[j + 1]; ++k) {
            const std::int32_t i = A_csc.row_idx()[k];
            const double a = A_csc.values()[k];
            if (a == 0.0) continue;
            const auto ii = static_cast<std::size_t>(i);
            if (!row_active[ii]) continue;
            if (std::isfinite(row_lo[ii])) row_lo[ii] -= a * value;
            if (std::isfinite(row_hi[ii])) row_hi[ii] -= a * value;
            --row_count[ii];
        }
    };

    const auto drop_row = [&](std::int32_t i) {
        const auto ii = static_cast<std::size_t>(i);
        if (!row_active[ii]) return;
        row_active[ii] = 0;
        for (std::int32_t k = A.row_ptr()[i]; k < A.row_ptr()[i + 1]; ++k) {
            const std::int32_t j = A.col_idx()[k];
            if (A.values()[k] == 0.0) continue;
            if (!col_active[static_cast<std::size_t>(j)]) continue;
            --col_count[static_cast<std::size_t>(j)];
        }
    };

    // Activity is computed ON DEMAND, per row, immediately before it is
    // used -- never cached across a pass. Fixing a column subtracts its
    // contribution from the row bounds of every row it appears in, and
    // singleton rows tighten column bounds; a cached activity snapshot goes
    // stale against those updates within the same pass, and comparing a
    // stale activity to updated row bounds yields bounds that are too
    // tight. That is exactly how a presolve turns a feasible model
    // infeasible, so freshness here is a correctness requirement, not an
    // optimization choice. Cost is unchanged: one O(nnz) sweep either way.
    const auto row_activity = [&](std::int32_t i) {
        Activity act;
        for (std::int32_t k = A.row_ptr()[i]; k < A.row_ptr()[i + 1]; ++k) {
            const std::int32_t j = A.col_idx()[k];
            const double a = A.values()[k];
            if (a == 0.0 || !col_active[static_cast<std::size_t>(j)]) continue;
            const double lo = col_lo[static_cast<std::size_t>(j)];
            const double hi = col_hi[static_cast<std::size_t>(j)];
            const double min_bound = (a > 0.0) ? lo : hi;
            const double max_bound = (a > 0.0) ? hi : lo;
            if (std::isfinite(min_bound)) {
                act.finite_min += a * min_bound;
            } else {
                ++act.n_min_inf;
            }
            if (std::isfinite(max_bound)) {
                act.finite_max += a * max_bound;
            } else {
                ++act.n_max_inf;
            }
        }
        return act;
    };

    // Bound updates go through these two helpers exclusively, so the
    // relative-improvement rule and the outward safety margin are applied
    // uniformly and cannot be forgotten at one call site.
    // A DERIVED bound that would cross the variable's opposite bound is
    // REFUSED, never applied and never treated as proof of infeasibility.
    //
    // Rationale, and it is a policy decision rather than a tolerance tweak:
    // rest_min/rest_max are computed by subtracting one term from an
    // accumulated sum over the whole row. On a row mixing large and small
    // coefficients that subtraction suffers catastrophic cancellation, and
    // the derived limit can be wrong by far more than any fixed tolerance
    // allows for. Netlib maros fails exactly this way -- propagation on row
    // 161 crosses column 822's bounds on a model that is provably feasible.
    //
    // Since presolve's infeasibility detection is an OPTIMIZATION (phase 1
    // of the simplex detects infeasibility on its own, from the unreduced
    // row), the safe resolution is to decline the reduction and leave the
    // row in the problem. We lose a tightening; we never lose correctness.
    // Concluding INFEASIBLE from a derived quantity would trade a missed
    // reduction for a wrong answer.
    const auto tighten_upper = [&](std::size_t jj, double limit) {
        if (!std::isfinite(limit)) return false;
        double relaxed = limit + kBoundRelax * (1.0 + std::fabs(limit));
        // Integer columns: tighten further to the nearest integer, applied
        // to the already-outward-padded `relaxed` value rather than the raw
        // `limit` -- this is what lets a limit that is really exactly
        // integer N, but landed at N-epsilon from floating-point noise in
        // the derivation above, still round to N rather than N-1. See
        // Presolve.hpp's own comment on `integer_columns` for the soundness
        // argument (unconditional: any integer-feasible x_j already
        // respects floor(relaxed) whenever it respects the un-rounded
        // bound).
        if (is_integer_column(jj)) relaxed = std::floor(relaxed);
        if (relaxed < col_lo[jj]) return false; // would cross: numerically unreliable
        const double current = col_hi[jj];
        if (!std::isfinite(current)) {
            col_hi[jj] = relaxed;
            return true;
        }
        if (relaxed < current - kBoundImproveTol * (1.0 + std::fabs(current))) {
            col_hi[jj] = relaxed;
            return true;
        }
        return false;
    };
    const auto tighten_lower = [&](std::size_t jj, double limit) {
        if (!std::isfinite(limit)) return false;
        double relaxed = limit - kBoundRelax * (1.0 + std::fabs(limit));
        if (is_integer_column(jj)) relaxed = std::ceil(relaxed); // see tighten_upper's comment
        if (relaxed > col_hi[jj]) return false; // would cross: numerically unreliable
        const double current = col_lo[jj];
        if (!std::isfinite(current)) {
            col_lo[jj] = relaxed;
            return true;
        }
        if (relaxed > current + kBoundImproveTol * (1.0 + std::fabs(current))) {
            col_lo[jj] = relaxed;
            return true;
        }
        return false;
    };

    // Infeasibility tests are scaled by the magnitude of the quantities
    // involved, for the same reason the improvement threshold is.
    const auto column_infeasible = [&](std::size_t jj) {
        return col_lo[jj] > col_hi[jj] + kInfeasTol * (1.0 + std::fabs(col_hi[jj]));
    };

    bool infeasible = false;
    bool unbounded = false;
    int current_pass = 0;

    const auto fail = [&](const char* why, std::int32_t row, std::int32_t col) {
        if (result.reason == nullptr) {
            result.reason = why;
            result.reason_row = row;
            result.reason_col = col;
            result.reason_pass = current_pass;
        }
        infeasible = true;
    };

    for (int pass = 0; pass < max_passes && !infeasible && !unbounded; ++pass) {
        current_pass = pass;
        bool changed = false;

        // --- Fixed columns -------------------------------------------------
        for (std::int32_t j = 0; j < n && !infeasible; ++j) {
            const auto jj = static_cast<std::size_t>(j);
            if (!col_active[jj]) continue;
            if (column_infeasible(jj)) {
                fail("column bounds crossed", -1, j);
                break;
            }
            if (std::isfinite(col_lo[jj]) && std::isfinite(col_hi[jj]) &&
                col_hi[jj] - col_lo[jj] <= kFixTol) {
                fix_column(j, 0.5 * (col_lo[jj] + col_hi[jj]));
                changed = true;
            }
        }
        if (infeasible) break;

        // --- Empty columns -------------------------------------------------
        // A column in no active row affects only the objective, so it goes
        // straight to whichever bound minimizes its own cost. An infinite
        // bound in that direction means the LP is genuinely unbounded.
        for (std::int32_t j = 0; j < n && !unbounded; ++j) {
            const auto jj = static_cast<std::size_t>(j);
            if (!col_active[jj] || col_count[jj] != 0) continue;
            const double c = working_obj[jj];
            if (c > 0.0) {
                if (!std::isfinite(col_lo[jj])) {
                    unbounded = true;
                    break;
                }
                fix_column(j, col_lo[jj]);
            } else if (c < 0.0) {
                if (!std::isfinite(col_hi[jj])) {
                    unbounded = true;
                    break;
                }
                fix_column(j, col_hi[jj]);
            } else {
                const double value = std::isfinite(col_lo[jj])
                                          ? col_lo[jj]
                                          : (std::isfinite(col_hi[jj]) ? col_hi[jj] : 0.0);
                fix_column(j, value);
            }
            changed = true;
        }
        if (unbounded) break;

        // --- Empty / infeasible / redundant / forcing rows -----------------
        for (std::int32_t i = 0; i < m && !infeasible; ++i) {
            const auto ii = static_cast<std::size_t>(i);
            if (!row_active[ii]) continue;

            // Row tests are scaled by the row's own magnitude, for the same
            // reason the column ones are: a fixed absolute tolerance is
            // meaningless on a row whose bounds are ~1e7.
            const double row_scale =
                1.0 + std::max(std::isfinite(row_lo[ii]) ? std::fabs(row_lo[ii]) : 0.0,
                                std::isfinite(row_hi[ii]) ? std::fabs(row_hi[ii]) : 0.0);

            if (row_lo[ii] > row_hi[ii] + kInfeasTol * row_scale) {
                fail("row bounds crossed", i, -1);
                break;
            }

            if (row_count[ii] == 0) {
                // No variables left: the row asserts 0 in [lo, hi].
                if (row_lo[ii] > kInfeasTol * row_scale ||
                    row_hi[ii] < -kInfeasTol * row_scale) {
                    fail("empty row violates its bounds", i, -1);
                    break;
                }
                drop_row(i);
                changed = true;
                continue;
            }

            const Activity act = row_activity(i);
            const double min_act = act.min_activity();
            const double max_act = act.max_activity();

            const double act_scale =
                row_scale + std::max(act.n_min_inf ? 0.0 : std::fabs(act.finite_min),
                                      act.n_max_inf ? 0.0 : std::fabs(act.finite_max));

            if (min_act > row_hi[ii] + kInfeasTol * act_scale ||
                max_act < row_lo[ii] - kInfeasTol * act_scale) {
                fail("row activity cannot reach its bounds", i, -1);
                break;
            }

            // Redundancy uses the CONSERVATIVE tolerance against the row's
            // own magnitude, not the activity magnitude: a row is dropped
            // only when it is unambiguously implied.
            if (min_act >= row_lo[ii] - kReductionTol * row_scale &&
                max_act <= row_hi[ii] + kReductionTol * row_scale) {
                drop_row(i);
                changed = true;
                continue;
            }

            // Forcing row: the activity can only just reach the row bound,
            // so every variable in the row is pinned to the bound that
            // achieves that extreme. Requires all contributions finite,
            // which n_*_inf == 0 guarantees.
            // Forcing pins EVERY variable in the row simultaneously, so it
            // is the single most destructive reduction here if it fires
            // wrongly. It therefore uses the conservative tolerance against
            // the row's own magnitude: near-equality is not enough,
            // essentially exact equality is required.
            const bool forcing_at_upper =
                act.n_min_inf == 0 && std::isfinite(row_hi[ii]) &&
                std::fabs(act.finite_min - row_hi[ii]) <= kReductionTol * row_scale;
            const bool forcing_at_lower =
                act.n_max_inf == 0 && std::isfinite(row_lo[ii]) &&
                std::fabs(act.finite_max - row_lo[ii]) <= kReductionTol * row_scale;

            if (forcing_at_upper || forcing_at_lower) {
                std::vector<std::pair<std::int32_t, double>> to_fix;
                for (std::int32_t k = A.row_ptr()[i]; k < A.row_ptr()[i + 1]; ++k) {
                    const std::int32_t j = A.col_idx()[k];
                    const double a = A.values()[k];
                    if (a == 0.0 || !col_active[static_cast<std::size_t>(j)]) continue;
                    const double lo = col_lo[static_cast<std::size_t>(j)];
                    const double hi = col_hi[static_cast<std::size_t>(j)];
                    const double pinned = forcing_at_upper ? ((a > 0.0) ? lo : hi)
                                                            : ((a > 0.0) ? hi : lo);
                    if (!std::isfinite(pinned)) continue;
                    to_fix.emplace_back(j, pinned);
                }
                for (const auto& [j, value] : to_fix) fix_column(j, value);
                drop_row(i);
                changed = true;
                continue;
            }

            // --- GCD-based row tightening (see Presolve.hpp's own comment
            // on `enable_gcd_tightening`). Scoped to row_count >= 2: a
            // row_count == 1 row is already handled exactly by the
            // singleton-row block below (which divides by that single
            // coefficient directly), so running this first would only
            // duplicate that work.
            if (enable_gcd_tightening && row_count[ii] >= 2) {
                bool applicable = true;
                std::int64_t g = 0;
                for (std::int32_t k = A.row_ptr()[i]; k < A.row_ptr()[i + 1] && applicable; ++k) {
                    const std::int32_t j = A.col_idx()[k];
                    const double a = A.values()[k];
                    if (a == 0.0) continue;
                    if (!col_active[static_cast<std::size_t>(j)]) continue;
                    if (!is_integer_column(static_cast<std::size_t>(j)) ||
                        std::fabs(a) > kGcdMaxCoeffMagnitude) {
                        applicable = false;
                        break;
                    }
                    const double rounded = std::round(a);
                    if (rounded == 0.0 ||
                        std::fabs(a - rounded) > kGcdCoeffIntTol * std::max(1.0, std::fabs(a))) {
                        applicable = false;
                        break;
                    }
                    const std::int64_t ai = static_cast<std::int64_t>(rounded < 0 ? -rounded : rounded);
                    g = (g == 0) ? ai : std::gcd(g, ai);
                }

                if (applicable && g > 1) {
                    const double gd = static_cast<double>(g);
                    // Tightened toward the row's own interior, then relaxed
                    // outward by kBoundRelax exactly like tighten_upper/
                    // tighten_lower above -- this value is exact arithmetic
                    // given the preconditions just verified, but row_hi/
                    // row_lo themselves may already carry floating-point
                    // noise from an earlier pass's propagation, and the
                    // outward pad is what this project's own policy
                    // (kBoundRelax's doc comment) requires of every
                    // reduction that tightens a bound.
                    if (std::isfinite(row_hi[ii])) {
                        const double tightened = gd * std::floor(row_hi[ii] / gd + kReductionTol);
                        if (tightened < row_hi[ii] - kReductionTol * row_scale) {
                            if (tightened < row_lo[ii] - kInfeasTol * row_scale) {
                                fail("gcd tightening: no multiple of the row's coefficient "
                                     "gcd lies within its bounds",
                                     i, -1);
                            } else {
                                row_hi[ii] = tightened + kBoundRelax * (1.0 + std::fabs(tightened));
                                changed = true;
                            }
                        }
                    }
                    if (!infeasible && std::isfinite(row_lo[ii])) {
                        const double tightened = gd * std::ceil(row_lo[ii] / gd - kReductionTol);
                        if (tightened > row_lo[ii] + kReductionTol * row_scale) {
                            if (tightened > row_hi[ii] + kInfeasTol * row_scale) {
                                fail("gcd tightening: no multiple of the row's coefficient "
                                     "gcd lies within its bounds",
                                     i, -1);
                            } else {
                                row_lo[ii] = tightened - kBoundRelax * (1.0 + std::fabs(tightened));
                                changed = true;
                            }
                        }
                    }
                }
            }
            if (infeasible) break;

            // --- Singleton row: a single coefficient turns the row bounds
            // directly into bounds on that one variable.
            if (row_count[ii] == 1) {
                std::int32_t j = -1;
                double a = 0.0;
                for (std::int32_t k = A.row_ptr()[i]; k < A.row_ptr()[i + 1]; ++k) {
                    const std::int32_t candidate = A.col_idx()[k];
                    if (A.values()[k] == 0.0) continue;
                    if (!col_active[static_cast<std::size_t>(candidate)]) continue;
                    j = candidate;
                    a = A.values()[k];
                    break;
                }
                if (j < 0) continue;
                const auto jj = static_cast<std::size_t>(j);

                double implied_lo = -kInfinity;
                double implied_hi = kInfinity;
                if (a > 0.0) {
                    if (std::isfinite(row_lo[ii])) implied_lo = row_lo[ii] / a;
                    if (std::isfinite(row_hi[ii])) implied_hi = row_hi[ii] / a;
                } else {
                    if (std::isfinite(row_hi[ii])) implied_lo = row_hi[ii] / a;
                    if (std::isfinite(row_lo[ii])) implied_hi = row_lo[ii] / a;
                }
                // A singleton row's implied bound is a single division, not
                // an accumulated sum, so it is far better conditioned than
                // propagation -- but it is still derived, so a crossing is
                // still declined rather than declared infeasible. The row
                // is only dropped once its content has been absorbed into
                // the bounds; if tighten_* declined, the row must STAY.
                const bool took_lower = tighten_lower(jj, implied_lo);
                const bool took_upper = tighten_upper(jj, implied_hi);
                const bool absorbed = (took_lower || !std::isfinite(implied_lo) ||
                                        implied_lo <= col_lo[jj] + kBoundRelax * (1.0 + std::fabs(implied_lo))) &&
                                       (took_upper || !std::isfinite(implied_hi) ||
                                        implied_hi >= col_hi[jj] - kBoundRelax * (1.0 + std::fabs(implied_hi)));
                if (!absorbed) continue; // keep the row; the simplex will enforce it
                drop_row(i);
                changed = true;
                continue;
            }

            // --- Doubleton row substitution (SCOPED, see Presolve.hpp's
            // own comment on `enable_doubleton_substitution`): an EQUALITY
            // row with exactly two active nonzero coefficients, where one
            // of the two variables appears in NO other active row, is
            // eliminated by expressing it as an exact affine function of
            // the other. Only the isolated-variable case is handled here;
            // if NEITHER variable is isolated, eliminating either would
            // require mutating some OTHER row's coefficients, which this
            // file does not support (see Presolve.hpp) -- the row is left
            // untouched in that case, a deliberate scope boundary, not a
            // missed case.
            if (enable_doubleton_substitution && row_count[ii] == 2 &&
                row_hi[ii] - row_lo[ii] <= kFixTol) {
                std::int32_t p = -1, q = -1;
                double a_p = 0.0, a_q = 0.0;
                for (std::int32_t k = A.row_ptr()[i]; k < A.row_ptr()[i + 1]; ++k) {
                    const std::int32_t candidate = A.col_idx()[k];
                    if (A.values()[k] == 0.0) continue;
                    if (!col_active[static_cast<std::size_t>(candidate)]) continue;
                    if (p < 0) {
                        p = candidate;
                        a_p = A.values()[k];
                    } else {
                        q = candidate;
                        a_q = A.values()[k];
                        break;
                    }
                }

                if (p >= 0 && q >= 0) {
                    const auto pp = static_cast<std::size_t>(p);
                    const auto qq = static_cast<std::size_t>(q);
                    const bool p_isolated = col_count[pp] == 1;
                    const bool q_isolated = col_count[qq] == 1;

                    std::int32_t e = -1, s = -1;
                    double a_e = 0.0, a_s = 0.0;
                    if (p_isolated && q_isolated) {
                        // Both isolated: eliminate the larger-magnitude
                        // coefficient for a better-conditioned 1/a_e.
                        if (std::fabs(a_p) >= std::fabs(a_q)) {
                            e = p; a_e = a_p; s = q; a_s = a_q;
                        } else {
                            e = q; a_e = a_q; s = p; a_s = a_p;
                        }
                    } else if (p_isolated) {
                        e = p; a_e = a_p; s = q; a_s = a_q;
                    } else if (q_isolated) {
                        e = q; a_e = a_q; s = p; a_s = a_p;
                    }
                    // else: neither isolated -- deferred, see comment above.

                    // Only FINITE bounds contribute to the leak this guards
                    // against -- an infinite bound is skipped entirely by
                    // both the implied-bound derivation below (isfinite-
                    // gated) and by postsolve's clamp (a no-op against
                    // infinity), so it carries no risk regardless of being
                    // "infinitely large" in a literal sense.
                    const bool bound_magnitude_safe = [&]() {
                        if (e < 0) return false;
                        const auto ee2 = static_cast<std::size_t>(e);
                        double worst_finite_bound = 0.0;
                        if (std::isfinite(col_lo[ee2])) {
                            worst_finite_bound = std::max(worst_finite_bound, std::fabs(col_lo[ee2]));
                        }
                        if (std::isfinite(col_hi[ee2])) {
                            worst_finite_bound = std::max(worst_finite_bound, std::fabs(col_hi[ee2]));
                        }
                        return worst_finite_bound <= kDoubletonMaxEliminatedBoundMagnitude;
                    }();

                    if (e >= 0 && bound_magnitude_safe &&
                        std::fabs(a_e) >= kDoubletonPivotFloor &&
                        std::fabs(a_e) >= kDoubletonRelativeFloor * std::fabs(a_s)) {
                        const auto ee = static_cast<std::size_t>(e);
                        const auto ss = static_cast<std::size_t>(s);
                        const double c = 0.5 * (row_lo[ii] + row_hi[ii]);
                        const double intercept = c / a_e;
                        const double slope = -a_s / a_e;

                        // Implied bound on x_s from x_e's OWN current
                        // bounds: x_e = intercept + slope*x_s must stay
                        // within [col_lo[e], col_hi[e]].
                        double implied_s_lo = -kInfinity;
                        double implied_s_hi = kInfinity;
                        if (slope > 0.0) {
                            if (std::isfinite(col_lo[ee])) implied_s_lo = (col_lo[ee] - intercept) / slope;
                            if (std::isfinite(col_hi[ee])) implied_s_hi = (col_hi[ee] - intercept) / slope;
                        } else {
                            if (std::isfinite(col_hi[ee])) implied_s_lo = (col_hi[ee] - intercept) / slope;
                            if (std::isfinite(col_lo[ee])) implied_s_hi = (col_lo[ee] - intercept) / slope;
                        }

                        // Same "absorbed" pattern as the singleton-row
                        // block above: a tighten_* that returns false
                        // because the current bound was ALREADY at least
                        // as tight is fine (nothing lost); one that
                        // returns false while the implied bound WOULD
                        // have been a genuine improvement can only mean it
                        // was declined as numerically unreliable (would
                        // cross) -- in which case the whole elimination
                        // must be declined too, not just the bound, or a
                        // point reachable in the reduced problem could
                        // postsolve to an x_e outside its own true range.
                        const bool took_lower = tighten_lower(ss, implied_s_lo);
                        const bool took_upper = tighten_upper(ss, implied_s_hi);
                        const bool absorbed_s =
                            (took_lower || !std::isfinite(implied_s_lo) ||
                             implied_s_lo <= col_lo[ss] + kBoundRelax * (1.0 + std::fabs(implied_s_lo))) &&
                            (took_upper || !std::isfinite(implied_s_hi) ||
                             implied_s_hi >= col_hi[ss] - kBoundRelax * (1.0 + std::fabs(implied_s_hi)));

                        if (absorbed_s) {
                            result.doubleton_eliminations.push_back(
                                {e, s, intercept, slope, col_lo[ee], col_hi[ee]});
                            result.column_removed[ee] = 1;
                            result.column_is_doubleton[ee] = 1;
                            col_active[ee] = 0;
                            // See working_obj's own comment above: x_e's
                            // cost contributes c_e*slope*x_s to the true
                            // objective once substituted, which must be
                            // folded into x_s's coefficient here, using
                            // x_e's CURRENT (not necessarily original)
                            // working_obj so a chain of eliminations
                            // accumulates correctly.
                            working_obj[ss] += working_obj[ee] * slope;
                            drop_row(i);
                            changed = true;
                            continue;
                        }
                        // else: declined (would cross); leave the row for
                        // the simplex, same policy as everywhere else in
                        // this file.
                    }
                }
            }
        }
        if (infeasible) break;

        // --- Activity-based bound tightening (bound propagation) -----------
        // For each variable in each row, subtract that variable's own
        // extreme contribution from the row's activity bound; what remains
        // is a bound on its coefficient times the variable. Valid when the
        // REST of the row has a finite extreme -- either because no
        // contribution is infinite, or because this variable is the only
        // infinite one.
        // Proposed bound updates are collected during a row's scan and
        // applied only after it completes. Applying them in place would
        // corrupt the very computation that produced them: `act` is a
        // snapshot of the row's activity under the bounds as they stood at
        // row entry, and rest_min/rest_max subtract THIS variable's
        // contribution from that snapshot. If an earlier variable in the
        // same row had already had its bound mutated, the snapshot and the
        // freshly-read bound disagree, the subtraction is inconsistent, and
        // the resulting limit is too tight -- which is a silent
        // over-tightening of a feasible region, observed as a false
        // INFEASIBLE on Netlib maros (row 161, column 822).
        std::vector<std::pair<std::size_t, double>> proposed_lower, proposed_upper;

        for (std::int32_t i = 0; i < m; ++i) {
            const auto ii = static_cast<std::size_t>(i);
            if (!row_active[ii]) continue;
            const Activity act = row_activity(i);
            proposed_lower.clear();
            proposed_upper.clear();

            for (std::int32_t k = A.row_ptr()[i]; k < A.row_ptr()[i + 1]; ++k) {
                const std::int32_t j = A.col_idx()[k];
                const double a = A.values()[k];
                if (a == 0.0 || !col_active[static_cast<std::size_t>(j)]) continue;
                const auto jj = static_cast<std::size_t>(j);
                const double lo = col_lo[jj];
                const double hi = col_hi[jj];
                const double min_bound = (a > 0.0) ? lo : hi;
                const double max_bound = (a > 0.0) ? hi : lo;

                double rest_min = -kInfinity;
                if (act.n_min_inf == 0) {
                    rest_min = act.finite_min - a * min_bound;
                } else if (act.n_min_inf == 1 && !std::isfinite(min_bound)) {
                    rest_min = act.finite_min;
                }

                double rest_max = kInfinity;
                if (act.n_max_inf == 0) {
                    rest_max = act.finite_max - a * max_bound;
                } else if (act.n_max_inf == 1 && !std::isfinite(max_bound)) {
                    rest_max = act.finite_max;
                }

                if (std::isfinite(rest_min) && std::isfinite(row_hi[ii])) {
                    const double limit = (row_hi[ii] - rest_min) / a;
                    (a > 0.0 ? proposed_upper : proposed_lower).emplace_back(jj, limit);
                }
                if (std::isfinite(rest_max) && std::isfinite(row_lo[ii])) {
                    const double limit = (row_lo[ii] - rest_max) / a;
                    (a > 0.0 ? proposed_lower : proposed_upper).emplace_back(jj, limit);
                }
            }

            // `changed` is set only when a bound genuinely moved; setting it
            // on every attempted tightening would make the fixed-point loop
            // run its full pass cap on every model.
            for (const auto& [jj, limit] : proposed_lower) changed |= tighten_lower(jj, limit);
            for (const auto& [jj, limit] : proposed_upper) changed |= tighten_upper(jj, limit);
            // No infeasibility conclusion here by construction: tighten_*
            // declines any update that would cross, so propagation can never
            // manufacture an empty domain.
        }

        if (!changed) break;
    }

    if (infeasible) {
        result.status = PresolveStatus::INFEASIBLE;
        return result;
    }
    if (unbounded) {
        result.status = PresolveStatus::UNBOUNDED;
        return result;
    }

    // --- Emit the reduced model ------------------------------------------
    std::vector<std::int32_t> new_col_index(static_cast<std::size_t>(n), -1);
    for (std::int32_t j = 0; j < n; ++j) {
        if (!col_active[static_cast<std::size_t>(j)]) continue;
        new_col_index[static_cast<std::size_t>(j)] =
            static_cast<std::int32_t>(result.kept_columns.size());
        result.kept_columns.push_back(j);
    }
    for (std::int32_t i = 0; i < m; ++i) {
        if (row_active[static_cast<std::size_t>(i)]) result.kept_rows.push_back(i);
    }

    const auto rn = static_cast<std::int32_t>(result.kept_columns.size());
    const auto rm = static_cast<std::int32_t>(result.kept_rows.size());

    std::vector<Triplet> triplets;
    for (std::int32_t r = 0; r < rm; ++r) {
        const std::int32_t i = result.kept_rows[static_cast<std::size_t>(r)];
        for (std::int32_t k = A.row_ptr()[i]; k < A.row_ptr()[i + 1]; ++k) {
            const std::int32_t j = A.col_idx()[k];
            const double a = A.values()[k];
            if (a == 0.0 || !col_active[static_cast<std::size_t>(j)]) continue;
            triplets.push_back({r, new_col_index[static_cast<std::size_t>(j)], a});
        }
    }

    LpProblem& out = result.reduced;
    out.A = CSRMatrix::from_triplets(rm, rn, triplets);
    out.obj.resize(static_cast<std::size_t>(rn));
    out.lower.resize(static_cast<std::size_t>(rn));
    out.upper.resize(static_cast<std::size_t>(rn));
    for (std::int32_t k = 0; k < rn; ++k) {
        const auto orig = static_cast<std::size_t>(result.kept_columns[static_cast<std::size_t>(k)]);
        out.obj[static_cast<std::size_t>(k)] = working_obj[orig];
        out.lower[static_cast<std::size_t>(k)] = col_lo[orig];
        out.upper[static_cast<std::size_t>(k)] = col_hi[orig];
    }

    out.rhs.resize(static_cast<std::size_t>(rm));
    out.row_types.resize(static_cast<std::size_t>(rm));
    out.slack_lower.resize(static_cast<std::size_t>(rm));
    out.slack_upper.resize(static_cast<std::size_t>(rm));
    for (std::int32_t r = 0; r < rm; ++r) {
        const auto orig = static_cast<std::size_t>(result.kept_rows[static_cast<std::size_t>(r)]);
        const auto rr = static_cast<std::size_t>(r);
        const double lo = row_lo[orig];
        const double hi = row_hi[orig];

        // Re-encode  lo <= A x <= hi  as  A x + s = rhs. Anchoring rhs to
        // whichever bound is finite keeps rhs finite in every case,
        // including one-sided rows.
        if (std::isfinite(hi)) {
            out.rhs[rr] = hi;
            out.slack_lower[rr] = 0.0;
            out.slack_upper[rr] = std::isfinite(lo) ? (hi - lo) : kInfinity;
            out.row_types[rr] = std::isfinite(lo) && hi - lo <= kFixTol ? 'E' : 'L';
        } else if (std::isfinite(lo)) {
            out.rhs[rr] = lo;
            out.slack_lower[rr] = -kInfinity;
            out.slack_upper[rr] = 0.0;
            out.row_types[rr] = 'G';
        } else {
            // Free row: no restriction at all. Kept only so row indexing
            // stays consistent; the slack absorbs everything.
            out.rhs[rr] = 0.0;
            out.slack_lower[rr] = -kInfinity;
            out.slack_upper[rr] = kInfinity;
            out.row_types[rr] = 'N';
        }
    }

    return result;
}

std::vector<double> postsolve(const PresolveResult& result,
                               const std::vector<double>& reduced_x) {
    std::vector<double> x(static_cast<std::size_t>(result.original_n_cols), 0.0);
    for (std::int32_t j = 0; j < result.original_n_cols; ++j) {
        const auto jj = static_cast<std::size_t>(j);
        // A doubleton-eliminated column's value is not a constant -- see
        // the reverse-replay loop below -- so fixed_value[jj] is skipped
        // for it here, matching PresolveResult's own documented contract.
        const bool is_doubleton =
            jj < result.column_is_doubleton.size() && result.column_is_doubleton[jj];
        if (result.column_removed[jj] && !is_doubleton) x[jj] = result.fixed_value[jj];
    }
    const auto kept = result.kept_columns.size();
    for (std::size_t k = 0; k < kept && k < reduced_x.size(); ++k) {
        x[static_cast<std::size_t>(result.kept_columns[k])] = reduced_x[k];
    }
    // Doubleton eliminations were recorded in DISCOVERY order, and a later
    // elimination's target_col is guaranteed to have been active (hence
    // already correctly valued above, either directly or by an EARLIER
    // entry in this same list) at the moment it was recorded -- replaying
    // in REVERSE therefore always resolves a dependency before it is
    // needed, with no separate topological sort. See PresolveResult's own
    // comment on doubleton_eliminations for the full argument.
    for (auto it = result.doubleton_eliminations.rbegin();
         it != result.doubleton_eliminations.rend(); ++it) {
        double value = it->intercept + it->slope * x[static_cast<std::size_t>(it->target_col)];
        // Clamp into eliminated_col's own recorded range -- see
        // DoubletonElimination's own comment (Presolve.hpp) for why this is
        // necessary, not merely defensive: the implied bound placed on
        // target_col during presolve is deliberately relaxed outward, and
        // that slack, carried through intercept/slope, can otherwise land
        // `value` fractionally outside eliminated_col's true bound.
        value = std::min(value, it->eliminated_upper);
        value = std::max(value, it->eliminated_lower);
        x[static_cast<std::size_t>(it->eliminated_col)] = value;
    }
    return x;
}

} // namespace sihps
