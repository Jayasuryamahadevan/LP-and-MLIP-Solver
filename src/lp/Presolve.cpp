#include "Presolve.hpp"

#include "../sparse/CSCMatrix.hpp"
#include "../sparse/Convert.hpp"

#include <algorithm>
#include <cmath>

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
                         const std::vector<char>& integer_columns) {
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
            const double c = problem.obj[jj];
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
        out.obj[static_cast<std::size_t>(k)] = problem.obj[orig];
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
        if (result.column_removed[jj]) x[jj] = result.fixed_value[jj];
    }
    const auto kept = result.kept_columns.size();
    for (std::size_t k = 0; k < kept && k < reduced_x.size(); ++k) {
        x[static_cast<std::size_t>(result.kept_columns[k])] = reduced_x[k];
    }
    return x;
}

} // namespace sihps
