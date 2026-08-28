#include "MilpSolver.hpp"

#include "../parallel/Parallel.hpp"
#include "ExactBinarySplit.hpp"
#include "ParallelSearch.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sihps {
namespace {

constexpr double kInfinityValue = std::numeric_limits<double>::infinity();

// Genuine floating-point cancellation noise on a row scales with that row's
// own component-wise term magnitude (|A|*|x|), but by an amount tied to
// machine precision, not to `feasibility_tolerance` (1e-6) -- MEASURED
// directly on Netlib `shell`/`perold`: their legitimate solver rounding
// noise, across dozens of affected rows each, lands at a strikingly
// consistent ratio of ~5e-10 relative to the row's own |A|*|x| activity
// (never above 7e-10 in either instance). A genuine infeasibility, by
// contrast, is not bounded by machine precision at all -- MEASURED on
// MIPLIB `ej` (a deliberately adversarial "numerics" instance, per its own
// MIPLIB tag): an incumbent that violates its single equality row by a
// real 20 units, against a row activity of ~2.05e7, is a ratio of ~9.8e-7,
// nearly 2000x above the noise ceiling above. `kActivityNoiseRatio` sits
// in between on a log scale (~20x above the measured noise ceiling, ~100x
// below the measured violation ratio) -- comfortable margin either way,
// not a value cut exactly at the boundary of the two measurements it is
// derived from. Used ADDITIVELY with `feasibility_tolerance`, not as a
// replacement multiplier: near-zero-activity rows still get exactly
// `feasibility_tolerance`'s own original, unweakened absolute allowance.
constexpr double kActivityNoiseRatio = 1e-8;

struct BoundChange {
    std::int32_t variable = -1;
    double lower = -kInfinityValue;
    double upper = kInfinityValue;
};

struct SearchNode {
    std::shared_ptr<const SearchNode> parent;
    BoundChange change;
    int depth = 0;
    std::uint64_t order = 0;

    // Branch metadata used to learn pseudocosts when this node's relaxation
    // is solved. It is not used for correctness or pruning.
    std::int32_t branch_variable = -1;
    int branch_direction = 0; // -1: x <= floor(parent x), +1: x >= ceil(parent x)
    double branch_distance = 0.0;

    // A parent's LP lower bound is also a valid lower bound for either child.
    // It is used for best-bound ordering before the child relaxation is run.
    double priority_bound = -kInfinityValue;

    // The parent's own terminal basis, for warm-starting this node's
    // relaxation (only set when warm_start_node_relaxations is on). Carried
    // on the node itself -- rather than in a side map keyed by node order --
    // because under parallel search a node's two children go into the
    // SHARED cross-worker queue and are very often popped by a DIFFERENT
    // worker than the one that created them (see docs/architecture/MILP.md's
    // parallel-B&B section on the shared-queue design). A side map keyed by
    // order and stored per-WorkerContext would leave most entries inserted
    // by the creating worker permanently unconsumed by whichever worker
    // actually pops the node -- an unbounded leak that scales with total
    // node count, not a mere warm-start-hit-rate degradation.
    std::shared_ptr<const Simplex::Basis> parent_basis;
};

struct NodeCompare {
    bool operator()(const std::shared_ptr<const SearchNode>& lhs,
                    const std::shared_ptr<const SearchNode>& rhs) const {
        if (lhs->priority_bound != rhs->priority_bound) {
            return lhs->priority_bound > rhs->priority_bound;
        }
        // Deterministic tie-breaks avoid pointer-address ordering, which can
        // change between processes and would make a benchmark irreproducible.
        if (lhs->depth != rhs->depth) return lhs->depth > rhs->depth;
        return lhs->order > rhs->order;
    }
};

using NodeQueue = std::priority_queue<std::shared_ptr<const SearchNode>,
                                      std::vector<std::shared_ptr<const SearchNode>>,
                                      NodeCompare>;

bool within(double value, double target, double tolerance) {
    return std::fabs(value - target) <= tolerance * (1.0 + std::fabs(target));
}

double objective_value(const LpProblem& problem, const std::vector<double>& x) {
    double value = 0.0;
    for (std::int32_t j = 0; j < problem.n_cols(); ++j) {
        value += problem.obj[static_cast<std::size_t>(j)] * x[static_cast<std::size_t>(j)];
    }
    return value;
}

void materialize_bounds(const SearchNode& node, const std::vector<double>& root_lower,
                        const std::vector<double>& root_upper, std::vector<double>& lower,
                        std::vector<double>& upper) {
    lower = root_lower;
    upper = root_upper;

    std::vector<const SearchNode*> path;
    for (const SearchNode* current = &node; current != nullptr; current = current->parent.get()) {
        path.push_back(current);
    }
    std::reverse(path.begin(), path.end());
    for (const SearchNode* current : path) {
        if (current->change.variable < 0) continue;
        const auto j = static_cast<std::size_t>(current->change.variable);
        lower[j] = std::max(lower[j], current->change.lower);
        upper[j] = std::min(upper[j], current->change.upper);
    }
}

bool bounds_are_valid(const std::vector<double>& lower, const std::vector<double>& upper) {
    if (lower.size() != upper.size()) return false;
    for (std::size_t j = 0; j < lower.size(); ++j) {
        if (std::isnan(lower[j]) || std::isnan(upper[j]) || lower[j] > upper[j]) return false;
    }
    return true;
}

bool feasible_point(const MilpProblem& problem, const std::vector<double>& x,
                    const std::vector<double>& lower, const std::vector<double>& upper,
                    double feasibility_tolerance, ParallelMode parallel_mode) {
    const LpProblem& lp = problem.relaxation;
    if (x.size() != static_cast<std::size_t>(lp.n_cols())) return false;

    for (std::int32_t j = 0; j < lp.n_cols(); ++j) {
        const auto jj = static_cast<std::size_t>(j);
        if (!std::isfinite(x[jj])) return false;
        if (x[jj] < lower[jj] - feasibility_tolerance * (1.0 + std::fabs(lower[jj])) ||
            x[jj] > upper[jj] + feasibility_tolerance * (1.0 + std::fabs(upper[jj]))) {
            return false;
        }
    }

    std::vector<double> ax(static_cast<std::size_t>(lp.n_rows()), 0.0);
    if (lp.n_rows() > 0) lp.A.multiply(x.data(), ax.data(), parallel_mode);
    // PER-ROW relative violation, scaled by that row's own component-wise
    // term magnitude (|A|*|x| -- Higham's standard scaled-residual
    // denominator, "Accuracy and Stability of Numerical Algorithms" \S7.1),
    // not a single violation normalized by the model-wide max |rhs|. A
    // shared global norm lets one large-RHS row (e.g. a plant-wide cost
    // cap) silently inflate the effective absolute tolerance for every
    // other row in the SAME model -- MEASURED to let a genuinely violated
    // equality row (violation ~0.1-0.7, terms of magnitude O(10-100)) pass
    // as "feasible" on MIPLIB's `flugplinf`, because an unrelated row's
    // RHS of 1.2e6 pushed the shared denominator that high. But the row's
    // OWN |rhs| alone is not the right per-row scale either: a row can have
    // rhs=0 while still summing large-magnitude cancelling terms (measured
    // on Netlib `shell`/`perold`, where legitimate solver rounding noise
    // on exactly such rows was wrongly flagged once scaled by |rhs| alone,
    // since |rhs|=0 gives no headroom for the cancellation that produced
    // it). Summing |coefficient * x| over the row's own nonzeros gives the
    // magnitude of what was actually being cancelled, which is the
    // quantity a rounding-error argument is properly relative to.
    const auto* row_ptr = lp.A.row_ptr();
    const auto* col_idx = lp.A.col_idx();
    const auto* values = lp.A.values();
    for (std::int32_t i = 0; i < lp.n_rows(); ++i) {
        const auto ii = static_cast<std::size_t>(i);
        double abs_activity = std::fabs(lp.rhs[ii]);
        for (std::int32_t k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
            abs_activity += std::fabs(values[k]) * std::fabs(x[static_cast<std::size_t>(col_idx[k])]);
        }
        const double allowed = feasibility_tolerance + kActivityNoiseRatio * abs_activity;
        const double lo = lp.rhs[ii] - lp.slack_upper[ii];
        const double hi = lp.rhs[ii] - lp.slack_lower[ii];
        if (std::isfinite(lo) && lo - ax[ii] > allowed) return false;
        if (std::isfinite(hi) && ax[ii] - hi > allowed) return false;
    }
    return true;
}

bool integral_point(const MilpProblem& problem, const std::vector<double>& x,
                    double integrality_tolerance) {
    for (std::int32_t j = 0; j < problem.n_cols(); ++j) {
        const auto jj = static_cast<std::size_t>(j);
        if (problem.variable_types[jj] == VariableType::CONTINUOUS) continue;
        if (!std::isfinite(x[jj]) ||
            !within(x[jj], std::round(x[jj]), integrality_tolerance)) {
            return false;
        }
    }
    return true;
}

std::vector<double> rounded_point(const MilpProblem& problem, const std::vector<double>& x,
                                  const std::vector<double>& lower,
                                  const std::vector<double>& upper) {
    std::vector<double> rounded = x;
    for (std::int32_t j = 0; j < problem.n_cols(); ++j) {
        const auto jj = static_cast<std::size_t>(j);
        if (problem.variable_types[jj] == VariableType::CONTINUOUS) continue;
        double value = std::round(x[jj]);
        if (problem.variable_types[jj] == VariableType::BINARY) {
            value = std::clamp(value, 0.0, 1.0);
        }
        if (std::isfinite(lower[jj])) value = std::max(value, std::ceil(lower[jj]));
        if (std::isfinite(upper[jj])) value = std::min(value, std::floor(upper[jj]));
        rounded[jj] = value;
    }
    return rounded;
}

struct FractionalCandidate {
    std::int32_t variable = -1;
    double fraction = 0.0;
    double fractionality = 0.0;
};

struct CoverCut {
    std::vector<std::int32_t> variables;
};

bool binary_domain(const MilpProblem& problem, const LpProblem& lp, std::int32_t variable) {
    const auto j = static_cast<std::size_t>(variable);
    if (problem.variable_types[j] == VariableType::BINARY) return true;
    // MPS commonly encodes binary variables as INTEGER inside INTORG/INTEND
    // with explicit [0,1] bounds. Treating those as binary is a semantic
    // classification for valid cover separation, not a relaxation change.
    return problem.variable_types[j] == VariableType::INTEGER && lp.lower[j] >= 0.0 &&
           lp.upper[j] <= 1.0;
}

std::vector<CoverCut> separate_cover_cuts(const MilpProblem& problem,
                                           const std::vector<double>& x,
                                           double violation_tolerance,
                                           std::uint32_t limit) {
    std::vector<CoverCut> cuts;
    const LpProblem& lp = problem.relaxation;
    for (std::int32_t i = 0; i < lp.n_rows() && cuts.size() < limit; ++i) {
        const auto ii = static_cast<std::size_t>(i);
        const double upper = lp.rhs[ii] - lp.slack_lower[ii];
        if (!std::isfinite(upper)) continue;

        struct Term {
            std::int32_t variable;
            double coefficient;
            double value;
        };
        std::vector<Term> terms;
        const std::int32_t begin = lp.A.row_ptr()[i];
        const std::int32_t end = lp.A.row_ptr()[i + 1];
        bool valid_cover_row = true;
        double noncover_minimum = 0.0;
        for (std::int32_t k = begin; k < end; ++k) {
            const auto kk = static_cast<std::size_t>(k);
            const std::int32_t j = lp.A.col_idx()[kk];
            const auto jj = static_cast<std::size_t>(j);
            const double coefficient = lp.A.values()[kk];
            const double lower_term = coefficient * lp.lower[jj];
            const double upper_term = coefficient * lp.upper[jj];
            const double minimum_term = std::min(lower_term, upper_term);
            if (!std::isfinite(minimum_term)) {
                // A term with an unbounded contribution below could cancel
                // the cover activity, so no cover inequality is inferred.
                valid_cover_row = false;
                break;
            }
            if (binary_domain(problem, lp, j) && coefficient > 0.0) {
                terms.push_back({j, coefficient, x[jj]});
            } else {
                noncover_minimum += minimum_term;
            }
        }
        if (!valid_cover_row || terms.size() < 2) continue;
        const double effective_upper = upper - noncover_minimum;
        std::vector<std::vector<Term>> orderings;
        orderings.push_back(terms);
        orderings.push_back(terms);
        orderings.push_back(terms);
        orderings.push_back(terms);
        std::sort(orderings[0].begin(), orderings[0].end(), [](const Term& lhs, const Term& rhs) {
            if (lhs.value != rhs.value) return lhs.value > rhs.value;
            return lhs.variable < rhs.variable;
        });
        std::sort(orderings[1].begin(), orderings[1].end(), [](const Term& lhs, const Term& rhs) {
            if (lhs.coefficient != rhs.coefficient) return lhs.coefficient > rhs.coefficient;
            return lhs.variable < rhs.variable;
        });
        std::sort(orderings[2].begin(), orderings[2].end(), [](const Term& lhs, const Term& rhs) {
            if (lhs.coefficient != rhs.coefficient) return lhs.coefficient < rhs.coefficient;
            return lhs.variable < rhs.variable;
        });
        std::sort(orderings[3].begin(), orderings[3].end(), [](const Term& lhs, const Term& rhs) {
            if (lhs.value != rhs.value) return lhs.value < rhs.value;
            return lhs.variable < rhs.variable;
        });

        for (auto& ordering : orderings) {
            if (cuts.size() >= limit) break;
            double coefficient_sum = 0.0;
            CoverCut cut;
            for (const Term& term : ordering) {
                coefficient_sum += term.coefficient;
                cut.variables.push_back(term.variable);
                if (coefficient_sum > effective_upper + violation_tolerance) break;
            }
            if (coefficient_sum <= effective_upper + violation_tolerance ||
                cut.variables.empty()) {
                continue;
            }

            // Remove redundant cover members. The resulting inequality is
            // still valid, and minimal covers are generally stronger than a
            // greedy nonminimal superset.
            for (std::size_t p = 0; p < cut.variables.size();) {
                const auto variable = cut.variables[p];
                double without = 0.0;
                for (const Term& term : terms) {
                    if (std::find(cut.variables.begin(), cut.variables.end(), term.variable) !=
                            cut.variables.end() &&
                        term.variable != variable) {
                        without += term.coefficient;
                    }
                }
                if (without > effective_upper + violation_tolerance) {
                    cut.variables.erase(cut.variables.begin() + static_cast<std::ptrdiff_t>(p));
                } else {
                    ++p;
                }
            }

            double fractional_activity = 0.0;
            for (std::int32_t variable : cut.variables) {
                fractional_activity += x[static_cast<std::size_t>(variable)];
            }
            if (cut.variables.size() < 2 ||
                fractional_activity <= static_cast<double>(cut.variables.size() - 1) +
                                           violation_tolerance) {
                continue;
            }
            std::sort(cut.variables.begin(), cut.variables.end());
            const bool duplicate = std::any_of(
                cuts.begin(), cuts.end(), [&](const CoverCut& existing) {
                    return existing.variables == cut.variables;
                });
            if (!duplicate) cuts.push_back(std::move(cut));
        }
    }
    return cuts;
}

void append_cover_cuts(LpProblem& workspace, const std::vector<CoverCut>& cuts) {
    const std::int32_t old_rows = workspace.n_rows();
    std::vector<Triplet> entries;
    entries.reserve(static_cast<std::size_t>(workspace.A.nnz()) +
                    std::accumulate(cuts.begin(), cuts.end(), std::size_t{0},
                                    [](std::size_t total, const CoverCut& cut) {
                                        return total + cut.variables.size();
                                    }));
    for (std::int32_t i = 0; i < old_rows; ++i) {
        for (std::int32_t k = workspace.A.row_ptr()[i]; k < workspace.A.row_ptr()[i + 1]; ++k) {
            const auto kk = static_cast<std::size_t>(k);
            entries.push_back({i, workspace.A.col_idx()[kk], workspace.A.values()[kk]});
        }
    }
    for (std::size_t cut_index = 0; cut_index < cuts.size(); ++cut_index) {
        const auto row = old_rows + static_cast<std::int32_t>(cut_index);
        for (std::int32_t variable : cuts[cut_index].variables) {
            entries.push_back({row, variable, 1.0});
        }
    }
    workspace.A = CSRMatrix::from_triplets(
        old_rows + static_cast<std::int32_t>(cuts.size()), workspace.n_cols(), entries);
    for (const CoverCut& cut : cuts) {
        workspace.rhs.push_back(static_cast<double>(cut.variables.size() - 1));
        workspace.row_types.push_back('L');
        workspace.slack_lower.push_back(0.0);
        workspace.slack_upper.push_back(kInfinityValue);
    }
}

// A cut in ORIGINAL structural-variable space: sum_k coeff_k * x_k >= rhs
// (row_type 'G') or <= rhs (row_type 'L').
struct GeneralCut {
    std::vector<std::pair<std::int32_t, double>> terms;
    double rhs = 0.0;
    char row_type = 'G';
};

void append_general_cuts(LpProblem& workspace, const std::vector<GeneralCut>& cuts) {
    const std::int32_t old_rows = workspace.n_rows();
    std::size_t new_nnz = 0;
    for (const GeneralCut& cut : cuts) new_nnz += cut.terms.size();
    std::vector<Triplet> entries;
    entries.reserve(static_cast<std::size_t>(workspace.A.nnz()) + new_nnz);
    for (std::int32_t i = 0; i < old_rows; ++i) {
        for (std::int32_t k = workspace.A.row_ptr()[i]; k < workspace.A.row_ptr()[i + 1]; ++k) {
            const auto kk = static_cast<std::size_t>(k);
            entries.push_back({i, workspace.A.col_idx()[kk], workspace.A.values()[kk]});
        }
    }
    for (std::size_t cut_index = 0; cut_index < cuts.size(); ++cut_index) {
        const auto row = old_rows + static_cast<std::int32_t>(cut_index);
        for (const auto& [variable, coefficient] : cuts[cut_index].terms) {
            entries.push_back({row, variable, coefficient});
        }
    }
    workspace.A = CSRMatrix::from_triplets(
        old_rows + static_cast<std::int32_t>(cuts.size()), workspace.n_cols(), entries);
    for (const GeneralCut& cut : cuts) {
        workspace.rhs.push_back(cut.rhs);
        workspace.row_types.push_back(cut.row_type);
        if (cut.row_type == 'L') {
            workspace.slack_lower.push_back(0.0);
            workspace.slack_upper.push_back(kInfinityValue);
        } else {
            workspace.slack_lower.push_back(-kInfinityValue);
            workspace.slack_upper.push_back(0.0);
        }
    }
}

// Gomory mixed-integer cuts (ESTABLISHED METHOD: Gomory, "An algorithm for
// the mixed integer problem", RAND P-1885, 1960; closed-form coefficients
// per Wolsey, "Integer Programming", 1998, Thm 5.1, and Marchand & Wolsey,
// "Aggregation and mixed integer rounding to solve MIPs", Math.
// Programming 91(1), 2001, eq. (5)-(6); docs/architecture/MILP.md \S2.2).
//
// For basis row r whose basic variable x_B[r] is integer-restricted with
// fractional value beta_r = floor(beta_r) + f_r, the exact tableau
// identity (valid on the whole affine subspace {Ax+s=rhs}, not only at
// the current vertex -- it is a change of basis, not a linearization)
// beta_r - x_B[r] = sum_{j nonbasic} bar_rho_{r,j} * d_j
// (bar_rho_{r,j} = rho_{r,j} if j rests at its lower bound, -rho_{r,j} if
// at its upper bound; d_j = |x_j - resting bound| >= 0 for every feasible
// x) combines with x_B[r] integer to give a two-branch disjunction whose
// MIR closed form is:
//   sum_{j in Z, f_j<=f_r} (f_j/f_r) d_j + sum_{j in Z, f_j>f_r} ((1-f_j)/(1-f_r)) d_j
// + sum_{j not in Z, bar_rho_j>=0} (bar_rho_j/f_r) d_j + sum_{j not in Z, bar_rho_j<0} (-bar_rho_j/(1-f_r)) d_j
//   >= 1
// where f_j = frac(bar_rho_{r,j}) for integer-restricted nonbasic j.
// Slacks are always treated via the continuous branch (safe: it is a
// strict relaxation of the integer branch, never invalid, only weaker --
// no attempt is made here to detect an incidentally-integer slack row).
// A nonbasic FREE variable (VarStatus::AT_ZERO) with a nonzero tableau
// coefficient has no one-sided d_j >= 0 bound to derive from, so the
// whole row is rejected rather than silently dropping that term (which
// would not be a valid relaxation) -- a documented, deliberate scope
// limit, not an oversight.
//
// The resulting inequality is over the tableau's own nonbasic columns
// (structural AND slack); nonbasic slack terms are substituted back via
// s_i = rhs_i - (Ax)_i to produce a cut purely in ORIGINAL structural
// variables, the space append_general_cuts appends new rows into.
//
// Numerical safety (ENGINEERING DECISION; both mechanisms below were
// directly MEASURED, not assumed -- see docs/architecture/MILP.md \S2.2
// for the full account and the diagnostic runs that produced these
// numbers, on gen-ip054 from bench_miplib's 5-instance MIPLIB set):
//
// 1. kGmiMinFractionality. Every branch-coefficient formula below divides
// by f_r or (1-f_r); a basic variable that is only barely fractional
// amplifies bar_rho by up to 1/f_r before it becomes a cut term -- the
// classic, well-documented numerical weakness of Gomory-derived cuts
// (Cornuejols, "Valid inequalities for mixed integer linear programs",
// 4OR 6, 2008, notes coefficient-magnitude disparity from exactly this
// mechanism as a standard practical hazard). MEASURED: a row with
// f_r = 1.00491e-07 (barely above integrality_tolerance's own 1e-7
// eligibility floor) produced a cut coefficient of 3.7e7 from a
// bar_rho of order 1 -- the 1/f_r amplification accounts for essentially
// all of it. 0.01 bounds that amplification to 100x, comfortably below
// where it becomes the dominant term.
//
// 2. kGmiRelativeZeroTolerance. A coefficient smaller than this fraction
// of the cut's own largest surviving coefficient is treated as the
// floating-point-roundoff residue it almost certainly is (an exact
// algebraic zero, from cancellation in the slack-substitution
// arithmetic, has no reason to land near double's ~2.2e-16 machine
// epsilon by chance) rather than a real term. MEASURED: a coefficient of
// 4.49e-16 was observed sitting next to O(1) terms in an otherwise
// unrelated cut.
//
// 3. kGmiMaxDynamicRange / kGmiMaxRelativeMagnitude. A final,
// independent safety net over the assembled cut, applied after (1) and
// (2): slack substitution sums a term across every nonzero in a row, so
// even a bounded per-term coefficient can compound into a cut whose
// scale is far beyond the original model's own. Both checks REJECT
// rather than rescale -- dynamic range is scale-invariant (uniform
// rescaling cannot change the ratio between a cut's own largest and
// smallest coefficients), and discarding a numerically dangerous cut is
// always safe, exactly the standard mitigation in the cutting-plane
// literature (Cornuejols 2008; cut-quality-based selection in
// Achterberg, "Constraint Integer Programming", PhD thesis, TU Berlin,
// 2007, rejects rather than repairs). kGmiMaxRelativeMagnitude compares
// against THIS root relaxation's own largest matrix coefficient (not an
// arbitrary constant), matching the same MPS instance's own units.
constexpr double kGmiMinFractionality = 0.01;
constexpr double kGmiRelativeZeroTolerance = 1e-9;
constexpr double kGmiMaxDynamicRange = 1e8;
constexpr double kGmiMaxRelativeMagnitude = 1e4;

std::vector<GeneralCut> separate_gmi_cuts(const MilpProblem& problem, const LpProblem& workspace,
                                           const Simplex& simplex, const std::vector<double>& x,
                                           double violation_tolerance, std::uint32_t limit) {
    std::vector<GeneralCut> cuts;
    const std::int32_t n_struct = workspace.n_cols();
    const std::int32_t n_slack = workspace.n_rows();

    double matrix_max_abs = 0.0;
    {
        const double* values = workspace.A.values();
        for (std::int32_t k = 0; k < workspace.A.nnz(); ++k) {
            matrix_max_abs = std::max(matrix_max_abs, std::fabs(values[k]));
        }
    }
    if (matrix_max_abs <= 0.0) return cuts; // no structural coefficients at all

    std::vector<double> struct_coeff(static_cast<std::size_t>(n_struct));

    for (std::int32_t basic_col = 0; basic_col < n_struct && cuts.size() < limit; ++basic_col) {
        const auto bj = static_cast<std::size_t>(basic_col);
        if (problem.variable_types[bj] == VariableType::CONTINUOUS) continue;
        const std::int32_t row = simplex.basic_row_of(basic_col);
        if (row < 0) continue; // not basic in this tableau

        const double beta = x[bj];
        if (!std::isfinite(beta)) continue;
        const double floor_beta = std::floor(beta);
        const double f_r = beta - floor_beta;
        if (f_r < kGmiMinFractionality || 1.0 - f_r < kGmiMinFractionality) continue;

        const std::vector<double> rho = simplex.tableau_row(row);
        std::fill(struct_coeff.begin(), struct_coeff.end(), 0.0);
        double constant = 0.0;
        bool reject_row = false;

        for (std::int32_t j = 0; j < simplex.n_total() && !reject_row; ++j) {
            if (j == basic_col) continue;
            const double rho_j = rho[static_cast<std::size_t>(j)];
            if (rho_j == 0.0) continue;
            const Simplex::VarStatus status = simplex.status_of(j);
            if (status == Simplex::VarStatus::BASIC) continue;
            if (status == Simplex::VarStatus::AT_ZERO) {
                // Free nonbasic variable, nonzero coefficient: no valid
                // one-sided cut from this row (see function comment).
                reject_row = true;
                break;
            }

            const bool is_lower = (status == Simplex::VarStatus::AT_LOWER);
            double bound_j;
            if (j < n_struct) {
                bound_j = is_lower ? workspace.lower[static_cast<std::size_t>(j)]
                                    : workspace.upper[static_cast<std::size_t>(j)];
            } else if (j < n_struct + n_slack) {
                const auto row_i = static_cast<std::size_t>(j - n_struct);
                bound_j = is_lower ? workspace.slack_lower[row_i] : workspace.slack_upper[row_i];
            } else {
                continue; // artificial: pinned to [0,0], contributes nothing
            }
            if (!std::isfinite(bound_j)) {
                // A nonbasic variable resting at an infinite bound cannot
                // occur at a genuine optimum; guard rather than propagate
                // an Inf/NaN cut coefficient.
                reject_row = true;
                break;
            }

            const double bar_rho = is_lower ? rho_j : -rho_j;
            const bool is_integer_col =
                (j < n_struct) &&
                problem.variable_types[static_cast<std::size_t>(j)] != VariableType::CONTINUOUS;
            double coeff;
            if (is_integer_col) {
                const double fj = bar_rho - std::floor(bar_rho);
                coeff = (fj <= f_r) ? (fj / f_r) : ((1.0 - fj) / (1.0 - f_r));
            } else {
                coeff = (bar_rho >= 0.0) ? (bar_rho / f_r) : (-bar_rho / (1.0 - f_r));
            }
            if (!std::isfinite(coeff) || coeff == 0.0) continue;

            // The branch formula above yields the coefficient of d_j, the
            // ALWAYS-NONNEGATIVE deviation from j's resting bound: d_j =
            // (x_j - bound_j) when resting at the LOWER bound, but d_j =
            // (bound_j - x_j) -- the OPPOSITE sign of (x_j - bound_j) --
            // when resting at the UPPER bound. term_coeff below is
            // coeff*d_j re-expressed as an affine function of x_j itself
            // (term_coeff*x_j - term_coeff*bound_j), so it must carry that
            // same sign flip; using coeff directly, unflipped, silently
            // produces the mirror-image (invalid) inequality for every
            // upper-resting nonbasic column.
            const double term_coeff = is_lower ? coeff : -coeff;

            if (j < n_struct) {
                struct_coeff[static_cast<std::size_t>(j)] += term_coeff;
                constant -= term_coeff * bound_j;
            } else {
                // Substitute s_i = rhs_i - (A x)_i for nonbasic slack i:
                // term_coeff*(s_i - bound_i)
                //   = term_coeff*rhs_i - term_coeff*bound_i - term_coeff*sum_k A_ik x_k.
                const std::int32_t row_i = j - n_struct;
                const auto row_ii = static_cast<std::size_t>(row_i);
                constant += term_coeff * workspace.rhs[row_ii];
                constant -= term_coeff * bound_j;
                const std::int32_t begin = workspace.A.row_ptr()[row_i];
                const std::int32_t end = workspace.A.row_ptr()[row_i + 1];
                for (std::int32_t k = begin; k < end; ++k) {
                    const auto kk = static_cast<std::size_t>(k);
                    struct_coeff[static_cast<std::size_t>(workspace.A.col_idx()[kk])] -=
                        term_coeff * workspace.A.values()[kk];
                }
            }
        }
        if (reject_row) continue;

        // Numerical cleanup 1: drop coefficients that are almost certainly
        // floating-point-roundoff residue rather than real terms (see the
        // function-level comment, kGmiRelativeZeroTolerance).
        double row_max_abs = 0.0;
        for (std::int32_t k = 0; k < n_struct; ++k) {
            row_max_abs = std::max(row_max_abs, std::fabs(struct_coeff[static_cast<std::size_t>(k)]));
        }
        if (row_max_abs <= 0.0) continue; // every term cancelled exactly; no cut
        const double zero_floor = kGmiRelativeZeroTolerance * row_max_abs;
        double cut_min_abs = std::numeric_limits<double>::infinity();
        double cut_max_abs = 0.0;
        for (std::int32_t k = 0; k < n_struct; ++k) {
            const auto kk = static_cast<std::size_t>(k);
            const double c = struct_coeff[kk];
            if (c == 0.0) continue;
            if (std::fabs(c) < zero_floor) {
                struct_coeff[kk] = 0.0;
                continue;
            }
            cut_min_abs = std::min(cut_min_abs, std::fabs(c));
            cut_max_abs = std::max(cut_max_abs, std::fabs(c));
        }
        // Numerical cleanup 2/3: reject (never rescale -- see the
        // function-level comment) a cut whose surviving coefficients still
        // span too wide a range, or are too large relative to this
        // relaxation's own matrix, to trust numerically.
        if (cut_max_abs <= 0.0) continue;
        if (cut_max_abs / cut_min_abs > kGmiMaxDynamicRange) continue;
        if (cut_max_abs > kGmiMaxRelativeMagnitude * matrix_max_abs) continue;

        const double cut_rhs = 1.0 - constant;
        double lhs = 0.0;
        for (std::int32_t k = 0; k < n_struct; ++k) {
            const double c = struct_coeff[static_cast<std::size_t>(k)];
            if (c != 0.0) lhs += c * x[static_cast<std::size_t>(k)];
        }
        // Defense in depth: an algebraically valid cut must be violated by
        // the point it was derived from. Reject anything that is not,
        // rather than trust the derivation blindly.
        if (!(lhs < cut_rhs - violation_tolerance)) continue;

        GeneralCut cut;
        cut.row_type = 'G';
        cut.rhs = cut_rhs;
        for (std::int32_t k = 0; k < n_struct; ++k) {
            const double c = struct_coeff[static_cast<std::size_t>(k)];
            if (c != 0.0) cut.terms.push_back({k, c});
        }
        if (!cut.terms.empty()) cuts.push_back(std::move(cut));
    }
    return cuts;
}

std::vector<FractionalCandidate> fractional_candidates(const MilpProblem& problem,
                                                        const std::vector<double>& x,
                                                        double integrality_tolerance) {
    std::vector<FractionalCandidate> candidates;
    for (std::int32_t j = 0; j < problem.n_cols(); ++j) {
        const auto jj = static_cast<std::size_t>(j);
        if (problem.variable_types[jj] == VariableType::CONTINUOUS) continue;
        if (!std::isfinite(x[jj])) continue;
        const double floor_value = std::floor(x[jj]);
        const double fraction = x[jj] - floor_value;
        if (fraction <= integrality_tolerance || 1.0 - fraction <= integrality_tolerance) {
            continue;
        }
        const double fractionality = std::min(fraction, 1.0 - fraction);
        candidates.push_back({j, fraction, fractionality});
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.fractionality != rhs.fractionality) {
            return lhs.fractionality > rhs.fractionality;
        }
        return lhs.variable < rhs.variable;
    });
    return candidates;
}

using SharedNodePtr = std::shared_ptr<const SearchNode>;
using ConcurrentNodeQueue = ConcurrentPriorityQueue<SharedNodePtr, NodeCompare>;

double current_best_bound(ConcurrentNodeQueue& queue, bool has_incumbent, double incumbent) {
    const auto top_bound =
        queue.best_priority_bound([](const SharedNodePtr& node) { return node->priority_bound; });
    if (top_bound.has_value()) return *top_bound;
    return has_incumbent ? incumbent : kInfinityValue;
}

double relative_gap(bool has_incumbent, double incumbent, double best_bound) {
    if (!has_incumbent || !std::isfinite(best_bound)) return kInfinityValue;
    return std::max(0.0, (incumbent - best_bound) / (1.0 + std::fabs(incumbent)));
}

// Thread-safe incumbent acceptance -- the ONLY place `objective` and `x`
// are updated together (see IncumbentState's own comment in
// ParallelSearch.hpp for why that pairing needs one lock rather than a
// lock-free CAS on the double alone). Behaviorally identical to the
// pre-parallel `consider_incumbent` closure: same feasibility/integrality
// gate, same strict-improvement comparison against the CURRENT incumbent
// re-read under the lock (so two workers racing to install an improving
// incumbent cannot both "win" -- only a genuine improvement over
// whatever is there AT COMMIT TIME is accepted).
bool consider_incumbent_mt(const MilpProblem& problem, const std::vector<double>& candidate,
                            const std::vector<double>& lower, const std::vector<double>& upper,
                            const LpProblem& workspace, const MilpSolverOptions& options,
                            IncumbentState& inc) {
    if (!feasible_point(problem, candidate, lower, upper, options.feasibility_tolerance,
                         ParallelMode::SERIAL) ||
        !integral_point(problem, candidate, options.integrality_tolerance)) {
        return false;
    }
    const double candidate_objective = objective_value(workspace, candidate);
    if (inc.clearly_worse(candidate_objective, options.objective_tolerance)) return false;

    std::lock_guard<std::mutex> lock(inc.mutex);
    const double current = inc.objective.load(std::memory_order_relaxed);
    if (!std::isfinite(current) ||
        candidate_objective <
            current - options.objective_tolerance * (1.0 + std::fabs(current))) {
        inc.x = candidate;
        inc.objective.store(candidate_objective, std::memory_order_release);
        ++inc.incumbent_updates;
        return true;
    }
    return false;
}

// What happened to one popped node. `Branched` carries its two children;
// every other outcome carries none. `Fatal` means a condition requiring
// the WHOLE search to stop was detected (UNBOUNDED relaxation, or an
// internal inconsistency this project treats as NUMERICAL_FAILURE rather
// than guessing at); the specific status is recorded into the shared
// SearchOutcome below by the caller, not returned here, since multiple
// workers could each independently reach a Fatal outcome and only the
// first one's status should stick.
enum class NodeOutcome { Pruned, Requeue, Branched, Fatal, TimedOutMidHeuristic };

struct NodeResult {
    NodeOutcome outcome = NodeOutcome::Pruned;
    MilpStatus fatal_status = MilpStatus::NUMERICAL_FAILURE;
    SharedNodePtr left, right;
    // Root-only reporting (a Requeue outcome from root's own cut
    // separation): how many cuts were just added, so the single-threaded
    // caller can attribute them to the right MilpSolution counter without
    // process_node needing to know about MilpSolution's field layout.
    std::size_t cover_cuts_added = 0;
    std::size_t gmi_cuts_added = 0;
};

// Shared, read-mostly state every call to process_node needs, gathered
// into one struct so the lambda capture list below stays legible rather
// than an ever-growing by-reference capture of a dozen loose locals.
struct SharedSearchState {
    const MilpProblem& problem;
    const MilpSolverOptions& options;
    const LpSolverOptions& relaxation_options;
    const std::vector<double>& root_lower;
    const std::vector<double>& root_upper;
    bool has_integer_variables = false;
    ConcurrentNodeQueue& open;
    IncumbentState& incumbent;
    std::atomic<std::uint64_t>& next_node_order;
    std::chrono::steady_clock::time_point start;
    double time_limit_seconds = 0.0;

    // Lazily-computed-once Ruiz scaling for warm-started direct-Simplex
    // node solves (docs/architecture/LP.md \S1/\S2) -- std::call_once is
    // exactly the right primitive for "compute once, then every reader
    // sees a fully-published, read-only result," and avoids a bespoke
    // double-checked-lock. Irrelevant (never touched) when
    // warm_start_node_relaxations is off, its own default.
    std::once_flag node_scale_once;
    ScaleFactors node_scale;

    // Root-only ("at most once each") guards, exactly mirroring the
    // pre-parallel code's own root_cuts_separated/root_gmi_separated
    // locals. Safe as plain (non-atomic) fields: node->depth == 0 is only
    // ever true during the mandatory serial root phase, which is
    // single-threaded by construction -- no worker thread ever calls
    // process_node with a depth-0 node once the root has branched.
    bool root_cuts_separated = false;
    bool root_gmi_separated = false;

    // Live count behind MilpSolverOptions::max_live_warm_start_bases. Held
    // by shared_ptr, not as a plain member, because each retained basis's
    // deleter must decrement it: a node (and therefore its basis) can in
    // principle outlive this struct during teardown, and capturing a
    // shared_ptr copy makes that ordering irrelevant rather than merely
    // unlikely.
    std::shared_ptr<std::atomic<std::uint64_t>> live_parent_bases =
        std::make_shared<std::atomic<std::uint64_t>>(0);

    bool timed_out() const {
        return time_limit_seconds > 0.0 &&
               std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() >=
                   time_limit_seconds;
    }
};

// Processes exactly one popped node: solves its relaxation, runs root-only
// cut separation/RENS/local-improvement (self-gating on node->depth == 0,
// so this is safe to call from a worker thread on a non-root node -- those
// blocks simply never fire there), runs the shallow-depth LP dive, and
// either prunes, requeues (root cuts need the same node reprocessed once
// its workspace has cuts appended), or branches into two children.
//
// Every mutable quantity this function touches beyond `shared` is either
// `ctx` (this call's own WorkerContext -- never shared with another
// thread) or `shared.incumbent`/`shared.open` (both independently
// thread-safe by construction, per their own types). This is what makes
// calling it from multiple worker threads simultaneously safe: no two
// calls ever touch the same WorkerContext, and the two pieces of state
// they DO share serialize their own critical sections internally.
NodeResult process_node(SharedSearchState& shared, WorkerContext& ctx, const SharedNodePtr& node) {
    const MilpProblem& problem = shared.problem;
    const MilpSolverOptions& options = shared.options;
    LpProblem& workspace = ctx.workspace;

    const auto record_pseudocost = [&](const SearchNode& child, double child_bound,
                                       bool infeasible) {
        if (child.branch_variable < 0 || child.branch_distance <= 0.0 ||
            !std::isfinite(child.priority_bound)) {
            return;
        }
        const auto j = static_cast<std::size_t>(child.branch_variable);
        const double unit_cost = infeasible
                                     ? kInfinityValue
                                     : std::max(0.0, child_bound - child.priority_bound) /
                                           child.branch_distance;
        if (child.branch_direction < 0) {
            ctx.down_pseudocost[j] += unit_cost;
            ++ctx.down_observations[j];
        } else {
            ctx.up_pseudocost[j] += unit_cost;
            ++ctx.up_observations[j];
        }
    };
    const auto reliable = [&](std::size_t j) {
        return ctx.down_observations[j] >= options.reliability_threshold &&
               ctx.up_observations[j] >= options.reliability_threshold;
    };
    const auto pseudocost_score = [&](const FractionalCandidate& candidate) {
        const auto j = static_cast<std::size_t>(candidate.variable);
        if (ctx.down_observations[j] == 0 || ctx.up_observations[j] == 0) return -1.0;
        const double down = (ctx.down_pseudocost[j] / ctx.down_observations[j]) * candidate.fraction;
        const double up =
            (ctx.up_pseudocost[j] / ctx.up_observations[j]) * (1.0 - candidate.fraction);
        return std::min(down, up);
    };
    const auto observe_pseudocost = [&](std::int32_t variable, int direction, double unit_cost) {
        const auto j = static_cast<std::size_t>(variable);
        if (direction < 0) {
            ctx.down_pseudocost[j] += unit_cost;
            ++ctx.down_observations[j];
        } else {
            ctx.up_pseudocost[j] += unit_cost;
            ++ctx.up_observations[j];
        }
    };
    const auto consider_incumbent = [&](const std::vector<double>& candidate,
                                        const std::vector<double>& lower,
                                        const std::vector<double>& upper) {
        return consider_incumbent_mt(problem, candidate, lower, upper, workspace, options,
                                      shared.incumbent);
    };
    const auto has_incumbent = [&] {
        return std::isfinite(shared.incumbent.objective.load(std::memory_order_acquire));
    };
    const auto incumbent_value = [&] {
        return shared.incumbent.objective.load(std::memory_order_acquire);
    };

    bool heuristic_timeout = false;
    const auto attempt_lp_dive = [&](const LpSolution& starting,
                                     const std::vector<double>& starting_lower,
                                     const std::vector<double>& starting_upper) {
        if (!options.use_diving_heuristic || options.diving_max_depth == 0 ||
            options.diving_max_lp_relaxations == 0 || has_incumbent()) {
            return;
        }
        LpSolution current = starting;
        std::vector<double> dive_lower = starting_lower;
        std::vector<double> dive_upper = starting_upper;
        std::uint32_t dive_relaxations = 0;
        for (std::uint32_t depth = 0; depth < options.diving_max_depth; ++depth) {
            if (shared.timed_out()) {
                heuristic_timeout = true;
                return;
            }
            if (integral_point(problem, current.x, options.integrality_tolerance)) {
                consider_incumbent(current.x, dive_lower, dive_upper);
                return;
            }
            const auto candidates =
                fractional_candidates(problem, current.x, options.integrality_tolerance);
            if (candidates.empty()) return;

            const FractionalCandidate& candidate = candidates.front();
            const auto j = static_cast<std::size_t>(candidate.variable);
            const double floor_value = std::floor(current.x[j]);
            const double ceil_value = std::ceil(current.x[j]);
            const int preferred_direction = workspace.obj[j] < 0.0 ? +1 : -1;
            const auto solve_dive_child = [&](int direction) -> bool {
                std::vector<double> child_lower = dive_lower;
                std::vector<double> child_upper = dive_upper;
                if (direction < 0) {
                    child_upper[j] = std::min(child_upper[j], floor_value);
                } else {
                    child_lower[j] = std::max(child_lower[j], ceil_value);
                }
                if (!bounds_are_valid(child_lower, child_upper)) return false;
                workspace.lower = child_lower;
                workspace.upper = child_upper;
                ++ctx.lp_relaxations;
                ++ctx.diving_heuristic_lp_relaxations;
                ++dive_relaxations;
                const LpSolution child = solve_lp(workspace, shared.relaxation_options);
                workspace.lower = dive_lower;
                workspace.upper = dive_upper;
                if (child.status != LpStatus::OPTIMAL ||
                    child.x.size() != static_cast<std::size_t>(problem.n_cols())) {
                    return false;
                }
                current = child;
                dive_lower = std::move(child_lower);
                dive_upper = std::move(child_upper);
                return true;
            };
            if (dive_relaxations >= options.diving_max_lp_relaxations ||
                (!solve_dive_child(preferred_direction) &&
                 (dive_relaxations >= options.diving_max_lp_relaxations ||
                  !solve_dive_child(-preferred_direction)))) {
                return;
            }
        }
        if (integral_point(problem, current.x, options.integrality_tolerance)) {
            consider_incumbent(current.x, dive_lower, dive_upper);
        }
    };

    const auto attempt_local_improvement = [&](const std::vector<double>& node_lower,
                                               const std::vector<double>& node_upper) {
        if (!options.use_local_improvement || options.local_improvement_passes == 0 ||
            options.local_improvement_max_trials == 0 || !has_incumbent()) {
            return;
        }
        std::vector<double> current;
        {
            std::lock_guard<std::mutex> lock(shared.incumbent.mutex);
            current = shared.incumbent.x;
        }
        for (std::uint32_t pass = 0; pass < options.local_improvement_passes; ++pass) {
            bool improved = false;
            std::uint32_t trials = 0;
            for (std::int32_t j = 0; j < problem.n_cols() &&
                                      trials < options.local_improvement_max_trials;
                 ++j) {
                const auto jj = static_cast<std::size_t>(j);
                if (problem.variable_types[jj] == VariableType::CONTINUOUS) continue;
                if (!std::isfinite(current[jj])) continue;

                std::vector<double> trial_lower = node_lower;
                std::vector<double> trial_upper = node_upper;
                for (std::int32_t k = 0; k < problem.n_cols(); ++k) {
                    const auto kk = static_cast<std::size_t>(k);
                    if (problem.variable_types[kk] == VariableType::CONTINUOUS) continue;
                    const double fixed = std::round(current[kk]);
                    trial_lower[kk] = std::max(trial_lower[kk], fixed);
                    trial_upper[kk] = std::min(trial_upper[kk], fixed);
                }
                const double current_value = std::round(current[jj]);
                double trial_value = current_value;
                if (binary_domain(problem, problem.relaxation, j)) {
                    trial_value = current_value <= 0.5 ? 1.0 : 0.0;
                } else {
                    const double up = current_value + 1.0;
                    const double down = current_value - 1.0;
                    if (up <= trial_upper[jj]) trial_value = up;
                    else if (down >= trial_lower[jj]) trial_value = down;
                    else continue;
                }
                trial_lower[jj] = std::max(trial_lower[jj], trial_value);
                trial_upper[jj] = std::min(trial_upper[jj], trial_value);
                if (!bounds_are_valid(trial_lower, trial_upper)) continue;

                LpProblem local = workspace;
                local.lower = trial_lower;
                local.upper = trial_upper;
                ++trials;
                ++ctx.lp_relaxations;
                ++ctx.local_improvement_lp_relaxations;
                const LpSolution local_solution = solve_lp(local, shared.relaxation_options);
                if (local_solution.status != LpStatus::OPTIMAL ||
                    local_solution.x.size() != static_cast<std::size_t>(problem.n_cols())) {
                    continue;
                }
                const double before = incumbent_value();
                if (consider_incumbent(local_solution.x, node_lower, node_upper) &&
                    incumbent_value() < before) {
                    std::lock_guard<std::mutex> lock(shared.incumbent.mutex);
                    current = shared.incumbent.x;
                    improved = true;
                }
                if (shared.timed_out()) return;
            }
            if (!improved) break;
        }
    };

    const auto attempt_rens = [&](const LpSolution& starting,
                                  const std::vector<double>& starting_lower,
                                  const std::vector<double>& starting_upper) {
        if (!options.use_rens_heuristic) return;
        if (integral_point(problem, starting.x, options.integrality_tolerance)) return;
        std::vector<double> rens_lower = starting_lower;
        std::vector<double> rens_upper = starting_upper;
        for (std::int32_t j = 0; j < problem.n_cols(); ++j) {
            const auto jj = static_cast<std::size_t>(j);
            if (problem.variable_types[jj] == VariableType::CONTINUOUS) continue;
            const double value = starting.x[jj];
            const double floor_value = std::floor(value);
            const double ceil_value = std::ceil(value);
            if (ceil_value - value <= options.integrality_tolerance ||
                value - floor_value <= options.integrality_tolerance) {
                const double fixed = std::round(value);
                rens_lower[jj] = std::max(rens_lower[jj], fixed);
                rens_upper[jj] = std::min(rens_upper[jj], fixed);
            } else {
                rens_lower[jj] = std::max(rens_lower[jj], floor_value);
                rens_upper[jj] = std::min(rens_upper[jj], ceil_value);
            }
        }
        if (!bounds_are_valid(rens_lower, rens_upper)) return;
        LpProblem restricted = workspace;
        restricted.lower = std::move(rens_lower);
        restricted.upper = std::move(rens_upper);
        ++ctx.lp_relaxations;
        ++ctx.rens_heuristic_lp_relaxations;
        const LpSolution restricted_solution = solve_lp(restricted, shared.relaxation_options);
        if (restricted_solution.status != LpStatus::OPTIMAL ||
            restricted_solution.x.size() != static_cast<std::size_t>(problem.n_cols())) {
            return;
        }
        consider_incumbent(restricted_solution.x, starting_lower, starting_upper);
    };

    NodeResult result;
    ++ctx.nodes_processed;

    const std::shared_ptr<const Simplex::Basis>& node_parent_basis = node->parent_basis;

    if (has_incumbent() &&
        node->priority_bound >=
            incumbent_value() - options.objective_tolerance * (1.0 + std::fabs(incumbent_value()))) {
        ++ctx.nodes_pruned;
        result.outcome = NodeOutcome::Pruned;
        return result;
    }

    std::vector<double> lower;
    std::vector<double> upper;
    materialize_bounds(*node, shared.root_lower, shared.root_upper, lower, upper);
    if (!bounds_are_valid(lower, upper)) {
        ++ctx.nodes_pruned;
        result.outcome = NodeOutcome::Pruned;
        return result;
    }
    workspace.lower = lower;
    workspace.upper = upper;

    ++ctx.lp_relaxations;
    LpSolution relaxation;
    std::shared_ptr<const Simplex::Basis> node_basis;
    if (node->depth == 0 || !options.warm_start_node_relaxations) {
        relaxation = solve_lp(workspace, shared.relaxation_options);
    } else {
        std::call_once(shared.node_scale_once, [&] {
            shared.node_scale = shared.relaxation_options.use_ruiz_scaling
                                     ? compute_ruiz_scaling(workspace.A)
                                     : ScaleFactors::identity(workspace.n_rows(), workspace.n_cols());
        });
        Simplex simplex(workspace, shared.relaxation_options.backend,
                        shared.relaxation_options.use_ruiz_scaling,
                        shared.relaxation_options.pricing_rule, LpAlgorithm::AUTO,
                        shared.relaxation_options.parallel_mode, &shared.node_scale);
        if (shared.relaxation_options.simplex_time_budget_seconds > 0.0) {
            simplex.set_time_budget(shared.relaxation_options.simplex_time_budget_seconds);
        }
        if (node_parent_basis) simplex.set_warm_start_basis(node_parent_basis.get());

        const LpResult lp = simplex.solve();
        relaxation.status = lp.status;
        relaxation.x = lp.x;
        relaxation.objective_value = lp.objective_value;
        if (lp.used_warm_start) ++ctx.warm_started_relaxations;
        if (lp.warm_start_attempted && !lp.used_warm_start) {
            ++ctx.warm_start_verification_fallbacks;
        }
        if (lp.status == LpStatus::OPTIMAL) {
            // Retain this basis for the children only while the live-basis
            // budget allows (MilpSolverOptions::max_live_warm_start_bases).
            // Past the cap the children simply warm-start from nothing and
            // take the already-tested cold path -- a speed degradation
            // under memory pressure, never a change to pruning or to the
            // certified answer.
            const std::uint64_t cap = options.max_live_warm_start_bases;
            auto& live = *shared.live_parent_bases;
            if (cap == 0 || live.load(std::memory_order_relaxed) < cap) {
                live.fetch_add(1, std::memory_order_relaxed);
                node_basis = std::shared_ptr<const Simplex::Basis>(
                    new Simplex::Basis(simplex.export_basis()),
                    [counter = shared.live_parent_bases](const Simplex::Basis* b) {
                        counter->fetch_sub(1, std::memory_order_relaxed);
                        delete b;
                    });
            }
        }
    }

    if (relaxation.status == LpStatus::INFEASIBLE) {
        record_pseudocost(*node, node->priority_bound, true);
        ++ctx.nodes_pruned;
        result.outcome = NodeOutcome::Pruned;
        return result;
    }
    if (relaxation.status == LpStatus::UNBOUNDED) {
        result.outcome = NodeOutcome::Fatal;
        result.fatal_status =
            shared.has_integer_variables ? MilpStatus::UNBOUNDED_RELAXATION : MilpStatus::UNBOUNDED;
        return result;
    }
    if (relaxation.status != LpStatus::OPTIMAL ||
        relaxation.x.size() != static_cast<std::size_t>(problem.n_cols())) {
        result.outcome = NodeOutcome::Fatal;
        result.fatal_status = MilpStatus::NUMERICAL_FAILURE;
        return result;
    }

    const double lower_bound = relaxation.objective_value;
    record_pseudocost(*node, lower_bound, false);

    if (node->depth == 0 && !shared.root_cuts_separated && options.enable_root_cover_cuts) {
        shared.root_cuts_separated = true;
        const auto cuts = separate_cover_cuts(problem, relaxation.x, options.cut_violation_tolerance,
                                              options.max_root_cover_cuts);
        if (!cuts.empty()) {
            append_cover_cuts(workspace, cuts);
            result.outcome = NodeOutcome::Requeue;
            result.cover_cuts_added = cuts.size();
            return result;
        }
    }
    if (node->depth == 0 && !shared.root_gmi_separated && options.enable_root_gmi_cuts) {
        shared.root_gmi_separated = true;
        Simplex gmi_simplex(workspace, PricingBackend::CPU, /*use_ruiz_scaling=*/false,
                            shared.relaxation_options.pricing_rule, LpAlgorithm::AUTO,
                            shared.relaxation_options.parallel_mode);
        ++ctx.lp_relaxations;
        const LpResult gmi_lp = gmi_simplex.solve();
        if (gmi_lp.status == LpStatus::OPTIMAL &&
            gmi_lp.x.size() == static_cast<std::size_t>(problem.n_cols())) {
            const auto cuts =
                separate_gmi_cuts(problem, workspace, gmi_simplex, gmi_lp.x,
                                  options.cut_violation_tolerance, options.max_root_gmi_cuts);
            if (!cuts.empty()) {
                append_general_cuts(workspace, cuts);
                result.outcome = NodeOutcome::Requeue;
                result.gmi_cuts_added = cuts.size();
                return result;
            }
        }
    }

    if (has_incumbent() &&
        lower_bound >=
            incumbent_value() - options.objective_tolerance * (1.0 + std::fabs(incumbent_value()))) {
        ++ctx.nodes_pruned;
        result.outcome = NodeOutcome::Pruned;
        return result;
    }

    const bool candidate_integral =
        integral_point(problem, relaxation.x, options.integrality_tolerance);
    const std::vector<double> rounded = rounded_point(problem, relaxation.x, lower, upper);
    if (candidate_integral || options.use_rounding_heuristic) {
        bool accepted = consider_incumbent(rounded, lower, upper);
        if (!accepted && candidate_integral) {
            // `rounded` snaps every already-near-integral column to its
            // exact nearest integer. For a row with large coefficients
            // (MEASURED on MIPLIB `ej`: up to ~51015), even a per-column
            // shift within `integrality_tolerance` can move that row's Ax
            // by an amount comparable to -- or exceeding -- the feasibility
            // gate's own noise budget, even though `relaxation.x` itself
            // (unrounded) was already certified OPTIMAL/feasible by Simplex
            // using the identical scaled-residual formula
            // (Simplex.cpp::finalize_result). Retry with the RAW,
            // already-certified point before treating this as a fatal,
            // search-aborting failure -- `candidate_integral` was computed
            // against `relaxation.x` itself, so it is exactly as
            // "integral enough" as the rounded version, just without the
            // rounding-induced perturbation.
            accepted = consider_incumbent(relaxation.x, lower, upper);
        }
        if (!accepted && candidate_integral) {
            result.outcome = NodeOutcome::Fatal;
            result.fatal_status = MilpStatus::NUMERICAL_FAILURE;
            return result;
        }
    }
    if (candidate_integral) {
        result.outcome = NodeOutcome::Pruned;
        return result;
    }

    if (node->depth == 0) {
        attempt_rens(relaxation, lower, upper);
        if (shared.timed_out()) {
            result.outcome = NodeOutcome::TimedOutMidHeuristic;
            return result;
        }
    }
    if (node->depth <= 2 && !has_incumbent()) {
        attempt_lp_dive(relaxation, lower, upper);
        if (heuristic_timeout) {
            result.outcome = NodeOutcome::TimedOutMidHeuristic;
            return result;
        }
    }
    if (node->depth == 0 && has_incumbent()) {
        attempt_local_improvement(lower, upper);
        if (shared.timed_out()) {
            result.outcome = NodeOutcome::TimedOutMidHeuristic;
            return result;
        }
    }

    for (std::int32_t j = 0; j < problem.n_cols(); ++j) {
        const auto jx = static_cast<std::size_t>(j);
        if (problem.variable_types[jx] != VariableType::CONTINUOUS &&
            !std::isfinite(relaxation.x[jx])) {
            result.outcome = NodeOutcome::Fatal;
            result.fatal_status = MilpStatus::NUMERICAL_FAILURE;
            return result;
        }
    }

    const std::vector<FractionalCandidate> candidates =
        fractional_candidates(problem, relaxation.x, options.integrality_tolerance);
    if (candidates.empty()) {
        result.outcome = NodeOutcome::Fatal;
        result.fatal_status = MilpStatus::NUMERICAL_FAILURE;
        return result;
    }

    std::int32_t branch_variable = candidates.front().variable;
    if (options.branching_rule == MilpBranchingRule::PSEUDOCOST ||
        options.branching_rule == MilpBranchingRule::RELIABILITY) {
        if (options.branching_rule == MilpBranchingRule::RELIABILITY) {
            std::uint32_t probes = 0;
            for (const FractionalCandidate& candidate : candidates) {
                const auto j = static_cast<std::size_t>(candidate.variable);
                if (reliable(j)) continue;
                if (probes >= options.strong_branching_candidates) break;
                if (shared.timed_out()) {
                    result.outcome = NodeOutcome::TimedOutMidHeuristic;
                    return result;
                }
                const double floor_probe = std::floor(relaxation.x[j]);
                const double ceil_probe = std::ceil(relaxation.x[j]);
                const double down_distance = candidate.fraction;
                const double up_distance = 1.0 - candidate.fraction;

                auto probe_child = [&](int direction, double bound,
                                       double distance) -> std::pair<bool, double> {
                    std::vector<double> probe_lower = lower;
                    std::vector<double> probe_upper = upper;
                    if (direction < 0) {
                        probe_upper[j] = std::min(probe_upper[j], bound);
                    } else {
                        probe_lower[j] = std::max(probe_lower[j], bound);
                    }
                    if (!bounds_are_valid(probe_lower, probe_upper)) return {true, kInfinityValue};
                    workspace.lower = probe_lower;
                    workspace.upper = probe_upper;
                    ++ctx.lp_relaxations;
                    ++ctx.strong_branching_probes;
                    const LpSolution probe = solve_lp(workspace, shared.relaxation_options);
                    workspace.lower = lower;
                    workspace.upper = upper;
                    if (probe.status == LpStatus::INFEASIBLE) return {true, kInfinityValue};
                    if (probe.status != LpStatus::OPTIMAL || distance <= 0.0) return {false, 0.0};
                    return {true, std::max(0.0, probe.objective_value - lower_bound) / distance};
                };
                const auto down_probe = probe_child(-1, floor_probe, down_distance);
                const auto up_probe = probe_child(+1, ceil_probe, up_distance);
                if (down_probe.first) observe_pseudocost(candidate.variable, -1, down_probe.second);
                if (up_probe.first) observe_pseudocost(candidate.variable, +1, up_probe.second);
                ++probes;
            }
        }
        double best_score = -1.0;
        for (const FractionalCandidate& candidate : candidates) {
            const auto j = static_cast<std::size_t>(candidate.variable);
            if (options.branching_rule == MilpBranchingRule::RELIABILITY && !reliable(j)) continue;
            const double score = pseudocost_score(candidate);
            if (score > best_score) {
                best_score = score;
                branch_variable = candidate.variable;
            }
        }
    }

    const auto jj = static_cast<std::size_t>(branch_variable);
    const double floor_value = std::floor(relaxation.x[jj]);
    const double ceil_value = std::ceil(relaxation.x[jj]);
    if (floor_value >= ceil_value || !std::isfinite(floor_value) || !std::isfinite(ceil_value)) {
        result.outcome = NodeOutcome::Fatal;
        result.fatal_status = MilpStatus::NUMERICAL_FAILURE;
        return result;
    }

    auto left = std::make_shared<SearchNode>();
    left->parent = node;
    left->depth = node->depth + 1;
    left->order = shared.next_node_order.fetch_add(1, std::memory_order_relaxed);
    left->priority_bound = lower_bound;
    left->change.variable = branch_variable;
    left->change.upper = floor_value;
    left->branch_variable = branch_variable;
    left->branch_direction = -1;
    left->branch_distance = relaxation.x[jj] - floor_value;

    auto right = std::make_shared<SearchNode>();
    right->parent = node;
    right->depth = node->depth + 1;
    right->order = shared.next_node_order.fetch_add(1, std::memory_order_relaxed);
    right->priority_bound = lower_bound;
    right->change.variable = branch_variable;
    right->change.lower = ceil_value;
    right->branch_variable = branch_variable;
    right->branch_direction = +1;
    right->branch_distance = ceil_value - relaxation.x[jj];

    if (node_basis) {
        left->parent_basis = node_basis;
        right->parent_basis = node_basis;
    }

    result.outcome = NodeOutcome::Branched;
    result.left = std::move(left);
    result.right = std::move(right);
    return result;
}

} // namespace

MilpSolution solve_milp(const MilpProblem& problem, const MilpSolverOptions& options) {
    validate_milp_problem(problem);
    if (options.integrality_tolerance < 0.0 || options.feasibility_tolerance < 0.0 ||
        options.objective_tolerance < 0.0 || options.time_limit_seconds < 0.0) {
        throw std::invalid_argument("MilpSolverOptions: tolerances and limits must be nonnegative");
    }
    if (options.reliability_threshold == 0 &&
        options.branching_rule == MilpBranchingRule::RELIABILITY) {
        throw std::invalid_argument("MilpSolverOptions: reliability threshold must be positive");
    }

    // Exact enumeration path, when the caller opted in AND the model is
    // exactly the structure ExactBinarySplit.hpp documents. A refusal here
    // is ordinary: it just falls through to branch-and-bound below.
    if (options.enable_exact_binary_split) {
        const auto exact = try_exact_binary_split(problem, options.exact_binary_split_memory_bytes,
                                                  options.exact_binary_split_threads);
        if (exact.applicable && exact.solved) {
            MilpSolution s;
            s.status = MilpStatus::OPTIMAL;
            s.has_incumbent = true;
            s.x = exact.x;
            s.objective_value = problem.maximize ? -exact.objective : exact.objective;
            s.best_bound = s.objective_value;   // proven, so bound == value
            s.relative_gap = 0.0;
            s.lp_relaxations = 0;
            s.nodes_processed = exact.subsets_examined;
            s.incumbent_updates = 1;
            return s;
        }
        if (exact.applicable && !exact.solved) {
            MilpSolution s;
            s.status = MilpStatus::INFEASIBLE;
            return s;
        }
    }

    MilpSolution solution;
    // This is the conservative terminal value if every queued relaxation is
    // proven infeasible. It is replaced by an explicit limit/failure status
    // whenever the search stops for any other reason.
    solution.status = MilpStatus::INFEASIBLE;
    const auto start = std::chrono::steady_clock::now();
    const auto timed_out = [&]() {
        return options.time_limit_seconds > 0.0 &&
               std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() >=
                   options.time_limit_seconds;
    };

    // One mutable workspace, copied once per worker context (root's own,
    // plus one per parallel worker if parallel_worker_count > 1) rather
    // than per node -- copying the sparse matrix per node would make a
    // large B&B tree memory-bound before the LP solver had a chance to
    // work; copying it per WORKER is a small, fixed cost paid once at
    // startup, given this project's own MILP benchmark instances (7-164
    // rows) are small.
    LpProblem workspace = problem.relaxation;
    // LP is a minimization engine. Keep the public MILP model in its natural
    // objective sense and normalize only this private relaxation workspace.
    if (problem.maximize) {
        for (double& coefficient : workspace.obj) coefficient = -coefficient;
    }
    std::vector<double> root_lower = workspace.lower;
    std::vector<double> root_upper = workspace.upper;
    for (std::int32_t j = 0; j < problem.n_cols(); ++j) {
        const auto jj = static_cast<std::size_t>(j);
        if (problem.variable_types[jj] == VariableType::BINARY) {
            root_lower[jj] = std::max(root_lower[jj], 0.0);
            root_upper[jj] = std::min(root_upper[jj], 1.0);
        }
    }
    if (!bounds_are_valid(root_lower, root_upper)) {
        solution.status = MilpStatus::INFEASIBLE;
        return solution;
    }

    auto root = std::make_shared<SearchNode>();
    std::atomic<std::uint64_t> next_node_order{1};
    ConcurrentNodeQueue open;

    IncumbentState incumbent_state;

    LpSolverOptions relaxation_options = options.lp_options;
    relaxation_options.method = LpMethod::SIMPLEX;
    bool has_integer_variables = false;
    for (VariableType type : problem.variable_types) {
        has_integer_variables |= type != VariableType::CONTINUOUS;
    }
    // Computed once and reused for every solve_lp() call this search makes
    // (node relaxations, diving, local improvement, strong-branching
    // probes) -- see MilpSolverOptions::enable_integer_bound_rounding and
    // ::enable_gcd_tightening, which share this one mask (both need exactly
    // "which columns are integer-restricted", so populating it twice under
    // two different flags would be redundant, not safer). Direct Simplex
    // construction elsewhere in this file (the GMI cut path) bypasses
    // solve_lp/presolve entirely already, so this mask has no effect there,
    // by design.
    if (options.enable_integer_bound_rounding || options.enable_gcd_tightening) {
        relaxation_options.integer_columns.assign(static_cast<std::size_t>(problem.n_cols()), 0);
        for (std::int32_t j = 0; j < problem.n_cols(); ++j) {
            relaxation_options.integer_columns[static_cast<std::size_t>(j)] =
                problem.variable_types[static_cast<std::size_t>(j)] != VariableType::CONTINUOUS ? 1 : 0;
        }
    }
    relaxation_options.enable_gcd_tightening = options.enable_gcd_tightening;

    const std::uint32_t hw = std::max(1u, std::thread::hardware_concurrency());
    const std::uint32_t n_workers =
        options.parallel_worker_count == 0 ? std::min(8u, hw) : options.parallel_worker_count;
    // Force SERIAL per-node LP solves whenever more than one thread will
    // ever touch this search (including the mandatory serial root phase,
    // so root and workers behave identically) -- see
    // MilpSolverOptions::parallel_worker_count's own doc comment. GPU
    // pricing under concurrent host threads has not been verified safe
    // and is out of scope for this increment (docs/architecture/MILP.md),
    // so it is disabled outright rather than silently risked.
    if (n_workers > 1) {
        relaxation_options.parallel_mode = ParallelMode::SERIAL;
        if (relaxation_options.backend == PricingBackend::GPU) {
            relaxation_options.backend = PricingBackend::CPU;
        }
    }

    SharedSearchState shared{
        problem, options, relaxation_options, root_lower, root_upper,
        has_integer_variables, open, incumbent_state, next_node_order,
        start, options.time_limit_seconds,
    };

    WorkerContext root_ctx(workspace, static_cast<std::size_t>(problem.n_cols()));

    // ---- mandatory serial root phase -------------------------------------
    // Root's own cuts/RENS/local-improvement are one-shot, root-only
    // heuristics regardless of parallel_worker_count -- processed here,
    // single-threaded, exactly as the pre-parallel code always did, before
    // any worker thread exists. `continue_search` becomes true only when
    // the root actually branched with no other condition (a limit, a
    // fatal error) preempting it first.
    bool continue_search = false;
    for (;;) {
        if (timed_out()) {
            solution.status = MilpStatus::TIME_LIMIT;
            break;
        }
        if (options.node_limit > 0 && root_ctx.nodes_processed >= options.node_limit) {
            solution.status = MilpStatus::NODE_LIMIT;
            break;
        }
        const NodeResult result = process_node(shared, root_ctx, root);
        solution.root_cover_cuts += result.cover_cuts_added;
        solution.cover_cuts += result.cover_cuts_added;
        solution.root_gmi_cuts += result.gmi_cuts_added;
        if (result.outcome == NodeOutcome::Requeue) continue;
        if (result.outcome == NodeOutcome::TimedOutMidHeuristic) {
            solution.status = MilpStatus::TIME_LIMIT;
            break;
        }
        if (result.outcome == NodeOutcome::Fatal) {
            solution.status = result.fatal_status;
            break;
        }
        if (result.outcome == NodeOutcome::Pruned) {
            // Root itself proved infeasible, or its own integral candidate
            // was accepted as the incumbent -- either way, nothing is left
            // to explore.
            break;
        }
        // Branched.
        open.push_pair(std::move(result.left), std::move(result.right));
        continue_search = true;
        break;
    }

    if (continue_search) {
        if (n_workers <= 1) {
            // Continue on THIS thread, reusing root_ctx so pseudocosts
            // and every counter accumulate exactly as the pre-parallel
            // single-threaded code always did -- no
            // std::thread is spawned at all, so parallel_worker_count's
            // default (1) costs nothing beyond the ConcurrentPriorityQueue's
            // own mutex, uncontended with a single caller. WorkerCoordinator
            // with n_workers=1 correctly treats "queue empty" as immediate,
            // successful termination.
            WorkerCoordinator solo_coordinator(1);
            while (true) {
                auto popped = open.pop_or_wait(solo_coordinator);
                if (!popped.has_value()) break;
                if (timed_out()) {
                    solution.status = MilpStatus::TIME_LIMIT;
                    break;
                }
                if (options.node_limit > 0 && root_ctx.nodes_processed >= options.node_limit) {
                    solution.status = MilpStatus::NODE_LIMIT;
                    break;
                }
                const NodeResult result = process_node(shared, root_ctx, *popped);
                if (result.outcome == NodeOutcome::Fatal) {
                    solution.status = result.fatal_status;
                    break;
                }
                if (result.outcome == NodeOutcome::TimedOutMidHeuristic) {
                    solution.status = MilpStatus::TIME_LIMIT;
                    break;
                }
                if (result.outcome == NodeOutcome::Branched) {
                    open.push_pair(std::move(result.left), std::move(result.right));
                }
            }
        } else {
            // Genuine multi-threaded phase: N fresh WorkerContexts (NOT
            // continuing root_ctx's own pseudocost history -- starting
            // blank is a real, explicitly accepted difference
            // from single-threaded search that affects tree shape/timing
            // only, never final-answer correctness; see
            // docs/architecture/MILP.md's parallel-B&B section).
            WorkerCoordinator coordinator(n_workers);
            std::vector<std::unique_ptr<WorkerContext>> contexts;
            contexts.reserve(n_workers);
            for (std::uint32_t i = 0; i < n_workers; ++i) {
                contexts.push_back(std::make_unique<WorkerContext>(
                    workspace, static_cast<std::size_t>(problem.n_cols())));
            }

            std::atomic<bool> worker_fatal{false};
            std::mutex fatal_status_mutex;
            MilpStatus worker_fatal_status = MilpStatus::NUMERICAL_FAILURE;
            // Best-effort GLOBAL node count across every worker: exactly
            // hitting node_limit under concurrency is inherently
            // approximate (several workers may each process one more node
            // past it before all notice) -- a real, stated tradeoff, not a
            // bug (docs/architecture/MILP.md).
            std::atomic<std::uint64_t> global_nodes_processed{0};

            const auto worker_loop = [&](std::uint32_t worker_index) {
                WorkerContext& ctx = *contexts[worker_index];
                while (true) {
                    auto popped = open.pop_or_wait(coordinator);
                    if (!popped.has_value()) return;
                    if (timed_out() ||
                        (options.node_limit > 0 &&
                         global_nodes_processed.load(std::memory_order_relaxed) >=
                             options.node_limit)) {
                        open.request_stop(coordinator);
                        return;
                    }
                    const NodeResult result = process_node(shared, ctx, *popped);
                    global_nodes_processed.fetch_add(1, std::memory_order_relaxed);
                    if (result.outcome == NodeOutcome::Fatal ||
                        result.outcome == NodeOutcome::TimedOutMidHeuristic) {
                        bool expected = false;
                        if (worker_fatal.compare_exchange_strong(expected, true)) {
                            std::lock_guard<std::mutex> lock(fatal_status_mutex);
                            worker_fatal_status =
                                result.outcome == NodeOutcome::TimedOutMidHeuristic
                                    ? MilpStatus::TIME_LIMIT
                                    : result.fatal_status;
                        }
                        open.request_stop(coordinator);
                        return;
                    }
                    if (result.outcome == NodeOutcome::Branched) {
                        open.push_pair(std::move(result.left), std::move(result.right));
                    }
                    // Pruned/Requeue: Requeue structurally never happens
                    // here (root-only cut separation self-gates on
                    // depth == 0, and the root never reaches a worker).
                }
            };

            std::vector<std::thread> workers;
            workers.reserve(n_workers);
            for (std::uint32_t i = 0; i < n_workers; ++i) {
                workers.emplace_back(worker_loop, i);
            }
            for (auto& t : workers) t.join();

            for (auto& ctx : contexts) {
                solution.nodes_processed += ctx->nodes_processed;
                solution.nodes_pruned += ctx->nodes_pruned;
                solution.lp_relaxations += ctx->lp_relaxations;
                solution.strong_branching_probes += ctx->strong_branching_probes;
                solution.warm_started_relaxations += ctx->warm_started_relaxations;
                solution.warm_start_verification_fallbacks += ctx->warm_start_verification_fallbacks;
                solution.diving_heuristic_lp_relaxations += ctx->diving_heuristic_lp_relaxations;
                solution.local_improvement_lp_relaxations += ctx->local_improvement_lp_relaxations;
                solution.rens_heuristic_lp_relaxations += ctx->rens_heuristic_lp_relaxations;
            }

            if (worker_fatal.load()) {
                solution.status = worker_fatal_status;
            } else if (timed_out()) {
                solution.status = MilpStatus::TIME_LIMIT;
            } else if (options.node_limit > 0 &&
                       global_nodes_processed.load(std::memory_order_relaxed) >= options.node_limit) {
                solution.status = MilpStatus::NODE_LIMIT;
            }
        }
    }

    // Merge the root phase's own counters exactly once, regardless of
    // which path above ran (single-threaded continuation reuses root_ctx
    // directly for everything after the root too, so this still correctly
    // captures the WHOLE search's work in that case).
    solution.nodes_processed += root_ctx.nodes_processed;
    solution.nodes_pruned += root_ctx.nodes_pruned;
    solution.lp_relaxations += root_ctx.lp_relaxations;
    solution.strong_branching_probes += root_ctx.strong_branching_probes;
    solution.warm_started_relaxations += root_ctx.warm_started_relaxations;
    solution.warm_start_verification_fallbacks += root_ctx.warm_start_verification_fallbacks;
    solution.diving_heuristic_lp_relaxations += root_ctx.diving_heuristic_lp_relaxations;
    solution.local_improvement_lp_relaxations += root_ctx.local_improvement_lp_relaxations;
    solution.rens_heuristic_lp_relaxations += root_ctx.rens_heuristic_lp_relaxations;

    solution.has_incumbent =
        std::isfinite(incumbent_state.objective.load(std::memory_order_acquire));
    if (solution.has_incumbent) {
        std::lock_guard<std::mutex> lock(incumbent_state.mutex);
        solution.x = incumbent_state.x;
        solution.objective_value = problem.maximize
                                        ? -incumbent_state.objective.load(std::memory_order_relaxed)
                                        : incumbent_state.objective.load(std::memory_order_relaxed);
    }
    const double minimization_incumbent = incumbent_state.objective.load(std::memory_order_acquire);
    const double minimization_bound =
        current_best_bound(open, solution.has_incumbent, minimization_incumbent);
    solution.best_bound = problem.maximize ? -minimization_bound : minimization_bound;
    solution.relative_gap =
        relative_gap(solution.has_incumbent, minimization_incumbent, minimization_bound);

    if (solution.status == MilpStatus::UNBOUNDED ||
        solution.status == MilpStatus::UNBOUNDED_RELAXATION) {
        return solution;
    }
    if (solution.status == MilpStatus::TIME_LIMIT || solution.status == MilpStatus::NODE_LIMIT ||
        solution.status == MilpStatus::NUMERICAL_FAILURE) {
        return solution;
    }
    if (open.size_unsafe() == 0) {
        solution.status = solution.has_incumbent ? MilpStatus::OPTIMAL : MilpStatus::INFEASIBLE;
    } else {
        solution.status = MilpStatus::NUMERICAL_FAILURE;
    }
    return solution;
}

} // namespace sihps
