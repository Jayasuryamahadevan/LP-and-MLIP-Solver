#include "MilpSolver.hpp"

#include "../parallel/Parallel.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sihps {
namespace {

constexpr double kInfinityValue = std::numeric_limits<double>::infinity();

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
    double rhs_norm = 0.0;
    for (double value : lp.rhs) rhs_norm = std::max(rhs_norm, std::fabs(value));
    double row_violation = 0.0;
    for (std::int32_t i = 0; i < lp.n_rows(); ++i) {
        const auto ii = static_cast<std::size_t>(i);
        const double lo = lp.rhs[ii] - lp.slack_upper[ii];
        const double hi = lp.rhs[ii] - lp.slack_lower[ii];
        if (std::isfinite(lo)) row_violation = std::max(row_violation, lo - ax[ii]);
        if (std::isfinite(hi)) row_violation = std::max(row_violation, ax[ii] - hi);
    }
    return std::max(0.0, row_violation) / (1.0 + rhs_norm) <= feasibility_tolerance;
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

double current_best_bound(const NodeQueue& queue, bool has_incumbent, double incumbent) {
    if (!queue.empty()) return queue.top()->priority_bound;
    return has_incumbent ? incumbent : kInfinityValue;
}

double relative_gap(bool has_incumbent, double incumbent, double best_bound) {
    if (!has_incumbent || !std::isfinite(best_bound)) return kInfinityValue;
    return std::max(0.0, (incumbent - best_bound) / (1.0 + std::fabs(incumbent)));
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

    // One mutable workspace reuses the original sparse matrix for every
    // node. Copying the matrix per node would make a large B&B tree
    // memory-bound before the LP solver had a chance to work.
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
    std::uint64_t next_node_order = 1;
    NodeQueue open;
    open.push(root);

    double incumbent = kInfinityValue;
    std::vector<double> incumbent_x;
    bool relaxation_unbounded = false;
    bool root_cuts_separated = false;
    bool root_gmi_separated = false;

    // Warm-started dual simplex for node relaxations
    // (docs/architecture/LP.md \S1/\S2). Keyed by SearchNode::order,
    // populated when a node's children are created and consumed-and-erased
    // the moment that child is popped -- NOT a SearchNode field, since
    // SearchNode::parent already keeps the whole ancestor chain alive for
    // the rest of the search, and a Basis stored there would outlive its
    // usefulness. This bounds the map to roughly the current queue width
    // rather than the size of the whole tree.
    std::unordered_map<std::uint64_t, std::shared_ptr<const Simplex::Basis>> pending_basis;
    // Ruiz factors for workspace.A, computed once (lazily, on first use)
    // AFTER root cover cuts have settled its final shape, and reused by
    // every subsequent node's direct Simplex construction. workspace.A
    // never changes again once cuts are separated (only lower_/upper_ do,
    // one variable at a time), so recomputing this per node would spend
    // exactly the cost warm-starting exists to avoid.
    bool node_scale_ready = false;
    ScaleFactors node_scale;

    LpSolverOptions relaxation_options = options.lp_options;
    relaxation_options.method = LpMethod::SIMPLEX;
    bool has_integer_variables = false;
    for (VariableType type : problem.variable_types) {
        has_integer_variables |= type != VariableType::CONTINUOUS;
    }

    std::vector<double> down_pseudocost(static_cast<std::size_t>(problem.n_cols()), 0.0);
    std::vector<double> up_pseudocost(static_cast<std::size_t>(problem.n_cols()), 0.0);
    std::vector<std::uint32_t> down_observations(static_cast<std::size_t>(problem.n_cols()), 0);
    std::vector<std::uint32_t> up_observations(static_cast<std::size_t>(problem.n_cols()), 0);

    auto record_pseudocost = [&](const SearchNode& child, double child_bound,
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
            down_pseudocost[j] += unit_cost;
            ++down_observations[j];
        } else {
            up_pseudocost[j] += unit_cost;
            ++up_observations[j];
        }
    };

    const auto reliable = [&](std::size_t j) {
        return down_observations[j] >= options.reliability_threshold &&
               up_observations[j] >= options.reliability_threshold;
    };
    const auto pseudocost_score = [&](const FractionalCandidate& candidate) {
        const auto j = static_cast<std::size_t>(candidate.variable);
        if (down_observations[j] == 0 || up_observations[j] == 0) return -1.0;
        const double down = (down_pseudocost[j] / down_observations[j]) * candidate.fraction;
        const double up = (up_pseudocost[j] / up_observations[j]) * (1.0 - candidate.fraction);
        return std::min(down, up);
    };

    const auto observe_pseudocost = [&](std::int32_t variable, int direction,
                                        double unit_cost) {
        const auto j = static_cast<std::size_t>(variable);
        if (direction < 0) {
            down_pseudocost[j] += unit_cost;
            ++down_observations[j];
        } else {
            up_pseudocost[j] += unit_cost;
            ++up_observations[j];
        }
    };

    const auto consider_incumbent = [&](const std::vector<double>& candidate,
                                        const std::vector<double>& candidate_lower,
                                        const std::vector<double>& candidate_upper) {
        if (!feasible_point(problem, candidate, candidate_lower, candidate_upper,
                            options.feasibility_tolerance, options.lp_options.parallel_mode) ||
            !integral_point(problem, candidate, options.integrality_tolerance)) {
            return false;
        }
        const double candidate_objective = objective_value(workspace, candidate);
        if (!std::isfinite(incumbent) ||
            candidate_objective < incumbent -
                                     options.objective_tolerance * (1.0 + std::fabs(incumbent))) {
            incumbent = candidate_objective;
            incumbent_x = candidate;
            ++solution.incumbent_updates;
            return true;
        }
        return false;
    };

    bool heuristic_timeout = false;
    const auto attempt_lp_dive = [&](const LpSolution& starting,
                                     const std::vector<double>& starting_lower,
                                     const std::vector<double>& starting_upper) {
        if (!options.use_diving_heuristic || options.diving_max_depth == 0 ||
            options.diving_max_lp_relaxations == 0 || std::isfinite(incumbent)) {
            return;
        }

        LpSolution current = starting;
        std::vector<double> dive_lower = starting_lower;
        std::vector<double> dive_upper = starting_upper;
        std::uint32_t dive_relaxations = 0;
        for (std::uint32_t depth = 0; depth < options.diving_max_depth; ++depth) {
            if (timed_out()) {
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
            // Prefer the side indicated by the objective, then try the other
            // side if the preferred LP is infeasible. This is a deterministic
            // objective-guided dive, not a relaxation bound used for proof.
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
                ++solution.lp_relaxations;
                ++solution.diving_heuristic_lp_relaxations;
                ++dive_relaxations;
                const LpSolution child = solve_lp(workspace, relaxation_options);
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
            options.local_improvement_max_trials == 0 || !std::isfinite(incumbent)) {
            return;
        }

        std::vector<double> current = incumbent_x;
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
                ++solution.lp_relaxations;
                ++solution.local_improvement_lp_relaxations;
                const LpSolution local_solution = solve_lp(local, relaxation_options);
                if (local_solution.status != LpStatus::OPTIMAL ||
                    local_solution.x.size() != static_cast<std::size_t>(problem.n_cols())) {
                    continue;
                }
                const double before = incumbent;
                if (consider_incumbent(local_solution.x, node_lower, node_upper) &&
                    incumbent < before) {
                    current = incumbent_x;
                    improved = true;
                }
                if (timed_out()) return;
            }
            if (!improved) break;
        }
    };

    while (!open.empty()) {
        if (timed_out()) {
            solution.status = MilpStatus::TIME_LIMIT;
            break;
        }
        if (options.node_limit > 0 && solution.nodes_processed >= options.node_limit) {
            solution.status = MilpStatus::NODE_LIMIT;
            break;
        }

        const auto node = open.top();
        open.pop();
        ++solution.nodes_processed;

        std::shared_ptr<const Simplex::Basis> node_parent_basis;
        if (options.warm_start_node_relaxations) {
            auto pending_it = pending_basis.find(node->order);
            if (pending_it != pending_basis.end()) {
                node_parent_basis = pending_it->second;
                pending_basis.erase(pending_it);
            }
        }

        if (std::isfinite(incumbent) &&
            node->priority_bound >= incumbent -
                                         options.objective_tolerance * (1.0 + std::fabs(incumbent))) {
            ++solution.nodes_pruned;
            continue;
        }

        std::vector<double> lower;
        std::vector<double> upper;
        materialize_bounds(*node, root_lower, root_upper, lower, upper);
        if (!bounds_are_valid(lower, upper)) {
            ++solution.nodes_pruned;
            continue;
        }
        workspace.lower = lower;
        workspace.upper = upper;

        ++solution.lp_relaxations;
        LpSolution relaxation;
        std::shared_ptr<const Simplex::Basis> node_basis;
        if (node->depth == 0 || !options.warm_start_node_relaxations) {
            // Root always takes this path: its solve goes through
            // solve_lp's presolve, and a warm basis is only valid for a
            // child that solves over the SAME augmented column space --
            // not guaranteed once presolve's bound-dependent reductions
            // are in the picture. Every node also takes this path when the
            // feature is off, which is exactly today's behavior.
            relaxation = solve_lp(workspace, relaxation_options);
        } else {
            if (!node_scale_ready) {
                node_scale = relaxation_options.use_ruiz_scaling
                                 ? compute_ruiz_scaling(workspace.A)
                                 : ScaleFactors::identity(workspace.n_rows(), workspace.n_cols());
                node_scale_ready = true;
            }
            Simplex simplex(workspace, relaxation_options.backend,
                             relaxation_options.use_ruiz_scaling, relaxation_options.pricing_rule,
                             LpAlgorithm::AUTO, relaxation_options.parallel_mode, &node_scale);
            if (relaxation_options.simplex_time_budget_seconds > 0.0) {
                simplex.set_time_budget(relaxation_options.simplex_time_budget_seconds);
            }
            if (node_parent_basis) simplex.set_warm_start_basis(node_parent_basis.get());

            const LpResult lp = simplex.solve();
            relaxation.status = lp.status;
            relaxation.x = lp.x;
            relaxation.objective_value = lp.objective_value;

            if (lp.used_warm_start) ++solution.warm_started_relaxations;
            if (lp.warm_start_attempted && !lp.used_warm_start) {
                ++solution.warm_start_verification_fallbacks;
            }
            if (lp.status == LpStatus::OPTIMAL) {
                node_basis = std::make_shared<const Simplex::Basis>(simplex.export_basis());
            }
        }
        if (relaxation.status == LpStatus::INFEASIBLE) {
            record_pseudocost(*node, node->priority_bound, true);
            ++solution.nodes_pruned;
            continue;
        }
        if (relaxation.status == LpStatus::UNBOUNDED) {
            relaxation_unbounded = true;
            solution.status = has_integer_variables ? MilpStatus::UNBOUNDED_RELAXATION
                                                     : MilpStatus::UNBOUNDED;
            break;
        }
        if (relaxation.status != LpStatus::OPTIMAL ||
            relaxation.x.size() != static_cast<std::size_t>(problem.n_cols())) {
            solution.status = MilpStatus::NUMERICAL_FAILURE;
            break;
        }

        const double lower_bound = relaxation.objective_value;
        record_pseudocost(*node, lower_bound, false);

        if (node->depth == 0 && !root_cuts_separated && options.enable_root_cover_cuts) {
            root_cuts_separated = true;
            const auto cuts = separate_cover_cuts(
                problem, relaxation.x, options.cut_violation_tolerance,
                options.max_root_cover_cuts);
            if (!cuts.empty()) {
                append_cover_cuts(workspace, cuts);
                solution.root_cover_cuts = cuts.size();
                solution.cover_cuts = cuts.size();
                open.push(node);
                continue;
            }
        }
        if (node->depth == 0 && !root_gmi_separated && options.enable_root_gmi_cuts) {
            root_gmi_separated = true;
            // A dedicated unscaled solve, bypassing solve_lp's presolve:
            // Gomory cuts need the tableau (B^-1), which solve_lp does not
            // expose (its LpSolution carries only status/x/objective), and
            // presolve's column-space reductions would make a tableau
            // built there invalid for THIS workspace's own row/column
            // space. Unscaled so tableau coefficients and nonbasic bounds
            // are directly in original model units, with no scale-factor
            // bookkeeping in the cut derivation itself. Root-only, so the
            // extra LP solve is a bounded, one-time cost, counted honestly
            // below rather than hidden from lp_relaxations.
            Simplex gmi_simplex(workspace, PricingBackend::CPU, /*use_ruiz_scaling=*/false,
                                relaxation_options.pricing_rule, LpAlgorithm::AUTO,
                                relaxation_options.parallel_mode);
            ++solution.lp_relaxations;
            const LpResult gmi_lp = gmi_simplex.solve();
            if (gmi_lp.status == LpStatus::OPTIMAL &&
                gmi_lp.x.size() == static_cast<std::size_t>(problem.n_cols())) {
                const auto cuts =
                    separate_gmi_cuts(problem, workspace, gmi_simplex, gmi_lp.x,
                                      options.cut_violation_tolerance, options.max_root_gmi_cuts);
                if (!cuts.empty()) {
                    append_general_cuts(workspace, cuts);
                    solution.root_gmi_cuts = cuts.size();
                    open.push(node);
                    continue;
                }
            }
        }
        if (std::isfinite(incumbent) &&
            lower_bound >= incumbent -
                               options.objective_tolerance * (1.0 + std::fabs(incumbent))) {
            ++solution.nodes_pruned;
            continue;
        }

        const bool candidate_integral =
            integral_point(problem, relaxation.x, options.integrality_tolerance);
        const std::vector<double> rounded = rounded_point(problem, relaxation.x, lower, upper);
        if (candidate_integral || options.use_rounding_heuristic) {
            const bool accepted = consider_incumbent(rounded, lower, upper);
            if (!accepted && candidate_integral) {
                // The LP claimed an integral point, but the exact integer
                // candidate did not clear the original-model gate. Do not
                // branch on a point that should already be terminal: this is
                // a numerical inconsistency, not proof of infeasibility.
                solution.status = MilpStatus::NUMERICAL_FAILURE;
                break;
            }
        }

        if (candidate_integral) continue;

        // Dive at the root and at only the first few levels when no
        // incumbent exists. Repeating a failed dive at every deep node can
        // spend more LP work on heuristics than on certified search.
        if (node->depth <= 2 && !std::isfinite(incumbent)) {
            attempt_lp_dive(relaxation, lower, upper);
            if (heuristic_timeout) {
                open.push(node);
                solution.status = MilpStatus::TIME_LIMIT;
                break;
            }
        }
        if (node->depth == 0 && std::isfinite(incumbent)) {
            attempt_local_improvement(lower, upper);
            if (timed_out()) {
                open.push(node);
                solution.status = MilpStatus::TIME_LIMIT;
                break;
            }
        }

        for (std::int32_t j = 0; j < problem.n_cols(); ++j) {
            const auto jx = static_cast<std::size_t>(j);
            if (problem.variable_types[jx] != VariableType::CONTINUOUS &&
                !std::isfinite(relaxation.x[jx])) {
                solution.status = MilpStatus::NUMERICAL_FAILURE;
                break;
            }
        }
        if (solution.status == MilpStatus::NUMERICAL_FAILURE) {
            break;
        }

        const std::vector<FractionalCandidate> candidates =
            fractional_candidates(problem, relaxation.x, options.integrality_tolerance);
        if (candidates.empty()) {
            solution.status = MilpStatus::NUMERICAL_FAILURE;
            break;
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
                    if (timed_out()) {
                        open.push(node);
                        solution.status = MilpStatus::TIME_LIMIT;
                        break;
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
                        if (!bounds_are_valid(probe_lower, probe_upper)) {
                            return {true, kInfinityValue};
                        }
                        workspace.lower = probe_lower;
                        workspace.upper = probe_upper;
                        ++solution.lp_relaxations;
                        ++solution.strong_branching_probes;
                        const LpSolution probe = solve_lp(workspace, relaxation_options);
                        workspace.lower = lower;
                        workspace.upper = upper;
                        if (probe.status == LpStatus::INFEASIBLE) {
                            return {true, kInfinityValue};
                        }
                        if (probe.status != LpStatus::OPTIMAL || distance <= 0.0) {
                            return {false, 0.0};
                        }
                        return {true, std::max(0.0, probe.objective_value - lower_bound) /
                                            distance};
                    };

                    const auto down_probe = probe_child(-1, floor_probe, down_distance);
                    const auto up_probe = probe_child(+1, ceil_probe, up_distance);
                    if (down_probe.first) {
                        observe_pseudocost(candidate.variable, -1, down_probe.second);
                    }
                    if (up_probe.first) {
                        observe_pseudocost(candidate.variable, +1, up_probe.second);
                    }
                    ++probes;
                }
                if (solution.status == MilpStatus::TIME_LIMIT) break;
            }

            double best_score = -1.0;
            for (const FractionalCandidate& candidate : candidates) {
                const auto j = static_cast<std::size_t>(candidate.variable);
                if (options.branching_rule == MilpBranchingRule::RELIABILITY &&
                    !reliable(j)) {
                    continue;
                }
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
        if (floor_value >= ceil_value || !std::isfinite(floor_value) ||
            !std::isfinite(ceil_value)) {
            solution.status = MilpStatus::NUMERICAL_FAILURE;
            break;
        }

        auto left = std::make_shared<SearchNode>();
        left->parent = node;
        left->depth = node->depth + 1;
        left->order = next_node_order++;
        left->priority_bound = lower_bound;
        left->change.variable = branch_variable;
        left->change.upper = floor_value;
        left->branch_variable = branch_variable;
        left->branch_direction = -1;
        left->branch_distance = relaxation.x[jj] - floor_value;

        auto right = std::make_shared<SearchNode>();
        right->parent = node;
        right->depth = node->depth + 1;
        right->order = next_node_order++;
        right->priority_bound = lower_bound;
        right->change.variable = branch_variable;
        right->change.lower = ceil_value;
        right->branch_variable = branch_variable;
        right->branch_direction = +1;
        right->branch_distance = ceil_value - relaxation.x[jj];

        if (node_basis) {
            // Same shared_ptr, refcounted rather than duplicated -- both
            // children start from the same parent basis, one bound-change
            // delta apart from it in opposite directions.
            pending_basis.emplace(left->order, node_basis);
            pending_basis.emplace(right->order, node_basis);
        }

        open.push(std::move(left));
        open.push(std::move(right));
    }

    solution.has_incumbent = std::isfinite(incumbent);
    if (solution.has_incumbent) {
        solution.x = std::move(incumbent_x);
        solution.objective_value = problem.maximize ? -incumbent : incumbent;
    }
    const double minimization_bound = current_best_bound(open, solution.has_incumbent, incumbent);
    solution.best_bound = problem.maximize ? -minimization_bound : minimization_bound;
    solution.relative_gap = relative_gap(solution.has_incumbent, incumbent, minimization_bound);

    if (relaxation_unbounded) return solution;
    if (solution.status == MilpStatus::TIME_LIMIT || solution.status == MilpStatus::NODE_LIMIT ||
        solution.status == MilpStatus::NUMERICAL_FAILURE) {
        return solution;
    }
    if (open.empty()) {
        solution.status = solution.has_incumbent ? MilpStatus::OPTIMAL : MilpStatus::INFEASIBLE;
    } else {
        solution.status = MilpStatus::NUMERICAL_FAILURE;
    }
    return solution;
}

} // namespace sihps
