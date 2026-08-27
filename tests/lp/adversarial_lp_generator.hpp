#pragma once

// Deterministic (seeded) synthetic LP generator for Phase 1 correctness
// hardening (prompt.md validation LEVEL 3 / docs/ROADMAP_STATUS.md's own
// KNOWN LIMITATION: "no false INFEASIBLE"/"no false UNBOUNDED" claims are
// currently only asserted against the Netlib set, which is small,
// historical, and not adversarial by construction).
//
// Every generator here builds an instance whose TRUE status is known BY
// CONSTRUCTION, not by trusting any solver (this project's own or an
// external one) -- the standard technique for a correctness fuzzer: derive
// the instance FROM a known answer, then check the solver reproduces it,
// rather than solve first and hope. Specifically:
//
// - FEASIBLE_BOUNDED / ILL_CONDITIONED / DEGENERATE: every variable has a
//   finite box bound [lower, upper], and every row is constructed to be
//   satisfied by a known feasible point `known_feasible_point`. A finite
//   box is compact; row constraints (<=, >=, =) are closed conditions, so
//   the feasible region is closed and bounded, and it is nonempty because
//   the known point satisfies it. A closed, bounded, nonempty polytope
//   guarantees the LP is both FEASIBLE and BOUNDED (Weierstrass extreme
//   value theorem applied to a linear objective on a compact set) --
//   ESTABLISHED, elementary convex-optimization fact, not specific to this
//   solver -- so the correct status is OPTIMAL unconditionally, regardless
//   of the objective's sign or the specific coefficients drawn.
// - INFEASIBLE_BOUNDS / INFEASIBLE_ROWS: built to be infeasible by an
//   explicit, checkable contradiction (a variable's lower bound exceeds
//   its upper bound; or two rows jointly bound a linear combination both
//   above and below a crossing threshold), independent of any bound.
// - UNBOUNDED: one variable has an infinite upper bound, a zero column in
//   the constraint matrix (no row limits it), and a negative objective
//   coefficient (minimizing) -- driving it to infinity always strictly
//   improves the objective with nothing to stop it, so the LP is
//   unbounded regardless of what the rest of the instance looks like.

#include "lp/LpProblem.hpp"
#include "sparse/Triplet.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace sihps_test {

enum class AdversarialKind {
    FEASIBLE_BOUNDED,
    ILL_CONDITIONED,
    DEGENERATE,
    INFEASIBLE_BOUNDS,
    INFEASIBLE_ROWS,
    UNBOUNDED
};

struct GeneratedLp {
    sihps::LpProblem problem;
    AdversarialKind kind;
    // Populated for FEASIBLE_BOUNDED / ILL_CONDITIONED / DEGENERATE only:
    // a point known to satisfy every bound and row by construction, used
    // by the test to independently re-verify the solver's reported
    // optimum without trusting the solver's own residual fields.
    std::vector<double> known_feasible_point;
};

namespace detail {

inline char random_row_type(std::mt19937& rng) {
    static const char kTypes[3] = {'L', 'G', 'E'};
    std::uniform_int_distribution<int> pick(0, 2);
    return kTypes[pick(rng)];
}

// Builds n_cols box-bounded variables with a known feasible point, then
// n_rows rows over them (density controls nonzeros/row), each constructed
// so the known point satisfies it exactly (E rows) or with random slack
// (L/G rows) -- see the file header for why this guarantees FEASIBLE +
// BOUNDED regardless of what follows. coeff_dist draws each nonzero's
// magnitude; sign is drawn independently and uniformly.
inline GeneratedLp build_compact_box_lp(std::mt19937& rng, int n_rows, int n_cols, double density,
                                         std::uniform_real_distribution<double>& coeff_magnitude,
                                         AdversarialKind kind) {
    GeneratedLp out;
    out.kind = kind;
    sihps::LpProblem& p = out.problem;

    std::uniform_real_distribution<double> bound_width(1.0, 100.0);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::uniform_real_distribution<double> obj_coeff(-10.0, 10.0);
    std::uniform_real_distribution<double> slack_extra(0.0, 5.0);

    p.lower.resize(static_cast<std::size_t>(n_cols));
    p.upper.resize(static_cast<std::size_t>(n_cols));
    p.obj.resize(static_cast<std::size_t>(n_cols));
    out.known_feasible_point.resize(static_cast<std::size_t>(n_cols));
    for (int j = 0; j < n_cols; ++j) {
        const double width = bound_width(rng);
        p.lower[static_cast<std::size_t>(j)] = 0.0;
        p.upper[static_cast<std::size_t>(j)] = width;
        out.known_feasible_point[static_cast<std::size_t>(j)] = unit(rng) * width;
        p.obj[static_cast<std::size_t>(j)] = obj_coeff(rng);
    }

    std::vector<sihps::Triplet> triplets;
    p.row_types.resize(static_cast<std::size_t>(n_rows));
    p.rhs.resize(static_cast<std::size_t>(n_rows));
    for (int i = 0; i < n_rows; ++i) {
        std::vector<int> cols;
        for (int j = 0; j < n_cols; ++j) {
            if (unit(rng) < density) cols.push_back(j);
        }
        if (cols.empty()) cols.push_back(std::uniform_int_distribution<int>(0, n_cols - 1)(rng));

        double activity = 0.0;
        for (int j : cols) {
            double coeff = coeff_magnitude(rng);
            if (unit(rng) < 0.5) coeff = -coeff;
            triplets.push_back({i, j, coeff});
            activity += coeff * out.known_feasible_point[static_cast<std::size_t>(j)];
        }

        const char type = random_row_type(rng);
        p.row_types[static_cast<std::size_t>(i)] = type;
        if (type == 'L') {
            p.rhs[static_cast<std::size_t>(i)] = activity + slack_extra(rng);
        } else if (type == 'G') {
            p.rhs[static_cast<std::size_t>(i)] = activity - slack_extra(rng);
        } else {
            p.rhs[static_cast<std::size_t>(i)] = activity; // E: exact
        }
    }

    p.A = sihps::CSRMatrix::from_triplets(n_rows, n_cols, triplets);
    sihps::apply_default_row_bounds(p);
    return out;
}

} // namespace detail

// Moderate, well-scaled coefficients (0.1 to 10): a plain correctness
// baseline, not adversarial on its own -- the control group.
inline GeneratedLp generate_feasible_bounded(std::mt19937& rng, int n_rows, int n_cols,
                                              double density = 0.3) {
    std::uniform_real_distribution<double> mag(0.1, 10.0);
    return detail::build_compact_box_lp(rng, n_rows, n_cols, density, mag,
                                         AdversarialKind::FEASIBLE_BOUNDED);
}

// Coefficient magnitudes drawn log-uniformly across 1e-6..1e6 -- a
// 12-order-of-magnitude spread, well beyond what Ruiz scaling
// (docs/architecture/NUMERICS.md \S2) is expected to fully equalize,
// stressing the scaling/factorization path specifically.
// build_compact_box_lp's magnitude parameter is a plain
// uniform_real_distribution (linear range), which cannot express a
// log-uniform draw, so this generator is constructed directly rather
// than by reusing that helper.
inline GeneratedLp generate_ill_conditioned(std::mt19937& rng, int n_rows, int n_cols,
                                             double density = 0.3) {
    GeneratedLp out;
    out.kind = AdversarialKind::ILL_CONDITIONED;
    sihps::LpProblem& p = out.problem;

    std::uniform_real_distribution<double> bound_width(1.0, 100.0);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::uniform_real_distribution<double> obj_exponent(-3.0, 3.0);
    std::uniform_real_distribution<double> slack_extra(0.0, 5.0);
    std::uniform_real_distribution<double> exponent(-6.0, 6.0);

    p.lower.resize(static_cast<std::size_t>(n_cols));
    p.upper.resize(static_cast<std::size_t>(n_cols));
    p.obj.resize(static_cast<std::size_t>(n_cols));
    out.known_feasible_point.resize(static_cast<std::size_t>(n_cols));
    for (int j = 0; j < n_cols; ++j) {
        const double width = bound_width(rng);
        p.lower[static_cast<std::size_t>(j)] = 0.0;
        p.upper[static_cast<std::size_t>(j)] = width;
        out.known_feasible_point[static_cast<std::size_t>(j)] = unit(rng) * width;
        double c = std::pow(10.0, obj_exponent(rng));
        if (unit(rng) < 0.5) c = -c;
        p.obj[static_cast<std::size_t>(j)] = c;
    }

    std::vector<sihps::Triplet> triplets;
    p.row_types.resize(static_cast<std::size_t>(n_rows));
    p.rhs.resize(static_cast<std::size_t>(n_rows));
    for (int i = 0; i < n_rows; ++i) {
        std::vector<int> cols;
        for (int j = 0; j < n_cols; ++j) {
            if (unit(rng) < density) cols.push_back(j);
        }
        if (cols.empty()) cols.push_back(std::uniform_int_distribution<int>(0, n_cols - 1)(rng));

        double activity = 0.0;
        for (int j : cols) {
            double coeff = std::pow(10.0, exponent(rng));
            if (unit(rng) < 0.5) coeff = -coeff;
            triplets.push_back({i, j, coeff});
            activity += coeff * out.known_feasible_point[static_cast<std::size_t>(j)];
        }

        const char type = detail::random_row_type(rng);
        p.row_types[static_cast<std::size_t>(i)] = type;
        if (type == 'L') {
            p.rhs[static_cast<std::size_t>(i)] = activity + slack_extra(rng);
        } else if (type == 'G') {
            p.rhs[static_cast<std::size_t>(i)] = activity - slack_extra(rng);
        } else {
            p.rhs[static_cast<std::size_t>(i)] = activity;
        }
    }
    p.A = sihps::CSRMatrix::from_triplets(n_rows, n_cols, triplets);
    sihps::apply_default_row_bounds(p);
    return out;
}

// Same compact-box construction as generate_feasible_bounded, but roughly
// a third of the rows are exact duplicates (verbatim copies, including
// RHS) of an earlier row -- redundant constraints, a standard source of
// primal degeneracy (multiple optimal bases represent the same feasible
// region), without changing the feasible region itself, so FEASIBLE +
// BOUNDED still holds unconditionally.
inline GeneratedLp generate_degenerate(std::mt19937& rng, int n_rows, int n_cols,
                                        double density = 0.3) {
    std::uniform_real_distribution<double> mag(0.1, 10.0);
    GeneratedLp out =
        detail::build_compact_box_lp(rng, n_rows, n_cols, density, mag, AdversarialKind::DEGENERATE);
    if (n_rows < 2) return out;

    sihps::LpProblem& p = out.problem;
    // Grouped by row, and mutated in place as duplications happen, so a
    // duplication whose OWN source row was itself overwritten by an
    // earlier duplication in this same loop still copies that row's
    // CURRENT content. Reading coefficients from p.A directly (a fixed
    // snapshot never rebuilt mid-loop) while rhs/row_types were updated
    // incrementally was tried first and is exactly the bug this
    // structure avoids: it let a later duplication pair rhs/type from an
    // already-overwritten source row with STALE, pre-overwrite
    // coefficients from p.A, producing a row whose rhs no longer matched
    // any point satisfying its own coefficients -- an instance that was
    // not actually guaranteed feasible, caught by
    // tests/lp/test_adversarial.cpp reporting an unexplained INFEASIBLE
    // that traced back to exactly this inconsistency, not a solver bug.
    std::vector<std::vector<std::pair<int, double>>> row_terms(static_cast<std::size_t>(n_rows));
    for (std::int32_t i = 0; i < p.A.rows(); ++i) {
        for (std::int32_t k = p.A.row_ptr()[i]; k < p.A.row_ptr()[i + 1]; ++k) {
            row_terms[static_cast<std::size_t>(i)].push_back(
                {p.A.col_idx()[static_cast<std::size_t>(k)], p.A.values()[static_cast<std::size_t>(k)]});
        }
    }

    std::uniform_int_distribution<int> row_pick(0, n_rows - 1);
    const int duplicate_count = std::max(1, n_rows / 3);
    for (int d = 0; d < duplicate_count; ++d) {
        const int src = row_pick(rng);
        const int dst = row_pick(rng);
        if (src == dst) continue;
        row_terms[static_cast<std::size_t>(dst)] = row_terms[static_cast<std::size_t>(src)];
        p.row_types[static_cast<std::size_t>(dst)] = p.row_types[static_cast<std::size_t>(src)];
        p.rhs[static_cast<std::size_t>(dst)] = p.rhs[static_cast<std::size_t>(src)];
    }

    std::vector<sihps::Triplet> triplets;
    for (int i = 0; i < n_rows; ++i) {
        for (const auto& [col, val] : row_terms[static_cast<std::size_t>(i)]) {
            triplets.push_back({i, col, val});
        }
    }
    // apply_default_row_bounds recomputes every row's slack bounds from
    // row_types alone, so a duplicated row's slack bounds match its
    // source automatically.
    p.A = sihps::CSRMatrix::from_triplets(n_rows, n_cols, triplets);
    sihps::apply_default_row_bounds(p);
    return out;
}

// A variable's lower bound is set strictly above its upper bound --
// infeasible by definition, independent of any row.
inline GeneratedLp generate_infeasible_bounds(std::mt19937& rng, int n_cols) {
    GeneratedLp out;
    out.kind = AdversarialKind::INFEASIBLE_BOUNDS;
    sihps::LpProblem& p = out.problem;
    std::uniform_int_distribution<int> pick(0, n_cols - 1);

    p.lower.assign(static_cast<std::size_t>(n_cols), 0.0);
    p.upper.assign(static_cast<std::size_t>(n_cols), 10.0);
    p.obj.assign(static_cast<std::size_t>(n_cols), 1.0);
    const int contradicted = pick(rng);
    p.lower[static_cast<std::size_t>(contradicted)] = 5.0;
    p.upper[static_cast<std::size_t>(contradicted)] = 2.0; // < lower: contradiction

    p.A = sihps::CSRMatrix::from_triplets(0, n_cols, {});
    p.row_types.clear();
    p.rhs.clear();
    sihps::apply_default_row_bounds(p);
    return out;
}

// Two rows jointly contradict regardless of bounds: one forces
// sum(x) >= BIG, the other forces sum(x) <= SMALL, with BIG > SMALL and
// every x_j >= 0, so no point can satisfy both.
inline GeneratedLp generate_infeasible_rows(std::mt19937& rng, int n_cols) {
    GeneratedLp out;
    out.kind = AdversarialKind::INFEASIBLE_ROWS;
    sihps::LpProblem& p = out.problem;
    std::uniform_real_distribution<double> unit(0.0, 1.0);

    p.lower.assign(static_cast<std::size_t>(n_cols), 0.0);
    p.upper.assign(static_cast<std::size_t>(n_cols), 1000.0);
    p.obj.assign(static_cast<std::size_t>(n_cols), 1.0);

    std::vector<sihps::Triplet> triplets;
    for (int j = 0; j < n_cols; ++j) triplets.push_back({0, j, 1.0});
    for (int j = 0; j < n_cols; ++j) triplets.push_back({1, j, 1.0});
    p.A = sihps::CSRMatrix::from_triplets(2, n_cols, triplets);
    p.row_types = {'G', 'L'};
    const double small = 1.0 + unit(rng);
    const double big = small + 100.0 + unit(rng) * 100.0;
    p.rhs = {big, small}; // sum(x) >= big  AND  sum(x) <= small, big > small
    sihps::apply_default_row_bounds(p);
    return out;
}

// The last variable has an infinite upper bound, a zero column (no row
// touches it), and a negative objective coefficient: driving it to
// infinity always strictly decreases the (minimized) objective, so the
// LP is unbounded regardless of the rest of the instance. The other
// variables get an ordinary compact-box sub-instance so the rest of the
// problem is unambiguously feasible on its own.
inline GeneratedLp generate_unbounded(std::mt19937& rng, int n_rows, int n_cols_other) {
    std::uniform_real_distribution<double> mag(0.1, 10.0);
    GeneratedLp base = detail::build_compact_box_lp(rng, n_rows, n_cols_other, 0.3, mag,
                                                      AdversarialKind::UNBOUNDED);
    sihps::LpProblem& p = base.problem;
    const int n_cols = n_cols_other + 1;

    std::vector<sihps::Triplet> triplets;
    for (std::int32_t i = 0; i < p.A.rows(); ++i) {
        for (std::int32_t k = p.A.row_ptr()[i]; k < p.A.row_ptr()[i + 1]; ++k) {
            triplets.push_back({i, p.A.col_idx()[static_cast<std::size_t>(k)],
                                 p.A.values()[static_cast<std::size_t>(k)]});
        }
    }
    // No triplet touches column n_cols_other (the new, unbounded one) --
    // a genuinely zero column.
    p.A = sihps::CSRMatrix::from_triplets(n_rows, n_cols, triplets);

    p.lower.push_back(0.0);
    p.upper.push_back(sihps::kInfinity);
    p.obj.push_back(-1.0); // minimize -x_unbounded: unbounded below as x_unbounded -> +inf
    base.known_feasible_point.clear(); // not applicable: this instance is UNBOUNDED, not OPTIMAL
    sihps::apply_default_row_bounds(p);
    return base;
}

} // namespace sihps_test
