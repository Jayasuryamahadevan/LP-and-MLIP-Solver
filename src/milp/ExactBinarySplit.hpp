#pragma once

#include "MilpProblem.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace sihps {

// Exact meet-in-the-middle solver for "binary system with unit slacks":
//
//     minimize  sum_i s_i
//     s.t.      A y + s = d,  y in {0,1}^n,  s >= 0,
//               A integral and >= 0, d integral, one unit slack per row.
//
// This is the market-split / multi-knapsack shape (Cornuejols & Dawande,
// "A class of hard small 0-1 programs", 1998). It is called out separately
// because LP-based branch-and-bound is provably the wrong tool for it: the
// relaxation attains slack 0 fractionally on every such instance, so the
// dual bound sits at 0 and never moves, whatever branching or low-rank cut
// family is applied. MEASURED on markshare2 (docs/architecture/MILP.md):
// dual bound exactly 0.00000000 after 8.68M nodes, gap 99.78%, incumbent
// never better than 444 against a true optimum of 1.
//
// The method here is COMPLETE ENUMERATION, not a heuristic: the n columns
// are split into L and R; every subset of L with A y_L <= d is stored in a
// hash table keyed by its exact m-vector; then for every subset of R with
// A y_R <= d we look up the exact complement. Since A >= 0 the partial sums
// are monotone, so "exceeds d" prunes soundly on both halves. Targets are
// enumerated by increasing objective v (t = d - r, r >= 0, sum r = v), so
// the first v that is achievable IS the optimum -- every smaller v has been
// exhausted first. A "no solution" answer is therefore a proof, and the
// returned assignment is re-verified against the original coefficients
// before it is handed back.
//
// Cost is exponential in n/2 and the table is sized against an explicit
// memory budget, so this never runs unless the caller opts in AND the
// structure and size both fit. It is not a general MILP method and makes no
// claim to be one.
struct ExactBinarySplitResult {
    bool applicable = false;  // structure/size gate passed
    std::string reason;       // why the gate refused, when !applicable
    bool solved = false;      // optimum proven
    double objective = 0.0;   // certified optimal objective
    std::vector<double> x;    // original column space, verified feasible
    std::uint64_t subsets_examined = 0;
    std::uint64_t probes = 0;
};

// Returns applicable=false (with a reason) whenever the model is not
// exactly this shape, or would exceed `memory_budget_bytes`. Never throws
// on a merely-unsupported model.
ExactBinarySplitResult try_exact_binary_split(const MilpProblem& problem,
                                              std::uint64_t memory_budget_bytes,
                                              unsigned threads);

} // namespace sihps
