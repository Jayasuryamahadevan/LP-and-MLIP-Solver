# LP Engine Architecture

**Status:** PHASE 2 architecture. Per prompt.md §2.7, the LP engine is designed independently of the MILP engine that will call it repeatedly, and no single algorithm is assumed universally optimal — the choice is adaptive, based on model characteristics, and stated as such rather than picked by default preference.

---

## 1. Algorithms Evaluated

| Algorithm | Status for v1 | Rationale |
|---|---|---|
| **Dual simplex** | **v1 default for warm starts** (implemented; see §2.1 — cold starts use primal, on measured evidence) | Warm-start-friendly from a parent B&B node's basis (the dominant workload shape in this project — thousands of related re-solves); matches the documented default reasoning in both HiGHS and CPLEX (SOTA.md §1.1, §1.2) |
| **Primal simplex** | v1 fallback, and v1 cold-start default (implemented; §2.1) | Used when dual simplex cannot establish a feasible starting basis, or as the recovery path in `NUMERICS.md` §5's fallback chain |
| **Barrier / interior-point** | **Deferred** | A from-scratch IPM (predictor-corrector, normal-equations or KKT solve) is a large independent engineering effort; SOTA.md §1.3b's literature review found GPU-accelerated IPM factorization speedups to be modest in practice (fill-in erodes sparsity advantage) — not obviously worth building before the simplex core is validated. Revisit once v1 core is benchmarked (Level 6+) |
| **First-order primal-dual (PDLP-style)** | **Deferred (SOTA.md KS-7)** | Not warm-start-friendly for repeated B&B relaxations (iterates are not basic solutions); would introduce a second numerical code path requiring its own validation before the first is even benchmarked |

**Why dual simplex over primal as the default, specifically:** in this project's dominant workload (a B&B node relaxation is the parent's relaxation plus one tightened bound), the parent's optimal basis is typically dual-feasible for the child (only primal feasibility is disturbed by the bound change) — exactly the condition dual simplex exploits for warm starts. This is the same reasoning documented for CPLEX's default (SOTA.md §1.2) and is treated here as LITERATURE EVIDENCE supporting the choice, not a novel claim.

## 2. Adaptive Selection Strategy

```cpp
enum class LPMethod { DUAL_SIMPLEX, PRIMAL_SIMPLEX };

LPMethod select_method(const LPWarmStartContext& ctx) {
    if (ctx.has_parent_basis && ctx.parent_basis_dual_feasible)
        return LPMethod::DUAL_SIMPLEX;
    if (!ctx.has_parent_basis)
        return LPMethod::PRIMAL_SIMPLEX; // cold start: build initial feasible basis directly
    return LPMethod::DUAL_SIMPLEX; // default; NUMERICS.md §5 governs runtime fallback
}
```

### 2.1 Implementation status (Phase 3) — `src/lp/Simplex.cpp`, `src/lp/LpSolver.cpp`

Both algorithms are implemented and both are reachable: `LpAlgorithm::{PRIMAL, DUAL, AUTO}` on `Simplex`, surfaced as `LpSolverOptions::algorithm` on the `solve_lp` pipeline.

**`AUTO` resolves to `PRIMAL` on a cold start, and gains the parent-basis branch when one is supplied.** `Simplex::set_warm_start_basis`/`export_basis` (`Simplex.hpp`) are now implemented: a caller exports a solved instance's basis and seats it into a fresh instance over a bound-tightened child, and `solve()` runs the dual-simplex repair from there — ahead of both cold paths — before falling back if verification fails. `MilpSolverOptions::warm_start_node_relaxations` wires this into the B&B node loop (`MILP.md` §1.4). See §8 below for what this measured.

**EXPERIMENTAL RESULT — cold-start primal vs dual** (`benchmarks/bench_lp_algorithm.cpp`, Netlib feasible set to 2600 rows, full `solve_lp` pipeline including presolve, Devex pricing, Ruiz scaling, 79 instances where the dual path was actually entered and both algorithms reached `OPTIMAL`):

| | total iterations | total wall-clock |
|---|---|---|
| Primal two-phase | 164,580 | 32.99 s |
| Dual simplex | 446,641 (**2.71×**) | 106.28 s (**3.22×**) |

The aggregate is not the whole finding, and the spread is the part that matters:

| instance | primal | dual | |
|---|---|---|---|
| `d6cube` | 31,033 it / 5.18 s | 1,070 it / 0.19 s | dual **29× fewer iterations** |
| `degen3` | 6,845 it / 1.10 s | 12,359 it / 2.02 s | dual 1.8× worse |
| `pilotnov` | 3,110 it / 0.29 s | 86,420 it / 6.78 s | dual runs, fails verification, primal fallback answers |
| `pilot87` | 14,071 it / 13.1 s | 110,278 it / 74.8 s | dual 5.7× slower **and** disagrees on the objective beyond 1e-6 |

So cold-start dual is not uniformly worse — it is *unpredictable*, and on a cold start that unpredictability buys nothing, because there is no warm basis whose retained dual feasibility is the entire reason to prefer it. This measurement supports the cold-start branch of the table above; it says **nothing** about the warm-start branch, which cannot be measured until warm starts exist.

Two classification details in that benchmark, both of which change the numbers if got wrong:

- **16 instances admit no dual-feasible start at all** and were excluded. `setup_dual_feasible_start` requires every cost to point toward a *finite* bound; where it does not, `DUAL` runs the primal path and comparing it against itself measures nothing.
- **Whether a model admits that start is a property of the model the simplex actually receives, so presolve changes it.** `pilot87` has no dual-feasible start unreduced but does after presolve fixes columns and tightens bounds. An earlier version of this benchmark drove `Simplex` directly, classified `pilot87` as "no-start", and reported dual as a 0.78× *win* overall — the opposite conclusion, from measuring a configuration that does not ship.

### 2.2 Dual simplex correctness work (Phase 3)

Three defects were found and fixed by wiring the dual path in and measuring it. All three shared a shape worth naming: each produced a *plausible* answer rather than an obvious crash.

1. **Single-pass dual ratio test → divergence and false infeasibility.** The test took the minimum ratio with only a 1e-12 tie-break toward larger pivots, so any pivot above `kPivotTol` (1e-9) was acceptable. Since the primal step is $t = \delta / \alpha_{\text{enter}}$, one such pivot converts a small bound violation into an enormous one, which becomes the next iteration's leaving row. **MEASURED on `grow15`:** worst basic infeasibility reached 1.4e+12 over 5,219 iterations, dual feasibility was lost entirely (max dual infeasibility 7.0 against a 1e-9 tolerance), and the solver then reported `INFEASIBLE` **on a feasible model** — one of 10 such false infeasibility claims across the 89-instance set. Fixed by the Harris two-pass form (Koberstein 2005 §3.3, the dual counterpart of §3's primal test): pass 1 relaxes each ratio by a dual-feasibility tolerance, pass 2 takes the largest pivot in that set. False infeasibility claims went 10 → 0. Pinned by `dual_simplex_ratio_test_rejects_tiny_pivots`.
2. **Termination measured in scaled units, verification in original units.** The dual stopped when every basic variable was within tolerance of its bounds *in Ruiz-scaled space*, while `finalize_result` verifies in original units — a 1e-9 scaled violation is $C_j$ times that once unscaled. Fixed by applying `unscale_factor` in the leaving-row test, so the termination criterion and the verification gate measure the same thing.
3. **Termination decided on drifted values.** `value_` is carried forward by incremental pivot updates between refactorizations. Both terminal conclusions — `OPTIMAL` *and* `INFEASIBLE` — were being decided on those drifted numbers. **MEASURED on `grow15`:** the drifted values passed the feasibility test while values re-derived from the same basis were 1.2e-05 infeasible in original units. Both exits now refactorize, re-derive, and re-scan before concluding; an infeasibility claim decided on drift would be a wrong answer, not merely a slow one.

**Open item, not fixed:** under explicit `DUAL`, `pilot87` reports an objective differing from the primal path's by more than 1e-6 relative while passing its own verification gate. The shipping path is unaffected (`AUTO` is primal), so this is recorded rather than worked around.

This is intentionally a small, explicit decision table rather than a learned or heuristic-scored selector (cf. SOTA.md §1.3's rejection of learned branching/selection for v1 on auditability grounds) — every selection is traceable to a stated structural reason. **RESEARCH HYPOTHESIS, not yet validated:** that this simple rule captures most of the achievable benefit versus a more elaborate model-characteristic classifier; revisit if benchmarking (Level 6) shows a large fraction of solves taking the "wrong" branch.

## 3. Degeneracy Handling (mandatory, not optional)

Per SOTA.md §1.4.2 and the project's own ASSUMPTION that refinery scheduling/blending models are frequently degenerate (interchangeable units/periods), the v1 simplex core includes anti-degeneracy machinery from the start rather than as a later optimization:

- **Ratio test:** Harris two-pass ratio test (SOTA.md §1.4.2, ESTABLISHED METHOD) — first pass computes a relaxed step-length bound tolerating a small, bounded constraint violation; second pass selects, among rows within that bound, the largest-magnitude pivot. This directly counters the numerically fragile "exact tightest ratio" tie-breaking that a naive ratio test would use.
- **Pricing:** Devex pricing (SOTA.md §1.4.2, ESTABLISHED METHOD) as the v1 default — a cheaper approximation to exact steepest-edge that still substantially reduces degenerate-tie iteration counts relative to naive Dantzig pricing, at lower per-iteration bookkeeping cost than full steepest-edge. **IMPLEMENTATION DECISION:** start with Devex; steepest-edge (Goldfarb & Reid 1977 / Forrest & Goldfarb 1992) is a candidate upgrade if benchmarking shows Devex's approximation quality is insufficient on refinery-structured degenerate instances — not built preemptively.
- **Anti-cycling fallback:** Bland's rule (smallest-index tie-breaking, SOTA.md §1.4.2) as the guaranteed-termination fallback when the Harris/Devex combination still stalls beyond a configured iteration count — Bland's rule is slow but is the only one of the surveyed methods with a finite-termination proof, making it the correct last resort rather than the default (which would sacrifice speed everywhere to guard against a failure mode Harris+Devex already handles in the common case).

## 3.1 Initial basis (crash) — implemented

**ESTABLISHED METHOD:** Bixby, "Implementing the Simplex Method: The Initial Basis," *ORSA Journal on Computing* 4(4), 1992.

The naive start seats an artificial in every row, so phase 1 must drive out one artificial per row — at least $m$ pivots before phase 2 can begin, and in practice far more. Netlib's own index records Bixby measuring 465,810 phase-1 iterations on `dfl001` ($m = 6071$) from exactly this cause.

An inequality row's slack already has room to absorb its own residual (`L` row: $[0,+\infty)$; `G` row: $(-\infty,0]$). Wherever the required residual falls inside the slack's bounds, that row starts **feasible** with the slack basic and needs no artificial. Only equalities and violated inequalities get one. The resulting basis is still one unit column per row, so $B$ stays diagonal and the initial factorization stays trivial.

Two conditions are enforced, both discovered by test failures rather than anticipated:

- **The slack must have room to move, not merely cover the residual.** An equality row's slack is fixed at $[0,0]$; crashing that into the basis seats a variable that can never change, so every ratio test through it forces a zero step and the basis is degenerate from iteration one. This made `grow7` fail verification (`NUMERICAL_FAILURE`) until the freedom test was added.
- **An artificial displaced by a crashed slack is frozen at $[0,0]$ for phase 1.** Under the all-artificial start every artificial began *basic*; under a crash basis many begin *nonbasic* and therefore remain eligible to **enter**, reintroducing infeasibility into a row that started feasible. Freezing is sound because the ratio test keeps the basic slack within its own bounds, so the row stays satisfied throughout phase 1.

## 4. Basis Management and Factorization

- **Representation:** LU factorization of the basis matrix, maintained via sparsity-preserving updates (Bartels & Golub 1969 / Forrest & Tomlin 1972 style — SOTA.md §1.1, ESTABLISHED METHOD) rather than full re-factorization at every pivot.
- **Periodic re-factorization:** triggered on a fixed iteration count *and* on the condition-number-monitoring signal from `NUMERICS.md` §5 — whichever comes first. This bounds both the fill-in growth from repeated updates and the numerical drift that motivates `NUMERICS.md`'s continuous condition monitoring.
- **Pivoting for the factorization itself:** Markowitz threshold pivoting (SOTA.md §1.4.1, ESTABLISHED METHOD) balancing fill-in against numerical stability.

### 4.1 Implementation status (Phase 3) — `src/lp/BasisFactorization.{hpp,cpp}`

Implemented and integrated into `Simplex`. Three deviations from the specification above are deliberate and are recorded here rather than left as silent gaps:

| Specified | Implemented | Why |
|---|---|---|
| Bartels–Golub / Forrest–Tomlin update | **Product form of the inverse** (Dantzig & Orchard-Hays 1954), eta file discarded at each refactorization | PFI is markedly easier to make numerically correct, and its known weakness — eta-file growth — is already bounded by the periodic-refactorization policy this section mandates. Forrest–Tomlin is the correct upgrade **if** measurement shows eta application, not factorization, dominates. That is an experiment to run, not an assumption to act on. |
| Markowitz threshold pivoting | **Gilbert–Peierls left-looking LU** (SIAM J. Sci. Stat. Comput. 9(5), 1988) with threshold partial pivoting (factor 0.01) and a row-count sparsity tie-break; columns pre-ordered by ascending nonzero count | Gilbert–Peierls costs time proportional to the arithmetic actually performed. The ascending-nnz column order makes the many unit columns (slacks, artificials) pivot immediately, which triangularizes most of a simplex basis before any general elimination — a cheap stand-in for the explicit triangularization phase production codes run first. |
| Refactorize on iteration count | Refactorize on **accumulated eta count** | PFI solve cost and numerical drift both track the eta file. Bound-flip iterations advance the iteration counter but push no eta, so an iteration-based trigger refactorizes at the wrong times. |

**Basis repair** (`NUMERICS.md` §5) is implemented here rather than deferred: a dependent basis column is reported by the factorization (not thrown), the unmatched row's artificial is substituted, and the basis is refactorized once. A second failure is reported as `NUMERICAL_FAILURE` rather than retried.

**EXPERIMENTAL RESULT — what this replaced and what it bought.** The prior implementation stored $B^{-1}$ explicitly as a dense $m \times m$ array: $O(m^2)$ memory and per-pivot work, $O(m^3)$ refactorization. Measured effect of the replacement, same machine, same instances, Netlib feasible set at row cap 1300, **77/77 passing in both cases** (so this is a pure speed comparison at equal correctness):

| Instance | rows | dense inverse | sparse LU | speedup |
|---|---|---|---|---|
| `pilot.ja` | 940 | 12.344 s | 0.869 s | 14.2× |
| `25fv47` | 821 | 4.742 s | 0.380 s | 12.5× |
| `d6cube` | 415 | 46.936 s | 4.433 s | 10.6× |
| `pilot.we` | 722 | 4.222 s | 0.501 s | 8.4× |
| `maros` | 846 | 0.679 s | 0.118 s | 5.8× |
| `pilotnov` | 975 | 1.884 s | 0.380 s | 5.0× |

The `d6cube` row also closes an open regression previously recorded in `NUMERICS.md` §2 (Ruiz scaling had pushed it 5.7 s → 46.9 s); the sparse factorization removes it, which indicates the regression was fill/inverse-update cost rather than anything to do with scaling.

The asymptotic point matters more than any single row above: at $m = 6072$ (`dfl001`) the dense inverse alone is 295 MB with ~$2.2\times10^{11}$ flops per refactorization, and at $m = 105127$ (`ken-18`) it is 88 TB. Those instances were not slow — they were unreachable in principle.

**Reachability, measured on the five instances that had been excluded by the 1300-row cap:**

| Instance | rows | cols | dense inverse | sparse LU |
|---|---|---|---|---|
| `bnl2` | 2324 | 3489 | not solved in 900 s | **1.37 s**, verified optimal |
| `greenbea` | 2392 | 5405 | not solved in 900 s | **1.97 s**, verified optimal |
| `d2q06c` | 2171 | 5167 | not solved in 900 s | **4.66 s**, verified optimal |
| `pilot87` | 2030 | 4883 | not solved in 900 s | **14.76 s**, verified optimal |
| `dfl001` | 6071 | 12230 | not solved in 900 s | `ITERATION_LIMIT` at 765 s (~549k iterations, ~1.4 ms/iter) |

Baseline: a single 900 s run over all five completed **0 of 5** (output was still buffered when the timeout killed it — nothing had finished). After: **4 of 5** verified optimal, three of them in under 5 s.

**`dfl001` is an open failure and is recorded as one.** It is not a surprise: Netlib's own index file documents it as pathological — Bixby reports 465,810 phase-I iterations under reduced-cost pricing, 94,337 with CPLEX defaults, and 25,803 only after switching to steepest-edge pricing *and* a different scaling, calling it "a nasty problem." Our run exhausted its iteration cap rather than stalling numerically, and per-iteration throughput was healthy, so the deficit is iteration *count*, not iteration cost. The two documented levers — presolve (`SYSTEM.md` §2.3, designed but not implemented) and pricing strength beyond Devex's steepest-edge approximation — are exactly the ones Bixby's numbers implicate. Neither is yet built; until one is, `dfl001` stays a known failure rather than a tuned-around one.

Two further observations, recorded rather than resolved: `d2q06c` (1.99e-07) and `pilot87` (3.26e-07) pass but sit an order of magnitude closer to the 1e-6 acceptance threshold than the rest of the set, which is consistent with `pilot87`'s documented bad scaling (Lustig, quoted in the Netlib index) and warrants attention when accuracy targets are tightened toward `NUMERICS.md` §3's 1e-8 goal.

## 5. Interface

```cpp
struct LPWarmStartContext {
    std::optional<Basis> parent_basis;
    bool has_parent_basis;
    bool parent_basis_dual_feasible;
};

struct LPResult {
    SolveStatus status;          // NUMERICS.md §6
    std::vector<double> x, y_dual, s_reduced_cost;
    Basis basis;                 // handed to children as their warm start
    int iterations;
    Residuals residuals;         // NUMERICS.md §3, in original-model units
};

class LPEngine {
public:
    LPResult solve(const SparseMatrixCSR& A, const SparseMatrixCSR& A_T,
                    const Bounds& bounds, const Objective& c,
                    const LPWarmStartContext& warm_start);
private:
    // owns: current basis, LU factors, Devex reference weights,
    // scratch vectors from the node-scratch arena (MEMORY.md §3.1 tier 2)
};
```

`LPEngine` does not own the matrix — it receives a view into the device/host-resident `SparseMatrixCSR`/`CSC` pair owned by `SolveContext` (`SYSTEM.md` §3), since the same matrix (or a node-local restriction of it) is shared across every node in the B&B tree. This avoids the zero-allocation-during-solve violation that would result from copying the matrix per node.

## 6. What v1 Explicitly Does Not Include

QP support (prompt.md's stated long-term target includes QP) is **not** part of the v1 LP engine. A QP extension requires either an active-set method built on this same simplex machinery or a barrier method handling the Hessian term — both are deferred alongside barrier/IPM (§1) until the LP core is validated. This is stated explicitly rather than left as a silent gap, per prompt.md's instruction to distinguish what is built from what is claimed.

---

## 7. `LpMethod::HYBRID` — structure-aware lead, verified fallback

**MEASURED.** Two solvers whose failures are disjoint, a cheap feature that
predicts which will win, and a fallback that makes a wrong prediction cost one
budget instead of a wrong answer.

```
HYBRID:
    rows(presolved) >= 3000  ->  first-order leads, simplex is the fallback
    otherwise                ->  simplex leads, first-order is the fallback

    the LEAD runs under a wall-clock budget
    its answer is checked against the ORIGINAL-SPACE gate before commitment
    if it does not converge, or converges but fails that gate, the other runs
```

### The predictor is row count, and it was measured

Sorted by rows across the 21 Kennington/QAP models, the first-order path wins
every instance from 4,350 rows upward, and the simplex wins most below 3,000.

**Nonzero count is the wrong feature.** `osa-07` has 143,694 nonzeros and the
simplex wins 2.2×; `ken-11` has 49,058 and the first-order path wins 18×. Rows
is right because it is the quantity the two costs actually scale with: a simplex
iteration solves against an $m \times m$ basis and its iteration count grows with
$m$, whereas a PDHG iteration costs the same 57–111 µs almost regardless of size.

Threshold chosen against an oracle that picks the better method per instance:

| lead policy | total over the 21 models |
|---|---|
| oracle (perfect choice) | 56.0 s |
| always simplex first | 419.9 s |
| **rows ≥ 3000 → first-order leads** | **67.8 s** |
| rows ≥ 4000 | 117.9 s |

4,000 is worse because `qap12` (3,192 rows) is solved *only* by the first-order
path and would fall on the wrong side of it.

### Measured effect

| set | metric | simplex-first | structure-aware |
|---|---|---|---|
| Netlib (93) | solved | 93 / 93 | **93 / 93** |
| | total | 115.90 s | **72.59 s** |
| | p95 | 6.71 s | **4.03 s** |
| | worst objective error | 5.779e-07 | **5.779e-07** |
| Kennington + QAP (21) | solved | — | **21 / 21** |
| | total, simplex alone | 388.0 s | — |
| | total, HYBRID | — | **43.9 s** |

Per instance on Netlib: `dfl001` 34.77 → **4.01 s**, `maros-r7` 5.30 → **1.04 s**,
`fit2p` 8.05 → **2.14 s**. `stocfor3` regresses, 15.03 → **24.74 s** — it clears
the row threshold but is one of the few large models the simplex handles well.
That is the cost of a one-feature predictor and it is bounded by the budget.

`KNOWN LIMITATION`: the cross-method benchmark reports `hybrid / oracle = 0.78×`,
which is **not** a claim that the hybrid beats a perfect chooser. The oracle is
built from standalone solves at `eps = 1e-8` while HYBRID runs its first-order
path at `1e-7`, so it is allowed a cheaper stopping test. Both clear the same
gate. Read that ratio as an upper bound on the quality of the method *choice*.

### Convergence is not verification — a defect this design exposed

The first version of the structure-aware hybrid fell back only when the leading
method **failed to converge**. That is the wrong condition, and an A/B with the
adaptive step size disabled produced the counterexample:

```
dfl001  NUM_FAILURE  objerr 4.16e-07  4.945s  [first-order]
```

The first-order path led, converged on its own KKT test, and was then rejected
by the original-space gate — with the simplex never tried, on a model the
simplex path had previously handled. PDLP's KKT test is a statement about the
**scaled, presolved** problem; the gate is a statement about the problem the
**caller** posed. They are different questions, and a fallback keyed only on the
first is not a fallback.

The gate is now factored out (`candidate_primal_residual`) and applied to the
leading method's answer *while the other method is still available*.

### Why sequential, and not concurrent

An earlier design raced the two solvers on separate threads, reasoning that the
simplex is CPU-bound and PDLP GPU-bound so they would not contend.

**They contend badly.** PDLP's host thread drives scaling, a transpose build, a
power iteration and a spinning stream synchronize, while the simplex's pricing
loops already use every core. `woodw` went 0.174 s → 5.6 s (PDLP then won with an
answer that failed verification); `greenbeb` 1.3 s → 62 s; validation 92/93 →
91/93. An earlier version was worse still — it let PDLP cancel the simplex on
*finishing* rather than *succeeding*, so a non-converged PDLP killed solves about
to succeed: 86/93.

Two rules generalize past this solver:

- **A loser must never be able to stop a winner.** Cancellation may only be
  triggered by a result the canceller could actually return.
- **Parallelism between algorithms is only free when they use disjoint
  resources.** "CPU-bound" and "GPU-bound" describe where the *work* happens, not
  where the *thread waits*.

### Per-domain CPU parallelism policy

The deterministic inner loops are exposed through `LpSolverOptions::parallel_mode`:

- `AUTO` (the default) uses OpenMP only for sufficiently large sparse passes,
  using the measured `kParallelNnzThreshold` gate.
- `SERIAL` disables CPU inner-loop teams for this solve. A caller that is
  solving independent domains or subproblems concurrently should use this
  mode to avoid nested OpenMP oversubscription.
- `PARALLEL` explicitly enables those deterministic loops, including for a
  small domain when the caller has measured that the extra team overhead is
  acceptable.

Only loops with one writer per output element and a fixed summation order use
this policy. It cannot change pivot choices, objective values, feasibility
checks, or reported statuses. Parallelism between independent solver calls is
not enabled implicitly because the surrounding MILP/domain scheduler owns the
resource budget and must decide how many domains to run at once.

---

## 8. Warm-started dual simplex for MILP nodes — implemented, measured, **not** adopted by default

**IMPLEMENTED.** `Simplex::Basis`/`set_warm_start_basis`/`export_basis` (this
file's long-stated missing piece, and `MILP.md`'s stated prerequisite) exist
and are unit-tested (`tests/lp/test_simplex.cpp`'s `warm_start_*` cases,
`tests/milp/test_milp.cpp`'s `milp_warm_start_*` cases): a non-root B&B node
constructs a `Simplex` directly over the shared node workspace (bypassing
`solve_lp`'s presolve — see the reasoning below), seats its parent's exported
basis, and runs the dual-simplex repair. The result passes the exact same
original-space verification gate (`NUMERICS.md` §6) as every other path;
`MILP.md` §4.1's invariant that every node bound comes from a certified
simplex solve is unchanged. Gated behind
`MilpSolverOptions::warm_start_node_relaxations`, **default `false`**.

**Why presolve is bypassed at node level, not made warm-start-aware.**
Presolve's reductions are bound-dependent — a child's tighter bound can fix a
column or drop a row the parent's presolve did not — so a warm basis is only
valid if parent and child solve over the *same* augmented column space.
Proving a specific presolve reduction is basis-preserving under one bound
tightening is a real but separate validation project; skipping presolve for
non-root nodes sidesteps the question by construction. The cost of that
choice is measured directly below.

**MEASURED, `benchmarks/bench_miplib.cpp` against `data/miplib2017_small`, 60 s
per instance, single process, nothing else running**
(`reports/runs/2026-08-25/miplib-warmstart-{off,on}.txt`):

| instance | nodes (off) | nodes (on) | certified (off) | certified (on) |
|---|---|---|---|---|
| `gen-ip002` | 298,448 | 792,647 (2.7×) | — (`TIME_LIMIT`) | — (`TIME_LIMIT`) |
| `gen-ip054` | 251,141 | 1,033,448 (4.1×) | — | — |
| `markshare2` | 669,274 | 2,390,406 (3.6×) | — | — |
| `neos859080` | 331 | 191,082 (577×) | **yes** — `INFEASIBLE`, 0.795 s | **no** — `TIME_LIMIT` |
| `pk1` | 126,318 | 468,345 (3.7×) | — | — |

Certified results across the set: **1/5 → 0/5.** Node throughput improved
2.7–577× everywhere it was measured, and the outcome got *worse* on the one
metric the roadmap's KPI gate actually cares about. `neos859080` is the sharp
case: cold, it proves infeasibility in 331 nodes because presolve-level
reductions (not deep B&B search) do most of the work; warm-started, each node
is faster but has to rediscover that structure by search alone, and 191,082
nodes in 60 s is not enough to finish what 331 presolved nodes did.

**KNOWN LIMITATION / RESEARCH HYPOTHESIS, not yet tested:** the working
hypothesis is that node-level presolve — not the per-node LP solve — is
carrying most of the pruning power on this instance set, and warm-starting
without it trades a cost that mattered less (LP re-solve time) for one that
matters more (lost structural reductions). A presolve-aware warm start
(verifying which reductions survive a specific bound tightening, or a
cheaper node-local propagation pass that does not require bypassing
`solve_lp` entirely) is the natural next increment if this path is revisited
— not a larger MILP heuristic/cut effort, which this measurement does not
implicate.

**Per the roadmap's own rule** ("no optimization is accepted unless it
improves a declared benchmark KPI without reducing correctness or
solvability"): this does not clear that bar, so the default stays `false`.
The code, tests, and this measurement are kept rather than reverted, because
a negative result recorded with its cause is worth more to the next attempt
than no result at all.

---

## 9. Hyper-sparse BTRAN — implemented, measured, a real win at scale

**IMPLEMENTED.** `BasisFactorization::btran` (`src/lp/BasisFactorization.cpp`)
now uses Gilbert & Peierls' 1988 DFS-reachability technique for its two
triangular solve phases (U^T then L^T), instead of the unconditional O(m)
sweep every prior version used regardless of how sparse the right-hand side
actually was. This is the roadmap's own long-standing "Hyper-sparse
FTRAN/BTRAN with active-pattern detection" item — scoped to **BTRAN only**
for v1 (see the design note below); FTRAN's `ftran_column` remains a named,
deferred v2 candidate.

The factorization already contained the exact DFS machinery this needed
(`sparse_lsolve`, built for the *factorization's own* sparse column solves)
— it was simply never wired into a post-factorization solve. Reusing it
directly turned out not to be possible: `sparse_lsolve` operates in
matrix-row space via `pinv_` indirection, which only holds during
construction; post-factorization, `Li_`/`Ui_` are already remapped into
pivot-step space. `btran` instead calls a new, simpler `reach()` — the same
DFS-with-explicit-stack shape, but working directly over row-major (CSR)
transposes of L's and U's off-diagonal structure (`Lt_p_/Lt_i_`,
`Ut_p_/Ut_i_`), built once per `factorize()` call since neither L nor U
changes again before the next refactorization. The eta phase is
deliberately **not** sparsified: eta count is bounded by the
refactorization cadence, not by `m_`, so it was never the O(m) cost this
targets.

### A real bug, caught by the existing differential test suite before it shipped

The first version had a mark-value collision: `sparse_lsolve` already uses
mark values `0..m-1` internally during `factorize()` (one per pivot step),
and the new `reach()`'s own mark counter started at 0 too, incrementing
per call. Its very first invocation could therefore reuse a mark value
`sparse_lsolve` had already written into the shared `mark_` array during
factorization — making a genuine DFS seed look "already visited" and
silently dropping it from the discovered pattern. `basis_factorization_
requires_pivoting` (`tests/lp/test_basis_factorization.cpp`, a 3x3 matrix
whose natural first pivot is zero) failed immediately: `check_btran`
verifies `B^T y = c` directly against the dense reference matrix, and the
corrupted pattern produced a wrong `y`. Fixed by starting the DFS mark
counter at `m_` (`solve_mark_ = m_;`, reset in `factorize()`) rather than
0, guaranteeing no future `reach()` call can ever collide with a mark
`sparse_lsolve` already used. Three new tests pin this specifically
(`basis_factorization_btran_matches_dense_on_unit_vector_rhs`,
`..._after_updates`, `..._when_pivot_order_permutes_rows` — the last being
this exact regression case, kept small and separate from the random trials
so a future break here fails immediately rather than inside a tolerance).

### MEASURED — large LP: a clean win; small MILP nodes: two more lessons

**Large LP (`stocfor3`, 16,675 rows), 3 clean single-process trials each,
median wall-clock:** 15.379 s (dense) → 12.852 s (hyper-sparse), a real
**16.4% improvement**. Per-stage: `pivot row: BTRAN` (`compute_binv_row`'s
RHS is always a unit vector — pattern size 1, the theoretically best case)
improved consistently, **~24%** across every trial. `duals (BTRAN)`
(`compute_duals`'s RHS is `c_B`) showed a small, noisy, roughly neutral
effect.

Getting a *reliable* read here took two false starts, both instructive:

1. **A single before/after pair showed `duals (BTRAN)` 67% SLOWER**, not
   faster. Investigation (not another retraction — the effect was real,
   just not what it looked like) found the actual cause: once phase 2 has
   many nonzero-cost structural variables basic, `c_B` is often not sparse
   at all, and the DFS then pays its own per-node bookkeeping on top of
   visiting nearly every node anyway — strictly more work than the dense
   sweep it replaced. Fixed with a density fallback
   (`kHyperSparseDensityThreshold = 0.3`, an engineering default from this
   one measurement, not a tuned optimum): when the seed pattern already
   exceeds 30% of `m_`, `btran` skips `reach()` entirely and processes
   every column in the same order the original dense sweep always used,
   by writing the identity (or reversed-identity, for the L^T phase)
   permutation into `pattern_` — reusing the same restricted-sweep loop
   rather than duplicating it.
2. **Even after that fix, one profiling run looked like an *overall*
   regression** (17.874 s, with stages the BTRAN change cannot touch —
   pricing, `A^T` assembly — also reading high). Three repeat trials of
   the *identical* binary showed 12.3–13.2 s: this was measurement noise
   (this repo's own documented risk — three previously retracted
   conclusions came from exactly this), not a code effect. The clean
   multi-trial comparison above is the one to trust.

**Small MILP node relaxations** (`bench_miplib`, the 5-instance MIPLIB set,
m in the 7–90 row range, warm start off): node throughput in the 60 s
budget dropped 4–7% at first. Root cause, confirmed by rereading the code
rather than re-guessing: `btran`'s new DFS path allocated fresh
`std::vector` locals (`seed`, `ut_pattern`) on *every call* — and `btran`
runs every simplex iteration, across hundreds of thousands of B&B nodes for
instances like `markshare2`. This is exactly what `BasisFactorization`'s
own existing scratch-member convention (`pattern_`, `dfs_node_`, etc., all
sized once in `factorize()`, per `prompt.md` §3.1's no-allocation-during-
solve rule) exists to prevent — the new code just didn't follow it. Fixed
by giving both arrays dedicated `mutable` scratch members
(`seed_work_`/`ut_pattern_work_`), reserved once in `factorize()` and
repopulated via `clear()`/`assign()` rather than reallocated. This
recovered most, not quite all, of the throughput gap — but the residual
(2–5%) is the same order of magnitude as this instance set's own natural
run-to-run variance under a time-limited budget (`gen-ip002`'s node count
alone ranged 287,498–298,448 across three *unrelated* runs of the
unmodified dense code earlier the same session), so it is recorded as
**within noise**, not as a confirmed further regression.

**Net assessment:** a genuine, reproducible win on the scale this project's
target workload (MRPL, large refinery models) actually cares about, with
two real defects found and fixed along the way (the mark-collision
correctness bug, and the allocation-driven small-instance slowdown) rather
than papered over. No KPI regression on the existing benchmark suite —
128/128 unit tests pass (3 new, added for this change), and every
Netlib/MIPLIB verdict is unchanged from before this change.

### Hyper-sparse FTRAN — attempted, mathematically correct, reverted: a real solvability regression on a degenerate instance

`IMPLEMENTED` (then reverted), `MEASURED`. The deferred v2 candidate named
above was attempted, following exactly the same pattern as BTRAN's own
hyper-sparse solve: `ftran()`'s L-phase and U-phase each gated by
reachability (Gilbert & Peierls 1988) with the same density fallback,
reusing `reach()` directly against `Lp_`/`Li_` and `Up_`/`Ui_` (their own
native column-major storage already is the right adjacency for a
FORWARD/BACKWARD solve in each factor's own natural direction — unlike
BTRAN's opposite-direction solves, no `build_transpose`-equivalent pass
was needed; a small `diagonal_skip` parameter on the existing `reach()`
let it walk `Lp_`/`Up_` directly instead of a dedicated transpose graph).

**Correctness: verified exhaustively, no defect found.** `check_ftran`
(this project's own dense-reference verifier — `Bx = rhs` checked
directly against the dense matrix, independent of any assumption about
DFS ordering) already covered identity, permutation, triangular,
pivoting-required, and 25 random-sparse-matrix cases; three more tests
were added specifically for genuinely sparse RHS (unit vectors, sparse
RHS after several PFI updates, and the exact pivot-permuting 3x3 case
that caught BTRAN's own mark-collision bug) to exercise the `reach()`
path rather than only the density fallback. **All 145 tests passed,
including every new one, with zero failures at any point in this work.**
This is stated explicitly because what killed this change was NOT a
logic bug of the kind unit testing catches.

**What actually happened: a real solvability regression, root-caused
precisely, not guessed at.** A full `validate_netlib` sweep after this
change showed `pilot87` (2,030 rows, one of the harder/more degenerate
Netlib instances — its own reference is sourced `summary`, not the
cleaner `cplex`/`minos` columns) intermittently failing: of several
repeated runs, some reported `OPTIMAL` with the objectively correct
answer, and some reported `ITER_LIMIT` — a genuine stall, not a
near-miss. This is not a tolerance-boundary flicker; `ITER_LIMIT` means
the search did not converge in budget at all. Compared directly against
`docs/measurements/netlib-hybrid-20000rows-repeats3.jsonl`, recorded
**before** this change: `pilot87` was 100% bit-identical across 3
repeats then (`repeats_deterministic: true`, 14,071 iterations every
time) — this instance was not already fragile; the regression is
attributable to this change.

Root cause, isolated by direct experiment rather than assumed:
`OMP_NUM_THREADS=1` against the **exact same hyper-sparse FTRAN binary**
made `pilot87` 100% stable again (2/2 identical runs, matching the
correct objective `301.71065562` exactly) — proving the mechanism is not
a defect in the hyper-sparse solve's arithmetic, but an *interaction*:
this project's existing OpenMP-parallel reductions (SpMV, pricing) are
already not guaranteed to sum in a fixed order across runs — an already-
known, already-documented characteristic (`docs/measurements/README.md`'s
own account of concurrent-benchmark contamination is a different symptom
of the same underlying fact: floating-point summation order is not fixed
under this project's parallel execution). Hyper-sparse FTRAN changes
*which* order entries are processed in (DFS/topological order instead of
always-ascending column order), which changes the floating-point rounding
of the computed pivot direction. On a well-conditioned instance this is
invisible. On `pilot87` specifically — large and degenerate enough that
its pivot sequence is already close to a ratio-test tie in multiple
places — the altered rounding was, in some but not all runs, enough to
flip a close tie and send the search down a path that stalls, where the
dense FTRAN's fixed summation order apparently did not (at least not with
comparable frequency in the runs measured).

**Disposition: reverted, not shipped behind a flag.** Unlike GMI cuts or
MILP node warm-starting (both shipped off-by-default, because their
failure mode was "doesn't help enough," never "sometimes doesn't
converge"), this failure mode — intermittent non-convergence on a real
Netlib instance, under this project's actual default multi-threaded
configuration — is exactly the kind of regression `docs/ROADMAP_STATUS.md`
Phase 1 exists to catch, and no default-off flag makes it acceptable to
leave implemented and reachable. `src/lp/BasisFactorization.{hpp,cpp}`
were reverted to the pre-change dense `ftran()` (verified: `pilot87`
stable again post-revert, matching the pre-change baseline). The three
new sparse-RHS FTRAN tests were **kept** (`tests/lp/test_basis_
factorization.cpp`) — they test the dense path correctly and add real
regression coverage regardless of this outcome.

**What this means for a future attempt, recorded as a `RESEARCH
HYPOTHESIS` rather than closing the door:** the failure mode here is
about *summation-order sensitivity interacting with a pre-existing
parallel-floating-point noise source*, not about the hyper-sparse
technique being wrong. A future attempt would need to either (a) make the
hyper-sparse solve's summation order match the dense sweep's own order
more closely when both visit the same entries (losing some of the
technique's benefit but preserving bitwise-closer behavior), or (b)
address the underlying parallel non-determinism directly rather than
adding a second computation path that turns out to be more sensitive to
it, or (c) accept the risk only behind an explicit opt-in flag with this
exact failure mode documented at the call site — not attempted in this
pass, and not recommended as the next default action given items 1-3 of
`docs/research/HIGHS_GAP_ANALYSIS.md`'s own ranked candidate list (§6) do
not depend on this line of work at all.

### Greedy minimum-degree column ordering — attempted, two real bugs fixed, reverted for the same class of solvability regression as hyper-sparse FTRAN

`IMPLEMENTED` (then reverted), `MEASURED`. `docs/ROADMAP_STATUS.md`
priority item 5 and `docs/research/HIGHS_GAP_ANALYSIS.md` §2.2/§6/§7 both
independently ranked a fill-reducing factorization ordering as the
highest-confidence remaining lever on the HiGHS LP-speed gap. `factorize()`
(`src/lp/BasisFactorization.cpp`) previously ordered columns once by
ascending original nonzero count only ("the poor man's version of the
explicit triangularization phase production codes run first," per its own
comment) with row pivots tie-broken by a *static* row count — no true
per-step fill prediction. Implemented `compute_min_degree_order()`: greedy
minimum-degree (Tinney & Walker 1967; surveyed in George & Liu 1981) on
the column-intersection graph, using the classical elimination-graph
clique rule (Markowitz 1957) at each step, computed once per `factorize()`
call and substituted for the ascending-nnz order.

**Deliberately scoped down from full AMD/COLAMD** (Amestoy, Davis & Duff
1996), stated explicitly per this project's own convention: AMD's own
contribution over the classical algorithm is (a) an *approximate* degree
update that avoids exact recomputation, and (b) a quotient-graph /
element-absorption representation that keeps each step cheap on large,
dense graphs. Both are substantial additional machinery with real bug
surface; implementing the *exact* classical algorithm first, protected by
a hard computational budget standing in for AMD's own approximation, was
judged the lower-risk first increment given `docs/ROADMAP_STATUS.md`'s
preference for measured, incremental changes.

**Two real bugs found and fixed before this got anywhere near a
benchmark** (exactly the discipline this project has now exercised
repeatedly — investigate and fix immediately, don't note and continue):

1. **A stale lazy-deletion key produced an invalid permutation.** The
   first version tried to keep a `std::set<{degree, column}>` priority
   structure in sync by erasing the exact old `{degree, column}` key
   whenever a column's adjacency changed, then reinserting the new one.
   This is wrong whenever a column's adjacency is touched more than once
   in the same elimination step (which the clique-formation step routinely
   does): by the time the "erase old key" call runs, the column's degree
   has already changed from what was actually stored, so the erase
   silently fails to find a match (`std::set::erase` on a missing key is a
   no-op, not an error) and leaves an orphaned stale entry. Popping that
   entry later produced a **duplicate column in the output order**, and
   since the order must be a permutation of every column, a corresponding
   column went **missing** — `factorize()` then ran out of columns for
   some row and reported a spurious singular basis on matrices that are
   provably nonsingular by construction. Caught immediately by this file's
   own dense-reference tests
   (`basis_factorization_matches_dense_on_random_sparse_matrices`,
   `..._ftran_matches_dense_on_unit_vector_rhs`, both asserting
   `result.singular.empty()` on a strongly-diagonal, genuinely nonsingular
   matrix) — 14 of 145 tests failed, none of them subtle. Fixed with the
   standard lazy-deletion pattern (Cormen et al.): a `current_degree[]`
   array as the single source of truth, entries validated on pop rather
   than precisely erased on every mutation.

2. **The fill-edge budget didn't bound the actual cost driver.** The
   safety cap (`kMinDegreeEdgeBudgetMultiplier`, mirroring
   `kHyperSparseDensityThreshold`'s own fallback discipline) originally
   charged only when a candidate pair became a genuinely *new* edge. But
   the per-step clique-formation loop performs an `O(log n)` set lookup
   for **every** candidate pair among a node's active neighbors, whether
   or not that pair turns out to be a new edge — so a step whose neighbor
   set had already saturated into a near-clique (mostly pre-existing
   edges, no new-edge charges) paid full `O(degree²)` cost with **no
   budget protection at all**. MEASURED on `pilot87` (2,030 rows,
   `profile_simplex`, single process, one clean run each, nothing else
   running): baseline is 28.819 s wall, with the refactorization stage at
   11.517 s (40.4% of total) over 136 refactorizations; the
   charge-on-insert-only version pushed the same stage to 266.075 s across
   131 refactorizations — 92.9% of a 290.854 s total, a real **~9x
   wall-clock regression**, root-caused by isolating the ordering as the
   only changed variable (identical binary, ordering toggled off via a
   one-line edit, same instance, same machine). Fixed by charging the
   budget for every candidate pair *examined*, not merely every pair
   *inserted* — this directly bounds the true `O(degree²)` cost driver.
   After the fix and a corresponding budget increase (32x → 64x original
   nonzero count, chosen empirically to restore full fill reduction on
   the synthetic test below), `pilot87` returned to 36.039 s wall / 18.693
   s refactorization (136 refactorizations, unchanged iteration count) —
   still a real ~25% wall-clock cost relative to the 28.819 s baseline on
   this specific instance, with no offsetting iteration-count reduction
   visible (see the aggregate-sweep finding below for why this was never
   resolved further).

**Fill reduction itself worked as designed.** A new test
(`basis_factorization_min_degree_order_reduces_grid_laplacian_fill`)
constructs a 12x12 five-point-stencil grid Laplacian (m=144, a textbook
naive-ordering fill stress case, George & Liu 1981) — MEASURED via a
standalone harness linking `BasisFactorization.cpp` directly with one
ordering flag flipped, otherwise identical code: the prior ascending-nnz
ordering produces 5,444 factor nonzeros; minimum-degree produces 2,741 —
a real **~49.6% fill reduction**, confirming the ordering mechanism itself
is correct and effective on a case chosen independently of this project's
own benchmark instances.

**What actually killed it: the same class of solvability regression as
hyper-sparse FTRAN, confirmed by the same isolation methodology.** A full
`validate_netlib` sweep (`presolve hybrid`, single process) with the fixed,
64x-budget ordering active showed `pilot87` failing — `ITER_LIMIT`, not a
near-miss, 13,088 iterations against the same instance that solves cleanly
otherwise (89/90, "FAILED: pilot87"). Root-caused by direct experiment,
not assumed: the **identical sweep with the ordering forced off** (its
result computed and then discarded, so `q_` is bit-identical to the old
ascending-nnz order) **also failed pilot87 the same way** (89/90, same
instance). A third run at pristine `HEAD` — `compute_min_degree_order()`
not even called — passed cleanly (90/90), and repeated twice for
stability (241,995 iterations, bit-identical both times). The conclusion
this triangulates to: merely **introducing the new ordering computation as
additional CPU-bound work in the refactorization path is enough to
perturb this project's existing OpenMP-parallel floating-point summation
order** on an already-marginal instance and tip it past its iteration
limit — the exact mechanism documented in the hyper-sparse FTRAN section
above, triggered here not by a changed pivot sequence but by changed
thread-scheduling/timing from the extra computation itself, even when
that computation's output goes unused.

**Disposition: reverted, not shipped behind a flag** — same reasoning as
hyper-sparse FTRAN: intermittent non-convergence on a real Netlib
instance under this project's actual default multi-threaded configuration
is a solvability regression, not a "doesn't help enough" outcome, and no
default-off flag makes a *reachable* code path carrying this failure mode
acceptable. `src/lp/BasisFactorization.cpp` was reverted in full (`git
stash` against the two changed files, verified diff was purely additive
to this attempt, then dropped) to the pre-change `HEAD`, re-verified
90/90 stable across two repeated sweeps post-revert. The grid-Laplacian
fill-reduction test was reverted along with it, since it tests a
mechanism no longer wired into `factorize()`.

**For a future attempt, recorded as a `RESEARCH HYPOTHESIS`:** this is now
the *second* independent case (after hyper-sparse FTRAN) where a
mathematically-correct, exhaustively-unit-tested change to the
factorization/solve path destabilized `pilot87` specifically via
parallel-floating-point non-determinism, not via a logic defect. That
instance's own sensitivity — not either individual technique — is
increasingly the load-bearing fact. A future attempt at either technique,
or at any change to `factorize()`'s hot path, should budget for this
directly: run `pilot87` (and ideally a small stable of similarly
degenerate/large instances) repeated 3-5x under the real default
multi-threaded configuration as a *required* gate before considering any
factorization-path change measured, not as an optional follow-up check.
A single-threaded (`OMP_NUM_THREADS=1`) isolation run, which resolved the
FTRAN case cleanly, was not re-attempted here given the additional
finding that even a *discarded* computation reproduces the failure — the
next attempt at either line of work should start there rather than
re-discover it. Separately: this session's own `HIGHS_GAP_ANALYSIS.md`
ranks MIP-specific presolve and a RENS-class primal heuristic (§6, items
2-3) above any further factorization-ordering work, and neither depends
on `factorize()`'s hot path at all — the more promising near-term
direction given this outcome.
