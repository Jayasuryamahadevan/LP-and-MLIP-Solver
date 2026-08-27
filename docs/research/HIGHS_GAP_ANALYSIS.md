# Closing the HiGHS Gap — Research Analysis

**Status:** research document, extends `docs/research/SOTA.md`; does not itself
implement or claim any benchmark improvement. Every claim below is labeled
per prompt.md's taxonomy (`ESTABLISHED METHOD` / `ENGINEERING TECHNIQUE` /
`PROPOSED MODIFICATION` / `RESEARCH HYPOTHESIS`) and, separately, by evidence
class: **VERIFIED** (the actual paper/thesis/source was read, not just a
search-engine summary of it), **SEARCH-SUMMARY** (found via a web search
result summary — the underlying document was not independently opened and
read in full, so treat specifics as probably-right, not certain), or
**MEASURED** (this project's own benchmark). Where a search-summary claim
could not be corroborated by opening the primary source, that is stated
explicitly rather than silently upgraded to VERIFIED.

**Trigger:** `docs/measurements/highs_comparison.md` — a real, single-machine
benchmark of this repo's solver against HiGHS 1.15.1 built from source and
run only through its own CLI — measured HiGHS **7.25x faster in aggregate
wall-clock** on 89 agreeing Netlib LP instances (78/89 head-to-head wins),
and a mixed 2-2-1 result on a 5-instance MIPLIB MILP subset.

---

## 0. Executive summary

**The gap is explained mostly by algorithmic maturity (ordering, updates,
presolve depth, pricing/ratio-test refinement accumulated over ~35 years of
continuous development), not by a single missing low-level trick (SIMD,
cache blocking).** This is the conclusion the research actually supports,
not an assumption going in:

- Bixby's own retrospective (`ESTABLISHED METHOD`, **SEARCH-SUMMARY** — its
  abstract/summary was read via search results; the full paper was not
  independently fetched and read in this pass, so treat the specific
  numbers below as probably-right, not certain) attributes LP solvers'
  historical 6-orders-of-magnitude speedup (1980s→2001) to **three orders
  of magnitude each** from raw machine speed and from **algorithmic**
  improvement — presolve, pricing, and factorization together, not any one
  of them alone, and not from micro-optimization. HiGHS's own origin story
  (Galabova's thesis, §1 below) is explicitly "presolve + crash + Huangfu's
  dual simplex" — three
  separate algorithmic components stacked, again not one trick.
- This project already has the *textbook* pieces (Devex, Harris ratio test,
  Ruiz scaling, Bixby crash, Gilbert-Peierls sparse LU, hyper-sparse BTRAN).
  What it is missing, per this research, are specifically the pieces that
  compound with those: a genuine fill-reducing pivot/column **ordering**
  (Markowitz/AMD-class — currently the factorization uses whatever order
  columns happen to arrive in), a denser **presolve** pass (this project's
  own LP presolve is not itself characterized in this document, but its
  **MILP** presolve is confirmed, by reading `docs/architecture/MILP.md`
  directly, to not exist at all beyond reusing LP presolve per node — no
  MIP-specific reductions: probing, clique merging, dual fixing on integer
  columns), and update-file management (PFI vs Forrest-Tomlin) tuned by
  measurement rather than left as a stated, unmeasured deviation.
- **No evidence was found, in this pass, that SIMD/cache-blocking of the
  sparse triangular solve itself is where the gap lives.** The 2024-2025
  literature on SpTRSV vectorization (§3 below) is almost entirely
  GPU-parallel-triangular-solve work aimed at *independent* solves (e.g.
  preconditioners in iterative methods) — it does not obviously transfer to
  simplex's FTRAN/BTRAN, which is one *sequential* chain of dependent
  solves per pivot on a single CPU thread, the exact regime this project's
  own H1/H5 findings (`docs/research/SOTA.md` §5) already established GPU
  parallelism does not help for reasons structural to simplex, not
  implementation quality.
- MILP's mixed 2-2-1 result is separately explained: HiGHS's win margin on
  its two instances lines up with a maturity gap in **cut breadth**
  (root-only cover + GMI here vs. a documented Gurobi-class MIR/clique/cover
  portfolio with cut selection heuristics elsewhere), **primal heuristics**
  (this project has safe rounding + basic LP diving; HiGHS ships RENS,
  RINS, feasibility-jump, and a reduced-cost heuristic, `mip_heuristic_
  effort` default 0.05 — confirmed by reading HiGHS's own GAMS-published
  option list, §5 below), and **MIP-specific presolve** (as above).

---

## 1. HiGHS's own documented internals

### 1.1 Origin and architecture (Galabova's PhD thesis)

`ESTABLISHED METHOD`, **SEARCH-SUMMARY** (thesis abstract/summary read via
search results and the University of Edinburgh repository page; the full
PDF was not opened in this pass — the thesis itself is public at
`era.ed.ac.uk/bitstream/handle/1842/39725/GalabovaI_2022.pdf`, a genuine
next step if deeper detail is needed later).

Ivet Galabova, *"Presolve, Crash and Software Engineering for HiGHS,"* PhD
thesis, University of Edinburgh, 2022/2023. Confirms HiGHS's own account of
its origin: in late 2016, Galabova's LP presolve was combined with Julian
Hall's simplex crash procedure and Qi Huangfu's dual simplex solver
specifically to beat the best open-source solvers of the time on an
industrial LP class. The thesis's three pillars — presolve (dimension
reduction before the algorithmic solve), crash (a cheap near-feasible
starting basis, the same problem this project's own Bixby-1992 crash
already solves — `docs/architecture/LP.md` §3.1), and postsolve (mapping a
presolved solution back to original space, which this project's own
`solve_lp` pipeline already does) — is architecturally identical to what
this project already targets. The thesis analyzes something this project
does not yet have any of: the **Idiot Crash Algorithm (ICA)**, a distinct
crash heuristic from the Bixby-1992 slack-crash this project uses.
`RESEARCH HYPOTHESIS`: whether ICA is complementary to (better on a
different instance class than) the existing Bixby crash is untested here
and not assumed.

### 1.2 Parallel dual simplex (PAMI/SIP)

`ESTABLISHED METHOD`, **VERIFIED** (the arXiv abstract page,
`arxiv.org/abs/1503.01889`, was fetched and read directly) plus
**SEARCH-SUMMARY** for the parallelized-stage detail (from search-result
summaries of the paper's body, not the full PDF).

Huangfu & Hall, *"Parallelizing the dual revised simplex method,"* Math.
Programming Computation 10(1), 2018 (arXiv:1503.01889). Two distinct
parallel strategies, both already noted by name in `docs/research/SOTA.md`
§1.1 but not previously broken down by mechanism:

- **PAMI** extends a pivoting strategy called *suboptimization*
  (choosing several entering variables per major iteration instead of one)
  to get parallelism *across* iterations, not just within one. Per
  search-summary of the paper body: the minor ratio test — comprising SpMV,
  `chuzc1` (a one-pass column selection), and `chuzc2` — is "a major source
  of parallelization," with SpMV and `chuzc1` data-parallel across a
  column partition; `chuzc2` is left serial as "relatively cheap." Reported
  result (search-summary): mean speedup 2.34x on 65% of a reference
  instance set — a real number, but not independently verified against the
  paper's own tables in this pass.
- **SIP** instead overlaps the computational components of a *single*
  iteration (FTRAN, pricing, ratio test, BTRAN-for-duals) where their data
  dependencies allow, rather than restructuring the pivoting strategy
  itself.
- **`RESEARCH HYPOTHESIS`, not previously stated in this repo's research
  docs:** this project's simplex is currently single-pivot, single-thread
  per LP solve (its existing OpenMP parallelism, per `src/parallel/
  Parallel.hpp` and this session's own earlier benchmark work, applies
  within a single dense sparse operation above a size threshold, not
  across pivots or across iteration components). PAMI/SIP-style
  parallelism is a genuinely distinct, unexplored lever — but it is also
  the single highest-*implementation-difficulty* item in this whole
  document: it requires restructuring the core pivot loop's control flow,
  not adding a preprocessing pass, and risks this project's own stated
  determinism requirement (Phase 1: "determinism under fixed
  configuration," MEASURED via 100% bit-identical repeated-run results in
  `docs/measurements/netlib-hybrid-20000rows-repeats3.jsonl`) if
  parallelism introduces order-dependent floating-point summation.
  Recommended: **not** a near-term candidate; flagged for a later,
  dedicated research pass if the ordering/presolve/update levers below are
  exhausted first.

### 1.3 HiPO — the newer interior-point solver

`ESTABLISHED METHOD`, **SEARCH-SUMMARY** (found via search results
referencing `ergo-code.github.io/HiGHS/dev/solvers/`; that page was not
independently fetched and read in this pass).

HiPO ("HiGHS Parallel interior-point Optimizer") is a parallel,
direct-factorization interior-point method, based on Zanetti & Gondzio,
*"A factorisation-based regularised interior point method using the
augmented system,"* 2025 (arXiv:2508.04370 — not independently opened in
this pass). It runs a parallel multifrontal factorization with two levels
of parallelism (across the elimination tree, and within each frontal
matrix's dense factorization — `hipo_parallel_type` controls tree/node/
both), falling back to a serial up-looking factorization for small/very
sparse problems. Per HiGHS's own docs (search-summary), HiPO's advantage
over the older `IPX` interior-point path grows with problem size, "more
than an order of magnitude" on large problems. This is **not** relevant to
this project's own near-term simplex gap (this project has no interior-
point path at all — a much larger, separate scope decision already flagged
as out of scope by `CLAUDE_OPUS_SOLVER_ROADMAP.md`'s own priority list, and
not revisited here), but it is worth recording as a data point for the
"is the gap in simplex specifically, or LP solving broadly" question: HiGHS
maintains **two** independently-optimized LP algorithms (simplex and
HiPO/IPX) and races or selects between them, which this project's own
`LpMethod::HYBRID` already does in spirit (simplex + first-order PDLP), so
the *architectural* pattern (offer more than one algorithm family, pick
by measured characteristics) is already validated by this project's own
`docs/architecture/LP.md` §7, not a new insight.

---

## 2. Sparse LU factorization and update strategy

### 2.1 Forrest-Tomlin vs. Product Form of the Inverse

`ESTABLISHED METHOD`, **SEARCH-SUMMARY** (Huangfu & Hall's own follow-up
paper on update techniques was found and its abstract/summary read; the
full PDF was not opened).

Huangfu & Hall, *"Novel update techniques for the revised simplex
method,"* Computational Optimization and Applications, 2015 (an earlier
preprint appears as `optimization-online.org/.../3774.pdf`). Directly
relevant: this is the SAME research group whose dual simplex HiGHS uses,
specifically comparing update strategies. Per search-summary, they
introduce *novel product-form variants* and found "one of the product form
variants is significantly more efficient than the traditional approach,
with its performance approaching that of the Forrest-Tomlin update for
some problems" — i.e. even this repo's own PFI-vs-FT deviation
(`src/lp/BasisFactorization.hpp`'s stated deviation from `docs/
architecture/LP.md` §4) is not a settled question in the literature
either: a *well-designed* PFI variant can approach FT's performance, and a
plain FT update reduces eta-file growth rate specifically (less frequent
refactorization, less per-pivot work as the eta chain lengthens). This
repo's own architecture doc already states the correct empirical stance
("the correct upgrade if benchmarking shows eta growth, not
factorization, dominates — that is a measurement to make, not an
assumption") — this research **confirms, but does not newly establish,**
that stance; nothing here overrides it. `PROPOSED MODIFICATION`, unchanged
from the existing doc: measure eta growth / refactorization frequency on
a large instance (`stocfor3`, matching this repo's own BTRAN benchmark
instance for direct comparability) before deciding whether FT is worth the
implementation cost.

### 2.2 Markowitz + AMD/COLAMD ordering

`ESTABLISHED METHOD`, **SEARCH-SUMMARY** (several distinct sources,
consistent with each other and with this project's own already-cited
Markowitz 1957/Gilbert-Peierls 1988 lineage).

Markowitz (1957) pivot selection minimizes a *local* fill-in count at each
elimination step; AMD (Amestoy, Davis, Duff) and COLAMD are *global*
approximate-minimum-degree preprocessing orderings applied once, before
factorization, to reduce the *total* fill-in the factorization will
produce regardless of pivot-order choices made during elimination itself.
Search results (`8.3 Pivoting To Preserve Sparsity`, `vismor.com`) frame
these as solving the same underlying problem — minimizing fill — at
different points in the pipeline (during elimination vs. before it), and
note the difficulty is a genuine trade-off between minimizing fill and
minimizing numerical element growth in the factors, not a free win.
**No specific, verified percentage fill-reduction number for LP simplex
bases specifically (as opposed to general sparse symmetric/unsymmetric
matrices) was found in this pass** — this is stated as an honest gap in
the research, not glossed over. `RESEARCH HYPOTHESIS`, correctly scoped:
whether AMD-class ordering meaningfully reduces fill for THIS project's
own basis matrices (which are the columns of whatever B the simplex
currently holds — refactorized repeatedly as the basis changes, unlike a
one-shot sparse solve) needs to be measured directly, the same way BTRAN's
win was measured, not assumed from general sparse-matrix literature.
Critically: this project's basis is **refactorized repeatedly** (a fresh
factorization roughly every refactorization interval, not once), so an
ordering computed once per refactorization pays its cost every time — the
benefit (less fill, cheaper subsequent FTRAN/BTRAN) must be weighed
against that recurring ordering-computation cost, which BTRAN's own
hyper-sparse work did not have to consider (it added no new
per-refactorization step, only a per-solve one).

---

## 3. Parallel/vectorized simplex and sparse triangular solve

`ESTABLISHED METHOD` for the general SpTRSV literature, **SEARCH-SUMMARY**.

The 2024-2025 literature found (`DaCP`, `AG-SpTRSV`, `CapelliniSpTRSV`,
SIMD-AVX2 SpTRSV work reporting 1.7-6x over Intel MKL) is concentrated on
**GPU-parallel** or **many-independent-solve** sparse triangular solve —
the target use case in nearly every result found is a preconditioner
step inside an iterative solver (many RHS vectors, or one RHS solved with
massive fine-grained parallelism across independent rows/colors). This
does **not** structurally match simplex's FTRAN/BTRAN, which is:
- one RHS per pivot (not a batch),
- solved on a **single CPU thread** sequentially within a pivot (the next
  pivot cannot start until this one's ratio test and update finish — this
  project's own H5 finding, `docs/research/SOTA.md` §5, already establishes
  *why* this project deliberately keeps pivot control flow off the GPU),
- and, per §2.1 above, itself the *target* of exactly the hyper-sparse
  reachability technique (Gilbert-Peierls DFS) this project already
  implemented for BTRAN — which is precisely a technique for exploiting
  sparsity in a *single-RHS, sequential* solve, the opposite regime from
  what the SIMD/GPU-SpTRSV literature found here targets.

**Conclusion, stated as a negative result rather than omitted:** this
research pass found no direct evidence that SIMD/cache-blocking
literature transfers to this project's specific FTRAN/BTRAN bottleneck.
The lever that *does* transfer directly — hyper-sparse reachability for a
single sequential RHS — is exactly what BTRAN already implemented and
what FTRAN (this iteration's own task, in progress alongside this
document) is evaluating next. This is evidence *against* prioritizing a
SIMD/vectorization investigation and *for* the ordering/presolve/update
levers in §1-2 as the higher-value remaining direction.

---

## 4. GPU first-order methods beyond SOTA.md

`ESTABLISHED METHOD`, **SEARCH-SUMMARY** throughout this section —
cuPDLPx's existence, headline numbers, and surrounding detail were all
found via search results; its arXiv page was not independently fetched
and read in this pass.

Since `docs/research/SOTA.md` §1.3/§1.3b was written, a further line of
work has appeared: **cuPDLP.jl** (Nov 2023) → **cuPDLP-C** (Dec 2023,
COPT's C reimplementation) → **cuPDLPx** (2025, arXiv:2507.14051),
"A Further Enhanced GPU-Based First-Order Solver for Linear Programming" —
adds a new restart criterion and a PID-controlled primal weight update on
top of "restarted Halpern PDHG," reporting **2.5-5x speedup on MIPLIB LP
relaxations and 3-6.8x on Mittelmann's benchmark set** relative to (per
the paper's own framing) prior GPU first-order baselines, not relative to
simplex. This project's own PDLP implementation (`src/cuda/PdlpKernels.cu`)
already implements the core Applegate et al. 2021 mechanisms (diagonal
preconditioning, adaptive restarts, adaptive step size) that this newer
work builds *on top of* — cuPDLPx's specific additions (a different
restart criterion, PID-controlled step-size/weight control) are a
`PROPOSED MODIFICATION` candidate for this project's own already-existing
PDLP path, not a new subsystem. `RESEARCH HYPOTHESIS`, unverified in this
pass: whether cuPDLPx's specific restart/weight mechanism would move this
project's own measured PDLP results (`docs/architecture/PDLP.md` §5 — the
6 stalling instances `pilot.we`, `pilot.ja`, `bnl2`, `greenbea`,
`greenbeb`, `pilot`, already flagged as a `KNOWN LIMITATION` with
"feasibility polishing" as the recorded untested hypothesis) is not
established here and would need its own paper read + implementation +
measurement pass.

**GPU crossover** (converting a first-order near-optimal interior point to
an exact vertex/basic solution): search results surfaced the concept
("smart crossover from an interior point to a corner point") as an
active area but returned **no specific paper or algorithm this pass could
verify or cite concretely** — recorded honestly as a `RESEARCH HYPOTHESIS`
with no supporting citation found yet, not asserted as if a specific
technique were identified. This matters to this project specifically
because its own HYBRID mode's PDLP fallback returns an approximate,
non-vertex point that still passes the same residual verification gate
(`docs/architecture/NUMERICS.md` §6) rather than being crossed over to an
exact basis — a real, already-acknowledged asymmetry between the two
methods' output character that a crossover step would remove, if one were
implemented and shown numerically safe.

---

## 5. MILP-specific findings

`ESTABLISHED METHOD`, **VERIFIED** for HiGHS's own published option list
(the GAMS-hosted HiGHS option documentation page was fetched and read
directly — GAMS mirrors HiGHS's own option definitions, not a third-party
reimplementation), **SEARCH-SUMMARY** for the Achterberg et al. 2020
presolve paper's content (its abstract/citations were found; the paper's
own body/tables were not opened in this pass).

### 5.1 HiGHS's MIP heuristic and cut portfolio, confirmed by reading its own option docs

- **Heuristics, by name, each independently toggleable:** feasibility jump
  (`mip_heuristic_run_feasibility_jump`), RENS (`mip_heuristic_run_rens`),
  RINS (`mip_heuristic_run_rins`), a root-reduced-cost heuristic
  (`mip_heuristic_run_root_reduced_cost`), plus Shifting and ZI-Round
  diving heuristics (present but **disabled by default**). Overall
  heuristic effort is a single dial, `mip_heuristic_effort`, **default
  0.05** — a small but nonzero default budget spent on primal heuristics
  every run.
- **This project's current MILP heuristic set, for direct comparison**
  (`docs/architecture/MILP.md` §4): a safe rounding heuristic (round +
  re-verify) and deterministic LP diving. **No RENS, no RINS, no
  feasibility pump, no feasibility-jump-class heuristic exist yet** — this
  is a real, specific, three-technique gap, not a vague "heuristics are
  weaker" impression.
- **Cut generation**: HiGHS's own docs (as mirrored by GAMS) describe
  root-level cut separation plus an aging mechanism that shrinks the cut
  pool after root separation, and an option to allow cut separation *at
  nodes*, not only the root (`mip_allow_cut_separation_at_nodes`). This
  project's own cuts (cover cuts, §2.1 of `docs/architecture/MILP.md`; GMI
  cuts, §2.2) are both **root-only**, already flagged as a known,
  deliberate v1 scope limit in `docs/ROADMAP_STATUS.md`'s own priority
  list ("separating cuts at more than one round/node ... remains a
  separate, later candidate") — this research **corroborates** that this
  is a real, named gap relative to HiGHS specifically, not merely a
  generic "more cuts would help" guess. The *specific* cut families
  HiGHS runs by default (beyond "cut separation" as a category) could not
  be confirmed in this pass — recorded as unverified rather than guessed.
- **Parallelism**: HiGHS allows up to 8 concurrent threads for its LP
  relaxation solves inside MIP (`simplex_max_concurrency`) — orthogonal to
  this project's own B&B-stays-CPU-single-threaded-per-node architecture
  question; not directly comparable without knowing whether HiGHS's MIP
  *tree* search itself is multi-threaded (search results did not confirm
  either way in this pass) versus only its per-node LP relaxation solves.

### 5.2 MIP-specific presolve — confirmed gap, read directly from this repo's own docs

Reading `docs/architecture/MILP.md` directly (not a new citation, an
in-repo verification) confirms: this project's "node presolve" (§1.4) is
**the existing LP presolve implementation, reapplied per node** — there is
no MIP-specific presolve reduction anywhere in the codebase (no probing,
no clique detection/merging on binary variables, no dual fixing exploiting
integrality, no coefficient tightening using integer bounds). Achterberg,
Bixby, Gu, Rothberg & Weninger, *"Presolve Reductions in Mixed Integer
Programming,"* INFORMS J. Computing 32(2), 2020 (already cited in this
project's own `docs/research/SOTA.md`) documents exactly this taxonomy of
MIP-specific reductions as what Gurobi's presolve implements beyond
generic LP presolve. Search-summary (not independently verified against
the paper's own experimental section in this pass) states "presolve
together with cutting plane techniques are by far the most important
individual tools contributing to the power of modern MIP solvers" — a
strong claim, correctly labeled here as a summary of the paper's framing,
not an independently re-derived result. Given this project's own MILP
KPI gap is real and measured (`docs/ROADMAP_STATUS.md`'s own §1: 1/5
certified, badly wrong incumbents on 3/5), and given cuts (both families,
already tried) and heuristics (none of RENS/RINS/feasibility-pump exist)
are both flagged gaps independently, MIP-specific presolve is a third,
independent, currently-completely-unaddressed lever.

---

## 6. Kill-shot candidates, ranked

Cross-referenced against `docs/ROADMAP_STATUS.md`'s live priority list as
of this document's writing (commit `314d39d` and this session's own
in-progress hyper-sparse FTRAN work) — nothing here duplicates GMI cuts or
MILP node warm-starting (both already tried, both did not clear their KPI
gate) or hyper-sparse FTRAN (already in progress).

| rank | candidate | mechanism | expected benefit | difficulty | numerical risk | status vs. this repo |
|---|---|---|---|---|---|---|
| 1 | **Markowitz/AMD fill-reducing ordering for basis factorization** | §2.2 — compute a fill-reducing column/pivot order once per refactorization, before Gilbert-Peierls elimination | Denser factors → less fill → cheaper subsequent FTRAN/BTRAN on every pivot until the next refactorization; this is the single item in this whole document with the most direct literature support for being a *standard, expected* lever in a mature simplex (§0, §1.3 architectural pattern) | Medium — AMD/COLAMD are well-documented, implementable algorithms; the integration point (feeding an ordering into the existing Gilbert-Peierls factorize()) is well-scoped | Low-medium — ordering choice interacts with pivot stability (numerical vs. sparsity trade-off, §2.2); needs the same threshold-pivoting safeguards already in place | Not attempted; explicitly next on `docs/ROADMAP_STATUS.md`'s own list after hyper-sparse FTRAN |
| 2 | **MIP-specific presolve (probing, clique merging, dual fixing on integers, coefficient tightening)** | §5.2 — Achterberg et al. 2020 taxonomy, applied at the MILP layer, not just reused LP presolve per node | Directly targets this project's own measured, named MILP KPI gap (1/5 certified); orthogonal to (compounds with) the cut-density and heuristic gaps below | High — this is a family of reductions, not one; needs its own reversible-postsolve mapping per reduction (this project's own existing correctness discipline) and its own correctness test suite, likely several loop iterations, not one | Medium — presolve bugs are historically this project's own worst failure mode (a false-INFEASIBLE risk); the adversarial-LP-generator infrastructure just built (this session) is directly reusable here as a correctness gate | Not attempted; a genuinely new gap this document identifies precisely (§5.2), not previously named in `docs/ROADMAP_STATUS.md` |
| 3 | **MILP primal heuristics: RENS and/or a feasibility-pump-class heuristic** | §5.1 — HiGHS ships RENS/RINS/feasibility-jump/root-reduced-cost by default; this project has only safe rounding + LP diving | Faster time-to-first-incumbent and better incumbent quality specifically on instances where cuts alone are insufficient — plausibly relevant to exactly the `pk1`/`gen-ip002` instances this session's own MILP work already struggled with | Medium — RENS (round + re-solve a restricted LP) is structurally close to this project's existing LP-diving machinery; lower incremental cost than presolve (#2) | Low — heuristics only *propose* incumbents, verified through the same existing original-space feasibility gate before acceptance (this project's own established safe-heuristic pattern, `docs/architecture/MILP.md` §4) | Not attempted; a concrete extension of an existing, working pattern rather than new infrastructure |
| 4 | **Multi-round / multi-node cut separation (not just root)** | Re-run cover-cut and GMI separation at additional B&B nodes, not only the root — already named as a deferred v1 scope limit in this repo's own docs | Tighter bounds deeper in the tree, where root-only cuts have already been exhausted | Medium — the separation *code* already exists (root-only call sites); the new work is deciding *when* to re-separate and managing a growing cut pool (aging, as HiGHS does per §5.1) without unbounded LP growth | Medium — more rows per node LP costs more per-node solve time; this project's own GMI-cut experience this session (net negative on the 5-instance set even after fixing a numerical bug) is a direct, sobering precedent that "more cuts" is not automatically a win — must be benchmarked, not assumed | Explicitly named as a "separate, later candidate" in `docs/ROADMAP_STATUS.md`; this document does not newly discover it, only re-confirms it's real |
| 5 | **PFI→Forrest-Tomlin update, or a better-tuned PFI variant** | §2.1 — reduce eta-file growth rate specifically | Fewer/cheaper refactorizations on long pivot sequences | Medium-high — FT modifies U in place; getting this numerically right is nontrivial (this project's own BasisFactorization doc already flags this) | Medium — update-strategy bugs are subtle; needs the same differential-testing discipline already used for hyper-sparse BTRAN | Already an explicitly named, deliberate, measured-not-assumed deviation in `docs/architecture/LP.md` §4 — this document adds one corroborating citation (§2.1) but does not change the recommended action (measure eta growth first) |
| 6 | **PAMI/SIP-style intra-solve parallelism** | §1.2 | Potentially large (2.34x reported for PAMI, unverified against this project's own workload) | Very high — restructures core pivot control flow; risks this project's own MEASURED determinism guarantee | Medium-high — order-dependent floating-point summation under parallelism is a real, specific risk to this project's own stated correctness bar | Not attempted; correctly the lowest-priority item here given effort/risk relative to items 1-4 |
| 7 | **cuPDLPx-style restart/step-size refinements to the existing PDLP path** | §4 | Could move the 6 stalling PDLP instances (`docs/architecture/PDLP.md` §5) | Medium — modifies existing, working code rather than building new infrastructure | Medium — PDLP's own correctness gate (original-space residual check) already catches a wrong step, but a stalling-vs-fixed distinction needs its own careful measurement | A concrete, scoped candidate for the *existing* "feasibility polishing" untested hypothesis already recorded in PDLP.md §5 — not new scope, a specific mechanism for already-named scope |

**Explicitly not re-ranked here, only cross-referenced:** GMI cut
numerical/density tuning (paused per `docs/architecture/MILP.md` §2.2.1),
MILP node warm-starting (measured, rejected, `docs/architecture/LP.md`
§8), GPU-simplex-pivoting or GPU-B&B-control-flow of any kind (rejected
by this project's own H5 finding and architectural mandate, not
reconsidered by anything found in this pass).

---

## 7. Recommended next 3 `/loop` iterations

Sized the same way this session's own history is sized — one coherent,
independently benchmarkable increment each, following the exact procedure
already established (implement → test → benchmark on a real instance,
clean/single-process/multi-trial → commit with the measured result stated
→ document, win or not).

1. **Markowitz/AMD fill-reducing ordering, benchmarked on `stocfor3`**
   (same instance as the BTRAN measurement, for direct comparability) —
   measure factor nonzero count and wall-clock before/after, exactly like
   `docs/architecture/LP.md` §9's own reporting format. This is
   `docs/ROADMAP_STATUS.md`'s own next-named item after hyper-sparse
   FTRAN and the single highest-confidence lever this document found.
2. **MIP-specific presolve, starting with the single reduction most
   likely to matter for this project's own named failing instances**:
   given `pk1`/`gen-ip002`/`gen-ip054` already get zero cover cuts because
   they lack binary-knapsack structure, a natural first probe is bound
   tightening / coefficient strengthening using **integer** rounding
   (tightening a continuous-looking bound to the nearest integer when the
   variable is integer-restricted) — small, well-scoped, directly testable
   against the adversarial-LP-generator infrastructure this session already
   built (extend it with a MILP variant), and independently re-measurable
   against `bench_miplib`'s existing 5-instance set for continuity with
   prior GMI-cut/warm-start measurements.
3. **One MILP primal heuristic — RENS** (round the LP relaxation's
   fractional variables to their nearest integer *only where safe*, fix
   the rest, re-solve the reduced LP for a candidate incumbent) — the
   heuristic closest in shape to this project's own existing LP-diving
   code, verified through the same existing original-space feasibility
   gate before any incumbent is accepted (no new correctness machinery
   needed, only a new proposal-generation path).

Each should be measured independently and honestly — including the
possibility, already this session's own repeated experience with GMI
cuts and MILP warm-starting, that a textbook-correct technique does not
clear this project's own KPI gate on its specific benchmark set. That
outcome is exactly as valuable to document as a win, per this project's
own established practice.

---

## References

- Bixby, R. E. "Solving Real-World Linear Programs: A Decade and More of
  Progress." *Operations Research* 50(1), 2002.
  <https://pubsonline.informs.org/doi/10.1287/opre.50.1.3.17780>
- Galabova, I. "Presolve, Crash and Software Engineering for HiGHS." PhD
  thesis, University of Edinburgh, 2022/2023.
  <https://era.ed.ac.uk/bitstream/handle/1842/39725/GalabovaI_2022.pdf>
- Huangfu, Q. & Hall, J. A. J. "Parallelizing the dual revised simplex
  method." *Mathematical Programming Computation* 10(1), 2018
  (arXiv:1503.01889). <https://arxiv.org/abs/1503.01889>
- Huangfu, Q. & Hall, J. A. J. "Novel update techniques for the revised
  simplex method." *Computational Optimization and Applications*, 2015.
  <https://webhomes.maths.ed.ac.uk/hall/HuHa12/>
- Zanetti, F. & Gondzio, J. "A factorisation-based regularised interior
  point method using the augmented system." 2025 (arXiv:2508.04370).
  <https://arxiv.org/pdf/2508.04370>
- HiGHS documentation, "Solvers" (HiPO).
  <https://ergo-code.github.io/HiGHS/dev/solvers/>
- HiGHS option reference, as mirrored by GAMS.
  <https://www.gams.com/latest/docs/S_HIGHS.html>
- Amestoy, P., Davis, T. & Duff, I. "An Approximate Minimum Degree
  Ordering Algorithm." SIAM J. Matrix Anal. Appl. 17(4), 1996 —
  established AMD reference; not independently opened in this pass,
  cited here for the algorithm's standard name/attribution only.
- Markowitz, H. "The elimination form of the inverse and its application
  to linear programming." *Management Science* 3(3), 1957 — already cited
  in `docs/architecture/BasisFactorization.hpp`'s own header comment;
  restated here for this document's self-containedness.
- Lu, H. & Yang, J. et al. cuPDLPx: "A Further Enhanced GPU-Based
  First-Order Solver for Linear Programming." 2025 (arXiv:2507.14051).
  <https://arxiv.org/abs/2507.14051>
- Achterberg, T., Bixby, R. E., Gu, Z., Rothberg, E. & Weninger, D.
  "Presolve Reductions in Mixed Integer Programming." *INFORMS Journal on
  Computing* 32(2), 2020. <https://pubsonline.informs.org/doi/10.1287/ijoc.2018.0857>
  — already cited in `docs/research/SOTA.md`; re-cited here with the
  specific taxonomy gap (§5.2) this document identifies.
