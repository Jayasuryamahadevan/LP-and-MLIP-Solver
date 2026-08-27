# Roadmap status

Status of this repository against `CLAUDE_OPUS_SOLVER_ROADMAP.md`, using that
document's own vocabulary: `IMPLEMENTED`, `MEASURED`, `ESTABLISHED METHOD`,
`ENGINEERING DECISION`, `RESEARCH HYPOTHESIS`, `KNOWN LIMITATION`.

Everything claimed here is traceable to a file in `docs/measurements/`. Nothing
in this document is an estimate.

---

## Headline

| KPI | value | source |
|---|---|---|
| Netlib validated instances solved | **93 / 93** | `netlib-hybrid-20000rows.jsonl` |
| — simplex alone | 92 / 93 | `netlib-validation-20000rows.txt` |
| Kennington + QAP, cross-method checked | **21 / 21**, 0 disagreements | `crossmethod-kennington.jsonl` |
| — HYBRID total on that set | **43.9 s** vs 388.0 s simplex-alone | `crossmethod-kennington.jsonl` |
| total models solved | **114 / 114** | both files |
| row cap | 20,000 | — |
| total time | **72.594 s** (was 115.900 s before the structure-aware lead) | JSONL |
| geometric mean | 0.030 s | JSONL |
| median | 0.020 s | JSONL |
| 95th percentile | **4.033 s** | JSONL |
| max | **25.066 s** (`stocfor3`) | JSONL |
| total iterations | 255,144 | JSONL |
| worst relative objective error | 5.779e-07 | JSONL |
| MIPLIB 2017 subset (5 instances, 60s budget) certified | **1 / 5** | `reports/runs/2026-08-25/miplib-raw.txt` |
| unit tests | 142 / 142 | `ctest` (128 pre-existing + 5 GMI-cut + 3 ResourceSnapshot + 6 adversarial-LP cases, the last covering 400 generated instances) |

`MEASURED`. Single process, nothing else running, build stamp
`1afe5bfa` recorded in the JSONL header.

This meets the roadmap's stated target — *"Maintain or exceed current 92/93
Netlib validation"* — by one instance.

---

## Environment fix that predates this revision's other work

Before any of the work below, this session's build showed **106/128 unit
tests failing** with `CUDA error: the provided PTX was compiled with an
unsupported toolchain`, contradicting the 128/128 claimed above. Root
cause: `CMAKE_CUDA_ARCHITECTURES` was populated by CMake's own CUDA-
language auto-defaulting (triggered by `project(... LANGUAGES CXX CUDA)`)
*before* this file's `if(NOT CMAKE_CUDA_ARCHITECTURES)` guard ever ran, so
the guard was dead code; on this machine (CMake 4.2.3 + CUDA 13.3) that
default picked `sm_75` (Turing) instead of the dev GPU's actual `sm_89`
(Ada RTX 3050 Laptop), producing a fatbin the driver rejects at
kernel-launch time rather than at compile time — which is why a stale
`build/` directory carrying the wrong cached architecture went
undetected for however long it existed. Fixed by moving the default
above `project()`, the only point at which CMake will actually honor it.
`MEASURED` after the fix: fatbin reports `sm_89`, 128/128 tests pass,
90/90 Netlib-validated (of 93 feasible instances with a published
reference — unaffected).

Separately, an uncommitted (but already measured-against) fix to
`src/io/MpsReader.cpp` was found in the working tree and verified rather
than discarded: a fixed-column MPS parsing fallback for classic strict-
format files (`dfl001`, `sierra`, `forplan`, `gfrd-pnc`, `blend`, …)
whose row/column names contain embedded blanks, or whose RHS/RANGES/
BOUNDS vector-name field is blank rather than absent — both illegal
under free-format whitespace tokenization. This is what the 93/93 /
90/90 figures above already depend on; it is now committed.

---

## Phase 0 — benchmark and observability infrastructure

The roadmap ranks this **priority 1**, ahead of all algorithmic work. That
ordering was earned the hard way here: three separate wrong conclusions in this
project came from measurements taken without recorded conditions (see
`docs/architecture/PDLP.md` §5).

| item | status |
|---|---|
| Structured JSON/CSV output | `IMPLEMENTED` — JSON Lines, `src/bench/RunMetadata.*` |
| Full configuration capture | `IMPLEMENTED` — method, pricing, presolve, scaling, budgets, tolerances |
| Reproducibility metadata | `IMPLEMENTED` — git commit + dirtiness, compiler, CUDA, GPU, driver, CPU, RAM, threads, OpenMP schedule |
| Instance hashing | `IMPLEMENTED` — FNV-1a 64 over file bytes |
| Per-stage timers | `IMPLEMENTED` — `SimplexProfile`, 9 stages, `benchmarks/profile_simplex.cpp` |
| Median / geometric mean / p95 summaries | `IMPLEMENTED` — `bench::summarize` |
| Memory statistics | `IMPLEMENTED`, `MEASURED` — `src/bench/ResourceSnapshot.{hpp,cpp}` (`InstanceRecord::peak_rss_kb`/`gpu_used_mb`), per-instance in `validate_netlib`'s JSONL. Peak RSS is the process's cumulative peak through that instance (`getrusage`'s own contract), not an isolated per-instance figure — stated in the header comment, not hidden |
| Repeated-run support, median over runs | `IMPLEMENTED`, `MEASURED` — `validate_netlib`'s new optional 7th argument (repeat count, default 1 so every prior measurement's meaning is unchanged); `wall_seconds` becomes the median, `wall_seconds_min`/`max` recorded alongside. `docs/measurements/netlib-hybrid-20000rows-repeats3.jsonl`: full 90-instance sweep at 3 repeats, 100% bit-identical objective/iteration/status across every repeat on every instance — a real, MEASURED confirmation of this project's stated determinism-under-fixed-configuration goal (Phase 1), not merely an assumption |
| Performance profile generation | **NOT IMPLEMENTED** |
| Benchmark regression comparison | **NOT IMPLEMENTED** |
| NVTX ranges | **NOT IMPLEMENTED** |

Acceptance criteria: *"a result can be traced to a commit and instance hash"* is
met. *"A regression shows exactly which stage changed"* is met for the simplex
only, and not across runs.

---

## Phase 1 — correctness hardening

| item | status |
|---|---|
| Independent primal / dual / objective verification | `IMPLEMENTED` — original-space gate, `NUMERICS.md` §6, applies to every result including first-order |
| Presolve on/off differential testing | `IMPLEMENTED` — `validate_netlib … nopresolve` |
| CPU/GPU differential testing | `IMPLEMENTED` — bit-identical assertions across thread counts and backends |
| Determinism under fixed configuration | `MEASURED` — exact-equality tests, including PDLP |
| Random / ill-conditioned / degenerate LP generators | `IMPLEMENTED`, `MEASURED` — `tests/lp/adversarial_lp_generator.hpp` + `tests/lp/test_adversarial.cpp`, see below |
| MPS and sparse-structure fuzzing | **NOT IMPLEMENTED** — the generator above builds `LpProblem` directly, not MPS text; a text-level MPS fuzzer is a distinct, still-open item |
| Compute Sanitizer in CI | **NOT IMPLEMENTED** — never run. Deliberately deferred out of this pass (recorded, not dropped): the generator work below was already one coherent increment, and Compute Sanitizer is independent enough to be its own |
| Brute-force checker for tiny MILPs | `IMPLEMENTED` — this line was stale; `tests/milp/test_milp.cpp` has brute-forceable tiny-MILP cases (integer optima, infeasibility proofs) since the MILP engine landed |

`MEASURED`: 6 generator categories (feasible/bounded, ill-conditioned,
degenerate, infeasible-by-bounds, infeasible-by-rows, unbounded), each with
a TRUE status known by construction (not by trusting any solver — see the
generator's own file header for the closed/bounded/nonempty-polytope
argument), seeded and deterministic. 400 generated instances total across
6 test cases, every accepted OPTIMAL independently re-verified in the test
file itself (recomputing bound/row feasibility and the objective from `x`
directly — not merely trusting `LpSolution`'s own residual fields):

| category | instances | result |
|---|---|---|
| feasible/bounded | 80 | 80/80 OPTIMAL, 80/80 independently verified (tol 1e-6) |
| ill-conditioned (coefficients 1e-6..1e6) | 60 | 60/60 correct status (never false INFEASIBLE/UNBOUNDED), 60/60 independently verified within a relaxed 1e-4 tolerance (conditioning bounds achievable accuracy for correct code too, not only buggy code) |
| degenerate (redundant/duplicate rows) | 80 | 80/80 OPTIMAL, 80/80 independently verified (tol 1e-6) |
| infeasible (bound contradiction) | 60 | 60/60 correctly detected INFEASIBLE |
| infeasible (row contradiction) | 60 | 60/60 correctly detected INFEASIBLE |
| unbounded (zero column, negative cost, no upper bound) | 60 | 60/60 correctly detected UNBOUNDED |

**One real bug found and fixed during this work — in the test generator,
not the solver.** The degenerate-LP generator's row-duplication loop
copied a source row's `rhs`/`row_types` from an up-to-date, incrementally
mutated array, but copied that same row's *coefficients* from a fixed
snapshot of the original matrix that was never rebuilt mid-loop. When a
duplication's source row had itself already been overwritten by an
earlier duplication in the same loop, this produced a row whose `rhs` and
coefficients came from two different points in time — an inequality no
point actually satisfied, i.e. a genuinely corrupted instance, not the
guaranteed-feasible one the generator claimed to produce. It surfaced as
an unexplained `INFEASIBLE` in `test_adversarial.cpp`; isolated with a
standalone reproduction (not guessed), confirmed by hand-checking the
corrupted row's coefficients against the known feasible point (activity
`-582.04` against a required RHS of `129.23` — no accuracy issue, an
actually-different linear constraint), and fixed by keeping row content
in one consistently-mutated structure instead of two independently-mutated
ones. This is exactly the outcome Phase 1 correctness infrastructure is
for — it does not matter that the bug was in the harness rather than the
solver; a fuzzer whose own instances are corrupted is worse than no
fuzzer, since it would have quietly recorded false “solver failures.”

`KNOWN LIMITATION` (unchanged by the above, restated precisely): the
solver-under-test claims themselves ("no false INFEASIBLE", "no false
UNBOUNDED") are now backed by 400 generated adversarial instances in
addition to the Netlib set (28/0/1 known-infeasible/unbounded/feasible),
which is real, new, stronger evidence — but MPS-text-level fuzzing and
Compute Sanitizer remain open, so this is not yet the roadmap's full
Phase 1 acceptance bar.

`KNOWN LIMITATION`, `MEASURED` via `docs/measurements/highs_comparison.md`:
`src/io/MpsReader.cpp` does not implement the MPS objective-row RHS
constant (a value on the RHS line for the objective/`N` row is a valid,
if uncommon, way to encode a constant term in the objective; HiGHS's own
parser implements this, `highs/io/HMpsFF.cpp:1081`). A full scan of the
Netlib feasible set found exactly one instance affected (`e226`,
constant `7.113`) out of 114 — narrow, not systemic — and this repo's
current (no-offset) behavior happens to match `data/netlib_readme.txt`'s
own published reference for that instance, so nothing here was silently
wrong on the existing validated set. Next concrete step: add an
`obj_offset` field to `MpsModel`/`LpProblem`, apply it when constructing
the reported objective in both `LpSolver.cpp` and `MilpSolver.cpp`, and
add a unit test against a small hand-built model with a nonzero
objective-row RHS.

---

## Phase 2 — high-performance LP core

| item | status |
|---|---|
| Sparse basis factorization + eta updates | `IMPLEMENTED` — no dense inverse |
| Devex pricing | `IMPLEMENTED`, `MEASURED` at 2.16× Dantzig per iteration |
| Ruiz scaling | `IMPLEMENTED`, `MEASURED` |
| Presolve (core reductions) | `IMPLEMENTED` |
| **Warm-started dual simplex** (`Simplex::set_warm_start_basis`/`export_basis`) | `IMPLEMENTED`, `MEASURED` — see below |
| Hyper-sparse **BTRAN** with active-pattern detection | `IMPLEMENTED`, `MEASURED` — see below; FTRAN's `ftran_column` deferred |
| Markowitz / AMD ordering, symbolic reuse | **NOT IMPLEMENTED** |
| Presolve expansion (doubleton, aggregation, probing, …) | **NOT IMPLEMENTED** |

`MEASURED` (`docs/architecture/LP.md` §9): `BasisFactorization::btran` now
uses Gilbert-Peierls DFS reachability instead of an unconditional O(m)
sweep, gated by a measured density fallback for RHS vectors that turn out
not to be sparse in practice. Clean 3-trial comparison on `stocfor3`
(16,675 rows): median wall-clock **15.379 s → 12.852 s (16.4% faster)**;
the always-maximally-sparse `compute_binv_row` call site improved ~24%
consistently. On small MILP node relaxations (MIPLIB, m in the 7–90 range)
the initial version cost 4–7% of node throughput from a real bug (fresh
per-call heap allocations violating this project's own no-allocation-
during-solve discipline); fixed, with the small residual left within this
instance set's own observed run-to-run noise. FTRAN's `ftran_column` RHS
is sparse too but its result typically fills in through `U`, so it is a
named, deferred v2 candidate rather than part of this pass.

`MEASURED` (`docs/architecture/LP.md` §8, `reports/runs/2026-08-25/
miplib-warmstart-{off,on}.txt`): wired into MILP node relaxations behind
`MilpSolverOptions::warm_start_node_relaxations` (default `false`). It
delivers the node-throughput gain the roadmap predicted — 2.7–577× more
nodes processed in the same 60 s budget across the 5-instance MIPLIB
set — and that gain is *worse* for MILP once measured end to end:
certified results went **1/5 → 0/5**, because non-root nodes bypass
presolve (a warm basis is only valid over the same column space presolve
produced it for) and lose more pruning power from that than the faster
per-node solve buys back. `KNOWN LIMITATION`, not a defect: this is a real
trade, recorded rather than hidden, and the default stays off per this
document's own KPI-gate rule. See `docs/architecture/LP.md` §8 for the
full measurement and the presolve-aware-warm-start hypothesis it points to
next.

---

## Phase 3–5 — MILP

**`IMPLEMENTED`, since this document was last synchronized against the
code.** `src/milp/MilpProblem.{hpp,cpp}` and `src/milp/MilpSolver.{hpp,cpp}`
exist: best-bound branch-and-bound, reliability branching (with
strong-branching probes and pseudocost fallback), root-only mixed-row cover
cuts (`docs/architecture/MILP.md` §2), a safe rounding heuristic, LP diving,
and local improvement. 116/116 unit tests pass, including MILP-specific
brute-forceable cases (tiny integer optima, infeasibility proofs, node-limit
handling, cover-cut validity) and the warm-start differential tests above.

`KNOWN LIMITATION`, and the one that actually matters right now:
**unit-correct is not benchmark-ready.** `benchmarks/bench_miplib.cpp`
against 5 real MIPLIB 2017 instances (60 s budget each,
`reports/runs/2026-08-25/miplib-raw.txt`) certifies only **1/5**
(`neos859080`, proven infeasible) and gets 3/5 badly wrong incumbents within
budget (`markshare2`: 231 vs. true optimum 1; `pk1`: 44 vs. 11). This, not
warm-starting or further cuts, is the benchmark gap that should govern what
gets built next for MILP.

**Root Gomory mixed-integer (GMI) cuts** (`docs/architecture/MILP.md`
§2.2) were built directly against this gap — cover cuts only separate
binary knapsack rows, and `pk1`/`gen-ip002`/`gen-ip054` generate zero
cover cuts for a structural reason (no binary variables, or non-
knapsack-shaped rows). GMI cuts separate from the tableau instead, so
they apply regardless of row shape or variable domain. `IMPLEMENTED`,
correctness `MEASURED` (two hand-derived unit tests verifying exact cut
coefficients by direct substitution, one per sign case — the upper-
bound-resting case caught and fixed a real sign bug during development).
Benchmark KPI gate: **not cleared**. `MEASURED` on the same 5-instance
set, single process, 60 s/instance: 3/5 instances get a *worse* final
gap or incumbent, `neos859080` regresses from a certified proof (0.87 s)
to a timeout, and `gen-ip054` hits `NUMERICAL_FAILURE`, root-caused to a
floating-point-noise coefficient (`4.5e-16` next to `O(1)` terms) and
unbounded cut coefficient magnitude — both well-documented cutting-plane
numerical hazards with no cleanup implemented yet. `KNOWN LIMITATION`,
default off (`MilpSolverOptions::enable_root_gmi_cuts = false`), exactly
the same KPI-gate discipline already applied to warm-started node
relaxations below. Full account in `docs/architecture/MILP.md` §2.2.

`ENGINEERING DECISION` (historical): branch-and-bound was intentionally not
built in this document's original pass because the benchmark set in question
(Netlib) is pure LP. It was subsequently built anyway, ahead of item 1 above
in this document's own priority order (warm-started dual simplex, which was
still unimplemented at the time) — a deviation from the stated sequencing,
recorded here rather than left implicit.

---

## Phase 6 — GPU specialization

| item | status |
|---|---|
| Device-resident PDLP, fused kernels | `IMPLEMENTED` — `src/cuda/PdlpKernels.cu` |
| Sync-free inner loop | `MEASURED` — **127 iterations per host synchronize** vs simplex's 1 |
| Diagonal preconditioning | `IMPLEMENTED` — Ruiz + Pock–Chambolle |
| Adaptive step size | `IMPLEMENTED`, `MEASURED`, **net positive in HYBRID** — see below |
| Adaptive restarts | `IMPLEMENTED` — Applegate sufficient/necessary/artificial |
| GPU pricing inside simplex | `MEASURED` at 3–5× **slower**; retained only behind a flag |
| Feasibility polishing | **NOT IMPLEMENTED** |
| CUDA Graphs | **NOT IMPLEMENTED** |
| SELL-C-σ / HYB / custom SpMV formats | **NOT IMPLEMENTED** — only CSR measured |
| Nsight Systems / Compute profiling | **NOT DONE** |

### Adaptive step size — implemented, measured, not adopted by default

`MEASURED`. Applegate et al. 2021 §3.1, implemented entirely on-device: η lives
in device memory, the accept/reject test is a reduction plus a small kernel, and
a rejected step leaves the iterate untouched so the next queued iteration *is*
the retry. No host synchronization anywhere — which refutes this repository's
earlier claim that the rule "needs a host decision every iteration".

Result on the Netlib feasible set under 2,500 rows:

| | fixed η | adaptive η |
|---|---|---|
| iterations | 1,854,720 | **1,451,008** (−22%) |
| time | **110.35 s** | 120.90 s (+10%) |
| converged | 20 / 26 | 20 / 26 |
| within 1e-6 of optimum | 11 | 12 |

Large per-instance swings in both directions: `maros-r7` 8,448 → **4,608**
iterations (2.3× faster), `ganges` 394,752 → **167,168**; but `stocfor3`
207,616 → **242,176** and `scfxm3` 51,712 → **80,128**.

`ENGINEERING DECISION`: kept and left **on**, and the justification is now
stronger than when this was written. Once the first-order path can *lead*
(`LP.md` §7) its behaviour on large models is on the critical path, and a direct
A/B over the whole Netlib set says:

| | adaptive on | adaptive off |
|---|---|---|
| solved | **93 / 93** | 92 / 93 |
| total | **72.59 s** | 88.86 s |

So it buys an instance *and* 1.22× wall clock in the configuration that ships.
The earlier "+10% and net negative" finding was measured with PDLP standing
alone at `eps = 1e-6`, which is not how the engine uses it.

`KNOWN LIMITATION`: the adaptive rule does **not** fix the six instances where
PDLP stalls (`pilot.we`, `pilot.ja`, `bnl2`, `greenbea`, `greenbeb`, `pilot`).
`greenbea` still ends at KKT 3.24 after 478,464 iterations. Whatever causes
those stalls, it is not the step size. Feasibility polishing is the untested
hypothesis; it has not been implemented, so nothing is claimed for it.

---

## What the honest GPU verdict is

`MEASURED`. Three separate questions, three different answers:

1. **GPU inside the simplex — rejected.** 3–5× slower, for a structural reason
   (a host decision per iteration) that no kernel work can reach.
2. **GPU as a synchronization argument — supported.** 127 iterations per host
   round trip, exactly as designed.
3. **GPU as a wall-clock win — depends entirely on scale.** On Netlib it
   loses: 4.4× slower than the simplex below 2,500 rows, 1.56× faster above it.
   On the **Kennington set it wins decisively** — see below.

Per-iteration cost is near-flat across a 20× range of nonzeros (57 µs at 7,777
nnz, 111 µs at 144,848), the signature of latency-bound execution. That is why
the small set loses: the device is mostly idle.

### The scale hypothesis — TESTED, and it holds

An earlier revision of this document recorded, as an untested `RESEARCH
HYPOTHESIS`, that the flat cost curve implied headroom at larger scale, and
asserted that **no model in this repository was large enough to test it**.

**That assertion was wrong.** 21 models were being skipped by `validate_netlib`
— not for size, but because `netlib_readme.txt` carries no reference objective
for them. They are the Kennington families and the QAP relaxations, and they are
far larger than anything in the validated set: `ken-18` has 105,127 rows,
`osa-60` has 1,397,793 nonzeros. Excluding the hardest models because they were
awkward to score is precisely the kind of coverage gap that makes an aggregate
look better than the engine is.

`benchmarks/validate_crossmethod.cpp` now solves them with both methods and
checks the objectives against each other. `MEASURED`, 60 s budget per method:

| | result |
|---|---|
| solved | **21 / 21** |
| objective disagreements | **0** |
| both methods solved (16) | simplex 106.56 s vs first-order 44.53 s — **2.39× faster** |
| solved only by the first-order path | 5, within the 60 s simplex budget |

| instance | rows | nnz | simplex | first-order | |
|---|---|---|---|---|---|
| `ken-11` | 14,694 | 49,058 | 21.67 s | **1.03 s** | 21.0× |
| `pds-10` | 16,558 | 106,436 | 22.29 s | **1.46 s** | 15.3× |
| `osa-60` | 10,280 | 1,397,793 | 14.52 s | **1.93 s** | 7.5× |
| `ken-18` | 105,127 | 358,171 | *budget* | **12.63 s** | solves what simplex could not |
| `pds-20` | 33,874 | 230,200 | *budget* | **6.94 s** | " |
| `ken-13` | 28,632 | 97,246 | *budget* | **3.84 s** | " |

`KNOWN LIMITATION`: the five "first-order only" results are relative to a **60 s
simplex budget**, not to a simplex that ran to completion. They show the GPU path
reaching an answer far sooner, not that the simplex cannot get there eventually.

`KNOWN LIMITATION`: cross-method agreement is weaker than a published optimum. It
shows two methods sharing no arithmetic reached the same point and both cleared
the original-space gate; it cannot prove that point optimal.

So H1's wall-clock claim **is supported at scale**, on real industrial-structured
models rather than by extrapolation. Layer D of the benchmark plan (a
refinery-style generator with controllable difficulty) still does not exist, so
no refinery-*specific* claim is admissible.

---

## Next, in the roadmap's own priority order

Items 1 and 5 from the prior revision of this list — warm-started dual
simplex and a minimal correct MILP B&B — are both now built. Warm-starting
was measured end to end (`docs/architecture/LP.md` §8) and did not clear the
KPI gate, so it ships off by default; the B&B exists but is not yet
benchmark-ready, which is now the top item below.

1. **Close the MILP benchmark gap — still open.** 1/5 certified and 3/5
   badly wrong incumbents on the 5-instance MIPLIB set within a 60 s
   budget (`reports/runs/2026-08-25/miplib-raw.txt`) is the real open MILP
   item — ahead of any further heuristics or symmetry work. Diagnosed
   (three independent passes: instance-difficulty research, cover-cut
   correctness audit, B&B bound-tracking audit — all in this repository's
   history, no fresh citation needed here): `markshare2`'s badness is
   *expected* — it is from Cornuéjols & Dawande's 1999 "hard small 0-1
   programs" paper, deliberately constructed to defeat conventional B&B;
   no cut or branching fix reaches it without a Feasibility-Pump-class
   method or a lattice/GCD reformulation. Cover-cut generation and B&B
   bound-tracking were both audited line-by-line and are correct — no bug.
   `pk1`/`gen-ip002`/`gen-ip054` generate **zero** cover cuts (`gen-ip*`
   have no binary variables at all; `pk1`'s precedence-shaped constraints
   don't trigger the cover condition either) — root GMI cuts
   (`docs/architecture/MILP.md` §2.2) were built directly against this,
   are correct (hand-verified), but **did not clear the KPI gate**: 3/5
   instances got worse, one regressed from certified to timeout, and one
   hit `NUMERICAL_FAILURE`. Root-caused precisely (not guessed) and
   fixed with three MEASURED-motivated numerical filters
   (`docs/architecture/MILP.md` §2.2.1: a fractionality floor bounding
   1/f_r amplification, a relative near-zero cleanup, and a dynamic-
   range/magnitude cap) — `NUMERICAL_FAILURE` is gone, but **the KPI
   gate is still not cleared** on re-measurement: aggregate incumbent
   quality is no better than before the fix, and worse on `pk1`
   specifically. `enable_root_gmi_cuts` stays off by default. This line
   of work is PAUSED, not abandoned: the next hypothesis (cut density,
   not numerical safety, per §2.2.1's closing paragraph) is recorded but
   unattempted. Per this document's own priority order, the next
   iteration moves to item 2 below rather than a third pass at MILP
   cuts against a set that includes a deliberately adversarial instance
   (`markshare2`) no cut approach reaches.
2. ~~Peak RSS / VRAM capture and repeated-run medians~~ — **done.** Both
   Phase 0 gaps now `IMPLEMENTED`, `MEASURED` (see the Phase 0 table
   above): `src/bench/ResourceSnapshot.{hpp,cpp}` (shared, deduplicated
   from what was previously two identical copies in
   `validate_netlib.cpp`/`bench_miplib.cpp`), `validate_netlib`'s new
   optional repeat-count argument, and a full 90-instance/3-repeat sweep
   (`docs/measurements/netlib-hybrid-20000rows-repeats3.jsonl`) that
   MEASURED 100% bit-identical determinism across every repeat of every
   instance. **Now the top item: item 3 below.**
3. **Generated adversarial LPs — done; Compute Sanitizer — still open.**
   400 generated instances across 6 categories (feasible/bounded,
   ill-conditioned, degenerate, infeasible×2, unbounded), all with a
   status known by construction, independently re-verified in the test
   itself; found and fixed one real bug (in the generator, not the
   solver — see the Phase 1 table above for the full account). Compute
   Sanitizer (memcheck/racecheck against a GPU-exercising benchmark) was
   deliberately deferred, not attempted — explicitly recorded as **now
   the top item** rather than silently dropped.
4. **Hyper-sparse FTRAN** (BTRAN is now done — `docs/architecture/LP.md`
   §9), then Markowitz/AMD ordering and presolve expansion.
5. Feasibility polishing for the six stalling PDLP instances.
6. Layer D refinery generator — without it, no refinery claim is admissible.
7. Resume MILP cut work (§1 above, PAUSED not abandoned) with the
   density-based hypothesis, once items 2-3 have made further correctness
   infrastructure progress.

The governing rule stands: *no optimization is accepted unless it improves a
declared benchmark KPI without reducing correctness or solvability.* Two changes
during this work were reverted under exactly that rule — window-granularity
adaptive step size, and concurrent simplex/PDLP racing.
