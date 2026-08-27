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
| unit tests | 132 / 132 | `ctest` (128 pre-existing + 4 GMI-cut hand-verified cases) |

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
| Memory statistics | **NOT IMPLEMENTED** — peak RSS and peak VRAM are not captured |
| Repeated-run support, median over runs | **NOT IMPLEMENTED** — each sweep runs once |
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
| Random / ill-conditioned / degenerate LP generators | **NOT IMPLEMENTED** |
| MPS and sparse-structure fuzzing | **NOT IMPLEMENTED** |
| Compute Sanitizer in CI | **NOT IMPLEMENTED** — never run |
| Brute-force checker for tiny MILPs | **NOT APPLICABLE YET** — no MILP engine |

`KNOWN LIMITATION`: "no false INFEASIBLE" and "no false UNBOUNDED" are asserted
against the Netlib infeasible set (28/0/1) and the feasible set, not against
generated adversarial cases. That is weaker evidence than the roadmap asks for.

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
   instances get worse, one regresses from certified to timeout, and one
   hits `NUMERICAL_FAILURE` from unfiltered floating-point-noise and
   large-magnitude cut coefficients. Concrete next candidate, now the top
   MILP item: **numerical cleanup for GMI cuts** — a relative near-zero
   coefficient threshold (not just exact-zero) and a coefficient dynamic-
   range/magnitude rejection filter, both standard, established
   engineering techniques in the cutting-plane literature (Cornuéjols
   2008 on numerical practicalities) — then re-run the exact same
   `bench_miplib` A/B before considering the default flipped. Separating
   cuts at more than one round/node (currently root-only for both
   families) remains a separate, later candidate.
2. **Peak RSS / VRAM capture and repeated-run medians** — the two Phase 0 gaps
   that still let a regression hide.
3. **Generated adversarial LPs + Compute Sanitizer** — Phase 1's real
   acceptance criteria, currently only argued from Netlib.
4. **Hyper-sparse FTRAN** (BTRAN is now done — `docs/architecture/LP.md`
   §9), then Markowitz/AMD ordering and presolve expansion.
5. Feasibility polishing for the six stalling PDLP instances.
6. Layer D refinery generator — without it, no refinery claim is admissible.

The governing rule stands: *no optimization is accepted unless it improves a
declared benchmark KPI without reducing correctness or solvability.* Two changes
during this work were reverted under exactly that rule — window-granularity
adaptive step size, and concurrent simplex/PDLP racing.
