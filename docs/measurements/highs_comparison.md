# Head-to-head against HiGHS

`MEASURED`. HiGHS ([ERGO-Code/HiGHS](https://github.com/ERGO-Code/HiGHS),
MIT licence) is one of the strongest openly available LP/MIP solvers and
is already analyzed architecturally in `docs/research/SOTA.md` §1.1. This
document is the direct, empirical follow-up: build the real upstream
project from source, run it through its own public CLI on the exact same
instances this repo already benchmarks itself against, and report the
result without editing a single line of HiGHS.

**Method, stated up front so every number below is reproducible:**

- HiGHS **1.15.1**, commit `73cac48`, built unmodified from source
  (`cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DFAST_BUILD=ON`) into
  `external/HiGHS/build` — gitignored, never committed, never edited.
  Invoked only via its own CLI: `bin/highs --presolve on --time_limit T
  model.mps`.
- SIHPS at commit `4f83aa2`, its normal shipped configuration
  (`LpMethod::HYBRID` + presolve for LP; default `MilpSolverOptions` —
  cover cuts on, GMI cuts off — for MIP), invoked through its own
  existing benchmark binaries (`validate_netlib`, `bench_miplib`), not a
  special-cased path.
- Both solvers parse the **same original `.mps` files independently** —
  no shared intermediate format — so this also exercises each project's
  own parser, the same way an external user would experience either tool.
- Single process, nothing else running, this machine (RTX 3050 Laptop,
  16 threads) — the same discipline `docs/measurements/README.md`
  documents as load-bearing for every other number in this repo.
- Orchestration script: `benchmarks/compare_highs.py` (Python — the one
  role prompt.md permits it: benchmarking orchestration, never linked
  into or wrapped by the C++ engine). Reproduce with:
  `python3 benchmarks/compare_highs.py both --time-limit 60`.
- Raw outputs: `reports/highs_comparison/{lp,mip}_comparison.csv`,
  `{lp,mip}_summary.json`, `ours_lp.jsonl`.

---

## LP: Netlib feasible set, 90 instances with a published reference

| | ours (HYBRID) | HiGHS |
|---|---|---|
| Solved OPTIMAL | 90/90 | 90/90 |
| Objective agreement (rel. err < 1e-5) | 89/90 (see below) | |
| Total wall-clock, 89 agreeing instances | 92.851 s | **12.800 s** |
| Per-instance wins | 11 | **78** |

**Verdict: HiGHS is faster — 7.25x in aggregate wall-clock, and faster on
78 of 89 instances.** This is not a surprising result and is reported as
such: HiGHS's simplex implementation (Huangfu & Hall 2018) represents
roughly two decades of continuous, heavily profiled engineering by a
dedicated team; this project's simplex is younger and has not yet
completed several items already on its own roadmap (hyper-sparse FTRAN,
Markowitz/AMD ordering — `docs/ROADMAP_STATUS.md` items 4). No claim of
LP speed superiority is made or implied by anything in this repository.

**The one objective disagreement, diagnosed, not hand-waved:**
`e226` — ours reports `-18.751929130010165` (agrees with
`data/netlib_readme.txt`'s published `-1.8751929066E+01` to 1e-8 relative
error — this is why `validate_netlib` marks it PASS in its own,
independent check); HiGHS reports `-11.638929066`. The difference is
exactly `7.113`, which is the RHS value on `e226.mps`'s objective (`N`)
row. Checked directly against HiGHS's own source
(`highs/io/HMpsFF.cpp:1081`, `obj_offset = -val`): **HiGHS implements the
standard MPS convention that an RHS entry on the objective row is a
constant term, added into the reported objective.** This repo's own MPS
reader does not implement that convention — an existing, already-flagged
gap (`src/io/MpsReader.cpp`'s COLUMNS/RHS handling has carried the
comment "objective constant shift; not used by anything yet" since
before this comparison). A file-by-file scan of the entire feasible set
found `e226` is the **only** instance with a nonzero objective-row RHS
(three `grow*` instances have an explicit but harmless `0.`) — so this is
a narrow, single-instance MPS-format completeness gap, not a systemic
correctness defect, and it happens that netlib's own published reference
for `e226` matches THIS repo's (no-offset) convention rather than HiGHS's
(with-offset) convention, which is itself informative about how the
original Netlib reference values were computed. `KNOWN LIMITATION`,
concrete and actionable: implement the objective-row RHS-as-constant
convention in `MpsReader.cpp` to match the MPS standard (and HiGHS).

---

## MILP: the 5-instance MIPLIB set, 60 s budget

| instance | reference | ours (obj / status) | HiGHS (obj / status) | who's ahead |
|---|---|---|---|---|
| `gen-ip002` | -4783.73 | **-4783.73** / TIME_LIMIT | -4772.26 / TIME_LIMIT | **ours** — found the true optimum as an incumbent; HiGHS did not |
| `gen-ip054` | 6840.97 | 6857.87 / TIME_LIMIT | 6858.26 / TIME_LIMIT | roughly tied, ours marginally closer |
| `markshare2` | 1 | 231 / TIME_LIMIT | **41** / TIME_LIMIT | **HiGHS**, by a wide margin |
| `neos859080` | infeasible | **INFEASIBLE, 0.87 s** | Infeasible, 1.29 s | **ours** — same correct proof, faster |
| `pk1` | 11 | 44 / TIME_LIMIT | **14** / TIME_LIMIT | **HiGHS**, clearly closer to true optimum |

Neither solver **certifies** (proves via an exhausted tree) any of the 5
within the 60 s budget — both report a time-limited incumbent or, for
`neos859080`, an actual proof of infeasibility.

**Verdict, stated exactly as measured, on a 5-instance sample too small
to generalize from:** 2 of 5 instances favor this repo's solver
(`gen-ip002`'s incumbent quality, `neos859080`'s proof speed), 2 of 5
favor HiGHS decisively (`markshare2`, `pk1` — both instances where HiGHS's
mature cut/heuristic arsenal clearly outperforms this repo's narrower
one), and 1 is close to a tie. This is a genuinely mixed result, not a
win or a loss in aggregate, and `docs/ROADMAP_STATUS.md`'s own diagnosis
of *why* — `markshare2` is a deliberately adversarial instance from
Cornuéjols & Dawande 1999, `pk1`/`gen-ip002` get zero cover cuts from this
solver's current (narrow) cut arsenal — is unchanged by this comparison;
if anything it is corroborated by an external, independent yardstick.

---

## Where this leaves the "are we better than Gurobi/HiGHS" question

**Not yet, in aggregate, on either LP speed or MILP certification.** That
is the honest answer this measurement supports. What it also supports:
this repo's solver is not merely "working" in some weak sense — it
matches HiGHS's LP objectives to 1e-5 on 89/90 real Netlib instances
(with the one disagreement fully explained above, not a solver defect),
proves the same infeasibility HiGHS proves and does so faster on the one
instance tested, and on one general-integer instance found a *better*
incumbent than HiGHS did in the same wall-clock budget. Those are
specific, narrow, real results — not a basis for a general superiority
claim, which prompt.md's own rules forbid asserting from a 5-instance
MIP sample or a single-machine LP sweep.

The gap that matters most, per `docs/ROADMAP_STATUS.md`'s own priority
order, remains the MILP cut/heuristic arsenal (`markshare2`, `pk1`) and
LP-core sparse linear algebra completeness (hyper-sparse FTRAN, ordering)
— this comparison sharpens the evidence for both, it does not change
which item is next.
