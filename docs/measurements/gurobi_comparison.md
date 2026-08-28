# Three-way comparison: SIHPS vs. HiGHS vs. Gurobi

`MEASURED`. Extends `docs/measurements/highs_comparison.md` with a third,
independent reference point. Gurobi is commercial; this comparison uses
`gurobipy`'s own automatic, no-registration **restricted license** ("for
non-production use only"), capped by Gurobi's own published terms at
**2000 variables / 2000 constraints** — not worked around, not a full
license. Reproduce with:

```
uv venv --python 3.11 external/gurobi-venv
uv pip install --python external/gurobi-venv/bin/python gurobipy
external/gurobi-venv/bin/python benchmarks/compare_gurobi.py --time-limit 60
```

Gurobi 13.0.3, invoked only through `gurobipy`'s own model-read/optimize
API on the original `.mps` files — never linked into, wrapped by, or
depended on by this repo's C++ solver, same rule as HiGHS. Raw results:
`reports/gurobi_comparison/lp_comparison_3way.csv` /
`lp_summary_3way.json`.

**Coverage: 63 of the 90 Netlib instances `compare_highs.py` already
measured fit under the 2000×2000 cap; 27 do not** (`ship04l`, `bnl2`,
`d2q06c`, `dfl001`, `pilot87`, `maros-r7`, and 21 others — all mid-to-large
instances). This is a real, stated limitation of this comparison, not
hidden: **a full-scale, all-90-instance Gurobi comparison needs an
academic or evaluation license**, which requires registration this
session cannot perform on the user's behalf. Ours-vs-HiGHS in
`highs_comparison.md` remains the only full-coverage comparison.

## Result

| | ours | HiGHS | Gurobi |
|---|---|---|---|
| Total wall-clock, 61 three-way-agreeing instances | 3.561 s | 0.690 s | **0.460 s** |
| Fastest-per-instance count (of 61) | 1 | 29 | **31** |

**Gurobi is faster than HiGHS here too**, and HiGHS remains well ahead of
this repo's solver. **This repo's solver beat BOTH HiGHS and Gurobi on 1
of 61 instances.** No claim of aggregate superiority over either is made
or supported by this measurement — the honest ranking on raw LP speed, on
this evidence, is Gurobi > HiGHS > this repo's solver, in that order, by a
wide margin against both.

**That one instance, checked in detail rather than left as a bare count,
does not survive scrutiny as a meaningful result.** It is `grow7` (140
rows, 301 cols): ours 4.05–4.18ms, Gurobi 4.6–5.3ms across 5 independent
trials each (both measured with a precise wall-clock timer around the
solve call, so this specific pairwise comparison is real, not noise —
HiGHS's own reported time for this instance is a suspicious, identical
"0.01 s" across 5 separate runs, which is a display/rounding floor in
HiGHS's CLI output, not a real measurement, and was excluded from this
specific claim for exactly that reason). So: yes, a genuine, reproducible
~15-20% speed edge over Gurobi exists on this one 140-row toy instance.
It is also the ENTIRE extent of any LP speed advantage found anywhere in
this comparison — a single small instance where fixed per-call overhead
plausibly dominates for both solvers, not evidence of any general
algorithmic edge. Presented honestly as one real data point, not as
"beating Gurobi" in any competitive sense.

Objective agreement: 61/63 all-three-agree (relative error < 1e-5, all
`OPTIMAL`/`Optimal`). Two exceptions:

- **`e226`**: already fully diagnosed in `highs_comparison.md` — HiGHS
  implements the standard MPS objective-row-RHS-as-constant convention
  and this repo's parser does not (`src/io/MpsReader.cpp`, a named, open
  gap in `docs/ROADMAP_STATUS.md`'s Phase 1 section). **Gurobi agrees
  with HiGHS's value here** (`-11.638929`, not this repo's
  `-18.751929`), a third independent data point corroborating that this
  repo's parser, not the other two, is the outlier on this specific MPS
  convention.
- **`forplan`**: ours and HiGHS agree with each other
  (`-664.218961`); Gurobi disagrees (`-1163.915769`). Checked the obvious
  suspect — `netlib_readme.txt` lists `forplan` among instances with
  "extra free rows," which can confuse a parser that assumes exactly one
  `N` row — but `forplan.mps`'s own `ROWS` section has only one `N` row
  (`OB1PNW20`), so that specific hypothesis does not hold. **Left
  unresolved and stated as such** rather than guessed at further in this
  pass: this is a genuine, open 2-vs-1 discrepancy where the "2" side
  (ours + HiGHS) happens to agree, which is suggestive but not proof of
  which value is actually correct — a `RESEARCH HYPOTHESIS` for a future
  pass, not a conclusion.

## MILP: three-way comparison on the 5-instance MIPLIB set

`MEASURED`. Unlike the LP comparison, all 5 instances in
`data/miplib2017_small` are well under Gurobi's restricted-license
2000x2000 cap (the largest, `neos859080`, is 164 rows x 160 cols) — this
is genuine full-coverage, not a size-filtered subset. Reproduce with
`benchmarks/compare_gurobi_mip.py`; raw results in
`reports/gurobi_comparison/mip_comparison_3way.csv`.

**An initial pass at this comparison overstated its own finding, and this
section corrects that rather than leaving it standing.** The first look
compared only the best-found INCUMBENT value per instance and noted that
on `gen-ip002` ours and Gurobi's incumbents were numerically identical.
Read on its own, that invites the conclusion "we matched Gurobi" — which
is incomplete enough to be misleading. The DUAL BOUND (Gurobi's own
`ObjBound`, this repo's own `best_bound`) is the actual measure of how
rigorously a solver has proven its solution is close to optimal, and
checking it changes the picture:

| instance | ours: obj / gap | Gurobi: obj / gap | verdict |
|---|---|---|---|
| `gen-ip002` | -4783.7333916 / **0.42%** | -4783.7333916 / **0.19%** | Incumbents tie exactly; Gurobi's *proof* is ~2x tighter |
| `gen-ip054` | 6857.8707 / 0.90% | 6840.9656 (= true optimum) / 0.41% | Gurobi clearly better |
| `markshare2` | 231 | 24 (true optimum: 1) | Gurobi's incumbent far better |
| `pk1` | 44 / 80% | **11 / 0%, CERTIFIED OPTIMAL in 4.0 s** | Gurobi fully solves it; ours is 4x off after the full 60 s budget |
| `neos859080` | INFEASIBLE, 0.87 s | Infeasible, 0.115 s | Both correct; trivial for both |

All entries reproduced independently: Gurobi's `gen-ip002` incumbent and
gap were confirmed bit-identical across 3 separate 60-second runs
(`-4783.7333916` every time); this repo's own `gen-ip002` incumbent was
confirmed bit-identical across 3 separate runs of `bench_miplib`.

**Honest verdict: this repo's solver does not solve more than Gurobi on
this set.** `gen-ip002` is a genuine tie on the number found, not a win —
Gurobi's own proof there is stronger. On the other three non-trivial
instances Gurobi is clearly ahead, and on `pk1` it is not close: Gurobi
certifies the true optimum in 4 seconds while this repo's solver is stuck
4x away from it after the full time budget. Framing `gen-ip002` as
"matches Gurobi's solution quality using ~12x fewer nodes" (this repo
explores ~292k nodes there vs. Gurobi's ~3.5M in the same 60 seconds) is
accurate as a per-node-efficiency observation but does not amount to
parity with Gurobi's overall MILP performance, which the dual-bound gap
and the `pk1` result both contradict.

## What this changes about the roadmap

Nothing in `docs/ROADMAP_STATUS.md`'s priority order — this measurement
confirms rather than redirects it. Beating HiGHS (the nearer, better-
characterized target, full 90-instance coverage) remains the stated
near-term goal; Gurobi is a further, harder target beyond that, per the
user's own explicit sequencing. A full-scale Gurobi comparison (all 90+
instances, matching HiGHS's coverage) is a concrete, ready-to-run next
step **if/when an academic or evaluation license is available** — the
harness (`benchmarks/compare_gurobi.py`) already supports it; only the
license-driven size filter would need to be lifted.

**Stated plainly, since an earlier pass at this comparison read more
optimistically than the numbers support: this repo's solver does not
currently beat Gurobi in any competitively meaningful sense, on either LP
or MILP.** The two positive-looking data points found in this pass
(`grow7`'s LP timing, `gen-ip002`'s MILP incumbent) are both real,
reproducible, and both checked in enough depth to know exactly how much
they do and don't mean: `grow7` is a ~15-20% edge on a single 140-row toy
instance, and `gen-ip002` is an incumbent tie where Gurobi's own proof is
still twice as tight. Neither changes the honest ranking Gurobi > HiGHS >
this repo's solver on speed, or Gurobi's clear lead on MILP solution
quality and proof strength on this 5-instance set (`pk1` decisively so).
