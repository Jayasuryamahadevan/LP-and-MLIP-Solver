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
| Total wall-clock, 61 three-way-agreeing instances | 4.724 s | 0.750 s | **0.456 s** |
| Fastest-per-instance count (of 61) | 1 | 27 | **33** |

**Gurobi is faster than HiGHS here too** — roughly 1.6x faster than HiGHS
in aggregate on this subset, and HiGHS is roughly 6.3x faster than this
repo's solver (a smaller multiple than the 7.25x measured on the full
90-instance set in `highs_comparison.md`, consistent with this being a
subset biased toward smaller instances, where this repo's own simplex has
comparatively less ground to lose per solve). **This repo's solver beat
BOTH HiGHS and Gurobi on 1 of 61 instances.** No claim of aggregate
superiority over either is made or supported by this measurement — the
honest ranking on raw LP speed, on this evidence, is Gurobi > HiGHS >
this repo's solver, in that order, by a wide margin against both.

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
