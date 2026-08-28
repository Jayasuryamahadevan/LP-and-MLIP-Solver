# MIPLIB 2017 broad diversity benchmark subset

24 instances downloaded live from MIPLIB 2017's own collection listing
(`https://miplib.zib.de/tag_collection.html`), plus `miplib2017-v36.solu`,
the same official solution/reference file used by
`data/miplib2017_small`'s runner (already covers all 1065 collection
instances by name, not just the original 5).

## Why this set exists

`data/miplib2017_small`'s original 5 instances are too narrow a base to
support any claim about correctness "across problem domains" — they
happen to share a fairly narrow structural profile (general-integer or
simple mixed-binary knapsack-style rows). This set was chosen
specifically for **structural diversity**, filtered from MIPLIB's own
`easy`-tagged, small (≤250 variables/rows) instances and selected to span
distinct MIPLIB structure tags: `set_covering`, `set_partitioning`,
`set_packing`, `knapsack`/`integer_knapsack`/`invariant_knapsack`,
`equation_knapsack`, `general_linear`, `precedence`, `cardinality`,
`variable_bound`, `aggregations`, `binpacking`, and — deliberately —
several `infeasible`-tagged instances (`flugplinf`, `g503inf`,
`stein9inf`, `enlight4`), since infeasibility detection is a distinct
correctness path from optimum-finding and deserves its own coverage.

## What this set already found

Building and validating against this set directly caught and fixed three
real correctness bugs in the row-feasibility verification gate shared by
the LP and MILP layers (`src/lp/LpSolver.cpp`, `src/lp/Simplex.cpp`,
`src/milp/MilpSolver.cpp`) — most notably, `flugplinf` (an intentionally
infeasible instance) was being reported `OPTIMAL`. Full account in
`docs/architecture/MILP.md`'s "Row-feasibility scaling" section and
`docs/ROADMAP_STATUS.md`.

## Provenance

- `https://miplib.zib.de/tag_collection.html` (instance listing, with
  size/tag/status metadata parsed directly from the page's own table)
- `https://miplib.zib.de/WebData/instances/<name>.mps.gz` (per-instance
  download; the same URL pattern `data/miplib2017_small`'s own instances
  came from)

## Running

```text
build/benchmarks/bench_miplib data/miplib2017_broad data/miplib2017_broad/miplib2017-v36.solu
```

Same runner, same 60-second-per-instance default, same certified-vs-
incumbent-only distinction as `data/miplib2017_small`'s own invocation.
