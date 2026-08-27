# Raw measurement outputs

Unedited stdout from the benchmark and validation runs that the claims in
`docs/architecture/` and `docs/research/SOTA.md` rest on. They are checked in so
that every quoted number can be traced to the run that produced it.

| file | command | what it establishes |
|---|---|---|
| `netlib-validation-20000rows.txt` | `validate_netlib data/netlib_lp 20000` | 92/93 pass at the 20,000-row cap (`NUMERICS.md` §3.2) |
| `pdlp-vs-cpu-under-2500rows.txt` | `bench_pdlp data/netlib_lp/feasible 2500 1e-6 256 40 700 2500` | GPU PDLP is 4.4× slower than the CPU simplex below 2,500 rows (`PDLP.md` §5) |
| `pdlp-vs-cpu-2500-20000rows.txt` | `bench_pdlp data/netlib_lp/feasible 20000 1e-6 256 60 2500 20000` | GPU PDLP is 1.56× faster from 2,500–20,000 rows, and solves `dfl001` where the simplex fails (`PDLP.md` §5) |

| `netlib-hybrid-20000rows.jsonl` | `validate_netlib data/netlib_lp/feasible data/netlib_readme.txt 20000 presolve hybrid <out.jsonl>` | 93/93 with `LpMethod::HYBRID`, with full reproducibility metadata (`NUMERICS.md` §3.2) |

| `crossmethod-kennington.jsonl` | `validate_crossmethod data/netlib_lp/feasible data/netlib_readme.txt 60 presolve <out.jsonl>` | 21/21 Kennington + QAP solved, 0 objective disagreements; first-order 2.39× faster where both solved (`ROADMAP_STATUS.md`) |

| `../highs_comparison.md` (+ `reports/highs_comparison/*`) | `benchmarks/compare_highs.py both --time-limit 60` | Head-to-head against HiGHS 1.15.1 built from source: HiGHS 7.25× faster in aggregate on 89 agreeing Netlib LPs; mixed 2-2-1 result on the 5-instance MIPLIB set; one objective disagreement (`e226`) diagnosed as a real, narrow MPS-format gap in this repo's parser (objective-row RHS constant not implemented), not a solver defect |

## Structured records

`validate_netlib` takes an optional sixth argument: a path for a **JSON Lines**
file. The first line is a header object carrying the roadmap's Phase 0
reproducibility fields — git commit and dirtiness, compiler, CUDA version, GPU
name and compute capability, driver, CPU model, RAM, thread count, OpenMP
schedule, and the full solver configuration. Every following line is one
instance, including its **FNV-1a 64 content hash**, so a record can be tied to
the exact file that produced it.

JSONL rather than one JSON document, and flushed per record, so a sweep killed
by a time limit still leaves valid parseable data for everything it finished.

## The one rule these files exist to enforce

**Every timing here was taken in a single process with nothing else running.**

That rule is not a stylistic preference. An earlier round of PDLP measurements
was taken with builds and test suites running concurrently, each spawning 16
OpenMP threads on a 16-core machine. Because pricing loops above
`kParallelNnzThreshold` genuinely parallelize, the contamination landed almost
entirely on the CPU column and produced a reported "39× GPU speedup on `degen3`"
that clean re-measurement showed to be a **tie** — and an "8× speedup on
`d2q06c`" that was really a **2.8× loss**. The conclusion drawn from those
numbers (that the GPU path wins on degenerate models) was wrong in mechanism as
well as magnitude.

`PDLP.md` §5 records the full account. The short version: on this machine
concurrent benchmark processes are not merely noisy, they are *actively
misleading*, because oversubscription hits size-gated parallel paths far harder
than serial ones — so the error has a consistent sign and looks like a result.

Anything added here must state its command line and must have been run alone.
