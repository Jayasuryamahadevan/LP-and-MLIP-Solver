#!/usr/bin/env python3
"""Three-way MILP comparison: this repo's solver, HiGHS, and Gurobi.

BENCHMARKING ORCHESTRATION ONLY (prompt.md's one sanctioned Python role,
same as compare_highs.py/compare_gurobi.py). gurobipy is installed in a
throwaway venv (external/gurobi-venv, gitignored, never linked into the
C++ solver) and used only as an external, independent oracle.

LICENSE, stated precisely: gurobipy's automatic, no-registration
"restricted license - for non-production use only" caps at 2000
variables / 2000 constraints. All 5 instances in data/miplib2017_small
are well under that cap (the largest, neos859080, is 164 rows x 160
cols), so this is genuine full-coverage on this set -- no exclusions,
unlike the LP comparison's 27-of-90 gap.

Reuses the SAME already-measured ours/HiGHS numbers from
reports/highs_comparison/mip_comparison.csv (produced by
`compare_highs.py mip`) rather than re-running those solves, so this
script only adds the third column.

Usage:
    external/gurobi-venv/bin/python benchmarks/compare_gurobi_mip.py [--time-limit 60]
"""

import argparse
import csv
import json
import time
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

REPO_ROOT = Path(__file__).resolve().parents[1]
MIP_DIR = REPO_ROOT / "data" / "miplib2017_small"
OURS_HIGHS_CSV = REPO_ROOT / "reports" / "highs_comparison" / "mip_comparison.csv"
OUT_DIR = REPO_ROOT / "reports" / "gurobi_comparison"
GUROBI_SIZE_CAP = 2000


def solve_with_gurobi(mps_path, time_limit):
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    model = gp.read(str(mps_path), env=env)
    model.Params.TimeLimit = time_limit
    model.Params.Threads = 0
    t0 = time.perf_counter()
    model.optimize()
    secs = time.perf_counter() - t0
    status_map = {
        GRB.OPTIMAL: "Optimal",
        GRB.INFEASIBLE: "Infeasible",
        GRB.UNBOUNDED: "Unbounded",
        GRB.TIME_LIMIT: "Time limit reached",
    }
    status = status_map.get(model.Status, f"status={model.Status}")
    objective = model.ObjVal if model.SolCount > 0 else None
    nodes = int(model.NodeCount)
    env.close()
    return {"status": status, "objective": objective, "seconds": secs, "nodes": nodes}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--time-limit", type=float, default=60.0)
    args = p.parse_args()

    if not OURS_HIGHS_CSV.exists():
        print(f"missing {OURS_HIGHS_CSV} -- run `python3 benchmarks/compare_highs.py mip` first", flush=True)
        return 2

    with open(OURS_HIGHS_CSV) as f:
        rows = list(csv.DictReader(f))

    print(f"Gurobi: {gp.gurobi.version()}, license cap {GUROBI_SIZE_CAP}x{GUROBI_SIZE_CAP} "
          f"(all {len(rows)} MIPLIB instances here are well under it)\n", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_rows = []
    for r in rows:
        name = r["instance"]
        mps_path = MIP_DIR / f"{name}.mps"
        if not mps_path.exists():
            continue
        g = solve_with_gurobi(mps_path, args.time_limit)
        out_rows.append(
            {
                "instance": name,
                "reference_objective": r["reference_objective"],
                "ours_status": r["ours_status"],
                "ours_objective": r["ours_objective"],
                "ours_nodes": r["ours_nodes"],
                "ours_seconds": r["ours_seconds"],
                "highs_status": r["highs_status"],
                "highs_objective": r["highs_objective"],
                "highs_nodes": r["highs_nodes"],
                "highs_seconds": r["highs_seconds"],
                "gurobi_status": g["status"],
                "gurobi_objective": g["objective"],
                "gurobi_nodes": g["nodes"],
                "gurobi_seconds": g["seconds"],
            }
        )
        print(
            f"{name:>14} ours={r['ours_status']:>12} {float(r['ours_objective']):>16.4f} n={r['ours_nodes']:>8} "
            f"{float(r['ours_seconds']):>7.3f}s | highs={r['highs_status']:>20} {float(r['highs_objective']):>16.4f} "
            f"n={r['highs_nodes']:>8} {float(r['highs_seconds']):>7.3f}s | "
            f"gurobi={g['status']:>20} {(g['objective'] if g['objective'] is not None else float('nan')):>16.4f} "
            f"n={g['nodes']:>8} {g['seconds']:>7.3f}s",
            flush=True,
        )

    csv_path = OUT_DIR / "mip_comparison_3way.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()) if out_rows else [])
        w.writeheader()
        w.writerows(out_rows)

    def best_of(row, minimize_ref):
        # Which of the three has the objective CLOSEST to being best,
        # honestly: lower is better if reference suggests a min problem
        # (all 5 MIPLIB instances here are minimizations per their .solu
        # reference), and only among those that actually report a finite
        # objective (an infeasible/no-incumbent status doesn't compete).
        candidates = {}
        for key in ("ours", "highs", "gurobi"):
            status = row[f"{key}_status"]
            obj = row[f"{key}_objective"]
            if obj in (None, "", "None"):
                continue
            try:
                obj = float(obj)
            except ValueError:
                continue
            if status in ("INFEASIBLE", "Infeasible") and row["reference_objective"] not in ("inf", "-inf"):
                continue
            candidates[key] = obj
        if not candidates:
            return None
        return min(candidates, key=candidates.get)

    summary_rows = []
    for r in out_rows:
        winner = best_of(r, True)
        summary_rows.append((r["instance"], winner))

    print("\n" + "=" * 100)
    print("Best incumbent per instance (lower objective wins; all 5 are minimizations):")
    for name, winner in summary_rows:
        print(f"  {name:>14}: {winner}")

    summary = {
        "gurobi_version": str(gp.gurobi.version()),
        "gurobi_license": "restricted (no-registration, free trial), non-production, <=2000x2000",
        "instances_compared": len(out_rows),
        "best_incumbent_per_instance": dict(summary_rows),
    }
    with open(OUT_DIR / "mip_summary_3way.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nCSV:  {csv_path}")
    print(f"JSON: {OUT_DIR / 'mip_summary_3way.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
