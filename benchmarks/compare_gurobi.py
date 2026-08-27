#!/usr/bin/env python3
"""Three-way LP comparison: this repo's solver, HiGHS, and Gurobi.

BENCHMARKING ORCHESTRATION ONLY (prompt.md's one sanctioned Python role,
same as compare_highs.py). gurobipy is installed in a throwaway venv
(external/gurobi-venv, gitignored, never linked into the C++ solver) and
used only as an external, independent oracle -- exactly like HiGHS's own
native CLI in compare_highs.py.

LICENSE, stated precisely: gurobipy ships with an automatic, no-
registration "restricted license - for non-production use only", capped
at 2000 variables / 2000 constraints (Gurobi's own published limit, not
worked around here). This script restricts itself to instances at or
under that cap and reports how many of the full set that covers. A
larger-scale comparison would need an academic or evaluation license,
which requires the user's own registration -- not attempted here.

Reuses the SAME instances and the SAME already-measured ours/HiGHS
numbers from reports/highs_comparison/lp_comparison.csv (produced by
compare_highs.py) rather than re-running those solves, so this script
only adds the third column.

Usage:
    external/gurobi-venv/bin/python benchmarks/compare_gurobi.py [--time-limit 60]
"""

import argparse
import csv
import json
import time
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

REPO_ROOT = Path(__file__).resolve().parents[1]
LP_DIR = REPO_ROOT / "data" / "netlib_lp" / "feasible"
OURS_HIGHS_CSV = REPO_ROOT / "reports" / "highs_comparison" / "lp_comparison.csv"
OUT_DIR = REPO_ROOT / "reports" / "gurobi_comparison"
GUROBI_SIZE_CAP = 2000  # rows AND cols, the free restricted license's own limit


def solve_with_gurobi(mps_path, time_limit):
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    model = gp.read(str(mps_path), env=env)
    model.Params.TimeLimit = time_limit
    model.Params.Threads = 0  # let Gurobi choose, same "let it use what it wants" spirit as HiGHS run
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
    env.close()
    return {"status": status, "objective": objective, "seconds": secs}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--time-limit", type=float, default=60.0)
    args = p.parse_args()

    if not OURS_HIGHS_CSV.exists():
        print(f"missing {OURS_HIGHS_CSV} -- run benchmarks/compare_highs.py lp first", flush=True)
        return 2

    with open(OURS_HIGHS_CSV) as f:
        rows = list(csv.DictReader(f))

    in_cap = [r for r in rows if int(r["rows"]) <= GUROBI_SIZE_CAP and int(r["cols"]) <= GUROBI_SIZE_CAP]
    out_of_cap = len(rows) - len(in_cap)
    print(
        f"Gurobi restricted (no-registration) license cap: {GUROBI_SIZE_CAP}x{GUROBI_SIZE_CAP}. "
        f"{len(in_cap)}/{len(rows)} instances fit ({out_of_cap} excluded -- would need a real license)."
    )
    print(f"Gurobi: {gp.gurobi.version()}\n", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_rows = []
    for r in in_cap:
        name = r["instance"]
        mps_path = LP_DIR / f"{name}.mps"
        if not mps_path.exists():
            continue
        g = solve_with_gurobi(mps_path, args.time_limit)
        out_rows.append(
            {
                "instance": name,
                "rows": r["rows"],
                "cols": r["cols"],
                "ours_status": r["ours_status"],
                "ours_objective": r["ours_objective"],
                "ours_seconds": r["ours_seconds"],
                "highs_status": r["highs_status"],
                "highs_objective": r["highs_objective"],
                "highs_seconds": r["highs_seconds"],
                "gurobi_status": g["status"],
                "gurobi_objective": g["objective"],
                "gurobi_seconds": g["seconds"],
            }
        )
        print(
            f"{name:>14} ours={r['ours_status']:>10} {float(r['ours_objective']):>16.6f} "
            f"{float(r['ours_seconds']):>8.4f}s | highs={r['highs_status']:>10} "
            f"{float(r['highs_objective']):>16.6f} {float(r['highs_seconds']):>8.4f}s | "
            f"gurobi={g['status']:>18} {(g['objective'] or float('nan')):>16.6f} {g['seconds']:>8.4f}s",
            flush=True,
        )

    csv_path = OUT_DIR / "lp_comparison_3way.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()) if out_rows else [])
        w.writeheader()
        w.writerows(out_rows)

    def agree_all_three(row):
        try:
            o, h, g = float(row["ours_objective"]), float(row["highs_objective"]), float(row["gurobi_objective"])
        except (TypeError, ValueError):
            return False
        if row["ours_status"] != "OPTIMAL" or row["highs_status"] != "Optimal" or row["gurobi_status"] != "Optimal":
            return False
        denom = 1.0 + abs(g)
        return abs(o - g) / denom < 1e-5 and abs(h - g) / denom < 1e-5

    agreeing = [r for r in out_rows if agree_all_three(r)]
    ours_total = sum(float(r["ours_seconds"]) for r in agreeing)
    highs_total = sum(float(r["highs_seconds"]) for r in agreeing)
    gurobi_total = sum(float(r["gurobi_seconds"]) for r in agreeing)
    ours_wins = sum(1 for r in agreeing if float(r["ours_seconds"]) < float(r["highs_seconds"]) and float(r["ours_seconds"]) < float(r["gurobi_seconds"]))
    fastest_counts = {"ours": 0, "highs": 0, "gurobi": 0}
    for r in agreeing:
        times = {"ours": float(r["ours_seconds"]), "highs": float(r["highs_seconds"]), "gurobi": float(r["gurobi_seconds"])}
        fastest_counts[min(times, key=times.get)] += 1

    summary = {
        "gurobi_version": str(gp.gurobi.version()),
        "gurobi_license": "restricted (no-registration, free trial), non-production, <=2000x2000",
        "instances_in_cap": len(in_cap),
        "instances_total": len(rows),
        "three_way_objective_agreement": len(agreeing),
        "total_seconds_ours": ours_total,
        "total_seconds_highs": highs_total,
        "total_seconds_gurobi": gurobi_total,
        "fastest_count": fastest_counts,
        "ours_beats_both_count": ours_wins,
    }
    with open(OUT_DIR / "lp_summary_3way.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 110)
    print(f"3-way agreement (all OPTIMAL, objectives within 1e-5 of each other): {len(agreeing)}/{len(out_rows)}")
    if agreeing:
        print(f"Total wall-clock: ours {ours_total:.3f}s | HiGHS {highs_total:.3f}s | Gurobi {gurobi_total:.3f}s")
        print(f"Fastest-per-instance counts: {fastest_counts}")
        print(f"Instances where ours beat BOTH HiGHS and Gurobi: {ours_wins}/{len(agreeing)}")
    print(f"CSV:  {csv_path}")
    print(f"JSON: {OUT_DIR / 'lp_summary_3way.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
