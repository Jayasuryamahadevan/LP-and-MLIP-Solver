#!/usr/bin/env python3
"""Head-to-head comparison against HiGHS (ERGO-Code/HiGHS, MIT licence),
built from source, unmodified, and invoked ONLY through its own public CLI
(`highs [options] file.mps`) -- exactly the same interface any external
user has. This is BENCHMARKING ORCHESTRATION ONLY, the one role prompt.md
permits Python to play in this project (see also compare_reference.py,
which does the equivalent thing via scipy's bundled/vendored HiGHS copy;
this script instead runs the actual upstream release binary, built by
scripts/build_highs.sh into external/HiGHS/build, so results reflect
HiGHS 1.15.1 itself rather than whatever version scipy happens to vendor).

Both solvers read the SAME original .mps file independently -- no shared
intermediate format -- so this also exercises each project's own parser,
which is how a real user would experience either tool.

Usage:
    python3 benchmarks/compare_highs.py lp   [--time-limit 60] [--row-cap 20000]
    python3 benchmarks/compare_highs.py mip  [--time-limit 60]
    python3 benchmarks/compare_highs.py both [--time-limit 60]

Writes CSV + JSON summaries to reports/highs_comparison/.
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HIGHS_BIN = REPO_ROOT / "external" / "HiGHS" / "build" / "bin" / "highs"
DEFAULT_VALIDATE_NETLIB = REPO_ROOT / "build" / "benchmarks" / "validate_netlib"
DEFAULT_BENCH_MILP = REPO_ROOT / "build" / "benchmarks" / "bench_miplib"
DEFAULT_LP_DIR = REPO_ROOT / "data" / "netlib_lp" / "feasible"
DEFAULT_README = REPO_ROOT / "data" / "netlib_readme.txt"
DEFAULT_MIP_DIR = REPO_ROOT / "data" / "miplib2017_small"
DEFAULT_MIP_SOLU = DEFAULT_MIP_DIR / "miplib2017-v36.solu"
OUT_DIR = REPO_ROOT / "reports" / "highs_comparison"


def run(cmd, timeout=None):
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    secs = time.perf_counter() - t0
    return proc.stdout, proc.stderr, secs, proc.returncode


def highs_version(highs_bin):
    out, _, _, _ = run([str(highs_bin), "-v"])
    return out.strip()


def git_commit():
    out, _, _, _ = run(["git", "rev-parse", "--short", "HEAD"])
    dirty_out, _, _, _ = run(["git", "status", "--porcelain"])
    return out.strip() + (" (dirty)" if dirty_out.strip() else " (clean)")


# ---------------------------------------------------------------------------
# LP comparison
# ---------------------------------------------------------------------------

LP_STATUS_RE = re.compile(r"Model status\s*:\s*(.+)")
LP_OBJ_RE = re.compile(r"Objective value\s*:\s*([-\d.eE+]+)")
LP_TIME_RE = re.compile(r"HiGHS run time\s*:\s*([\d.]+)")


def highs_solve_lp(highs_bin, mps_path, time_limit):
    cmd = [
        str(highs_bin),
        "--presolve", "on",
        "--time_limit", str(time_limit),
        str(mps_path),
    ]
    try:
        out, err, wall, rc = run(cmd, timeout=time_limit + 30)
    except subprocess.TimeoutExpired:
        return {"status": "PROCESS_TIMEOUT", "objective": None, "seconds": None}
    status_m = LP_STATUS_RE.search(out)
    obj_m = LP_OBJ_RE.search(out)
    time_m = LP_TIME_RE.search(out)
    return {
        "status": status_m.group(1).strip() if status_m else "UNKNOWN",
        "objective": float(obj_m.group(1)) if obj_m else None,
        # HiGHS's own reported run time (its internal clock); wall is our
        # process-level stopwatch around the whole subprocess, included too
        # since it also captures process launch overhead a real user pays.
        "highs_seconds": float(time_m.group(1)) if time_m else None,
        "wall_seconds": wall,
    }


def load_our_lp_jsonl(jsonl_path):
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("record") == "instance":
                records.append(rec)
    return records


def compare_lp(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ours_jsonl = OUT_DIR / "ours_lp.jsonl"

    print(f"[1/2] Running our solver (validate_netlib, HYBRID+presolve) over {args.lp_dir} ...")
    cmd = [
        str(args.our_validate_netlib),
        str(args.lp_dir),
        str(args.readme),
        str(args.row_cap),
        "presolve",
        "hybrid",
        str(ours_jsonl),
    ]
    out, err, wall, rc = run(cmd, timeout=args.time_limit * 200 + 300)
    print(out.strip().splitlines()[-1] if out.strip() else "(no output)")

    our_records = load_our_lp_jsonl(ours_jsonl)
    print(f"  {len(our_records)} instances with a published reference solved by ours.\n")

    print(f"[2/2] Running HiGHS {highs_version(args.highs_bin)} on the SAME instances ...")
    rows = []
    for rec in our_records:
        name = Path(rec["instance"]).stem
        mps_path = args.lp_dir / f"{name}.mps"
        if not mps_path.exists():
            continue
        h = highs_solve_lp(args.highs_bin, mps_path, args.time_limit)
        rows.append(
            {
                "instance": name,
                "rows": rec.get("rows"),
                "cols": rec.get("cols"),
                "ours_status": rec.get("status"),
                "ours_objective": rec.get("objective"),
                "ours_seconds": rec.get("wall_seconds"),
                "highs_status": h["status"],
                "highs_objective": h["objective"],
                "highs_seconds": h["highs_seconds"],
                "highs_wall_seconds": h["wall_seconds"],
            }
        )
    write_and_report_lp(rows, args)


def write_and_report_lp(rows, args):
    csv_path = OUT_DIR / "lp_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        w.writerows(rows)

    both_optimal = [
        r
        for r in rows
        if r["ours_status"] == "OPTIMAL"
        and r["highs_status"] == "Optimal"
        and r["ours_objective"] is not None
        and r["highs_objective"] is not None
    ]
    agree = 0
    disagree = []
    ours_total = highs_total = 0.0
    ours_wins = highs_wins = 0
    for r in both_optimal:
        denom = 1.0 + abs(r["highs_objective"])
        rel = abs(r["ours_objective"] - r["highs_objective"]) / denom
        if rel < 1e-5:
            agree += 1
            ours_total += r["ours_seconds"]
            highs_total += r["highs_seconds"]
            if r["ours_seconds"] < r["highs_seconds"]:
                ours_wins += 1
            else:
                highs_wins += 1
        else:
            disagree.append((r["instance"], rel))

    our_only = [r for r in rows if r["ours_status"] == "OPTIMAL" and r["highs_status"] != "Optimal"]
    highs_only = [r for r in rows if r["highs_status"] == "Optimal" and r["ours_status"] != "OPTIMAL"]

    summary = {
        "highs_version": highs_version(args.highs_bin),
        "git_commit": git_commit(),
        "instances_compared": len(rows),
        "both_optimal_and_agree": agree,
        "objective_disagreements": disagree,
        "solved_only_by_ours": [r["instance"] for r in our_only],
        "solved_only_by_highs": [r["instance"] for r in highs_only],
        "total_wall_seconds_ours": ours_total,
        "total_wall_seconds_highs": highs_total,
        "per_instance_wins_ours": ours_wins,
        "per_instance_wins_highs": highs_wins,
    }
    with open(OUT_DIR / "lp_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 100)
    print(f"LP: {len(rows)} instances compared against HiGHS {summary['highs_version']}")
    print(f"Both OPTIMAL and objective agrees (rel err < 1e-5): {agree}/{len(rows)}")
    if disagree:
        print(f"  DISAGREEMENTS: {disagree}")
    if our_only:
        print(f"  Solved ONLY by ours: {[r['instance'] for r in our_only]}")
    if highs_only:
        print(f"  Solved ONLY by HiGHS: {[r['instance'] for r in highs_only]}")
    if agree:
        faster = "ours" if ours_total < highs_total else "HiGHS"
        ratio = max(ours_total, highs_total) / max(min(ours_total, highs_total), 1e-9)
        print(
            f"Total wall-clock on the {agree} agreeing instances: "
            f"ours {ours_total:.3f}s vs HiGHS {highs_total:.3f}s "
            f"-> {faster} is {ratio:.2f}x faster in aggregate"
        )
        print(f"Per-instance wins: ours faster on {ours_wins}, HiGHS faster on {highs_wins}")
    print(f"CSV:  {OUT_DIR / 'lp_comparison.csv'}")
    print(f"JSON: {OUT_DIR / 'lp_summary.json'}")
    print("=" * 100)


# ---------------------------------------------------------------------------
# MILP comparison
# ---------------------------------------------------------------------------

OURS_ROW_RE = re.compile(
    r"^(?P<name>\S+)\s+(?P<status>\S+)\s+(?P<obj>[-\d.eE+]+)\s+(?P<ref>[-\d.eE+]+)\s+"
    r"(?P<abserr>[-\d.eE+]+)\s+(?P<nodes>\d+)\s+(?P<lps>\d+)\s+(?P<secs>[\d.]+)"
)
MIP_STATUS_RE = re.compile(r"^\s*Status\s+(.+)$", re.MULTILINE)
MIP_PRIMAL_RE = re.compile(r"^\s*Primal bound\s+(\S+)$", re.MULTILINE)
MIP_DUAL_RE = re.compile(r"^\s*Dual bound\s+(\S+)$", re.MULTILINE)
MIP_GAP_RE = re.compile(r"^\s*Gap\s+(\S+)")
MIP_TIMING_RE = re.compile(r"^\s*Timing\s+([\d.]+)", re.MULTILINE)
MIP_NODES_RE = re.compile(r"^\s*Nodes\s+(\d+)", re.MULTILINE)
LP_ONLY_MIP_STATUS_RE = re.compile(r"Model status\s*:\s*(.+)")  # a MIP with no integer gap printed


def parse_float_maybe_inf(s):
    if s in ("inf", "-inf"):
        return float(s)
    try:
        return float(s)
    except ValueError:
        return None


def highs_solve_mip(highs_bin, mps_path, time_limit):
    cmd = [str(highs_bin), "--presolve", "on", "--time_limit", str(time_limit), str(mps_path)]
    try:
        out, err, wall, rc = run(cmd, timeout=time_limit + 30)
    except subprocess.TimeoutExpired:
        return {"status": "PROCESS_TIMEOUT", "objective": None, "seconds": None, "nodes": None}

    status_m = MIP_STATUS_RE.search(out)
    primal_m = MIP_PRIMAL_RE.search(out)
    timing_m = MIP_TIMING_RE.search(out)
    nodes_m = MIP_NODES_RE.search(out)
    if status_m:
        status = status_m.group(1).strip()
        objective = parse_float_maybe_inf(primal_m.group(1)) if primal_m else None
        seconds = float(timing_m.group(1)) if timing_m else wall
        nodes = int(nodes_m.group(1)) if nodes_m else None
    else:
        # No integer variables detected / pure-LP fallback report format.
        lp_status_m = LP_ONLY_MIP_STATUS_RE.search(out)
        obj_m = LP_OBJ_RE.search(out)
        time_m = LP_TIME_RE.search(out)
        status = lp_status_m.group(1).strip() if lp_status_m else "UNKNOWN"
        objective = float(obj_m.group(1)) if obj_m else None
        seconds = float(time_m.group(1)) if time_m else wall
        nodes = 0
    return {"status": status, "objective": objective, "seconds": seconds, "nodes": nodes, "wall": wall}


def parse_our_miplib_table(stdout_text):
    rows = {}
    for line in stdout_text.splitlines():
        m = OURS_ROW_RE.match(line.strip())
        if not m:
            continue
        rows[m.group("name")] = {
            "status": m.group("status"),
            "objective": float(m.group("obj")),
            "reference": float(m.group("ref")),
            "nodes": int(m.group("nodes")),
            "lp_relaxations": int(m.group("lps")),
            "seconds": float(m.group("secs")),
        }
    return rows


def compare_mip(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/2] Running our solver (bench_miplib, default shipped config) over {args.mip_dir} ...")
    cmd = [
        str(args.our_bench_miplib),
        str(args.mip_dir),
        str(args.mip_solu),
        "",
        str(args.time_limit),
        "reliability",
        "off",
        "off",
    ]
    out, err, wall, rc = run(cmd, timeout=args.time_limit * 20 + 120)
    our_rows = parse_our_miplib_table(out)
    print(f"  parsed {len(our_rows)} instance rows from our output.\n")

    print(f"[2/2] Running HiGHS {highs_version(args.highs_bin)} on the SAME instances ...")
    rows = []
    for name, ours in sorted(our_rows.items()):
        mps_path = args.mip_dir / f"{name}.mps"
        if not mps_path.exists():
            continue
        h = highs_solve_mip(args.highs_bin, mps_path, args.time_limit)
        rows.append(
            {
                "instance": name,
                "reference_objective": ours["reference"],
                "ours_status": ours["status"],
                "ours_objective": ours["objective"],
                "ours_nodes": ours["nodes"],
                "ours_seconds": ours["seconds"],
                "highs_status": h["status"],
                "highs_objective": h["objective"],
                "highs_nodes": h["nodes"],
                "highs_seconds": h["seconds"],
            }
        )
    write_and_report_mip(rows, args)


def write_and_report_mip(rows, args):
    csv_path = OUT_DIR / "mip_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        w.writerows(rows)

    print("\n" + "=" * 120)
    print(
        f"{'instance':>14} {'ref obj':>14} | {'ours':>12} {'ours obj':>16} {'nodes':>8} {'secs':>7} | "
        f"{'HiGHS':>18} {'highs obj':>16} {'nodes':>8} {'secs':>7}"
    )
    print("-" * 120)
    ours_certified = highs_certified = 0
    for r in rows:
        ours_ok = r["ours_status"] == "OPTIMAL"
        highs_ok = r["highs_status"].lower() == "optimal"
        ours_certified += ours_ok
        highs_certified += highs_ok
        print(
            f"{r['instance']:>14} {r['reference_objective']:>14.6g} | "
            f"{r['ours_status']:>12} {r['ours_objective']:>16.6g} {r['ours_nodes']:>8} "
            f"{r['ours_seconds']:>7.2f} | "
            f"{r['highs_status']:>18} {(r['highs_objective'] if r['highs_objective'] is not None else float('nan')):>16.6g} "
            f"{(r['highs_nodes'] if r['highs_nodes'] is not None else -1):>8} {r['highs_seconds']:>7.2f}"
        )
    print("=" * 120)
    print(f"Certified OPTIMAL within {args.time_limit}s: ours {ours_certified}/{len(rows)}, HiGHS {highs_certified}/{len(rows)}")

    summary = {
        "highs_version": highs_version(args.highs_bin),
        "git_commit": git_commit(),
        "time_limit_seconds": args.time_limit,
        "instances_compared": len(rows),
        "ours_certified_optimal": ours_certified,
        "highs_certified_optimal": highs_certified,
    }
    with open(OUT_DIR / "mip_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"CSV:  {OUT_DIR / 'mip_comparison.csv'}")
    print(f"JSON: {OUT_DIR / 'mip_summary.json'}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["lp", "mip", "both"])
    p.add_argument("--time-limit", type=float, default=60.0)
    p.add_argument("--row-cap", type=int, default=20000)
    p.add_argument("--lp-dir", type=Path, default=DEFAULT_LP_DIR)
    p.add_argument("--readme", type=Path, default=DEFAULT_README)
    p.add_argument("--mip-dir", type=Path, default=DEFAULT_MIP_DIR)
    p.add_argument("--mip-solu", type=Path, default=DEFAULT_MIP_SOLU)
    p.add_argument("--highs-bin", type=Path, default=DEFAULT_HIGHS_BIN)
    p.add_argument("--our-validate-netlib", type=Path, default=DEFAULT_VALIDATE_NETLIB)
    p.add_argument("--our-bench-miplib", type=Path, default=DEFAULT_BENCH_MILP)
    args = p.parse_args()

    if not args.highs_bin.exists():
        print(f"HiGHS binary not found at {args.highs_bin}; build it first.", file=sys.stderr)
        return 2

    print(f"git commit: {git_commit()}")
    print(f"HiGHS: {highs_version(args.highs_bin)}  ({args.highs_bin})")
    print()

    if args.mode in ("lp", "both"):
        compare_lp(args)
    if args.mode in ("mip", "both"):
        print()
        compare_mip(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
