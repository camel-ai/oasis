#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
from igraph import Graph


def _read_edges(path: Path) -> List[Tuple[int, int]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        edges: List[Tuple[int, int]] = []
        for r in reader:
            try:
                u = int(r["follower_id"])
                v = int(r["followee_id"])
            except Exception:
                continue
            if u == v:
                continue
            edges.append((u, v))
    return edges


def _degree_powerlaw_slope(degrees: List[int], kmin: int = 3) -> float | None:
    vals = [d for d in degrees if d >= kmin]
    if len(vals) < 5:
        return None
    # CCDF
    counts: Dict[int, int] = {}
    for d in vals:
        counts[d] = counts.get(d, 0) + 1
    xs = sorted(counts.keys())
    n = len(vals)
    ccdf = []
    running = 0
    for x in reversed(xs):
        running += counts[x]
        ccdf.append((x, running / n))
    ccdf = list(reversed(ccdf))
    X = np.array([math.log(float(x)) for x, _ in ccdf if x > 0])
    Y = np.array([math.log(float(p)) for _, p in ccdf if p > 0.0])
    if len(X) < 3:
        return None
    slope, _ = np.polyfit(X, Y, deg=1)
    # CCDF ~ x^{-alpha+1} in some definitions; for rough check, we report |slope|
    return abs(float(slope))


def compute_metrics(edges: List[Tuple[int, int]]) -> Dict[str, Any]:
    if not edges:
        return {"nodes": 0, "edges": 0}
    n = max(max(u, v) for u, v in edges) + 1
    g_dir = Graph(n=n, directed=True)
    g_dir.add_edges(edges)
    g = g_dir.as_undirected(combine_edges="ignore")
    assort = g.assortativity_degree(directed=False)
    avg_clustering = g.transitivity_avglocal_undirected()
    degs = g.degree()
    slope = _degree_powerlaw_slope(degs, kmin=3)
    return {
        "nodes": n,
        "edges": len(edges),
        "assortativity": float(assort) if assort is not None else None,
        "avg_clustering": float(avg_clustering) if avg_clustering is not None else None,
        "degree_slope_abs": float(slope) if slope is not None else None,
    }


def validate(metrics: Dict[str, Any],
             assort_range: Tuple[float, float],
             slope_range: Tuple[float, float]) -> Tuple[bool, List[str]]:
    ok = True
    msgs: List[str] = []
    assort = metrics.get("assortativity")
    slope = metrics.get("degree_slope_abs")
    if assort is None or not (assort_range[0] <= assort <= assort_range[1]):
        ok = False
        msgs.append(f"assortativity {assort} not in [{assort_range[0]}, {assort_range[1]}]")
    if slope is None or not (slope_range[0] <= slope <= slope_range[1]):
        ok = False
        msgs.append(f"degree_slope_abs {slope} not in [{slope_range[0]}, {slope_range[1]}]")
    return ok, msgs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute and validate graph metrics for follow network.")
    p.add_argument("--edges", type=str, required=True, help="Path to edges CSV (follower_id,followee_id).")
    p.add_argument("--assort-range", type=str, default="0.18,0.32", help="Assortativity acceptable range (min,max).")
    p.add_argument("--slope-range", type=str, default="2.2,2.6", help="Degree slope(|slope|) acceptable range (min,max).")
    p.add_argument("--json-out", type=str, default="", help="Optional path to write metrics JSON.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    assort_tokens = [t.strip() for t in args.assort_range.split(",")]
    slope_tokens = [t.strip() for t in args.slope_range.split(",")]
    assort_range = (float(assort_tokens[0]), float(assort_tokens[1]))
    slope_range = (float(slope_tokens[0]), float(slope_tokens[1]))

    edges = _read_edges(Path(args.edges))
    metrics = compute_metrics(edges)
    ok, msgs = validate(metrics, assort_range, slope_range)
    print(json.dumps({"metrics": metrics, "ok": ok, "messages": msgs}, indent=2))
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "ok": ok, "messages": msgs}, f, indent=2)
    # Exit code for CI gating
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()


