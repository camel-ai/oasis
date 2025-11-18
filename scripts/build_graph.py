#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GraphParams:
    r"""Parameters controlling SBM + PA generation.

    Args:
        theta_b (float): Baseline edge probability.
        rho (float): Homophily strength in [0, 1]. Higher → more within-group edges.
        alpha (float): Approximate number of preferential-attachment edges added per node.
        avg_out_degree (float): Target average out-degree after SBM stage (used to scale `theta_b` if desired).
        seed (int): RNG seed for determinism.
    """
    theta_b: float
    rho: float
    alpha: float
    avg_out_degree: float
    seed: int


def _infer_group_from_username(username: str) -> str:
    r"""Infer persona group from username prefix (e.g., 'incel_0001...' -> 'incel').
    Falls back to 'unknown' if pattern not found.
    """
    try:
        prefix = username.split("_", 1)[0].strip().lower()
        return prefix if prefix else "unknown"
    except Exception:
        return "unknown"


def _read_personas(personas_csv: Path) -> Tuple[int, List[str], Dict[int, str], Dict[str, List[int]]]:
    r"""Read personas CSV and return:
    - n: number of agents
    - groups: sorted unique group names
    - node_to_group: mapping node index -> group name
    - group_to_nodes: mapping group name -> list of node indices
    """
    df = pd.read_csv(personas_csv)
    if "username" not in df.columns:
        raise ValueError("Expected 'username' column in personas CSV.")
    usernames = df["username"].astype(str).tolist()
    node_to_group: Dict[int, str] = {}
    group_to_nodes: Dict[str, List[int]] = {}
    for i, u in enumerate(usernames):
        g = _infer_group_from_username(u)
        node_to_group[i] = g
        group_to_nodes.setdefault(g, []).append(i)
    groups = sorted(group_to_nodes.keys())
    return len(usernames), groups, node_to_group, group_to_nodes


def _clip_prob(p: float) -> float:
    return float(min(1.0, max(0.0, p)))


def _sample_sbm_directed_edges(
    n: int,
    groups: List[str],
    node_to_group: Dict[int, str],
    params: GraphParams,
) -> List[Tuple[int, int]]:
    r"""Generate directed edges using a simple SBM with homophily.

    For each ordered pair (i, j), i != j:
      - If same group: p = theta_b * (1 + rho)
      - If different group: p = theta_b * (1 - rho)
    """
    rng = np.random.default_rng(params.seed)
    edges: List[Tuple[int, int]] = []
    # Precompute per-pair probabilities by group relation
    p_same = _clip_prob(params.theta_b * (1.0 + params.rho))
    p_diff = _clip_prob(params.theta_b * (1.0 - params.rho))
    # Vectorized generation by rows for cache locality
    for i in range(n):
        gi = node_to_group[i]
        # Draw j != i
        # We draw a full row of Bernoulli trials then clear self-loop
        probs = np.full(n, p_diff, dtype=np.float32)
        # Same-group positions switched to p_same
        same_mask = np.fromiter((node_to_group[j] == gi for j in range(n)),
                                dtype=bool, count=n)
        probs[same_mask] = p_same
        probs[i] = 0.0  # no self-loops
        trials = rng.random(n) < probs
        # Collect edges
        for j in np.nonzero(trials)[0].tolist():
            edges.append((i, j))
    return edges


def _preferential_attachment_edges(
    n: int,
    existing_edges: List[Tuple[int, int]],
    params: GraphParams,
    node_to_group: Dict[int, str],
) -> List[Tuple[int, int]]:
    r"""Add preferential-attachment edges.

    For each node i in random order, add m ≈ alpha out-edges to targets sampled
    proportional to current in-degree + 1 (to avoid zeros).
    A mild homophily reweight is applied by multiplying same-group targets by (1 + rho/2).
    """
    rng = np.random.default_rng(params.seed + 1)
    m = max(0, int(round(params.alpha)))
    if m == 0:
        return []

    # Build current in-degree
    in_deg = np.zeros(n, dtype=np.int64)
    for _, j in existing_edges:
        in_deg[j] += 1

    # For fast membership checking
    has_edge = set(existing_edges)
    added: List[Tuple[int, int]] = []

    nodes = list(range(n))
    rng.shuffle(nodes)

    for i in nodes:
        gi = node_to_group[i]
        # Candidate scores: in-degree + 1
        scores = in_deg.astype(np.float64) + 1.0
        # Apply mild homophily bias
        same_mask = np.fromiter((node_to_group[j] == gi for j in range(n)),
                                dtype=bool, count=n)
        scores[same_mask] *= (1.0 + params.rho / 2.0)
        scores[i] = 0.0  # no self-loops
        total = scores.sum()
        if total <= 0:
            continue
        probs = scores / total

        # Sample without replacement, avoid duplicates and existing edges
        # We will over-sample by a small factor and then filter to m distinct targets.
        sample_k = min(n - 1, max(m * 3, m))
        # Use choice with replacement then filter uniques to avoid zero-prob pitfalls
        cand = rng.choice(n, size=sample_k, replace=True, p=probs).tolist()
        uniq_targets: List[int] = []
        seen = set()
        for t in cand:
            if t == i:
                continue
            if (i, t) in has_edge:
                continue
            if t in seen:
                continue
            seen.add(t)
            uniq_targets.append(t)
            if len(uniq_targets) >= m:
                break

        for t in uniq_targets:
            added.append((i, t))
            has_edge.add((i, t))
            in_deg[t] += 1  # update degree for subsequent choices

    return added


def _dedup_edges(edges: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    seen = set()
    out: List[Tuple[int, int]] = []
    for e in edges:
        if e[0] == e[1]:
            continue
        if e in seen:
            continue
        seen.add(e)
        out.append(e)
    return out


def _write_edges_csv(edges: List[Tuple[int, int]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["follower_id", "followee_id"])
        writer.writerows(edges)


def build_graph(personas_csv: Path, out_csv: Path, params: GraphParams) -> Tuple[int, int]:
    r"""Generate a directed follow graph and write edge list CSV.

    Returns:
        n (int): Number of nodes.
        m (int): Number of edges written.
    """
    n, groups, node_to_group, _ = _read_personas(personas_csv)
    # Stage 1: SBM-directed
    sbm_edges = _sample_sbm_directed_edges(n=n, groups=groups, node_to_group=node_to_group, params=params)
    sbm_edges = _dedup_edges(sbm_edges)
    # Stage 2: PA augmentation
    pa_edges = _preferential_attachment_edges(
        n=n,
        existing_edges=sbm_edges,
        params=params,
        node_to_group=node_to_group,
    )
    all_edges = _dedup_edges([*sbm_edges, *pa_edges])
    _write_edges_csv(all_edges, out_csv)
    return n, len(all_edges)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SBM+PA initial follow network for OASIS MVP.")
    parser.add_argument("--personas", type=str, default="./data/personas_mvp.csv",
                        help="Path to personas CSV (expects 'username' column).")
    parser.add_argument("--out", type=str, default="./data/edges_mvp.csv",
                        help="Output edge list CSV (follower_id,followee_id).")
    parser.add_argument("--theta-b", type=float, default=0.015, dest="theta_b",
                        help="Baseline edge probability.")
    parser.add_argument("--rho", type=float, default=0.25,
                        help="Homophily strength in [0,1].")
    parser.add_argument("--alpha", type=float, default=2.0,
                        help="Approx edges per node added via preferential attachment.")
    parser.add_argument("--avg-out-degree", type=float, default=10.0, dest="avg_out_degree",
                        help="Targeted average out-degree (informative; not strictly enforced).")
    parser.add_argument("--seed", type=int, default=314159,
                        help="RNG seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    personas_csv = Path(os.path.abspath(args.personas))
    out_csv = Path(os.path.abspath(args.out))
    params = GraphParams(
        theta_b=float(args.theta_b),
        rho=float(args.rho),
        alpha=float(args.alpha),
        avg_out_degree=float(args.avg_out_degree),
        seed=int(args.seed),
    )
    n, m = build_graph(personas_csv=personas_csv, out_csv=out_csv, params=params)
    print(f"Generated directed follow graph: nodes={n}, edges={m} -> {out_csv}")


if __name__ == "__main__":
    main()


