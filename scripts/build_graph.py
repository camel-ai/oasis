#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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


def _read_personas(personas_csv: Path, class_col: Optional[str]) -> Tuple[int, List[str], Dict[int, str], Dict[str, List[int]]]:
    r"""Read personas CSV and return:
    - n: number of agents
    - groups: sorted unique group names
    - node_to_group: mapping node index -> group name
    - group_to_nodes: mapping group name -> list of node indices
    """
    df = pd.read_csv(personas_csv)
    # Always derive groups from an explicit class column; never parse usernames.
    accepted_cols = (
        "primary_label",
        "class",
        "persona_class",
        "category",
        "label",
    )
    chosen: Optional[str] = None
    if class_col:
        if class_col not in df.columns:
            raise ValueError(f"Specified class column '{class_col}' not found in personas CSV.")
        chosen = class_col
    else:
        for cand in accepted_cols:
            if cand in df.columns:
                chosen = cand
                break
        if chosen is None:
            raise ValueError(
                "No class column found. Provide one of "
                f"{accepted_cols} in the personas CSV, or pass --class-col."
            )
    classes = df[chosen].astype(str).tolist()
    node_to_group: Dict[int, str] = {}
    group_to_nodes: Dict[str, List[int]] = {}
    for i, c in enumerate(classes):
        g = str(c).strip().lower() if str(c).strip() else "unknown"
        node_to_group[i] = g
        group_to_nodes.setdefault(g, []).append(i)
    groups = sorted(group_to_nodes.keys())
    return len(classes), groups, node_to_group, group_to_nodes


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


def _triadic_closure_edges(
    n: int,
    existing_edges: List[Tuple[int, int]],
    node_to_group: Dict[int, str],
    *,
    tc_prob: float,
    tc_max_per_node: int,
    tc_homophily: float,
    assort_beta: float,
    seed: int,
) -> List[Tuple[int, int]]:
    r"""Add edges via a simple triadic-closure mechanism on a directed graph.

    For each node i, consider two-hop neighbors k reachable via i -> j and j -> k.
    Propose i -> k with probability that increases with the number of common
    intermediaries j and with a homophily bias for same-group targets.

    Args:
        n: Number of nodes.
        existing_edges: Current directed edges (i, j).
        node_to_group: Mapping node -> group label.
        tc_prob: Base per-common-neighbor probability (e.g., 0.02).
        tc_max_per_node: Cap edges added per source node.
        tc_homophily: Extra multiplicative weight for same-group targets in [0, 1].
        seed: RNG seed for determinism.
    """
    if tc_prob <= 0.0 or tc_max_per_node <= 0:
        return []
    rng = np.random.default_rng(seed + 2)

    # Build adjacency (out-neighbors) for fast two-hop enumeration
    out_neighbors: List[List[int]] = [[] for _ in range(n)]
    has_edge = set()
    in_deg = np.zeros(n, dtype=np.int64)
    for u, v in existing_edges:
        if u == v:
            continue
        if (u, v) in has_edge:
            continue
        has_edge.add((u, v))
        out_neighbors[u].append(v)
        in_deg[v] += 1
    out_deg = np.array([len(out_neighbors[u]) for u in range(n)], dtype=np.int64)
    total_deg = in_deg + out_deg

    added: List[Tuple[int, int]] = []
    # Iterate deterministically by node id
    for i in range(n):
        # Accumulate two-hop candidates and their multiplicities (common intermediaries)
        counts: Dict[int, int] = {}
        for j in out_neighbors[i]:
            # i -> j exists; consider j -> k
            for k in out_neighbors[j]:
                if k == i:
                    continue
                if (i, k) in has_edge:
                    continue
                counts[k] = counts.get(k, 0) + 1
        if not counts:
            continue

        # Sort candidates by number of common neighbors descending (tie-breaker: node id)
        candidates = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        added_for_i = 0
        gi = node_to_group[i]
        deg_i = int(total_deg[i])
        for k, common in candidates:
            if added_for_i >= tc_max_per_node:
                break
            # Probability increases with number of common intermediaries
            p = float(tc_prob) * float(common)
            if node_to_group.get(k) == gi:
                p *= (1.0 + float(tc_homophily))
            # Degree-assortative weighting: prefer similar-degree targets
            max_deg = int(total_deg.max()) if total_deg.size > 0 else 1
            if max_deg <= 0:
                max_deg = 1
            diff = abs(deg_i - int(total_deg[k])) / float(max_deg)
            if assort_beta > 0.0:
                p *= float(np.exp(-assort_beta * diff))
            # Clamp to [0, 1]
            if p > 1.0:
                p = 1.0
            if rng.random() < p:
                added.append((i, k))
                has_edge.add((i, k))
                out_neighbors[i].append(k)  # update adjacency incrementally
                # Update degrees for subsequent proposals
                out_deg[i] += 1
                in_deg[k] += 1
                total_deg[i] += 1
                total_deg[k] += 1
                added_for_i += 1

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
    n, groups, node_to_group, _ = _read_personas(personas_csv, class_col=None)
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
    return n, len(all_edges)


def build_graph_with_closure(
    personas_csv: Path,
    out_csv: Path,
    params: GraphParams,
    *,
    tc_prob: float,
    tc_max_per_node: int,
    tc_homophily: float,
    assort_beta: float,
    rewire_ratio: float = 0.0,
    class_col: Optional[str] = None,
) -> Tuple[int, int]:
    r"""Generate a directed follow graph with SBM + PA + triadic closure."""
    n, groups, node_to_group, _ = _read_personas(personas_csv, class_col=class_col)
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
    current_edges = _dedup_edges([*sbm_edges, *pa_edges])
    # Stage 3: Triadic closure
    tc_edges = _triadic_closure_edges(
        n=n,
        existing_edges=current_edges,
        node_to_group=node_to_group,
        tc_prob=tc_prob,
        tc_max_per_node=tc_max_per_node,
        tc_homophily=tc_homophily,
        assort_beta=assort_beta,
        seed=params.seed,
    )
    all_edges = _dedup_edges([*current_edges, *tc_edges])

    # Optional: degree-assortative rewiring while preserving node out-degrees
    if rewire_ratio and rewire_ratio > 0.0:
        all_edges = _assortative_rewire_edges(n=n, edges=all_edges, ratio=rewire_ratio, seed=params.seed + 3)

    _write_edges_csv(all_edges, out_csv)
    return n, len(all_edges)


def _assortative_rewire_edges(
    n: int,
    edges: List[Tuple[int, int]],
    ratio: float,
    seed: int,
) -> List[Tuple[int, int]]:
    r"""Increase degree assortativity by rewiring targets of pairs of edges.

    This preserves each source node's out-degree. It attempts up to
    target_accepts = int(ratio * len(edges)) accepted swaps that reduce the
    sum of absolute degree differences |deg(i)-deg(j)| across swapped pairs.
    """
    if ratio <= 0.0 or not edges:
        return edges
    rng = np.random.default_rng(seed)
    # Build adjacency (out-neighbors) and degree arrays
    out_neighbors: List[List[int]] = [[] for _ in range(n)]
    in_deg = np.zeros(n, dtype=np.int64)
    has_edge = set()
    for u, v in edges:
        if u == v:
            continue
        if (u, v) in has_edge:
            continue
        has_edge.add((u, v))
        out_neighbors[u].append(v)
        in_deg[v] += 1
    out_deg = np.array([len(out_neighbors[u]) for u in range(n)], dtype=np.int64)
    total_deg = in_deg + out_deg

    # Candidate sources with at least one out-edge
    sources = [u for u in range(n) if out_neighbors[u]]
    if len(sources) < 2:
        return list(has_edge)

    target_accepts = max(1, int(ratio * len(has_edge)))
    accepts = 0
    trials = 0
    max_trials = target_accepts * 20
    while accepts < target_accepts and trials < max_trials:
        trials += 1
        i, k = rng.choice(sources, size=2, replace=False)
        # pick random targets from each
        t1 = int(rng.choice(out_neighbors[i]))
        t2 = int(rng.choice(out_neighbors[k]))
        if t1 == t2:
            continue
        if i == t2 or k == t1:
            continue
        if (i, t2) in has_edge or (k, t1) in has_edge:
            continue
        # Compute objective before/after (degrees constant)
        cost_before = abs(int(total_deg[i]) - int(total_deg[t1])) + abs(int(total_deg[k]) - int(total_deg[t2]))
        cost_after = abs(int(total_deg[i]) - int(total_deg[t2])) + abs(int(total_deg[k]) - int(total_deg[t1]))
        if cost_after < cost_before:
            # Perform swap: remove (i,t1),(k,t2); add (i,t2),(k,t1)
            has_edge.remove((i, t1))
            has_edge.remove((k, t2))
            has_edge.add((i, t2))
            has_edge.add((k, t1))
            # Update adjacency lists
            # replace t1 in out_neighbors[i] with t2
            try:
                idx = out_neighbors[i].index(t1)
                out_neighbors[i][idx] = t2
            except ValueError:
                # fallback: rebuild list
                out_neighbors[i] = [t2 if x == t1 else x for x in out_neighbors[i]]
            try:
                idx2 = out_neighbors[k].index(t2)
                out_neighbors[k][idx2] = t1
            except ValueError:
                out_neighbors[k] = [t1 if x == t2 else x for x in out_neighbors[k]]
            accepts += 1
    return list(has_edge)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SBM+PA initial follow network for OASIS MVP.")
    parser.add_argument("--personas", type=str, default="./data/personas_mvp.csv",
                        help="Path to personas CSV (expects a class column, e.g., primary_label/class).")
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
    # Triadic closure controls
    parser.add_argument("--tc-prob", type=float, default=0.02, dest="tc_prob",
                        help="Base per-common-neighbor closure probability.")
    parser.add_argument("--tc-max-per-node", type=int, default=5, dest="tc_max_per_node",
                        help="Maximum number of closure edges to add per source node.")
    parser.add_argument("--tc-homophily", type=float, default=0.4, dest="tc_homophily",
                        help="Additional multiplicative weight for same-group closure targets.")
    parser.add_argument("--tc-assort-beta", type=float, default=2.0, dest="tc_assort_beta",
                        help="Strength of degree-similarity bias in closure (higher -> stronger assortativity).")
    parser.add_argument(
        "--class-col",
        type=str,
        default="",
        help="Column name for persona class (e.g., primary_label, class). Auto-detects if omitted.",
    )
    parser.add_argument(
        "--rewire-ratio",
        type=float,
        default=0.0,
        dest="rewire_ratio",
        help="Fraction of edges to accept as assortative rewirings (approximate).",
    )
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
    n, m = build_graph_with_closure(
        personas_csv=personas_csv,
        out_csv=out_csv,
        params=params,
        tc_prob=float(args.tc_prob),
        tc_max_per_node=int(args.tc_max_per_node),
        tc_homophily=float(args.tc_homophily),
        assort_beta=float(args.tc_assort_beta),
        rewire_ratio=float(args.rewire_ratio),
        class_col=(args.class_col.strip() or None),
    )
    print(f"Generated directed follow graph: nodes={n}, edges={m} -> {out_csv}")


if __name__ == "__main__":
    main()


