#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


def _load_rows(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute multi-label metrics from JSONL (sidecar or final).")
    ap.add_argument("--data", type=str, required=True, help="Path to sidecar.jsonl or posts.jsonl")
    args = ap.parse_args()
    path = Path(os.path.abspath(args.data))
    rows = _load_rows(path)
    harmful_total = 0
    harmful_ml = 0
    pair_counts = Counter()
    benign_none = 0
    for r in rows:
        labels = r.get("category_labels") or []
        labels = [str(x) for x in labels]
        if not labels:
            continue
        if labels == ["benign"]:
            benign_none += 1
        harm = [l for l in labels if l != "benign"]
        if harm:
            harmful_total += 1
            if len(harm) >= 2:
                harmful_ml += 1
                key = "↔".join(sorted(harm)[:2])
                pair_counts[key] += 1
    rate = (harmful_ml / harmful_total) if harmful_total > 0 else 0.0
    out = {
        "harmful_total": harmful_total,
        "harmful_multi_label": harmful_ml,
        "harmful_multi_label_rate": rate,
        "pair_cooccurrence_top": pair_counts.most_common(10),
        "benign_singleton_count": benign_none,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()


