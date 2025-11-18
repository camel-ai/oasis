#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple


def _load_sidecar(path: Path) -> List[Dict]:
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


def _measure_multi_label(rows: List[Dict]) -> Tuple[int, int, float]:
    harmful_total = 0
    harmful_ml = 0
    for r in rows:
        labs = r.get("category_labels") or []
        labs = [str(x) for x in labs]
        if not labs:
            continue
        harmful = any(l for l in labs if l != "benign")
        if not harmful:
            continue
        harmful_total += 1
        if len([l for l in labs if l != "benign"]) >= 2:
            harmful_ml += 1
    rate = (harmful_ml / harmful_total) if harmful_total > 0 else 0.0
    return harmful_total, harmful_ml, rate


def _suggest_mode_probs(current_double: float, measured_rate: float, target: float) -> Dict[str, float]:
    # Simple proportional controller for double-probability
    err = target - measured_rate
    adj = 0.5 * err  # gain
    new_double = min(0.5, max(0.05, current_double + adj))
    # Keep a fixed ratio between none and single for now (can be calibrated later)
    remaining = max(0.0, 1.0 - new_double)
    none = remaining * 0.55
    single = remaining * 0.45
    return {"none": none, "single": single, "double": new_double}


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate multi-label rate from sidecar.")
    ap.add_argument("--sidecar", type=str, required=True, help="Path to sidecar.jsonl")
    ap.add_argument("--current-double", type=float, default=0.10, help="Current 'double' probability")
    ap.add_argument("--target-rate", type=float, default=0.20, help="Target harmful multi-label rate")
    ap.add_argument("--out", type=str, default="", help="Optional path to write JSON overrides")
    args = ap.parse_args()

    sidecar_path = Path(os.path.abspath(args.sidecar))
    rows = _load_sidecar(sidecar_path)
    harmful_total, harmful_ml, rate = _measure_multi_label(rows)
    suggested = _suggest_mode_probs(args.current_double, rate, args.target_rate)
    result = {
        "harmful_total": harmful_total,
        "harmful_multi_label": harmful_ml,
        "measured_rate": rate,
        "suggested_post_label_mode_probs": suggested,
    }
    print(json.dumps(result, indent=2))
    if args.out:
        out_path = Path(os.path.abspath(args.out))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()


