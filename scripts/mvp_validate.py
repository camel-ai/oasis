#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

REQUIRED_KEYS = [
    "post_id",
    "thread_id",
    "user_id",
    "timestamp",
    "text",
    "category_labels",
]


def validate_jsonl(path: Path) -> Dict[str, int]:
    stats = {
        "lines": 0,
        "ok": 0,
        "missing_keys": 0,
        "has_tokens_in_text": 0,
        "empty_labels": 0,
    }
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stats["lines"] += 1
            try:
                rec = json.loads(line)
            except Exception:
                continue

            if any(k not in rec for k in REQUIRED_KEYS):
                stats["missing_keys"] += 1
                continue

            text = rec.get("text", "")
            if "<LBL:" in text:
                stats["has_tokens_in_text"] += 1
                continue

            labels = rec.get("category_labels", [])
            if not isinstance(labels, list) or len(labels) == 0:
                stats["empty_labels"] += 1
                continue

            stats["ok"] += 1
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate MVP JSONL output")
    parser.add_argument("--file", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(os.path.abspath(args.file))
    stats = validate_jsonl(path)
    print(json.dumps(stats, indent=2))
    if stats["ok"] == 0 or stats["has_tokens_in_text"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()


