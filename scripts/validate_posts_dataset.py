#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

HARMFUL_LABELS = {"eating_disorder_risk", "incel_misogyny", "misinformation", "conspiracy"}


def load_posts(path: Path) -> List[Dict[str, object]]:
    posts: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            posts.append(json.loads(line))
    return posts


def summarize(posts: List[Dict[str, object]]) -> Dict[str, object]:
    label_counts = Counter()
    split_counts = Counter()
    persona_counts = Counter()
    harmful_posts = 0
    multi_harmful = 0
    placeholders_missing = 0
    for record in posts:
        labels = record.get("labels", [])
        if isinstance(labels, str):
            labels = [labels]
        harm_labels = [label for label in labels if label in HARMFUL_LABELS]
        if harm_labels:
            harmful_posts += 1
            if len(harm_labels) > 1:
                multi_harmful += 1
            if not record.get("placeholders_applied"):
                placeholders_missing += 1
        for label in labels:
            label_counts[label] += 1
        split_counts[record.get("split", "train")] += 1
        persona_counts[record.get("persona_id", "")] += 1
    return {
        "total_posts": len(posts),
        "unique_personas": len(persona_counts),
        "harmful_posts": harmful_posts,
        "multi_label_harmful": multi_harmful,
        "label_distribution": dict(label_counts),
        "split_distribution": dict(split_counts),
        "persona_post_counts": dict(persona_counts),
        "placeholders_missing": placeholders_missing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate posts dataset counts")
    parser.add_argument("--input", type=str, default="./posts.jsonl")
    parser.add_argument("--output", type=str, default="./oasis/configs/lexicons/posts_dataset_metrics.json")
    args = parser.parse_args()

    posts = load_posts(Path(args.input))
    metrics = summarize(posts)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
