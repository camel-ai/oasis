#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAG_DIR = DATA_DIR / "rag_corpus"


def clean_text(text: str) -> str:
    text = text.strip()
    text = text.replace("URL", "").replace("rt ", "")
    text = re.sub(r"\s+", " ", text)
    return text


def from_flagged_messages(path: Path) -> Dict[str, List[Dict[str, str]]]:
    variant_map: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    if not path.exists():
        return variant_map

    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            text = clean_text(row.get("content", ""))
            reason = (row.get("reason") or "").lower()
            if not text:
                continue
            if "harassment" in reason or "hate" in reason:
                variant = "aggressor"
                tags = ["flagged", "harassment"]
            elif "self-harm" in reason:
                variant = "doomer"
                tags = ["flagged", "self_harm"]
            else:
                continue
            variant_map[variant].append(
                {
                    "text": text,
                    "source": "flagged_messages",
                    "tags": tags,
                }
            )
    return variant_map


def from_all_topics(path: Path) -> Dict[str, List[Dict[str, str]]]:
    variant_map: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    if not path.exists():
        return variant_map

    df = pd.read_csv(path)
    for _, row in df.iterrows():
        text = clean_text(str(row.get("source_tweet", "")))
        category = (str(row.get("topic_category", ""))).lower()
        if not text:
            continue
        entry = {
            "text": text,
            "source": "twitter_all_topics",
            "tags": [category] if category else ["twitter"],
        }
        if category in {"terrorism & war", "religion", "politics", "government"}:
            variant_map["contrarian"].append(entry)
        elif category in {"science & technology", "business", "finance"}:
            variant_map["theorist"].append(entry)
        elif category in {"entertainment", "lifestyle", "sports", "health"}:
            variant_map["supportive_generalist"].append(entry)
        else:
            variant_map["supportive_generalist"].append(entry)
    return variant_map


def from_personality(path: Path) -> Dict[str, List[Dict[str, str]]]:
    variant_map: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    if not path.exists():
        return variant_map

    df = pd.read_csv(path)
    for _, row in df.iterrows():
        persona_desc = clean_text(str(row.get("Persona", "")))
        chat_blob = str(row.get("chat", ""))
        chat_lines = [clean_text(line) for line in chat_blob.splitlines() if clean_text(line)]

        if persona_desc:
            variant_map["supportive_generalist"].append(
                {
                    "text": persona_desc,
                    "source": "facebook_persona_description",
                    "tags": ["persona_desc"],
                }
            )
        for line in chat_lines[:4]:
            variant_map["supportive_generalist"].append(
                {
                    "text": line,
                    "source": "facebook_persona_chat",
                    "tags": ["persona_chat"],
                }
            )
    return variant_map


def merge_sources(sources: Iterable[Dict[str, List[Dict[str, str]]]]) -> Dict[str, List[Dict[str, str]]]:
    merged: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for source in sources:
        for variant, entries in source.items():
            merged[variant].extend(entries)
    for variant, entries in merged.items():
        seen = set()
        unique_entries = []
        for entry in entries:
            text = entry["text"]
            if not text or text in seen:
                continue
            seen.add(text)
            unique_entries.append(entry)
        merged[variant] = unique_entries
    return merged


def enforce_limits(
    data: Dict[str, List[Dict[str, str]]],
    max_per_variant: int,
) -> Dict[str, List[Dict[str, str]]]:
    if max_per_variant <= 0:
        return data
    limited: Dict[str, List[Dict[str, str]]] = {}
    for variant, entries in data.items():
        limited[variant] = entries[:max_per_variant]
    return limited


def write_jsonl(data: Dict[str, List[Dict[str, str]]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for variant, entries in data.items():
            for entry in entries:
                payload = {
                    "persona_variant": variant,
                    **entry,
                }
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_summary(data: Dict[str, List[Dict[str, str]]], output_path: Path) -> None:
    summary = {
        variant: {
            "count": len(entries),
            "sample": entries[:5],
        }
        for variant, entries in data.items()
    }
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build persona RAG corpus from existing datasets.")
    parser.add_argument("--flagged", type=str, default=str(DATA_DIR / "flagged_messages_rows.csv"))
    parser.add_argument("--topics", type=str, default=str(DATA_DIR / "twitter_dataset" / "all_topics.csv"))
    parser.add_argument("--personality", type=str, default=str(DATA_DIR / "personality.csv"))
    parser.add_argument("--output", type=str, default=str(RAG_DIR / "persona_corpus.jsonl"))
    parser.add_argument("--summary", type=str, default=str(RAG_DIR / "persona_corpus_summary.json"))
    parser.add_argument("--max-per-variant", type=int, default=500)
    args = parser.parse_args()

    flagged = from_flagged_messages(Path(args.flagged))
    topics = from_all_topics(Path(args.topics))
    personality = from_personality(Path(args.personality))

    merged = merge_sources([flagged, topics, personality])
    merged = enforce_limits(merged, args.max_per_variant)
    write_jsonl(merged, Path(args.output))
    write_summary(merged, Path(args.summary))

    print("Persona corpus written to:", args.output)
    for variant, entries in merged.items():
        print(f"  {variant}: {len(entries)} entries")


if __name__ == "__main__":
    main()
