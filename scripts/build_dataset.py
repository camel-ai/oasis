#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

LABEL_TOKEN_PATTERN = re.compile(r"<LBL:([A-Z_]+)>")


DEFAULT_LABEL_MAPPING: Dict[str, List[str]] = {
    "LBL:INCEL_SLANG": ["incel"],
    "LBL:MISINFO_CLAIM": ["misinfo", "conspiracy"],
    "LBL:SUPPORTIVE": ["recovery", "benign"],
}


PERSONA_ALLOWED: Dict[str, List[str]] = {
    "incel": ["incel"],
    "misinfo": ["misinfo", "conspiracy"],
    "benign": ["benign"],
}


@dataclass
class StaticBank:
    bank: Dict[str, List[str]]

    @staticmethod
    def load_simple_yaml(path: Path) -> "StaticBank":
        bank: Dict[str, List[str]] = {}
        current: Optional[str] = None
        with path.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.rstrip("\n")
                if not line.strip():
                    continue
                if not line.startswith(" ") and line.endswith(":"):
                    current = line[:-1]
                    bank[current] = []
                    continue
                if current and line.lstrip().startswith("- "):
                    # Extract after '- ' and strip surrounding quotes if any
                    item = line.strip()[2:]
                    if (item.startswith("\"") and item.endswith("\"")) or (
                        item.startswith("'") and item.endswith("'")
                    ):
                        item = item[1:-1]
                    bank[current].append(item)
        return StaticBank(bank)


def det_phrase(
    choices: List[str], seed: int, post_id: int, token: str, occurrence_index: int
) -> str:
    key = f"{seed}:{post_id}:{token}:{occurrence_index}".encode()
    idx = int(hashlib.sha256(key).hexdigest(), 16) % max(1, len(choices))
    return choices[idx]


def extract_tokens(text: str) -> List[str]:
    return [f"LBL:{m.group(1)}" for m in LABEL_TOKEN_PATTERN.finditer(text)]


def impute_text(
    raw_text: str, static_bank: StaticBank, seed: int, post_id: int
) -> Tuple[str, List[str]]:
    tokens = extract_tokens(raw_text)
    occurrence: Dict[str, int] = {}

    def replace_match(m: re.Match[str]) -> str:
        full = f"LBL:{m.group(1)}"
        occurrence[full] = occurrence.get(full, 0) + 1
        choices = static_bank.bank.get(full, [""])
        return det_phrase(choices, seed, post_id, full, occurrence[full] - 1)

    new_text = LABEL_TOKEN_PATTERN.sub(replace_match, raw_text)
    return new_text, tokens


def token_to_categories(token: str, mapping: Dict[str, List[str]]) -> List[str]:
    return mapping.get(token, [])


def assign_labels(tokens: List[str], persona: Optional[str]) -> Tuple[List[str], float]:
    cats: List[str] = []
    for t in tokens:
        cats.extend(token_to_categories(t, DEFAULT_LABEL_MAPPING))
    cats = sorted(set(cats))
    if persona:
        allowed = PERSONA_ALLOWED.get(persona, cats)
        cats = [c for c in cats if c in allowed]
        if not cats and allowed:
            cats = [allowed[0]]
    confidence = 0.5 if not tokens else 0.8 + 0.1 * min(len(tokens), 2)
    confidence = float(min(confidence, 1.0))
    return cats, confidence


def infer_persona_from_username(username: str) -> Optional[str]:
    lowered = username.lower()
    if lowered.startswith("incel_"):
        return "incel"
    if lowered.startswith("misinfo_"):
        return "misinfo"
    if lowered.startswith("benign_"):
        return "benign"
    return None


def isoformat_timestamp(ts_val) -> str:
    # post.created_at may be integer (twitter clock) or datetime (reddit mode)
    # Map both to ISO-8601 string deterministically.
    if isinstance(ts_val, (int, float)):
        # Interpret as minutes since start; render as a pseudo-UTC time.
        base = datetime(2025, 1, 1)
        return (base.replace(microsecond=0) + (ts_val * 60) * 1e-6 * 0).isoformat() + "Z"
    try:
        return datetime.fromisoformat(str(ts_val)).isoformat() + "Z"
    except Exception:
        return str(ts_val)


def build_dataset(db_path: Path, out_path: Path, bank_path: Path, seed: int, skip_imputation: bool = False) -> None:
    static_bank = StaticBank.load_simple_yaml(bank_path)
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    # Fetch users for persona inference
    cur.execute("SELECT user_id, user_name, name FROM user")
    user_rows = cur.fetchall()
    uid_to_username: Dict[int, str] = {}
    for uid, uname, name in user_rows:
        uid_to_username[uid] = uname or name or f"user_{uid}"

    # Fetch posts
    cur.execute(
        "SELECT post_id, user_id, original_post_id, content, quote_content, created_at "
        "FROM post ORDER BY post_id"
    )
    post_rows = cur.fetchall()

    # Fetch comments
    cur.execute(
        "SELECT comment_id, post_id, user_id, content, created_at "
        "FROM comment ORDER BY comment_id"
    )
    comment_rows = cur.fetchall()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    num_written = 0
    with out_path.open("w", encoding="utf-8") as f:
        # Write posts
        for (post_id, user_id, original_post_id, content, quote_content, created_at) in post_rows:
            text_raw = content or ""
            username = uid_to_username.get(user_id, f"user_{user_id}")
            persona = infer_persona_from_username(username)

            if skip_imputation:
                imputed_text = text_raw
                tokens = extract_tokens(text_raw)
            else:
                imputed_text, tokens = impute_text(text_raw, static_bank, seed, int(post_id))
            labels, conf = assign_labels(tokens, persona)

            rec = {
                "post_id": f"p_{post_id}",
                "thread_id": f"p_{original_post_id}" if original_post_id else f"p_{post_id}",
                "user_id": str(user_id),
                "parent_id": None,
                "timestamp": str(created_at),
                "text": imputed_text,
                "category_labels": labels,
                "gold_confidence": conf,
                "split": "train",
                "provenance": f"gen:mvp persona:{persona or 'unknown'} | imputer:{'skip' if skip_imputation else 'v0-mvp'}",
                "generation_seed": int(seed),
                "persona_id": persona or "unknown",
                "needs_thread_context": False,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            num_written += 1

        # Write comments as child posts with parent_id
        for (comment_id, post_id, user_id, content, created_at) in comment_rows:
            text_raw = content or ""
            username = uid_to_username.get(user_id, f"user_{user_id}")
            persona = infer_persona_from_username(username)

            if skip_imputation:
                imputed_text = text_raw
                tokens = extract_tokens(text_raw)
            else:
                imputed_text, tokens = impute_text(text_raw, static_bank, seed, int(comment_id) + 100000)
            labels, conf = assign_labels(tokens, persona)

            rec = {
                "post_id": f"c_{comment_id}",
                "thread_id": f"p_{post_id}",
                "user_id": str(user_id),
                "parent_id": f"p_{post_id}",
                "timestamp": str(created_at),
                "text": imputed_text,
                "category_labels": labels,
                "gold_confidence": conf,
                "split": "train",
                "provenance": f"gen:mvp persona:{persona or 'unknown'} | imputer:{'skip' if skip_imputation else 'v0-mvp'}",
                "generation_seed": int(seed),
                "persona_id": persona or "unknown",
                "needs_thread_context": True,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            num_written += 1

    conn.close()
    print(f"Wrote {num_written} items (posts + comments) to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MVP posts JSONL from OASIS SQLite DB")
    parser.add_argument("--db", type=str, required=True, help="Path to OASIS SQLite DB")
    parser.add_argument("--out", type=str, default="./data/mvp/posts_mvp.jsonl")
    parser.add_argument(
        "--static-bank", type=str, default="./data/label_tokens_static_bank.yaml"
    )
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--skip-imputation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db_path = Path(os.path.abspath(args.db))
    out_path = Path(os.path.abspath(args.out))
    bank_path = Path(os.path.abspath(args.static_bank))
    build_dataset(db_path, out_path, bank_path, args.seed, skip_imputation=args.skip_imputation)


if __name__ == "__main__":
    main()


