#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def load_user_variant_map(persona_csv: Path) -> Dict[str, str]:
    df = pd.read_csv(persona_csv, keep_default_na=False)
    return {
        row["username"]: row.get("persona_variant", "")
        for _, row in df.iterrows()
    }


def load_user_lookup(conn: sqlite3.Connection) -> Dict[int, str]:
    lookup: Dict[int, str] = {}
    for user_id, user_name, name in conn.execute(
        "SELECT user_id, user_name, name FROM user"
    ):
        lookup[user_id] = user_name or name or ""
    return lookup


def clean_text(text: str) -> str:
    return text.replace("\n", " ").strip()


def gather_samples(
    conn: sqlite3.Connection,
    user_lookup: Dict[int, str],
    variant_map: Dict[str, str],
    limit: int = 3,
) -> Tuple[Dict[str, List[Tuple[int, str, str]]], Dict[str, List[Tuple[int, str, str]]]]:
    posts: Dict[str, List[Tuple[int, str, str]]] = defaultdict(list)
    comments: Dict[str, List[Tuple[int, str, str]]] = defaultdict(list)

    for user_id, post_id, content in conn.execute(
        "SELECT user_id, post_id, content FROM post ORDER BY post_id DESC"
    ):
        username = user_lookup.get(user_id, "")
        variant = variant_map.get(username, "unknown")
        if len(posts[variant]) < limit:
            posts[variant].append((post_id, username, clean_text(content)))

    for user_id, comment_id, content in conn.execute(
        "SELECT user_id, comment_id, content FROM comment ORDER BY comment_id DESC"
    ):
        username = user_lookup.get(user_id, "")
        variant = variant_map.get(username, "unknown")
        if len(comments[variant]) < limit:
            comments[variant].append((comment_id, username, clean_text(content)))

    return posts, comments


def render_markdown(
    posts: Dict[str, List[Tuple[int, str, str]]],
    comments: Dict[str, List[Tuple[int, str, str]]],
) -> str:
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Persona Sample Outputs",
        "",
        f"Latest run captured at {now}.",
    ]
    variants = sorted(v for v in set(posts.keys()) | set(comments.keys()) if v)
    for variant in variants:
        lines.append("")
        lines.append(f"## {variant}")
        post_samples = posts.get(variant, [])
        if post_samples:
            lines.append("- **Posts**")
            for post_id, username, content in post_samples:
                lines.append(f"  - post {post_id} ({username}): {content}")
        comment_samples = comments.get(variant, [])
        if comment_samples:
            lines.append("- **Comments**")
            for comment_id, username, content in comment_samples:
                lines.append(f"  - comment {comment_id} ({username}): {content}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export persona sample outputs to markdown.")
    parser.add_argument("--db", type=str, default="data/mvp/oasis_mvp_gemini.db")
    parser.add_argument("--personas", type=str, default="data/personas_mvp.csv")
    parser.add_argument("--output", type=str, default="docs/personas/mvp_sample_outputs.md")
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    db_path = Path(args.db)
    persona_csv = Path(args.personas)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    variant_map = load_user_variant_map(persona_csv)
    conn = sqlite3.connect(db_path)
    try:
        user_lookup = load_user_lookup(conn)
        posts, comments = gather_samples(conn, user_lookup, variant_map, limit=args.limit)
    finally:
        conn.close()

    markdown = render_markdown(posts, comments)
    output_path.write_text(markdown, encoding="utf-8")
    print(f"Wrote persona samples to {output_path}")


if __name__ == "__main__":
    main()
