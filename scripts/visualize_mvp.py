#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MVP Dataset Viewer</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            max-width: 900px;
            margin: 40px auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .thread {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .post {{
            border-left: 3px solid #1da1f2;
            padding-left: 15px;
            margin-bottom: 15px;
        }}
        .comment {{
            border-left: 3px solid #657786;
            padding-left: 15px;
            margin: 10px 0 10px 30px;
            font-size: 0.95em;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            font-size: 0.9em;
            color: #657786;
        }}
        .username {{
            font-weight: 600;
            color: #14171a;
        }}
        .persona {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.75em;
            font-weight: 600;
            margin-left: 8px;
        }}
        .persona-incel {{
            background: #fee;
            color: #c00;
        }}
        .persona-misinfo {{
            background: #ffeaa7;
            color: #d63031;
        }}
        .persona-benign {{
            background: #dfe6e9;
            color: #2d3436;
        }}
        .persona-recovery {{
            background: #d4f1f4;
            color: #05668d;
        }}
        .labels {{
            margin-top: 8px;
            font-size: 0.85em;
        }}
        .label {{
            display: inline-block;
            background: #e1e8ed;
            padding: 3px 10px;
            border-radius: 4px;
            margin-right: 5px;
            color: #14171a;
        }}
        .text {{
            margin: 10px 0;
            line-height: 1.5;
            color: #14171a;
        }}
        .label-token {{
            background: #fff3cd;
            border: 1px solid #ffc107;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 0.85em;
            font-weight: 600;
            color: #856404;
        }}
        .stats {{
            display: flex;
            gap: 15px;
            margin-top: 10px;
            font-size: 0.85em;
            color: #657786;
        }}
        .stat {{
            display: flex;
            align-items: center;
            gap: 4px;
        }}
        h1 {{
            color: #14171a;
            margin-bottom: 30px;
        }}
        .summary {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .summary-item {{
            padding: 10px;
            background: #f7f9fa;
            border-radius: 4px;
        }}
        .summary-label {{
            font-size: 0.8em;
            color: #657786;
            margin-bottom: 4px;
        }}
        .summary-value {{
            font-size: 1.4em;
            font-weight: 600;
            color: #14171a;
        }}
    </style>
</head>
<body>
    <h1>MVP Dataset Viewer</h1>
    <div class="summary">
        <h2 style="margin-top:0;">Summary</h2>
        <div class="summary-grid">
            <div class="summary-item">
                <div class="summary-label">Total Posts</div>
                <div class="summary-value">{total_posts}</div>
            </div>
            <div class="summary-item">
                <div class="summary-label">Threads</div>
                <div class="summary-value">{total_threads}</div>
            </div>
            <div class="summary-item">
                <div class="summary-label">Users</div>
                <div class="summary-value">{total_users}</div>
            </div>
        </div>
        <div style="margin-top:15px; font-size: 0.9em;">
            <strong>Label distribution:</strong> {label_dist}
        </div>
    </div>
    
    {threads_html}
</body>
</html>
"""


def load_posts(jsonl_path: Path) -> List[dict]:
    posts: List[dict] = []
    if not jsonl_path.exists():
        return posts
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                posts.append(json.loads(line))
    return posts


def group_by_thread(posts: List[dict]) -> Dict[str, List[dict]]:
    threads: Dict[str, List[dict]] = defaultdict(list)
    for p in posts:
        tid = p.get("thread_id", p.get("post_id", "unknown"))
        threads[tid].append(p)
    return threads


def render_post(p: dict, is_root: bool = True) -> str:
    import html
    persona = p.get("persona_id", "unknown")
    persona_class = "persona-benign"
    if "incel" in persona:
        persona_class = "persona-incel"
    elif "misinfo" in persona:
        persona_class = "persona-misinfo"
    elif "recovery" in persona:
        persona_class = "persona-recovery"

    username = p.get("user_id", "unknown")
    timestamp = p.get("timestamp", "")
    text_raw = p.get("text", "")
    # Escape HTML but preserve label tokens as plain text
    text_escaped = html.escape(text_raw)
    labels = p.get("category_labels", [])
    confidence = p.get("gold_confidence", 0.0)

    labels_html = " ".join([f'<span class="label">{lbl}</span>' for lbl in labels])
    container_class = "post" if is_root else "comment"

    return f"""
    <div class="{container_class}">
        <div class="header">
            <span>
                <span class="username">{username}</span>
                <span class="persona {persona_class}">{persona}</span>
            </span>
            <span>{timestamp[:19] if timestamp else ""}</span>
        </div>
        <div class="text">{text_escaped}</div>
        <div class="labels">
            {labels_html}
            <span style="color: #657786; margin-left: 10px;">confidence: {confidence:.2f}</span>
        </div>
    </div>
    """


def render_threads(threads: Dict[str, List[dict]]) -> str:
    html_parts: List[str] = []
    for tid, posts in sorted(threads.items(), key=lambda x: x[0]):
        # root is the post without parent_id
        root = next((p for p in posts if not p.get("parent_id")), posts[0] if posts else None)
        replies = [p for p in posts if p.get("parent_id")]

        thread_html = '<div class="thread">'
        if root:
            thread_html += render_post(root, is_root=True)
        for r in replies:
            thread_html += render_post(r, is_root=False)
        thread_html += "</div>"
        html_parts.append(thread_html)

    return "\n".join(html_parts)


def compute_stats(posts: List[dict]) -> dict:
    users = set(p.get("user_id") for p in posts)
    threads = set(p.get("thread_id") for p in posts)
    label_counts: Dict[str, int] = defaultdict(int)
    for p in posts:
        for lbl in p.get("category_labels", []):
            label_counts[lbl] += 1
    label_dist = ", ".join([f"{k}: {v}" for k, v in sorted(label_counts.items())])
    return {
        "total_posts": len(posts),
        "total_threads": len(threads),
        "total_users": len(users),
        "label_dist": label_dist if label_dist else "none",
    }


def build_html(jsonl_path: Path, out_path: Path) -> None:
    posts = load_posts(jsonl_path)
    threads = group_by_thread(posts)
    threads_html = render_threads(threads)
    stats = compute_stats(posts)

    html = HTML_TEMPLATE.format(threads_html=threads_html, **stats)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote HTML to: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate HTML view of MVP dataset")
    parser.add_argument("--file", type=str, default="./data/mvp/posts_mvp.jsonl")
    parser.add_argument("--out", type=str, default="./data/mvp/posts_mvp.html")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_html(Path(os.path.abspath(args.file)), Path(os.path.abspath(args.out)))


if __name__ == "__main__":
    main()

