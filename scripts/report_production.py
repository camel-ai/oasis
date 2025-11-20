#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx

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

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_minimal_static_bank(path: Path) -> None:
    if path.exists():
        return
    ensure_dir(path.parent)
    path.write_text(
        "LBL:INCEL_SLANG:\n"
        "  - 'blackpill'\n"
        "LBL:MISINFO_CLAIM:\n"
        "  - 'hidden agenda'\n"
        "LBL:SUPPORTIVE:\n"
        "  - 'stay strong'\n"
        "LBL:MISINFO_SOURCE:\n"
        "  - 'trusted source says'\n"
        "LBL:HARASSMENT:\n"
        "  - 'insult'\n",
        encoding="utf-8",
    )


def export_jsonl_with_labels(
    db_path: Path,
    out_jsonl: Path,
    sidecar_path: Optional[Path],
    static_bank: Optional[Path],
    skip_imputation: bool = True,
    seed: int = 314159,
) -> None:
    """
    Export posts/comments with labels to JSONL.
    Tries to import scripts/build_dataset.py; falls back to spawning a subprocess if import fails.
    """
    build_ds_file = Path(__file__).resolve().parent / "build_dataset.py"
    if not build_ds_file.exists():
        build_ds_file = Path(__file__).resolve().parent / "build_dataset.py"
    if not build_ds_file.exists():
        raise FileNotFoundError(f"Expected exporter at {build_ds_file}")
    bank_path = static_bank or (out_jsonl.parent / "label_tokens_static_bank.yaml")
    write_minimal_static_bank(bank_path)

    ensure_dir(out_jsonl.parent)
    sidecar = sidecar_path if (sidecar_path and sidecar_path.exists()) else None
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("oasis_build_dataset", str(build_ds_file))
        if spec is None or spec.loader is None:
            raise RuntimeError("spec loader missing")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        if not hasattr(module, "build_dataset"):
            raise AttributeError("build_dataset not found in module")
        module.build_dataset(  # type: ignore[attr-defined]
            db_path=db_path,
            out_path=out_jsonl,
            bank_path=bank_path,
            seed=seed,
            skip_imputation=skip_imputation,
            sidecar=sidecar,
        )
    except Exception:
        # Fallback to subprocess execution
        import subprocess, sys
        cmd: List[str] = [
            sys.executable,
            str(build_ds_file),
            "--db",
            str(db_path),
            "--out",
            str(out_jsonl),
            "--static-bank",
            str(bank_path),
        ]
        if skip_imputation:
            cmd.append("--skip-imputation")
        if sidecar:
            cmd.extend(["--sidecar", str(sidecar)])
        subprocess.run(cmd, check=True)


def plot_action_timeline(actions: List[Tuple[str, int]], out_path: Path) -> None:
    if not actions:
        return
    names = [a for a, _ in actions]
    counts = [n for _, n in actions]
    plt.figure(figsize=(10, 4))
    plt.bar(names, counts, color="steelblue")
    plt.title("Action counts")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def build_interaction_network(
    db_path: Path,
    out_path: Path,
) -> None:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    # Post author mapping
    cur.execute("SELECT post_id, user_id FROM post")
    post_author = {pid: uid for pid, uid in cur.fetchall()}
    # Likes
    cur.execute("SELECT user_id, post_id FROM like")
    likes = cur.fetchall()
    # Reposts/quotes (original_post_id != NULL)
    cur.execute(
        "SELECT user_id, original_post_id, quote_content FROM post WHERE original_post_id IS NOT NULL"
    )
    reposts_quotes = cur.fetchall()
    # Comments
    cur.execute("SELECT user_id, post_id FROM comment")
    comments = cur.fetchall()
    conn.close()

    G = nx.DiGraph()
    for uid in set(post_author.values()):
        G.add_node(uid)
    # Likes edges
    for liker, pid in likes:
        author = post_author.get(pid)
        if author is not None and liker != author:
            w = G.get_edge_data(liker, author, {}).get("weight", 0) + 1
            G.add_edge(liker, author, weight=w, interaction="like")
    # Reposts/quotes edges
    for uid, opid, quote in reposts_quotes:
        if opid is None:
            continue
        author = post_author.get(opid)
        if author is not None and uid != author:
            w_add = 2 if (quote is None or quote == "") else 1.5
            w = G.get_edge_data(uid, author, {}).get("weight", 0) + w_add
            G.add_edge(uid, author, weight=w, interaction="repost" if w_add == 2 else "quote")
    # Comments edges
    for uid, pid in comments:
        author = post_author.get(pid)
        if author is not None and uid != author:
            w = G.get_edge_data(uid, author, {}).get("weight", 0) + 1.5
            G.add_edge(uid, author, weight=w, interaction="comment")

    if G.number_of_nodes() == 0:
        return

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    weights = [G[u][v].get("weight", 1) for u, v in G.edges()]
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=400)
    nx.draw_networkx_edges(
        G, pos, width=[w * 0.5 for w in weights], alpha=0.4, arrows=True, arrowsize=10
    )
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.title("Interaction Network")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def read_sidecar_labels(
    sidecar_path: Optional[Path], post_ids: set[int], comment_ids: set[int]
) -> Tuple[Dict[int, List[str]], Dict[int, List[str]]]:
    post_labels: Dict[int, List[str]] = {}
    comment_labels: Dict[int, List[str]] = {}
    if not sidecar_path or not sidecar_path.exists():
        return post_labels, comment_labels
    with sidecar_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            pid = rec.get("post_id")
            cid = rec.get("comment_id")
            labs = rec.get("category_labels") or []
            if isinstance(pid, int) and pid in post_ids:
                post_labels[pid] = list(labs)
            if isinstance(cid, int) and cid in comment_ids:
                comment_labels[cid] = list(labs)
    return post_labels, comment_labels


def write_html_report(
    db_path: Path,
    sidecar_path: Optional[Path],
    out_dir: Path,
    counts: Dict[str, int],
    actions: List[Tuple[str, int]],
    post_rows: List[Tuple[int, int, int, str]],
    comment_rows: List[Tuple[int, int, int, str]],
    post_labels: Dict[int, List[str]],
    comment_labels: Dict[int, List[str]],
) -> Path:
    report_path = out_dir / "production_report.html"

    def esc(s: object) -> str:
        return html.escape(str(s)) if s is not None else ""

    parts: List[str] = []
    parts.append("<h1>Production Run Report</h1>")
    parts.append(
        f"<p><b>DB:</b> {esc(db_path)}"
        + (f" | <b>Sidecar:</b> {esc(sidecar_path)}" if sidecar_path else "")
        + "</p>"
    )
    parts.append("<h2>Counts</h2>")
    parts.append(
        "<ul>"
        + "".join(
            f"<li>{esc(k)}: {esc(v)}</li>" for k, v in counts.items()
        )
        + "</ul>"
    )
    parts.append(
        "<h2>Actions</h2><ul>"
        + "".join(f"<li>{esc(a)}: {esc(n)}</li>" for a, n in actions)
        + "</ul>"
    )

    # Embed images if present
    for title, img in [
        ("Action Timeline", out_dir / "action_timeline.png"),
        ("Interaction Network", out_dir / "interaction_network.png"),
    ]:
        if img.exists():
            parts.append(f"<h2>{esc(title)}</h2><img src=\"{esc(img)}\" width=\"900\" />")

    # Top posts
    parts.append("<h2>Top Posts (by likes)</h2>")
    for pid, uid, likes, content in post_rows:
        labs = ", ".join(post_labels.get(pid, [])) or "—"
        snippet = esc((content or "")[:1000]).replace("\n", " ")
        parts.append(
            f"<div><b>post {pid}</b> | user {uid} | likes {likes} | labels: {esc(labs)}<br>{snippet}</div><hr>"
        )

    # Comments
    parts.append("<h2>Latest Comments</h2>")
    for cid, uid, pid, content in comment_rows:
        labs = ", ".join(comment_labels.get(cid, [])) or "—"
        snippet = esc((content or "")[:1000]).replace("\n", " ")
        parts.append(
            f"<div><b>comment {cid}</b> | user {uid} | post {pid} | labels: {esc(labs)}<br>{snippet}</div><hr>"
        )

    report_path.write_text(
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>Production Report</title></head>"
        "<body>" + "".join(parts) + "</body></html>",
        encoding="utf-8",
    )
    return report_path

def infer_persona_from_username(name: Optional[str]) -> str:
    if not name:
        return "unknown"
    n = name.lower()
    if n.startswith("incel_"):
        return "incel"
    if n.startswith("misinfo_"):
        return "misinfo"
    if n.startswith("benign_"):
        return "benign"
    return "unknown"

def build_threaded_html(
    db_path: Path,
    sidecar_path: Optional[Path],
    out_path: Path,
    limit_threads: int = 50,
) -> None:
    # Load sidecar labels
    post_labels: Dict[int, List[str]] = {}
    comment_labels: Dict[int, List[str]] = {}
    if sidecar_path and sidecar_path.exists():
        with sidecar_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                pid = rec.get("post_id")
                cid = rec.get("comment_id")
                labs = rec.get("category_labels") or []
                if isinstance(pid, int):
                    post_labels[pid] = list(labs)
                if isinstance(cid, int):
                    comment_labels[cid] = list(labs)

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    # map user_id -> user_name (fallback to name)
    cur.execute("SELECT user_id, user_name, name FROM user")
    uid_to_name = {int(uid): (uname if uname else name) for uid, uname, name in cur.fetchall()}
    # latest root posts (original_post_id is NULL)
    cur.execute(
        "SELECT post_id, user_id, content, created_at "
        "FROM post WHERE original_post_id IS NULL ORDER BY post_id DESC LIMIT ?",
        (int(limit_threads),),
    )
    roots = cur.fetchall()
    # prefetch comments grouped by post_id
    cur.execute(
        "SELECT comment_id, post_id, user_id, content, created_at "
        "FROM comment ORDER BY comment_id ASC"
    )
    comments = cur.fetchall()
    conn.close()

    # Build stats and threads using visualize_mvp-style structure
    total_posts = 0
    users_set: set[str] = set()
    label_counts: Dict[str, int] = {}

    comments_by_post: Dict[int, List[Tuple[int, int, str, str]]] = {}
    for cid, pid, uid, ctext, cts in comments:
        comments_by_post.setdefault(int(pid), []).append(
            (int(cid), int(uid), str(ctext or ""), str(cts or ""))
        )

    def esc(s: str) -> str:
        return html.escape(s)

    def persona_class(pers: str) -> str:
        base = "persona"
        if pers == "incel":
            return f"{base} persona-incel"
        if pers == "misinfo":
            return f"{base} persona-misinfo"
        if pers == "benign":
            return f"{base} persona-benign"
        if pers == "recovery":
            return f"{base} persona-recovery"
        return base

    thread_parts: List[str] = []
    for pid, uid, text, ts in roots:
        uid = int(uid)
        pid = int(pid)
        uname = uid_to_name.get(uid, f"user_{uid}") or f"user_{uid}"
        persona = infer_persona_from_username(uname)
        root_labels = post_labels.get(pid, [])

        users_set.add(str(uname))
        for lbl in root_labels:
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
        total_posts += 1

        parts: List[str] = []
        parts.append('<div class="thread">')
        # Root post
        parts.append('<div class="post">')
        parts.append(
            f'<div class="header">'
            f'<span><span class="username">{esc(str(uname))}</span> '
            f'<span class="{persona_class(persona)}">{esc(persona)}</span></span>'
            f'<span>{esc(str(ts))[:19] if ts else ""}</span>'
            f'</div>'
        )
        parts.append(f'<div class="text">{esc(str(text or ""))}</div>')
        if root_labels:
            parts.append(
                '<div class="labels">'
                + " ".join(f'<span class="label">{esc(l)}</span>' for l in root_labels)
                + '<span style="color: #657786; margin-left: 10px;">confidence: {:.2f}</span>'.format(0.0)
                + "</div>"
            )
        parts.append("</div>")  # end root .post

        # Replies
        for cid, cuid, ctext, cts in comments_by_post.get(pid, []):
            cuname = uid_to_name.get(cuid, f"user_{cuid}") or f"user_{cuid}"
            cpersona = infer_persona_from_username(cuname)
            clabels = comment_labels.get(cid, [])
            users_set.add(str(cuname))
            for lbl in clabels:
                label_counts[lbl] = label_counts.get(lbl, 0) + 1
            total_posts += 1

            parts.append('<div class="comment">')
            parts.append(
                f'<div class="header">'
                f'<span><span class="username">{esc(str(cuname))}</span> '
                f'<span class="{persona_class(cpersona)}">{esc(cpersona)}</span></span>'
                f'<span>{esc(str(cts))[:19] if cts else ""}</span>'
                f'</div>'
            )
            parts.append(f'<div class="text">{esc(str(ctext))}</div>')
            if clabels:
                parts.append(
                    '<div class="labels">'
                    + " ".join(f'<span class="label">{esc(l)}</span>' for l in clabels)
                    + '<span style="color: #657786; margin-left: 10px;">confidence: {:.2f}</span>'.format(0.0)
                    + "</div>"
                )
            parts.append("</div>")  # end .comment

        parts.append("</div>")  # end .thread
        thread_parts.append("\n".join(parts))

    label_dist = ", ".join(f"{k}: {v}" for k, v in sorted(label_counts.items()))
    html_text = HTML_TEMPLATE.format(
        threads_html="\n".join(thread_parts),
        total_posts=total_posts,
        total_threads=len(roots),
        total_users=len(users_set),
        label_dist=label_dist if label_dist else "none",
    )
    out_path.write_text(html_text, encoding="utf-8")

def export_actions_jsonl(
    db_path: Path,
    out_path: Path,
) -> None:
    """
    Export actions (trace + table-derived) to a JSONL file.
    Includes:
      - trace actions (action, user_id, created_at, info)
      - likes (like_post)
      - comments (create_comment)
      - posts with original_post_id (repost_post / quote_post), plain posts as create_post
      - follows (follow)
    """
    ensure_dir(out_path.parent)
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    with out_path.open("w", encoding="utf-8") as fh:
        # trace
        cur.execute("SELECT user_id, created_at, action, info FROM trace ORDER BY created_at ASC")
        for uid, ts, action, info in cur.fetchall():
            obj = {"source":"trace","action": action, "user_id": uid, "created_at": str(ts)}
            # try parse info
            try:
                obj["info"] = json.loads(info) if info else None
            except Exception:
                obj["info"] = info
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        # likes
        cur.execute("SELECT user_id, post_id, created_at FROM like ORDER BY like_id ASC")
        for uid, pid, ts in cur.fetchall():
            fh.write(json.dumps({
                "source":"table",
                "action":"like_post",
                "user_id": uid,
                "post_id": pid,
                "created_at": str(ts),
            }, ensure_ascii=False) + "\n")
        # comments
        cur.execute("SELECT comment_id, user_id, post_id, content, created_at FROM comment ORDER BY comment_id ASC")
        for cid, uid, pid, text, ts in cur.fetchall():
            fh.write(json.dumps({
                "source":"table",
                "action":"create_comment",
                "comment_id": cid,
                "user_id": uid,
                "post_id": pid,
                "content": text,
                "created_at": str(ts),
            }, ensure_ascii=False) + "\n")
        # posts (create / repost / quote)
        cur.execute("SELECT post_id, user_id, original_post_id, quote_content, content, created_at FROM post ORDER BY post_id ASC")
        for pid, uid, opid, qtext, ptext, ts in cur.fetchall():
            if opid is None:
                action = "create_post"
            else:
                action = "quote_post" if (qtext not in (None, "")) else "repost_post"
            fh.write(json.dumps({
                "source":"table",
                "action": action,
                "post_id": pid,
                "user_id": uid,
                "original_post_id": opid,
                "content": ptext,
                "quote_content": qtext,
                "created_at": str(ts),
            }, ensure_ascii=False) + "\n")
        # follows (current follow edges as 'follow' snapshots)
        cur.execute("SELECT follower_id, followee_id, created_at FROM follow ORDER BY rowid ASC")
        for fid, feid, ts in cur.fetchall():
            fh.write(json.dumps({
                "source":"table",
                "action":"follow",
                "follower_id": fid,
                "followee_id": feid,
                "created_at": str(ts),
            }, ensure_ascii=False) + "\n")
    conn.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize and export a production run")
    ap.add_argument("--db", required=True, help="Path to production SQLite DB")
    ap.add_argument("--sidecar", default="", help="Path to sidecar.jsonl (optional)")
    ap.add_argument("--out-dir", default="data/production", help="Output directory")
    ap.add_argument(
        "--export-jsonl",
        default="data/production/production_export.jsonl",
        help="Path to write JSONL export with labels",
    )
    ap.add_argument(
        "--export-actions",
        default="",
        help="Optional path to write actions JSONL (trace + likes/comments/posts/follows)",
    )
    ap.add_argument(
        "--threads-html",
        default="",
        help="Optional path to write threaded HTML view (visualize_mvp-style layout)",
    )
    ap.add_argument(
        "--static-bank",
        default="data/production/label_tokens_static_bank.yaml",
        help="Static bank for token imputation (created if missing)",
    )
    ap.add_argument(
        "--no-export",
        action="store_true",
        help="Skip JSONL export step",
    )
    args = ap.parse_args()

    db_path = Path(args.db).resolve()
    sidecar_path = Path(args.sidecar).resolve() if args.sidecar else None
    out_dir = Path(args.out_dir).resolve()
    out_jsonl = Path(args.export_jsonl).resolve()
    static_bank = Path(args.static_bank).resolve() if args.static_bank else None

    ensure_dir(out_dir)

    # 1) Optional JSONL export with labels
    if not args.no_export:
        export_jsonl_with_labels(
            db_path=db_path,
            out_jsonl=out_jsonl,
            sidecar_path=sidecar_path,
            static_bank=static_bank,
            skip_imputation=True,
        )

    # 2) Read DB for counts and actions
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM user")
    users = int(cur.fetchone()[0])
    cur.execute("SELECT COUNT(*) FROM post")
    posts = int(cur.fetchone()[0])
    cur.execute("SELECT COUNT(*) FROM comment")
    comments = int(cur.fetchone()[0])
    cur.execute("SELECT action, COUNT(*) FROM trace GROUP BY action ORDER BY COUNT(*) DESC")
    actions = [(str(a), int(n)) for a, n in cur.fetchall()]
    # top posts by likes
    cur.execute(
        """
        SELECT p.post_id, p.user_id, IFNULL(l.cnt,0) AS likes, p.content
        FROM post p
        LEFT JOIN (SELECT post_id, COUNT(*) cnt FROM like GROUP BY post_id) l
        ON p.post_id=l.post_id
        ORDER BY likes DESC, p.post_id DESC LIMIT 15
        """
    )
    post_rows = [(int(pid), int(uid), int(lk), str(txt or "")) for pid, uid, lk, txt in cur.fetchall()]
    cur.execute("SELECT comment_id, user_id, post_id, content FROM comment ORDER BY comment_id DESC LIMIT 15")
    comment_rows = [
        (int(cid), int(uid), int(pid), str(txt or "")) for cid, uid, pid, txt in cur.fetchall()
    ]
    # used for filtering sidecar
    cur.execute("SELECT post_id FROM post")
    post_ids = {int(r[0]) for r in cur.fetchall()}
    cur.execute("SELECT comment_id FROM comment")
    comment_ids = {int(r[0]) for r in cur.fetchall()}
    conn.close()

    # 3) Plots
    plot_action_timeline(actions, out_dir / "action_timeline.png")
    build_interaction_network(db_path, out_dir / "interaction_network.png")

    # 4) Sidecar labels
    post_labels, comment_labels = read_sidecar_labels(sidecar_path, post_ids, comment_ids)

    # 5) HTML report
    counts = {"users": users, "posts": posts, "comments": comments}
    report_path = write_html_report(
        db_path=db_path,
        sidecar_path=sidecar_path,
        out_dir=out_dir,
        counts=counts,
        actions=actions,
        post_rows=post_rows,
        comment_rows=comment_rows,
        post_labels=post_labels,
        comment_labels=comment_labels,
    )
    print(f"Wrote {report_path}")

    # 6) Optional actions JSONL
    if args.export_actions:
        export_actions_jsonl(db_path=db_path, out_path=Path(args.export_actions).resolve())
        print(f"Wrote {Path(args.export_actions).resolve()}")

    # 7) Optional threaded HTML view
    if args.threads_html:
        threads_path = Path(args.threads_html).resolve()
        build_threaded_html(
            db_path=db_path,
            sidecar_path=sidecar_path,
            out_path=threads_path,
            limit_threads=50,
        )
        print(f"Wrote {threads_path}")


if __name__ == "__main__":
    main()


