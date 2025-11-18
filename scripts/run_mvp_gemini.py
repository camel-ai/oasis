#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from yaml import safe_load

import oasis
from oasis import ActionType, Platform
from oasis.generation.gemini_client import GeminiConfig, generate_text
from oasis.social_platform.typing import DefaultPlatformType


def _seed_initial_follows_from_csv(env, edges_csv: Path) -> int:
    """Seed initial follow edges into the SQLite DB and AgentGraph.

    This mirrors the built-in CSV seeding path:
      1) INSERT rows into `follow(follower_id, followee_id, created_at)`
      2) UPDATE `user.num_followings` and `user.num_followers`
      3) Add in-memory edges to `AgentGraph` for runtime consistency

    Args:
        env: Active OASIS environment (already reset with signed-up users).
        edges_csv: Path to CSV with header: follower_id,followee_id

    Returns:
        int: Number of edges successfully seeded.
    """
    if not edges_csv.exists():
        return 0
    rows = []
    update_followings = []
    update_followers = []
    # Use current platform time; exact value is not critical for refresh logic
    try:
        current_time = env.platform.sandbox_clock.get_time_step()
    except Exception:
        current_time = 0

    with edges_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                follower = int(r["follower_id"])
                followee = int(r["followee_id"])
            except Exception:
                continue
            if follower == followee:
                continue
            rows.append((follower, followee, current_time))
            update_followings.append((follower,))
            update_followers.append((followee,))

    if not rows:
        return 0

    # Write to DB using platform utils (keeps commits consistent)
    follow_insert = (
        "INSERT INTO follow (follower_id, followee_id, created_at) "
        "VALUES (?, ?, ?)"
    )
    env.platform.pl_utils._execute_many_db_command(follow_insert, rows, commit=True)

    # Update user counters to reflect the seeded edges
    inc_followings = (
        "UPDATE user SET num_followings = num_followings + 1 WHERE user_id = ?"
    )
    env.platform.pl_utils._execute_many_db_command(
        inc_followings, update_followings, commit=True
    )
    inc_followers = (
        "UPDATE user SET num_followers = num_followers + 1 WHERE user_id = ?"
    )
    env.platform.pl_utils._execute_many_db_command(
        inc_followers, update_followers, commit=True
    )

    # Sync in-memory AgentGraph
    for follower, followee, _ in rows:
        try:
            env.agent_graph.add_edge(follower, followee)
        except Exception:
            # If graph back-end changes or ids out of range, skip safely
            pass
    return len(rows)


async def sign_up_agents_from_csv(env, personas_csv: Path) -> List[int]:
    # The env.reset() will sign up agents created in AgentGraph, but here we
    # will create manual controllable agents if needed. For MVP, we reuse the
    # Twitter CSV loader to create the agents first.
    return []


BASE_POST_PROMPT = (
    "Write one short tweet-length post in your voice. Include inline label tokens "
    "as instructed (e.g., <LBL:INCEL_SLANG>, <LBL:MISINFO_CLAIM>, <LBL:SUPPORTIVE>) "
    "exactly once if it fits. Do not explain the token."
)
BASE_COMMENT_PROMPT = (
    "Reply in your voice to the following thread. Include at most one inline label token "
    "like <LBL:INCEL_SLANG>, <LBL:MISINFO_CLAIM>, or <LBL:SUPPORTIVE> if it fits."
)
BASE_COMMENT_PROMPT_NO_CONTEXT = (
    "Reply in your voice to the ongoing conversation. Include at most one inline label token "
    "if it fits. Do not explain the token."
)
DEFAULT_FALLBACK_POST = "Observing the platform today <LBL:SUPPORTIVE>"
DEFAULT_FALLBACK_COMMENT = "Interesting point <LBL:SUPPORTIVE>"


@dataclass
class PersonaContext:
    system_prompt: str
    post_prompt: str
    comment_prompt: str
    lexical_required: List[str]
    lexical_optional: List[str]
    style_quirks: List[str]
    goal: str
    personality: str
    fallback_post: str
    fallback_comment: str
    variant: str
    rag_samples: List[str]

    @classmethod
    def empty(cls) -> "PersonaContext":
        return cls("", "", "", [], [], [], "", "", "", "", "", [])

    def fallback_post_text(self) -> str:
        return self.fallback_post or DEFAULT_FALLBACK_POST

    def fallback_comment_text(self) -> str:
        return self.fallback_comment or DEFAULT_FALLBACK_COMMENT

    def build_post_prompt(self, base_prompt: str) -> str:
        extras = self._build_extras(self.post_prompt, include_optional=True)
        return self._assemble_prompt(base_prompt, extras)

    def build_comment_prompt(self, base_prompt: str, thread_context: str) -> str:
        base = base_prompt.strip()
        if thread_context:
            base = f"{base}\n\n{thread_context}"
        extras = self._build_extras(self.comment_prompt, include_optional=True)
        return self._assemble_prompt(base, extras)

    def _build_extras(self, persona_prompt: str, *, include_optional: bool) -> List[str]:
        extras: List[str] = []
        if persona_prompt:
            extras.append(persona_prompt.strip())
        else:
            extras.append(
                "Vary your sentence openings and tie the message to specific details; avoid repeating stock phrases."
            )
        if self.goal:
            extras.append(f"Goal: {self.goal}.")
        if self.personality:
            extras.append(f"Personality cues: {self.personality}.")
        if self.style_quirks:
            extras.append(
                "Style quirks to emphasize: " + "; ".join(self.style_quirks) + "."
            )
        if self.lexical_required:
            extras.append(
                "Use vocabulary such as "
                + ", ".join(self.lexical_required)
                + " when it fits naturally."
            )
        if include_optional and self.lexical_optional:
            extras.append(
                "Optional flavor words: "
                + ", ".join(self.lexical_optional)
                + "."
            )
        references = self.sample_references(2)
        if references:
            formatted = "\n".join(f"- {ref}" for ref in references)
            extras.append("Reference snippets for tone:\n" + formatted)
        extras.append(
            "Keep your phrasing fresh; avoid repeating identical openings across messages."
        )
        return extras

    @staticmethod
    def _assemble_prompt(base_prompt: str, extras: List[str]) -> str:
        segments = [base_prompt.strip()]
        segments.extend(extra.strip() for extra in extras if extra and extra.strip())
        return "\n\n".join(segment for segment in segments if segment)

    def sample_references(self, k: int) -> List[str]:
        if not self.rag_samples:
            return []
        return random.sample(self.rag_samples, min(k, len(self.rag_samples)))


def _normalize_string(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _split_field(value: object) -> List[str]:
    text = _normalize_string(value)
    if not text:
        return []
    return [part.strip() for part in text.split(";") if part.strip()]


def _clean_reference_text(value: str) -> str:
    value = _normalize_string(value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def load_persona_contexts(csv_path: Path, rag_map: Dict[str, List[str]]) -> List[PersonaContext]:
    df = pd.read_csv(csv_path, keep_default_na=False)
    contexts: List[PersonaContext] = []
    for _, row in df.iterrows():
        variant = _normalize_string(row.get("persona_variant"))
        contexts.append(
            PersonaContext(
                system_prompt=_normalize_string(row.get("user_char")),
                post_prompt=_normalize_string(row.get("persona_prompt_post")),
                comment_prompt=_normalize_string(row.get("persona_prompt_comment")),
                lexical_required=_split_field(row.get("persona_lexical_required")),
                lexical_optional=_split_field(row.get("persona_lexical_optional")),
                style_quirks=_split_field(row.get("persona_style_quirks")),
                goal=_normalize_string(
                    row.get("persona_goal") or row.get("variant_goal")
                ),
                personality=_normalize_string(
                    row.get("persona_personality") or row.get("variant_persona_traits")
                ),
                fallback_post=_normalize_string(row.get("persona_fallback_post")),
                fallback_comment=_normalize_string(row.get("persona_fallback_comment")),
                variant=variant,
                rag_samples=rag_map.get(variant, []) if variant else [],
            )
        )
    if not contexts:
        contexts.append(PersonaContext.empty())
    return contexts


def load_rag_samples(path: Path) -> Dict[str, List[str]]:
    if not path.exists():
        return {}
    samples: Dict[str, List[str]] = defaultdict(list)  # type: ignore[var-annotated]
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            variant = _normalize_string(payload.get("persona_variant"))
            text = _clean_reference_text(str(payload.get("text", "")))
            if not variant or not text:
                continue
            samples.setdefault(variant, []).append(text)
    return samples


def _weighted_sample(action_mix: dict[str, float]) -> str:
    items = list(action_mix.items())
    if not items:
        return "create_post"
    total = sum(w for _, w in items)
    if total <= 0:
        return "create_post"
    r = random.random() * total
    upto = 0.0
    for name, w in items:
        upto += w
        if upto >= r:
            return name
    return items[-1][0]


async def _pick_post_id(agent) -> Optional[int]:
    try:
        rec = await agent.env.action.refresh()
        if rec.get("success") and rec.get("posts"):
            posts = rec["posts"]
            # posts is a list of dicts with post_id
            choice = random.choice(posts)
            return int(choice.get("post_id"))
    except Exception:
        pass
    return None


def _get_post_thread_context(env, post_id: int, max_comments: Optional[int] = None) -> tuple[str, list[str]]:
    """Fetch original post content and replies for context.

    If max_comments is None, returns all replies ordered from oldest to newest.
    """
    try:
        cur = env.platform.db_cursor
        cur.execute("SELECT content, user_id, created_at FROM post WHERE post_id = ?", (post_id, ))
        row = cur.fetchone()
        post_content = row[0] if row else ""

        if max_comments is None:
            cur.execute(
                "SELECT content FROM comment WHERE post_id = ? ORDER BY comment_id ASC",
                (post_id, ),
            )
            comments = [r[0] for r in cur.fetchall()]
        else:
            cur.execute(
                "SELECT content FROM comment WHERE post_id = ? ORDER BY comment_id DESC LIMIT ?",
                (post_id, max_comments),
            )
            # reverse to present oldest -> newest for readability
            comments = [r[0] for r in cur.fetchall()][::-1]
        return post_content or "", comments
    except Exception:
        return "", []


def _hash_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _follow_table_empty(env) -> bool:
    try:
        cur = env.platform.db_cursor
        cur.execute("SELECT COUNT(*) FROM follow")
        cnt = cur.fetchone()[0]
        return int(cnt) == 0
    except Exception:
        return True


def _write_graph_provenance(db_path: Path,
                            graph_cfg: Dict[str, Any] | None,
                            edges_csv: Optional[Path],
                            env) -> None:
    try:
        out_dir = Path(db_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "dataset_provenance.json"
        record: Dict[str, Any] = {}
        record["db_path"] = str(db_path)
        record["graph"] = {}
        if graph_cfg:
            record["graph"]["theta_b"] = graph_cfg.get("theta_b")
            record["graph"]["rho"] = graph_cfg.get("rho")
            record["graph"]["alpha"] = graph_cfg.get("alpha")
            record["graph"]["seed"] = graph_cfg.get("seed")
        if edges_csv and edges_csv.exists():
            record["graph"]["edges_csv"] = str(edges_csv)
            record["graph"]["edges_sha256"] = _hash_file_sha256(edges_csv)
        # Basic counts from DB (authoritative after seeding)
        try:
            cur = env.platform.db_cursor
            cur.execute("SELECT COUNT(*) FROM follow")
            record["graph"]["edges_in_db"] = int(cur.fetchone()[0])
            cur.execute("SELECT COUNT(*) FROM user")
            record["graph"]["nodes_in_db"] = int(cur.fetchone()[0])
        except Exception:
            pass
        # Append or write fresh (keep last record simple for MVP)
        if out_file.exists():
            try:
                with out_file.open("r", encoding="utf-8") as f:
                    existing = json.load(f)
            except Exception:
                existing = {}
            if isinstance(existing, dict):
                existing.update(record)
                with out_file.open("w", encoding="utf-8") as f:
                    json.dump(existing, f, indent=2, ensure_ascii=False)
            else:
                with out_file.open("w", encoding="utf-8") as f:
                    json.dump(record, f, indent=2, ensure_ascii=False)
        else:
            with out_file.open("w", encoding="utf-8") as f:
                json.dump(record, f, indent=2, ensure_ascii=False)
    except Exception:
        # Fail-soft for provenance
        pass


async def _seed_initial_posts(
    env,
    agent_ids: List[int],
    persona_contexts: List[PersonaContext],
    cfg: GeminiConfig,
    n: int,
    allow_fallback_text: bool,
) -> int:
    """Create a small pool of initial posts so follow feeds aren't empty at t0."""
    if not agent_ids or n <= 0:
        return 0
    selected = random.sample(agent_ids, k=min(n, len(agent_ids)))
    actions = {}
    for aid in selected:
        agent = env.agent_graph.get_agent(aid)
        persona_ctx = (
            persona_contexts[aid] if aid < len(persona_contexts) else PersonaContext.empty()
        )
        system_instruction = persona_ctx.system_prompt or ""
        try:
            user_prompt = persona_ctx.build_post_prompt(BASE_POST_PROMPT)
            text = generate_text(
                system_instruction=system_instruction,
                user_text=user_prompt,
                config=cfg,
            )
        except Exception:
            text = ""
        if not text and not allow_fallback_text:
            continue
        if not text:
            text = persona_ctx.fallback_post_text()
        actions[agent] = oasis.ManualAction(
            action_type=ActionType.CREATE_POST, action_args={"content": text}
        )
    if actions:
        await env.step(actions)
        try:
            await env.platform.update_rec_table()
        except Exception:
            pass
        return len(actions)
    return 0


async def run(
    db_path: Path,
    personas_csv: Path,
    steps: int,
    posts_per_step: int,
    temperature: float,
    platform_cfg: dict | None = None,
    action_mix: Optional[dict] = None,
    allow_fallback_text: bool = True,
    rag_map: Dict[str, List[str]] | None = None,
    graph_cfg: dict | None = None,
    seed_only: bool = False,
    disable_initial_posts: bool = False,
) -> None:
    # Build empty agent graph via personas CSV (standard function)
    # We avoid providing a model; we will drive manual posts using Gemini.
    agent_graph = await oasis.generate_twitter_agent_graph(
        profile_path=str(personas_csv), model=None, available_actions=None
    )

    os.environ["OASIS_DB_PATH"] = os.path.abspath(str(db_path))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        os.remove(db_path)

    if platform_cfg:
        recsys_type = platform_cfg.get("recsys_type", "twhin-bert")
        refresh_rec_post_count = int(platform_cfg.get("refresh_rec_post_count", 2))
        max_rec_post_len = int(platform_cfg.get("max_rec_post_len", 2))
        following_post_count = int(platform_cfg.get("following_post_count", 3))
        allow_self_rating = bool(platform_cfg.get("allow_self_rating", True))
        show_score = bool(platform_cfg.get("show_score", False))
        platform_obj = Platform(
            db_path=str(db_path),
            recsys_type=recsys_type,
            refresh_rec_post_count=refresh_rec_post_count,
            max_rec_post_len=max_rec_post_len,
            following_post_count=following_post_count,
            allow_self_rating=allow_self_rating,
            show_score=show_score,
        )
        env = oasis.make(
            agent_graph=agent_graph,
            platform=platform_obj,
            database_path=str(db_path),
        )
    else:
        env = oasis.make(
            agent_graph=agent_graph,
            platform=DefaultPlatformType.TWITTER,
            database_path=str(db_path),
        )

    await env.reset()

    # Seed initial follow network if available (SBM+PA edges)
    try:
        # Only seed once per DB (skip if follow table already has rows)
        if _follow_table_empty(env):
            cfg_edges = (graph_cfg or {}).get("edges_csv") if graph_cfg else None
            edges_path = Path(os.getenv("MVP_EDGES_CSV", cfg_edges or "./data/edges_mvp.csv"))
            seeded = _seed_initial_follows_from_csv(env, edges_path)
            if seeded:
                # Basic stats print
                cur = env.platform.db_cursor
                cur.execute("SELECT COUNT(*) FROM user")
                n_users = int(cur.fetchone()[0])
                cur.execute("SELECT COUNT(*) FROM follow")
                n_edges = int(cur.fetchone()[0])
                avg_out = (n_edges / n_users) if n_users > 0 else 0.0
                print(f"Seeded {seeded} follow edges from {edges_path} | nodes={n_users} edges={n_edges} avg_outdeg={avg_out:.2f}")
            # Write graph provenance (params + file hash + counts)
            _write_graph_provenance(db_path=db_path, graph_cfg=graph_cfg, edges_csv=edges_path, env=env)
        else:
            print("Follow table is non-empty; skipping follow seeding.")
    except Exception as e:
        print(f"Seeding follows failed: {e}")

    persona_contexts = load_persona_contexts(personas_csv, rag_map or {})
    agent_ids = [agent.social_agent_id for _, agent in env.agent_graph.get_agents()]
    cfg = GeminiConfig(
        api_key=os.getenv("GEMINI_API_KEY", ""),
        temperature=temperature,
        model_id=os.getenv("GEMINI_MODEL", os.getenv("GEMINI_DEFAULT_MODEL", "gemini-2.5-flash")),
    )
    print(f"Using Gemini 2.5 Flash-Lite with temperature={temperature}, safety=OFF")

    # Seed a small pool of initial posts (so follow feeds aren't empty at t0) unless disabled
    if not disable_initial_posts:
        try:
            seed_count = max(5, min(50, len(agent_ids) // 3))
            seeded_posts = await _seed_initial_posts(
                env=env,
                agent_ids=agent_ids,
                persona_contexts=persona_contexts,
                cfg=cfg,
                n=seed_count,
                allow_fallback_text=allow_fallback_text,
            )
            if seeded_posts:
                print(f"Seeded {seeded_posts} initial posts")
        except Exception:
            pass

    # If seed-only, do a single recsys table update and exit
    if seed_only:
        try:
            await env.platform.update_rec_table()
        except Exception:
            pass
        await env.close()
        return

    for _ in range(steps):
        # keep recsys fresh before sampling targets
        try:
            await env.platform.update_rec_table()
        except Exception:
            pass
        selected = random.sample(agent_ids, k=min(posts_per_step, len(agent_ids)))
        actions = {}
        for aid in selected:
            agent = env.agent_graph.get_agent(aid)
            persona_ctx = (
                persona_contexts[aid]
                if aid < len(persona_contexts)
                else PersonaContext.empty()
            )
            system_instruction = persona_ctx.system_prompt or ""
            act_name = _weighted_sample(action_mix or {})
            if act_name == "create_post":
                try:
                    user_prompt = persona_ctx.build_post_prompt(BASE_POST_PROMPT)
                    text = generate_text(
                        system_instruction=system_instruction,
                        user_text=user_prompt,
                        config=cfg,
                    )
                except Exception:
                    text = ""
                if not text and not allow_fallback_text:
                    actions[agent] = oasis.ManualAction(
                        action_type=ActionType.DO_NOTHING, action_args={}
                    )
                else:
                    if not text:
                        text = persona_ctx.fallback_post_text()
                    actions[agent] = oasis.ManualAction(
                        action_type=ActionType.CREATE_POST, action_args={"content": text}
                    )
            elif act_name == "create_comment":
                pid = await _pick_post_id(agent)
                if pid is None:
                    actions[agent] = oasis.ManualAction(
                        action_type=ActionType.DO_NOTHING, action_args={}
                    )
                else:
                    post_text, recent_replies = _get_post_thread_context(env, pid, max_comments=None)
                    context_lines = []
                    if post_text:
                        context_lines.append(f"Original post: \"{post_text}\"")
                    if recent_replies:
                        context_lines.append("Recent replies:")
                        for rr in recent_replies:
                            context_lines.append(f"- {rr}")
                    thread_context = "\n".join(context_lines)
                    base_prompt = (
                        BASE_COMMENT_PROMPT
                        if thread_context
                        else BASE_COMMENT_PROMPT_NO_CONTEXT
                    )
                    user_prompt = persona_ctx.build_comment_prompt(
                        base_prompt, thread_context
                    )
                    try:
                        text = generate_text(
                            system_instruction=system_instruction,
                            user_text=user_prompt,
                            config=cfg,
                        )
                    except Exception:
                        text = ""
                    if not text and not allow_fallback_text:
                        actions[agent] = oasis.ManualAction(
                            action_type=ActionType.DO_NOTHING, action_args={}
                        )
                    else:
                        if not text:
                            text = persona_ctx.fallback_comment_text()
                        actions[agent] = oasis.ManualAction(
                            action_type=ActionType.CREATE_COMMENT,
                            action_args={"post_id": pid, "content": text},
                        )
            elif act_name == "like_post":
                pid = await _pick_post_id(agent)
                if pid is None:
                    actions[agent] = oasis.ManualAction(
                        action_type=ActionType.DO_NOTHING, action_args={}
                    )
                else:
                    actions[agent] = oasis.ManualAction(
                        action_type=ActionType.LIKE_POST, action_args={"post_id": pid}
                    )
            elif act_name == "follow":
                # pick a random other user id
                others = [uid for uid in agent_ids if uid != aid]
                if others:
                    target = random.choice(others)
                    actions[agent] = oasis.ManualAction(
                        action_type=ActionType.FOLLOW, action_args={"followee_id": int(target)}
                    )
                else:
                    actions[agent] = oasis.ManualAction(
                        action_type=ActionType.DO_NOTHING, action_args={}
                    )
            else:
                actions[agent] = oasis.ManualAction(
                    action_type=ActionType.DO_NOTHING, action_args={}
                )

        if actions:
            await env.step(actions)

    await env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MVP with Gemini-generated posts")
    parser.add_argument("--db", type=str, default="./data/mvp/oasis_mvp_gemini.db")
    parser.add_argument("--personas", type=str, default="./data/personas_mvp.csv")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--posts-per-step", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--rag-corpus", type=str, default="./data/rag_corpus/persona_corpus.jsonl")
    parser.add_argument("--seed-only", action="store_true", help="Initialize DB, seed follows, update rec table once, then exit.")
    parser.add_argument("--disable-initial-posts", action="store_true", help="Skip seeding the initial pool of posts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = safe_load(f) or {}
        sim = cfg.get("simulation", {})
        platform_cfg = cfg.get("platform", {})
        personas_cfg = cfg.get("personas", {})
        graph_cfg = cfg.get("graph", {})
        db = sim.get("db", args.db)
        personas = sim.get("personas", args.personas)
        steps = int(sim.get("steps", args.steps))
        posts_per_step = int(sim.get("posts_per_step", args.posts_per_step))
        temperature = float(sim.get("temperature", args.temperature))
        action_mix = sim.get("action_mix", {})
        allow_fallback_text = bool(sim.get("allow_fallback_text", True))
        gem_model = sim.get("gemini_model")
        if gem_model:
            os.environ["GEMINI_MODEL"] = str(gem_model)
        rag_corpus = personas_cfg.get("rag_corpus", args.rag_corpus)
    else:
        db = args.db
        personas = args.personas
        steps = args.steps
        posts_per_step = args.posts_per_step
        temperature = args.temperature
        platform_cfg = None
        action_mix = None
        allow_fallback_text = True
        rag_corpus = args.rag_corpus
        personas_cfg = {}

    rag_samples = load_rag_samples(Path(rag_corpus))

    asyncio.run(
        run(
            db_path=Path(db),
            personas_csv=Path(personas),
            steps=steps,
            posts_per_step=posts_per_step,
            temperature=temperature,
            platform_cfg=platform_cfg,
            action_mix=action_mix,
            allow_fallback_text=allow_fallback_text,
            rag_map=rag_samples,
            graph_cfg=graph_cfg,
            seed_only=args.seed_only,
            disable_initial_posts=args.disable_initial_posts,
        )
    )


if __name__ == "__main__":
    main()


