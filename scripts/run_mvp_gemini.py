#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

from yaml import safe_load

import oasis
from oasis import ActionType, Platform
from oasis.generation.gemini_client import GeminiConfig, generate_text
from oasis.social_platform.typing import DefaultPlatformType


async def sign_up_agents_from_csv(env, personas_csv: Path) -> List[int]:
    # The env.reset() will sign up agents created in AgentGraph, but here we
    # will create manual controllable agents if needed. For MVP, we reuse the
    # Twitter CSV loader to create the agents first.
    return []


def read_persona_cards(csv_path: Path) -> List[str]:
    cards: List[str] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cards.append(row.get("user_char", ""))
    return cards


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


async def run(db_path: Path, personas_csv: Path, steps: int, posts_per_step: int,
              temperature: float, platform_cfg: dict | None = None,
              action_mix: Optional[dict] = None,
              allow_fallback_text: bool = True) -> None:
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

    persona_cards = read_persona_cards(personas_csv)
    agent_ids = [agent.social_agent_id for _, agent in env.agent_graph.get_agents()]
    cfg = GeminiConfig(
        api_key=os.getenv("GEMINI_API_KEY", ""),
        temperature=temperature,
        model_id=os.getenv("GEMINI_MODEL", os.getenv("GEMINI_DEFAULT_MODEL", "gemini-2.5-flash")),
    )
    print(f"Using Gemini 2.5 Flash-Lite with temperature={temperature}, safety=OFF")

    prompt_suffix = (
        "Write one short tweet-length post in your voice. Include inline label tokens "
        "as instructed (e.g., <LBL:INCEL_SLANG>, <LBL:MISINFO_CLAIM>, <LBL:SUPPORTIVE>) "
        "exactly once if it fits. Do not explain the token."
    )

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
            persona = persona_cards[aid] if aid < len(persona_cards) else ""
            act_name = _weighted_sample(action_mix or {})
            if act_name == "create_post":
                try:
                    text = generate_text(
                        system_instruction=persona,
                        user_text=prompt_suffix,
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
                        text = "Observing the platform today <LBL:SUPPORTIVE>"
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
                    try:
                        text = generate_text(
                            system_instruction=persona,
                            user_text=(
                                "Reply in your voice to the following thread. "
                                "Include at most one inline label token like <LBL:INCEL_SLANG>, <LBL:MISINFO_CLAIM>, or <LBL:SUPPORTIVE> if it fits.\n\n"
                                f"{thread_context}" if thread_context else
                                "Write a short reply in your voice; include one inline label token if it fits."
                            ),
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
                            text = "Interesting point <LBL:SUPPORTIVE>"
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = safe_load(f) or {}
        sim = cfg.get("simulation", {})
        platform_cfg = cfg.get("platform", {})
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
    else:
        db = args.db
        personas = args.personas
        steps = args.steps
        posts_per_step = args.posts_per_step
        temperature = args.temperature
        platform_cfg = None
        action_mix = None
        allow_fallback_text = True

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
        )
    )


if __name__ == "__main__":
    main()


