#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

import pandas as pd
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


BASE_POST_PROMPT = (
    "Write one short tweet-length post in your voice. Include inline label tokens "
    "such as <LBL:BENIGN>, <LBL:SUPPORTIVE>, <LBL:ED_RISK>, <LBL:INCEL_SLANG>, <LBL:MISINFO_CLAIM>, <LBL:CONSPIRACY> "
    "exactly once per requested label set. Do not explain the token."
)
BASE_COMMENT_PROMPT = (
    "Reply in your voice to the following thread. Include at most one inline label token "
    "like <LBL:BENIGN>, <LBL:SUPPORTIVE>, <LBL:ED_RISK>, <LBL:INCEL_SLANG>, <LBL:MISINFO_CLAIM>, or <LBL:CONSPIRACY> if it fits."
)
BASE_COMMENT_PROMPT_NO_CONTEXT = (
    "Reply in your voice to the ongoing conversation. Include at most one inline label token "
    "matching the requested labels if it fits. Do not explain the token."
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


def load_persona_context_records(
    csv_path: Path, rag_map: Dict[str, List[str]]
) -> Tuple[List[PersonaContext], List[Dict[str, object]]]:
    df = pd.read_csv(csv_path, keep_default_na=False)
    contexts: List[PersonaContext] = []
    metadata: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        variant = _normalize_string(row.get("persona_variant"))
        context = PersonaContext(
            system_prompt=_normalize_string(row.get("user_char")),
            post_prompt=_normalize_string(row.get("persona_prompt_post")),
            comment_prompt=_normalize_string(row.get("persona_prompt_comment")),
            lexical_required=_split_field(row.get("persona_lexical_required")),
            lexical_optional=_split_field(row.get("persona_lexical_optional")),
            style_quirks=_split_field(row.get("persona_style_quirks")),
            goal=_normalize_string(row.get("persona_goal") or row.get("variant_goal")),
            personality=_normalize_string(
                row.get("persona_personality") or row.get("variant_persona_traits")
            ),
            fallback_post=_normalize_string(row.get("persona_fallback_post")),
            fallback_comment=_normalize_string(row.get("persona_fallback_comment")),
            variant=variant,
            rag_samples=rag_map.get(variant, []) if variant else [],
        )
        contexts.append(context)
        metadata.append(row.to_dict())
    if not contexts:
        contexts.append(PersonaContext.empty())
        metadata.append({})
    return contexts, metadata


def load_persona_contexts(csv_path: Path, rag_map: Dict[str, List[str]]) -> List[PersonaContext]:
    contexts, _ = load_persona_context_records(csv_path, rag_map)
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

    persona_contexts = load_persona_contexts(personas_csv, rag_map or {})
    agent_ids = [agent.social_agent_id for _, agent in env.agent_graph.get_agents()]
    cfg = GeminiConfig(
        api_key=os.getenv("GEMINI_API_KEY", ""),
        temperature=temperature,
        model_id=os.getenv("GEMINI_MODEL", os.getenv("GEMINI_DEFAULT_MODEL", "gemini-2.5-flash")),
    )
    print(f"Using Gemini 2.5 Flash-Lite with temperature={temperature}, safety=OFF")

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
    parser.add_argument("--personas", type=str, default="./data/personas_primary.csv")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--posts-per-step", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--rag-corpus", type=str, default="./data/rag_corpus/persona_corpus.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = safe_load(f) or {}
        sim = cfg.get("simulation", {})
        platform_cfg = cfg.get("platform", {})
        personas_cfg = cfg.get("personas", {})
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
        )
    )


if __name__ == "__main__":
    main()


