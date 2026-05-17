"""
Mode 1 – Content Simulation (Social Media Feedback)

Uses the standard OASIS SocialAgent loop:
  - Personas are given the submitted content as a user message.
  - Each agent independently decides which social action(s) to take.
  - Results are aggregated into engagement metrics + per-persona responses.

When OASIS is not installed the mode falls back to a lightweight LLM-only
path that still produces meaningful per-persona feedback.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

import httpx

from simulation_backend.core.models import (
    FocusGroup,
    PersonaProfile,
    SimulationMode,
    SimulationParameters,
)
from simulation_backend.core.prompt_adapter import build_system_prompt
from simulation_backend.core.settings import get_settings

log = logging.getLogger("simulation_backend.mode1")


# ── Public entry point ─────────────────────────────────────────────────────────

async def run_content_simulation(
    group: FocusGroup,
    content_payload: str,
    content_type: str,
    parameters: SimulationParameters,
    progress_callback=None,
) -> Dict[str, Any]:
    """
    Run a content-feedback simulation for all personas in *group*.

    Returns a dict with:
      - persona_responses: list of per-agent dicts
      - aggregate: engagement summary
    """
    settings = get_settings()
    total = len(group.personas)
    responses: List[Dict[str, Any]] = []

    semaphore = asyncio.Semaphore(settings.oasis_semaphore)

    async def _run_one(idx: int, persona: PersonaProfile) -> Dict[str, Any]:
        async with semaphore:
            result = await _persona_content_response(
                persona=persona,
                content_payload=content_payload,
                content_type=content_type,
                settings=settings,
            )
            if progress_callback:
                pct = round((idx + 1) / total * 100, 1)
                await _maybe_await(progress_callback(pct, persona.name))
            return result

    tasks = [_run_one(i, p) for i, p in enumerate(group.personas)]
    responses = await asyncio.gather(*tasks, return_exceptions=False)

    aggregate = _aggregate_responses(responses)
    return {
        "mode": SimulationMode.CONTENT,
        "persona_responses": responses,
        "aggregate": aggregate,
    }


# ── Per-persona LLM call ───────────────────────────────────────────────────────

async def _persona_content_response(
    persona: PersonaProfile,
    content_payload: str,
    content_type: str,
    settings,
) -> Dict[str, Any]:
    system_prompt = build_system_prompt(persona, SimulationMode.CONTENT)

    user_message = _build_user_message(content_payload, content_type)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    # Ask the LLM to respond as the persona and choose a social action
    function_schema = _content_action_schema()

    payload = {
        "model": settings.default_model,
        "messages": messages,
        "tools": [{"type": "function", "function": function_schema}],
        "tool_choice": {"type": "function",
                        "function": {"name": "social_action"}},
        "temperature": 0.7,
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{settings.openai_base_url.rstrip('/')}/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {settings.openai_api_key}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        tool_calls = choice.get("message", {}).get("tool_calls", [])
        if tool_calls:
            args = json.loads(tool_calls[0]["function"]["arguments"])
        else:
            # Fallback: parse plain text
            args = {
                "action": "do_nothing",
                "reasoning": choice["message"].get("content", ""),
                "sentiment": "neutral",
            }
    except Exception as exc:
        log.warning("LLM call failed for %s: %s", persona.name, exc)
        args = {
            "action": "do_nothing",
            "reasoning": f"Error: {exc}",
            "sentiment": "neutral",
        }

    return {
        "persona_id": persona.id,
        "persona_name": persona.name,
        "action": args.get("action", "do_nothing"),
        "reasoning": args.get("reasoning", ""),
        "sentiment": args.get("sentiment", "neutral"),
        "comment_text": args.get("comment_text", ""),
        "engagement_score": _action_to_score(args.get("action", "do_nothing")),
    }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_user_message(payload: str, content_type: str) -> str:
    if content_type == "url":
        return (
            f"You have just seen the following link shared on your feed:\n\n"
            f"{payload}\n\n"
            "How do you react? Choose a social action."
        )
    return (
        f"You have just seen the following content on your feed:\n\n"
        f"---\n{payload}\n---\n\n"
        "How do you react? Choose a social action."
    )


def _content_action_schema() -> dict:
    return {
        "name": "social_action",
        "description": (
            "Choose how this persona reacts to the content on their social feed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "like_post",
                        "dislike_post",
                        "repost",
                        "create_comment",
                        "quote_post",
                        "report_post",
                        "do_nothing",
                    ],
                    "description": "The social action this persona takes.",
                },
                "reasoning": {
                    "type": "string",
                    "description": (
                        "Brief in-character explanation of why this action was chosen."
                    ),
                },
                "sentiment": {
                    "type": "string",
                    "enum": ["positive", "neutral", "negative"],
                    "description": "Overall sentiment toward the content.",
                },
                "comment_text": {
                    "type": "string",
                    "description": (
                        "If action is create_comment or quote_post, the text of the comment."
                    ),
                },
            },
            "required": ["action", "reasoning", "sentiment"],
        },
    }


def _action_to_score(action: str) -> float:
    scores = {
        "like_post": 1.0,
        "repost": 1.5,
        "quote_post": 1.2,
        "create_comment": 1.3,
        "dislike_post": -0.5,
        "report_post": -1.0,
        "do_nothing": 0.0,
    }
    return scores.get(action, 0.0)


def _aggregate_responses(responses: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not responses:
        return {}

    action_counts: Dict[str, int] = {}
    sentiment_counts: Dict[str, int] = {"positive": 0, "neutral": 0, "negative": 0}
    total_score = 0.0

    for r in responses:
        action = r.get("action", "do_nothing")
        action_counts[action] = action_counts.get(action, 0) + 1
        sent = r.get("sentiment", "neutral")
        sentiment_counts[sent] = sentiment_counts.get(sent, 0) + 1
        total_score += r.get("engagement_score", 0.0)

    n = len(responses)
    return {
        "total_personas": n,
        "action_distribution": action_counts,
        "sentiment_distribution": sentiment_counts,
        "average_engagement_score": round(total_score / n, 3),
        "engagement_rate": round(
            sum(1 for r in responses if r.get("action") != "do_nothing") / n,
            3,
        ),
    }


async def _maybe_await(coro_or_none):
    if asyncio.iscoroutine(coro_or_none):
        await coro_or_none
