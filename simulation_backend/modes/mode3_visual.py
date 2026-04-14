"""
Mode 3 – Visual Input Test Simulation

Each persona analyses one or more images (advertisements, UI mockups,
brand assets, social media visuals) using a Vision-Language Model (VLM).

The persona's system prompt is adapted for visual analysis, and the images
are passed as base64-encoded content or URLs in the OpenAI vision message
format.  The model used defaults to `gpt-4o` (configurable via VISION_MODEL).
"""
from __future__ import annotations

import asyncio
import base64
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

log = logging.getLogger("simulation_backend.mode3")


# ── Public entry point ─────────────────────────────────────────────────────────

async def run_visual_simulation(
    group: FocusGroup,
    content_payload: str,
    parameters: SimulationParameters,
    progress_callback=None,
) -> Dict[str, Any]:
    """
    Run a visual-input simulation for all personas in *group*.

    Images are taken from:
      - parameters.image_urls   (list of publicly accessible URLs)
      - parameters.image_base64 (list of base64-encoded image strings)
      - content_payload          (treated as a URL if it starts with http)
    """
    settings = get_settings()
    total = len(group.personas)
    semaphore = asyncio.Semaphore(settings.oasis_semaphore)

    # Build the image content blocks once (shared across all personas)
    image_blocks = _build_image_blocks(content_payload, parameters)

    async def _run_one(idx: int, persona: PersonaProfile) -> Dict[str, Any]:
        async with semaphore:
            result = await _persona_visual_response(
                persona=persona,
                text_context=content_payload,
                image_blocks=image_blocks,
                settings=settings,
            )
            if progress_callback:
                pct = round((idx + 1) / total * 100, 1)
                coro = progress_callback(pct, persona.name)
                if asyncio.iscoroutine(coro):
                    await coro
            return result

    tasks = [_run_one(i, p) for i, p in enumerate(group.personas)]
    responses = await asyncio.gather(*tasks, return_exceptions=False)

    return {
        "mode": SimulationMode.VISUAL,
        "persona_responses": responses,
        "aggregate": _aggregate_visual_responses(responses),
    }


# ── Per-persona VLM call ───────────────────────────────────────────────────────

async def _persona_visual_response(
    persona: PersonaProfile,
    text_context: str,
    image_blocks: List[Dict[str, Any]],
    settings,
) -> Dict[str, Any]:
    system_prompt = build_system_prompt(persona, SimulationMode.VISUAL)

    # Build the user message: text + images
    user_content: List[Dict[str, Any]] = []

    if text_context and not text_context.startswith("http"):
        user_content.append(
            {
                "type": "text",
                "text": (
                    f"Please analyse the following visual content "
                    f"and provide your feedback as this persona.\n\n"
                    f"Additional context: {text_context}"
                ),
            }
        )
    else:
        user_content.append(
            {
                "type": "text",
                "text": (
                    "Please analyse the following visual content "
                    "and provide your in-character feedback."
                ),
            }
        )

    user_content.extend(image_blocks)

    # Ask the LLM to respond using a structured feedback schema
    feedback_schema = _visual_feedback_schema()

    payload = {
        "model": settings.vision_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "tools": [{"type": "function", "function": feedback_schema}],
        "tool_choice": {
            "type": "function",
            "function": {"name": "visual_feedback"},
        },
        "temperature": 0.7,
        "max_tokens": 1024,
    }

    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
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
        tool_calls = (
            data["choices"][0].get("message", {}).get("tool_calls", [])
        )
        if tool_calls:
            args = json.loads(tool_calls[0]["function"]["arguments"])
        else:
            # Fallback: plain text response
            content = data["choices"][0]["message"].get("content", "")
            args = {
                "first_impression": content[:300],
                "attention_elements": [],
                "resonance_score": 5,
                "engagement_likelihood": "maybe",
                "feedback": content,
                "sentiment": "neutral",
            }
    except Exception as exc:
        log.warning("VLM call failed for %s: %s", persona.name, exc)
        args = {
            "first_impression": f"Error: {exc}",
            "attention_elements": [],
            "resonance_score": 0,
            "engagement_likelihood": "no",
            "feedback": str(exc),
            "sentiment": "neutral",
        }

    return {
        "persona_id": persona.id,
        "persona_name": persona.name,
        **args,
    }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_image_blocks(
    content_payload: str,
    parameters: SimulationParameters,
) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []

    # From explicit URL list
    for url in (parameters.image_urls or []):
        blocks.append(
            {
                "type": "image_url",
                "image_url": {"url": url, "detail": "high"},
            }
        )

    # From base64 list
    for b64 in (parameters.image_base64 or []):
        # Ensure it has the data URI prefix
        if not b64.startswith("data:"):
            b64 = f"data:image/jpeg;base64,{b64}"
        blocks.append(
            {
                "type": "image_url",
                "image_url": {"url": b64, "detail": "high"},
            }
        )

    # If content_payload looks like an image URL, add it too
    if content_payload and content_payload.startswith("http"):
        lower = content_payload.lower()
        if any(
            lower.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"]
        ):
            blocks.append(
                {
                    "type": "image_url",
                    "image_url": {"url": content_payload, "detail": "high"},
                }
            )

    return blocks


def _visual_feedback_schema() -> dict:
    return {
        "name": "visual_feedback",
        "description": (
            "Structured visual feedback from a persona reacting to an image."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "first_impression": {
                    "type": "string",
                    "description": "Immediate emotional reaction to the visual.",
                },
                "attention_elements": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of elements that captured attention.",
                },
                "resonance_score": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "description": "How much this content resonates (1=not at all, 10=strongly).",
                },
                "engagement_likelihood": {
                    "type": "string",
                    "enum": ["definitely", "probably", "maybe", "unlikely", "no"],
                    "description": "Likelihood of engaging with this content.",
                },
                "feedback": {
                    "type": "string",
                    "description": "Detailed in-character feedback and suggestions.",
                },
                "sentiment": {
                    "type": "string",
                    "enum": ["positive", "neutral", "negative"],
                    "description": "Overall sentiment toward the visual.",
                },
            },
            "required": [
                "first_impression",
                "resonance_score",
                "engagement_likelihood",
                "feedback",
                "sentiment",
            ],
        },
    }


def _aggregate_visual_responses(responses: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not responses:
        return {}
    scores = [r.get("resonance_score", 0) for r in responses]
    sentiment_counts: Dict[str, int] = {
        "positive": 0, "neutral": 0, "negative": 0
    }
    engagement_counts: Dict[str, int] = {}
    for r in responses:
        s = r.get("sentiment", "neutral")
        sentiment_counts[s] = sentiment_counts.get(s, 0) + 1
        e = r.get("engagement_likelihood", "maybe")
        engagement_counts[e] = engagement_counts.get(e, 0) + 1

    return {
        "total_personas": len(responses),
        "average_resonance_score": round(sum(scores) / len(scores), 2),
        "sentiment_distribution": sentiment_counts,
        "engagement_distribution": engagement_counts,
    }
