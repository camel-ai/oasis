"""Thin async wrapper around the OpenAI chat-completions API."""
from __future__ import annotations
import json
import asyncio
from typing import Any, Dict, List, Optional

import httpx
from ux_sim_app.core.config import OPENAI_API_KEY, OPENAI_BASE_URL, TEXT_MODEL, LLM_SEMAPHORE

_sem = asyncio.Semaphore(LLM_SEMAPHORE)


async def chat(
    messages: List[Dict],
    model: Optional[str] = None,
    tools: Optional[List] = None,
    tool_choice: Optional[Any] = None,
    max_tokens: int = 1200,
    temperature: float = 0.7,
) -> Dict:
    """Call the OpenAI chat completions endpoint and return the full response dict."""
    payload: Dict[str, Any] = {
        "model": model or TEXT_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice or "auto"

    async with _sem:
        async with httpx.AsyncClient(timeout=90.0) as client:
            r = await client.post(
                f"{OPENAI_BASE_URL}/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
            )
            r.raise_for_status()
    return r.json()


def tool_args(response: Dict) -> Dict:
    """Extract tool-call arguments dict from a chat response."""
    tc = response["choices"][0]["message"].get("tool_calls", [])
    if tc:
        return json.loads(tc[0]["function"]["arguments"])
    return {}


def text_content(response: Dict) -> str:
    """Extract plain text content from a chat response."""
    return response["choices"][0]["message"].get("content", "")
