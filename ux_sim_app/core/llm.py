"""Async LLM client supporting OpenAI, OpenRouter, and custom providers.

The active provider, base URL, and API key are resolved at import time from
config.py but can be overridden at runtime via `set_runtime_config()` when
the user changes settings in the Gradio UI.
"""
from __future__ import annotations
import json
import asyncio
from typing import Any, Dict, List, Optional

import httpx
import ux_sim_app.core.config as _cfg

_sem = asyncio.Semaphore(_cfg.LLM_SEMAPHORE)

# ── Runtime overrides (set by Gradio Settings tab) ─────────────────────────────
# These shadow the module-level config values when the user changes settings.
_runtime: Dict[str, Any] = {}


def set_runtime_config(
    *,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    text_model: Optional[str] = None,
    vision_model: Optional[str] = None,
) -> None:
    """Override config at runtime from the Gradio Settings tab.
    Pass None for any field you do not want to change.
    """
    if provider is not None:
        _runtime["provider"] = provider.lower()
    if api_key is not None:
        _runtime["api_key"] = api_key
    if base_url is not None:
        _runtime["base_url"] = base_url
    if text_model is not None:
        _runtime["text_model"] = text_model
    if vision_model is not None:
        _runtime["vision_model"] = vision_model


def _effective_api_key() -> str:
    if "api_key" in _runtime and _runtime["api_key"]:
        return _runtime["api_key"]
    return _cfg.EFFECTIVE_API_KEY


def _effective_base_url() -> str:
    if "base_url" in _runtime and _runtime["base_url"]:
        return _runtime["base_url"].rstrip("/")
    return _cfg.EFFECTIVE_BASE_URL.rstrip("/")


def _effective_text_model() -> str:
    return _runtime.get("text_model") or _cfg.TEXT_MODEL


def _effective_vision_model() -> str:
    return _runtime.get("vision_model") or _cfg.VISION_MODEL


def _build_headers() -> Dict[str, str]:
    headers = {
        "Authorization": f"Bearer {_effective_api_key()}",
        "Content-Type": "application/json",
    }
    # OpenRouter requires these headers for proper attribution and routing
    provider = _runtime.get("provider") or _cfg.PROVIDER
    if provider == "openrouter":
        headers["HTTP-Referer"] = "https://github.com/Greene-ctrl/oasis"
        headers["X-Title"] = "OASIS UX Simulation"
    return headers


async def chat(
    messages: List[Dict],
    model: Optional[str] = None,
    tools: Optional[List] = None,
    tool_choice: Optional[Any] = None,
    max_tokens: int = 1200,
    temperature: float = 0.7,
    vision: bool = False,
) -> Dict:
    """Call the chat completions endpoint and return the full response dict.

    Args:
        messages:    OpenAI-format message list. For vision tasks, include
                     image_url content blocks in the user message.
        model:       Explicit model override. If None, uses VISION_MODEL when
                     vision=True, else TEXT_MODEL.
        vision:      When True, selects the VISION_MODEL by default.
        tools / tool_choice / max_tokens / temperature: standard OpenAI params.
    """
    resolved_model = model or (_effective_vision_model() if vision else _effective_text_model())

    payload: Dict[str, Any] = {
        "model": resolved_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice or "auto"

    base_url = _effective_base_url()
    headers = _build_headers()

    async with _sem:
        async with httpx.AsyncClient(timeout=90.0) as client:
            r = await client.post(
                f"{base_url}/chat/completions",
                json=payload,
                headers=headers,
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
