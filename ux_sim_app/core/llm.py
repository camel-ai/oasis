"""Async LLM client supporting per-model provider selection.

TEXT tasks  → TEXT_PROVIDER  (api key + base url resolved from config)
VISION tasks → VISION_PROVIDER (separate api key + base url)

Runtime overrides can be set from the Gradio Settings tab via set_runtime_config().
"""
from __future__ import annotations
import json
import asyncio
from typing import Any, Dict, List, Optional

import httpx
import ux_sim_app.core.config as _cfg

_sem = asyncio.Semaphore(_cfg.LLM_SEMAPHORE)

# ── Runtime overrides (set by Gradio Settings tab) ─────────────────────────────
_runtime: Dict[str, Any] = {}


def set_runtime_config(
    *,
    # Text provider
    text_provider: Optional[str] = None,
    text_api_key: Optional[str] = None,
    text_base_url: Optional[str] = None,
    text_model: Optional[str] = None,
    # Vision provider
    vision_provider: Optional[str] = None,
    vision_api_key: Optional[str] = None,
    vision_base_url: Optional[str] = None,
    vision_model: Optional[str] = None,
    # Legacy single-provider override (maps to text)
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> None:
    """Override config at runtime from the Gradio Settings tab."""
    # Text provider
    if text_provider is not None:
        _runtime["text_provider"] = text_provider.lower()
    if text_api_key is not None:
        _runtime["text_api_key"] = text_api_key
    if text_base_url is not None:
        _runtime["text_base_url"] = text_base_url
    if text_model is not None:
        _runtime["text_model"] = text_model
    # Vision provider
    if vision_provider is not None:
        _runtime["vision_provider"] = vision_provider.lower()
    if vision_api_key is not None:
        _runtime["vision_api_key"] = vision_api_key
    if vision_base_url is not None:
        _runtime["vision_base_url"] = vision_base_url
    if vision_model is not None:
        _runtime["vision_model"] = vision_model
    # Legacy single-provider (maps to text)
    if provider is not None:
        _runtime["text_provider"] = provider.lower()
    if api_key is not None:
        _runtime["text_api_key"] = api_key
    if base_url is not None:
        _runtime["text_base_url"] = base_url


def _resolve_provider_creds(vision: bool) -> tuple[str, str, str]:
    """Return (provider, api_key, base_url) for text or vision tasks."""
    _OR = "https://openrouter.ai/api/v1"
    _OA = "https://api.openai.com/v1"

    if vision:
        provider = _runtime.get("vision_provider") or _cfg.VISION_PROVIDER
        api_key = _runtime.get("vision_api_key") or _cfg.EFFECTIVE_VISION_API_KEY
        base_url = _runtime.get("vision_base_url") or _cfg.EFFECTIVE_VISION_BASE_URL
    else:
        provider = _runtime.get("text_provider") or _cfg.TEXT_PROVIDER
        api_key = _runtime.get("text_api_key") or _cfg.EFFECTIVE_TEXT_API_KEY
        base_url = _runtime.get("text_base_url") or _cfg.EFFECTIVE_TEXT_BASE_URL

    # Normalise base_url based on provider if it looks like a placeholder
    if provider == "openrouter" and not base_url.startswith("https://openrouter"):
        base_url = _OR
    elif provider == "openai" and not base_url:
        base_url = _OA

    return provider, api_key, base_url.rstrip("/")


def _effective_text_model() -> str:
    return _runtime.get("text_model") or _cfg.TEXT_MODEL


def _effective_vision_model() -> str:
    return _runtime.get("vision_model") or _cfg.VISION_MODEL


def _build_headers(provider: str, api_key: str) -> Dict[str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
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
        vision:      When True, routes to the VISION provider with VISION_MODEL.
        tools / tool_choice / max_tokens / temperature: standard OpenAI params.
    """
    provider, api_key, base_url = _resolve_provider_creds(vision)
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

    headers = _build_headers(provider, api_key)

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
