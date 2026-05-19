"""Async LLM client with Helmholtz-first model fallback hierarchy.

Model priority order (tried in sequence, falls through on any proxy /
gateway / 5xx / 501 error):

  Tier 1 — Helmholtz (primary)
    1. alias-qwen36-35b      Qwen3.6-35B-A3B-FP8  (April 2026, multimodal)
    2. alias-qwen36-27b      Qwen3.6-27B-FP8       (April 2026, multimodal)
    3. alias-qwen35-35b-a3b  Qwen3.5-35B-A3B       (Feb 2026, multimodal)

  Tier 2 — OpenAI (final fallback)
    4. gpt-4o
    5. gpt-4o-mini

Vision tasks follow the same order; all three Helmholtz models are
multimodal so they handle image_url content blocks natively.

Runtime overrides from the Gradio Settings tab are still respected:
  set_runtime_config(text_model="...", vision_model="...", ...)
  When a runtime model is set it is tried first, then the hierarchy.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import httpx

import ux_sim_app.core.config as _cfg

logger = logging.getLogger(__name__)

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
    if text_provider is not None:
        _runtime["text_provider"] = text_provider.lower()
    if text_api_key is not None:
        _runtime["text_api_key"] = text_api_key
    if text_base_url is not None:
        _runtime["text_base_url"] = text_base_url
    if text_model is not None:
        _runtime["text_model"] = text_model
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


# ── Model / credential catalogue ──────────────────────────────────────────────

def _build_fallback_chain() -> List[Tuple[str, str, str]]:
    """Return ordered list of (model_id, api_key, base_url) to try.

    If HELMHOLTZ_API_KEY is not set the Helmholtz tier is skipped and we
    fall straight through to OpenAI.
    """
    chain: List[Tuple[str, str, str]] = []

    helmholtz_key = _cfg.HELMHOLTZ_API_KEY
    helmholtz_url = _cfg.HELMHOLTZ_BASE_URL.rstrip("/")

    if helmholtz_key:
        for model in _cfg.HELMHOLTZ_MODELS:
            chain.append((model, helmholtz_key, helmholtz_url))

    openai_key = _cfg.OPENAI_API_KEY
    openai_url = "https://api.openai.com/v1"
    if openai_key:
        for model in _cfg.OPENAI_FALLBACK_MODELS:
            chain.append((model, openai_key, openai_url))

    return chain


def _is_retryable(exc: Exception) -> bool:
    """Return True if the error is a transient proxy / gateway issue."""
    if isinstance(exc, httpx.HTTPStatusError):
        # 501 Not Implemented, 502 Bad Gateway, 503 Service Unavailable,
        # 504 Gateway Timeout, 429 Rate Limit, 500 Internal Server Error
        return exc.response.status_code in {429, 500, 501, 502, 503, 504}
    if isinstance(exc, (httpx.ConnectError, httpx.ReadTimeout,
                        httpx.ConnectTimeout, httpx.RemoteProtocolError,
                        httpx.ProxyError)):
        return True
    # Catch-all for any other network-level error
    msg = str(exc).lower()
    return any(k in msg for k in ("proxy", "gateway", "timeout", "connection", "ssl"))


def _build_headers(model: str, api_key: str) -> Dict[str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    # OpenRouter needs extra headers
    if "openrouter" in api_key or "openrouter" in model:
        headers["HTTP-Referer"] = "https://github.com/Greene-ctrl/oasis"
        headers["X-Title"] = "OASIS UX Simulation"
    return headers


# ── Core chat function ─────────────────────────────────────────────────────────

async def chat(
    messages: List[Dict],
    model: Optional[str] = None,
    tools: Optional[List] = None,
    tool_choice: Optional[Any] = None,
    max_tokens: int = 1200,
    temperature: float = 0.7,
    vision: bool = False,
) -> Dict:
    """Call the chat completions endpoint with automatic model fallback.

    Args:
        messages:    OpenAI-format message list.  For vision tasks include
                     image_url content blocks in the user message.
        model:       Explicit model override.  If set, it is tried first
                     before the fallback chain.
        vision:      Hint that this is a vision task (no routing change —
                     all Helmholtz models are multimodal).
        tools / tool_choice / max_tokens / temperature: standard params.

    Returns:
        The raw OpenAI-compatible response dict from the first successful call.

    Raises:
        RuntimeError: if every model in the chain fails.
    """
    # Build the candidate list
    chain = _build_fallback_chain()

    # If a runtime override or explicit model arg is given, prepend it
    runtime_model = _runtime.get("vision_model" if vision else "text_model")
    explicit = model or runtime_model
    if explicit:
        # Use the same credentials as the first chain entry (Helmholtz if
        # available, else OpenAI) for the explicit model, then continue
        if chain:
            _, first_key, first_url = chain[0]
            chain = [(explicit, first_key, first_url)] + [
                (m, k, u) for m, k, u in chain if m != explicit
            ]
        else:
            # No keys at all — nothing we can do
            raise RuntimeError("No API keys configured. Set HELMHOLTZ_API_KEY or OPENAI_API_KEY.")

    if not chain:
        raise RuntimeError(
            "No API keys configured. Set HELMHOLTZ_API_KEY or OPENAI_API_KEY in .env."
        )

    payload_base: Dict[str, Any] = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        payload_base["tools"] = tools
        payload_base["tool_choice"] = tool_choice or "auto"

    last_exc: Optional[Exception] = None

    async with _sem:
        for attempt, (model_id, api_key, base_url) in enumerate(chain):
            payload = {**payload_base, "model": model_id}
            headers = _build_headers(model_id, api_key)
            try:
                async with httpx.AsyncClient(timeout=90.0) as client:
                    r = await client.post(
                        f"{base_url}/chat/completions",
                        json=payload,
                        headers=headers,
                    )
                    r.raise_for_status()
                if attempt > 0:
                    logger.info(
                        "LLM fallback: succeeded with model=%s after %d failure(s)",
                        model_id, attempt,
                    )
                return r.json()

            except Exception as exc:
                last_exc = exc
                if _is_retryable(exc):
                    status = getattr(getattr(exc, "response", None), "status_code", "?")
                    logger.warning(
                        "LLM fallback: model=%s failed (status=%s, err=%s), trying next…",
                        model_id, status, exc,
                    )
                    continue
                # Non-retryable (e.g. 400 Bad Request, auth error) — raise immediately
                raise

    raise RuntimeError(
        f"All LLM models exhausted. Last error: {last_exc}"
    ) from last_exc


# ── Convenience helpers ────────────────────────────────────────────────────────

def tool_args(response: Dict) -> Dict:
    """Extract tool-call arguments dict from a chat response."""
    tc = response["choices"][0]["message"].get("tool_calls", [])
    if tc:
        return json.loads(tc[0]["function"]["arguments"])
    return {}


def text_content(response: Dict) -> str:
    """Extract plain text content from a chat response."""
    return response["choices"][0]["message"].get("content", "")


# ── Legacy compatibility shims ─────────────────────────────────────────────────
# These keep older call sites that imported _resolve_provider_creds or
# _effective_text_model working without changes.

def _resolve_provider_creds(vision: bool = False) -> tuple[str, str, str]:
    """Legacy shim — returns (provider, api_key, base_url) for the primary model."""
    chain = _build_fallback_chain()
    if chain:
        model_id, api_key, base_url = chain[0]
        provider = "helmholtz" if _cfg.HELMHOLTZ_API_KEY else "openai"
        return provider, api_key, base_url
    return "openai", _cfg.OPENAI_API_KEY, "https://api.openai.com/v1"


def _effective_text_model() -> str:
    return _runtime.get("text_model") or (
        _cfg.HELMHOLTZ_MODELS[0] if _cfg.HELMHOLTZ_API_KEY else _cfg.TEXT_MODEL
    )


def _effective_vision_model() -> str:
    return _runtime.get("vision_model") or (
        _cfg.HELMHOLTZ_MODELS[0] if _cfg.HELMHOLTZ_API_KEY else _cfg.VISION_MODEL
    )
