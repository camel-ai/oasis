from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Literal, Optional

from dotenv import load_dotenv

load_dotenv()

from camel.models import BaseModelBackend, ModelFactory
from camel.types import ModelPlatformType, ModelType

from orchestrator.llm_config import LLM_CONFIG
try:
    # Reuse the centralized xAI limiter used by ExtendedSocialAgent to avoid duplicate logic
    from generation.extended_agent import _XAI_LIMITER as _GLOBAL_XAI_LIMITER  # type: ignore
except Exception:
    _GLOBAL_XAI_LIMITER = None  # Fallback if import path changes; wrapper will no-op

ProviderName = Literal["openai", "xai", "gemini"]


@dataclass(frozen=True)
class LLMProviderSettings:
    r"""Settings for building a CAMEL BaseModelBackend."""
    provider: ProviderName
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout_seconds: float = 3600.0


class _RateLimitedModelBackend:
    r"""Lightweight wrapper that enforces a per-request async rate limit.

    This wraps a CAMEL BaseModelBackend and acquires tokens on each call to
    `run`/`arun`. It is intentionally minimal and forwards all other
    attributes to the underlying backend.
    """

    def __init__(self, backend: BaseModelBackend) -> None:
        self._backend = backend

    async def _acquire(self) -> None:
        limiter = _GLOBAL_XAI_LIMITER
        if limiter is None:
            return
        try:
            est = limiter.estimate_tokens()
            await limiter.acquire(est)
        except Exception:
            # Fail-open on limiter issues
            return

    # Async inference path used by ChatAgent
    async def arun(self, messages, response_format=None, tools=None):
        await self._acquire()
        return await self._backend.arun(messages, response_format, tools)

    # Sync path (rarely used in our async flows, but keep parity)
    def run(self, messages, response_format=None, tools=None):
        # Use a blocking acquire on the event loop if available; otherwise skip
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Schedule and wait briefly
                loop.run_until_complete(self._acquire())  # type: ignore[call-arg]
            else:
                loop.run_until_complete(self._acquire())
        except Exception:
            pass
        return self._backend.run(messages, response_format, tools)

    # Forward everything else transparently
    def __getattr__(self, name):
        return getattr(self._backend, name)


def create_model_backend(settings: LLMProviderSettings) -> BaseModelBackend:
    r"""Create a CAMEL model backend for the given provider settings."""
    provider = settings.provider.lower()
    cfg = LLM_CONFIG
    # NOTE:
    # - OpenAI-compatible (xAI) and OpenAI SDKs expect `max_tokens` only.
    # - Gemini SDK expects `max_output_tokens`.
    # Build per-provider defaults to avoid passing unsupported kwargs.
    if provider == "xai":
        api_key = settings.api_key or os.getenv("XAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("xAI selected but no API key provided.")
        base_url = settings.base_url or "https://api.x.ai/v1"
        default_model_cfg = {"max_tokens": int(cfg.xai_max_tokens)}
        backend = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type=settings.model_name,
            api_key=api_key,
            url=base_url,
            model_config_dict=default_model_cfg,
            timeout=settings.timeout_seconds,
        )
        # Wrap with request-level rate limiter to prevent 429 bursts
        return _RateLimitedModelBackend(backend)
    if provider == "gemini":
        api_key = settings.api_key or os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError("Gemini selected but no API key provided.")
        # ModelFactory accepts string model types for Gemini
        default_model_cfg = {"max_output_tokens": int(cfg.gemini_max_output_tokens)}
        return ModelFactory.create(
            model_platform=ModelPlatformType.GEMINI,
            model_type=settings.model_name,
            api_key=api_key,
            model_config_dict=default_model_cfg,
            timeout=settings.timeout_seconds,
        )
    if provider == "openai":
        # Use standard OpenAI platform; model_name can be a ModelType or string
        model_type = getattr(ModelType, settings.model_name, settings.model_name)  # type: ignore[arg-type]
        api_key = settings.api_key or os.getenv("OPENAI_API_KEY", "")
        default_model_cfg = {"max_tokens": int(cfg.openai_max_tokens)}
        return ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=model_type,
            api_key=api_key,
            model_config_dict=default_model_cfg,
            timeout=settings.timeout_seconds,
        )
    raise ValueError(f"Unknown provider: {settings.provider}")


