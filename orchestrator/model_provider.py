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


def _attach_rate_limiters(backend: BaseModelBackend) -> BaseModelBackend:
    """Monkey-patch backend.run/arun to acquire the global xAI limiter per request."""
    limiter = _GLOBAL_XAI_LIMITER
    if limiter is None:
        return backend
    import types

    if hasattr(backend, "arun"):
        orig_arun = backend.arun

        async def limited_arun(self, messages, response_format=None, tools=None):
            try:
                await limiter.acquire(limiter.estimate_tokens())
            except Exception:
                pass
            return await orig_arun(messages, response_format, tools)

        backend.arun = types.MethodType(limited_arun, backend)  # type: ignore[assignment]

    if hasattr(backend, "run"):
        orig_run = backend.run

        def limited_run(self, messages, response_format=None, tools=None):
            try:
                loop = asyncio.get_event_loop()
                if not loop.is_running():
                    loop.run_until_complete(limiter.acquire(limiter.estimate_tokens()))
            except Exception:
                pass
            return orig_run(messages, response_format, tools)

        backend.run = types.MethodType(limited_run, backend)  # type: ignore[assignment]

    return backend


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
        # Attach request-level rate limiter to prevent 429 bursts
        return _attach_rate_limiters(backend)
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


