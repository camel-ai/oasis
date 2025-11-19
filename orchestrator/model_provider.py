from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Optional

from dotenv import load_dotenv

load_dotenv()

from camel.models import BaseModelBackend, ModelFactory
from camel.types import ModelPlatformType, ModelType

ProviderName = Literal["openai", "xai", "gemini"]


@dataclass(frozen=True)
class LLMProviderSettings:
    r"""Settings for building a CAMEL BaseModelBackend."""
    provider: ProviderName
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout_seconds: float = 3600.0


def create_model_backend(settings: LLMProviderSettings) -> BaseModelBackend:
    r"""Create a CAMEL model backend for the given provider settings."""
    provider = settings.provider.lower()
    if provider == "xai":
        api_key = settings.api_key or os.getenv("XAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("xAI selected but no API key provided.")
        base_url = settings.base_url or "https://api.x.ai/v1"
        return ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type=settings.model_name,
            api_key=api_key,
            url=base_url,
            timeout=settings.timeout_seconds,
        )
    if provider == "gemini":
        api_key = settings.api_key or os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError("Gemini selected but no API key provided.")
        # ModelFactory accepts string model types for Gemini
        return ModelFactory.create(
            model_platform=ModelPlatformType.GEMINI,
            model_type=settings.model_name,
            api_key=api_key,
            timeout=settings.timeout_seconds,
        )
    if provider == "openai":
        # Use standard OpenAI platform; model_name can be a ModelType or string
        model_type = getattr(ModelType, settings.model_name, settings.model_name)  # type: ignore[arg-type]
        api_key = settings.api_key or os.getenv("OPENAI_API_KEY", "")
        return ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=model_type,
            api_key=api_key,
            timeout=settings.timeout_seconds,
        )
    raise ValueError(f"Unknown provider: {settings.provider}")


