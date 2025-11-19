from __future__ import annotations

from camel.models import BaseModelBackend
from orchestrator.model_provider import LLMProviderSettings, create_model_backend

import pytest


pytestmark = pytest.mark.integration


def test_provider_instantiates_xai_backend_offline() -> None:
    settings = LLMProviderSettings(
        provider="xai",
        model_name="grok-4-fast-non-reasoning",
        api_key="sk-dummy",  # no network call at construction
        base_url="https://api.x.ai/v1",
        timeout_seconds=1.0,
    )
    backend = create_model_backend(settings)
    assert isinstance(backend, BaseModelBackend)


def test_provider_instantiates_openai_backend_offline() -> None:
    settings = LLMProviderSettings(
        provider="openai",
        model_name="GPT_4O_MINI",  # use ModelType name
        api_key="sk-dummy",
        timeout_seconds=1.0,
    )
    backend = create_model_backend(settings)
    assert isinstance(backend, BaseModelBackend)


def test_provider_instantiates_gemini_backend_offline() -> None:
    settings = LLMProviderSettings(
        provider="gemini",
        model_name="gemini-2.5-flash",
        api_key="sk-dummy",
        timeout_seconds=1.0,
    )
    backend = create_model_backend(settings)
    assert isinstance(backend, BaseModelBackend)


