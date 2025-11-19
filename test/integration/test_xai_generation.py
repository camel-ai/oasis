from __future__ import annotations

import os
import pytest

from oasis.generation.xai_client import XAIConfig, generate_text


pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    not os.getenv("XAI_API_KEY"),
    reason="XAI_API_KEY not set; skipping live xAI generation test",
)
def test_xai_generation_smoke() -> None:
    cfg = XAIConfig(
        api_key=os.getenv("XAI_API_KEY", ""),
        model=os.getenv("XAI_MODEL_NAME", "grok-4-fast-non-reasoning"),
        base_url=os.getenv("XAI_BASE_URL", "https://api.x.ai/v1"),
        timeout_seconds=float(os.getenv("XAI_TIMEOUT_SECONDS", "60")),
    )
    out = generate_text(
        system_instruction="You are a concise assistant.",
        user_text="Say 'ok' once.",
        config=cfg,
    )
    assert isinstance(out, str)
    assert len(out) > 0


