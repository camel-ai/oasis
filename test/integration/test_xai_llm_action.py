from __future__ import annotations

import os

os.environ["OASIS_DISABLE_RECSYS_IMPORT"] = "true"
import asyncio
from pathlib import Path

import pytest

import oasis
from oasis.social_agent.agent import SocialAgent
from oasis.social_agent.agent_graph import AgentGraph
from oasis.social_platform.channel import Channel
from oasis.social_platform.config import UserInfo
from oasis.social_platform.platform import Platform
from orchestrator.model_provider import (LLMProviderSettings,
                                         create_model_backend)

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    not os.getenv("XAI_API_KEY"),
    reason="XAI_API_KEY not set; skipping live xAI LLMAction integration test",
)
@pytest.mark.asyncio
async def test_xai_llm_action_single_step(tmp_path: Path) -> None:
    # Build model backend for xAI
    settings = LLMProviderSettings(
        provider="xai",
        model_name=os.getenv("XAI_MODEL_NAME", "grok-4-fast-non-reasoning"),
        api_key=os.getenv("XAI_API_KEY", ""),
        base_url=os.getenv("XAI_BASE_URL", "https://api.x.ai/v1"),
        timeout_seconds=float(os.getenv("XAI_TIMEOUT_SECONDS", "60")),
    )
    model = create_model_backend(settings)

    # Minimal platform + agent graph
    channel = Channel()
    platform = Platform(
        db_path=str(tmp_path / "xai_integration.db"),
        channel=channel,
        sandbox_clock=None,
        recsys_type="twhin-bert",
        refresh_rec_post_count=2,
        max_rec_post_len=2,
        following_post_count=1,
        show_score=False,
        allow_self_rating=False,
    )

    graph = AgentGraph()
    user_info = UserInfo(name="agent_0", description="test agent", profile={"other_info": {"user_profile": "neutral"}}, recsys_type="twitter")
    agent = SocialAgent(agent_id=0, user_info=user_info, channel=channel, model=model, agent_graph=graph, available_actions=None)
    graph.add_agent(agent)

    env = oasis.make(agent_graph=graph, platform=platform, database_path=str(tmp_path / "xai_integration.db"))
    await env.reset()

    # One LLMAction step (will call xAI)
    actions = {agent: oasis.LLMAction()}
    await env.step(actions)

    await env.close()


