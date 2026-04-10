# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
"""Unit and integration tests for MiniMax provider support."""

import os
from unittest.mock import MagicMock, patch

import pytest
from camel.models import BaseModelBackend
from camel.types import ModelPlatformType

from oasis.minimax import (
    MINIMAX_API_BASE_URL,
    MINIMAX_MODELS,
    create_minimax_model,
)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestMiniMaxModels:
    """Tests for the MiniMax model constants."""

    def test_models_dict_has_expected_models(self):
        assert "MiniMax-M2.7" in MINIMAX_MODELS
        assert "MiniMax-M2.7-highspeed" in MINIMAX_MODELS

    def test_models_have_description(self):
        for model_name, info in MINIMAX_MODELS.items():
            assert "description" in info, f"{model_name} missing description"
            assert isinstance(info["description"], str)

    def test_models_have_context_length(self):
        for model_name, info in MINIMAX_MODELS.items():
            assert "context_length" in info, (
                f"{model_name} missing context_length"
            )
            assert info["context_length"] > 0

    def test_api_base_url(self):
        assert MINIMAX_API_BASE_URL == "https://api.minimax.io/v1"


class TestCreateMiniMaxModelValidation:
    """Tests for argument validation in create_minimax_model()."""

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown MiniMax model"):
            create_minimax_model("nonexistent-model", api_key="test-key")

    def test_missing_api_key_raises(self):
        env = os.environ.copy()
        env.pop("MINIMAX_API_KEY", None)
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="API key is required"):
                create_minimax_model("MiniMax-M2.7")

    def test_api_key_from_env(self):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "env-key"}):
            with patch(
                "oasis.minimax.ModelFactory.create"
            ) as mock_create:
                mock_create.return_value = MagicMock(spec=BaseModelBackend)
                model = create_minimax_model("MiniMax-M2.7")
                mock_create.assert_called_once()
                call_kwargs = mock_create.call_args
                assert call_kwargs.kwargs["api_key"] == "env-key"
                assert model is not None

    def test_explicit_api_key_overrides_env(self):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "env-key"}):
            with patch(
                "oasis.minimax.ModelFactory.create"
            ) as mock_create:
                mock_create.return_value = MagicMock(spec=BaseModelBackend)
                create_minimax_model("MiniMax-M2.7", api_key="explicit-key")
                call_kwargs = mock_create.call_args
                assert call_kwargs.kwargs["api_key"] == "explicit-key"


class TestCreateMiniMaxModelFactory:
    """Tests for correct ModelFactory.create() invocation."""

    @patch("oasis.minimax.ModelFactory.create")
    def test_default_model(self, mock_create):
        mock_create.return_value = MagicMock(spec=BaseModelBackend)
        create_minimax_model(api_key="test-key")
        mock_create.assert_called_once_with(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type="MiniMax-M2.7",
            api_key="test-key",
            url=MINIMAX_API_BASE_URL,
            model_config_dict=None,
        )

    @patch("oasis.minimax.ModelFactory.create")
    def test_highspeed_model(self, mock_create):
        mock_create.return_value = MagicMock(spec=BaseModelBackend)
        create_minimax_model("MiniMax-M2.7-highspeed", api_key="test-key")
        call_kwargs = mock_create.call_args
        assert call_kwargs.kwargs["model_type"] == "MiniMax-M2.7-highspeed"

    @patch("oasis.minimax.ModelFactory.create")
    def test_custom_url(self, mock_create):
        mock_create.return_value = MagicMock(spec=BaseModelBackend)
        custom_url = "https://custom.minimax.io/v1"
        create_minimax_model(api_key="test-key", url=custom_url)
        call_kwargs = mock_create.call_args
        assert call_kwargs.kwargs["url"] == custom_url

    @patch("oasis.minimax.ModelFactory.create")
    def test_returns_base_model_backend(self, mock_create):
        mock_backend = MagicMock(spec=BaseModelBackend)
        mock_create.return_value = mock_backend
        result = create_minimax_model(api_key="test-key")
        assert result is mock_backend


class TestTemperatureClamping:
    """Tests for MiniMax temperature constraints."""

    @patch("oasis.minimax.ModelFactory.create")
    def test_zero_temperature_clamped(self, mock_create):
        mock_create.return_value = MagicMock(spec=BaseModelBackend)
        create_minimax_model(
            api_key="test-key",
            model_config_dict={"temperature": 0.0},
        )
        call_kwargs = mock_create.call_args
        config = call_kwargs.kwargs["model_config_dict"]
        assert config["temperature"] == 0.01

    @patch("oasis.minimax.ModelFactory.create")
    def test_negative_temperature_clamped(self, mock_create):
        mock_create.return_value = MagicMock(spec=BaseModelBackend)
        create_minimax_model(
            api_key="test-key",
            model_config_dict={"temperature": -1.0},
        )
        call_kwargs = mock_create.call_args
        config = call_kwargs.kwargs["model_config_dict"]
        assert config["temperature"] == 0.01

    @patch("oasis.minimax.ModelFactory.create")
    def test_high_temperature_clamped(self, mock_create):
        mock_create.return_value = MagicMock(spec=BaseModelBackend)
        create_minimax_model(
            api_key="test-key",
            model_config_dict={"temperature": 2.0},
        )
        call_kwargs = mock_create.call_args
        config = call_kwargs.kwargs["model_config_dict"]
        assert config["temperature"] == 1.0

    @patch("oasis.minimax.ModelFactory.create")
    def test_valid_temperature_unchanged(self, mock_create):
        mock_create.return_value = MagicMock(spec=BaseModelBackend)
        create_minimax_model(
            api_key="test-key",
            model_config_dict={"temperature": 0.7},
        )
        call_kwargs = mock_create.call_args
        config = call_kwargs.kwargs["model_config_dict"]
        assert config["temperature"] == 0.7

    @patch("oasis.minimax.ModelFactory.create")
    def test_no_temperature_no_config(self, mock_create):
        mock_create.return_value = MagicMock(spec=BaseModelBackend)
        create_minimax_model(api_key="test-key")
        call_kwargs = mock_create.call_args
        assert call_kwargs.kwargs["model_config_dict"] is None


class TestOasisImports:
    """Tests that MiniMax helpers are accessible from the oasis package."""

    def test_import_from_oasis(self):
        from oasis import create_minimax_model as fn
        assert callable(fn)

    def test_import_from_oasis_minimax(self):
        from oasis.minimax import (
            MINIMAX_API_BASE_URL,
            MINIMAX_MODELS,
            create_minimax_model,
        )
        assert callable(create_minimax_model)
        assert isinstance(MINIMAX_MODELS, dict)
        assert isinstance(MINIMAX_API_BASE_URL, str)


class TestSocialAgentWithMiniMax:
    """Tests that SocialAgent accepts a MiniMax model backend."""

    @patch("oasis.minimax.ModelFactory.create")
    def test_agent_accepts_minimax_model(self, mock_create):
        mock_backend = MagicMock(spec=BaseModelBackend)
        mock_backend.model_type = "MiniMax-M2.7"
        mock_create.return_value = mock_backend

        from oasis import SocialAgent, ActionType
        from oasis.social_platform.config import UserInfo

        model = create_minimax_model(api_key="test-key")
        agent = SocialAgent(
            agent_id=0,
            user_info=UserInfo(
                user_name="test_user",
                name="Test User",
                description="A test user",
                profile=None,
                recsys_type="reddit",
            ),
            model=model,
            available_actions=[ActionType.CREATE_POST, ActionType.DO_NOTHING],
        )
        assert agent is not None
        assert agent.social_agent_id == 0


# ---------------------------------------------------------------------------
# Integration tests (require MINIMAX_API_KEY)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("MINIMAX_API_KEY"),
    reason="MINIMAX_API_KEY not set",
)
class TestMiniMaxIntegration:
    """Integration tests that call the real MiniMax API."""

    def test_create_model_real(self):
        model = create_minimax_model("MiniMax-M2.7")
        assert isinstance(model, BaseModelBackend)

    def test_create_highspeed_model_real(self):
        model = create_minimax_model("MiniMax-M2.7-highspeed")
        assert isinstance(model, BaseModelBackend)

    @pytest.mark.asyncio
    async def test_agent_with_minimax_model(self):
        import asyncio
        from oasis import (
            ActionType, AgentGraph, ManualAction, SocialAgent, UserInfo,
        )
        from oasis.social_platform.channel import Channel
        from oasis.social_platform.platform import Platform

        model = create_minimax_model("MiniMax-M2.7")
        channel = Channel()

        test_db = os.path.join(
            os.path.dirname(__file__), "test_minimax_integration.db"
        )
        if os.path.exists(test_db):
            os.remove(test_db)

        try:
            infra = Platform(
                db_path=test_db, channel=channel, recsys_type="reddit"
            )
            task = asyncio.create_task(infra.running())

            agent = SocialAgent(
                agent_id=0,
                user_info=UserInfo(
                    user_name="minimax_user",
                    name="MiniMax Tester",
                    description="An agent using MiniMax M2.7",
                    profile=None,
                    recsys_type="reddit",
                ),
                channel=channel,
                model=model,
                available_actions=[
                    ActionType.CREATE_POST,
                    ActionType.DO_NOTHING,
                ],
            )

            await agent.env.action.sign_up(
                "minimax_user", "MiniMax Tester", "Testing MiniMax."
            )
            await agent.env.action.create_post(
                "Hello from MiniMax-M2.7!"
            )

            await channel.write_to_receive_queue((None, None, "exit"))
            await task
        finally:
            if os.path.exists(test_db):
                os.remove(test_db)
