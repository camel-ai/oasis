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
"""MiniMax model integration for OASIS.

This module provides a convenience function for creating MiniMax LLM models
using CAMEL's ``ModelFactory`` with the ``OPENAI_COMPATIBLE_MODEL`` platform
type. MiniMax offers an OpenAI-compatible API at ``https://api.minimax.io/v1``.

Available models:

* **MiniMax-M2.7** -- Latest flagship model with 1M context window.
* **MiniMax-M2.7-highspeed** -- Faster variant for latency-sensitive workloads.

Usage::

    from oasis.models import create_minimax_model

    model = create_minimax_model("MiniMax-M2.7")

The ``MINIMAX_API_KEY`` environment variable must be set.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from camel.models import BaseModelBackend, ModelFactory
from camel.types import ModelPlatformType

MINIMAX_API_BASE_URL = "https://api.minimax.io/v1"

MINIMAX_MODELS: Dict[str, Dict[str, Any]] = {
    "MiniMax-M2.7": {
        "description": "Flagship model with 1M context window",
        "context_length": 1_000_000,
    },
    "MiniMax-M2.7-highspeed": {
        "description": "Faster variant for latency-sensitive workloads",
        "context_length": 1_000_000,
    },
}


def create_minimax_model(
    model_type: str = "MiniMax-M2.7",
    api_key: Optional[str] = None,
    url: Optional[str] = None,
    model_config_dict: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> BaseModelBackend:
    """Create a MiniMax model backend via CAMEL's ``ModelFactory``.

    MiniMax provides an OpenAI-compatible API, so this function uses
    ``ModelPlatformType.OPENAI_COMPATIBLE_MODEL`` under the hood.

    Args:
        model_type: MiniMax model identifier. Defaults to ``"MiniMax-M2.7"``.
            Supported values: ``"MiniMax-M2.7"``,
            ``"MiniMax-M2.7-highspeed"``.
        api_key: MiniMax API key. If *None*, reads from the
            ``MINIMAX_API_KEY`` environment variable.
        url: API base URL. Defaults to ``https://api.minimax.io/v1``.
        model_config_dict: Extra model configuration passed to CAMEL's
            ``ModelFactory.create()``.  Temperature is automatically
            clamped to the MiniMax-supported range ``(0.0, 1.0]``.
        **kwargs: Additional keyword arguments forwarded to
            ``ModelFactory.create()``.

    Returns:
        A CAMEL ``BaseModelBackend`` instance configured for MiniMax.

    Raises:
        ValueError: If *model_type* is not a recognized MiniMax model.
        ValueError: If no API key is provided or found in the environment.
    """
    if model_type not in MINIMAX_MODELS:
        raise ValueError(
            f"Unknown MiniMax model: {model_type!r}. "
            f"Supported models: {list(MINIMAX_MODELS.keys())}"
        )

    resolved_key = api_key or os.environ.get("MINIMAX_API_KEY")
    if not resolved_key:
        raise ValueError(
            "MiniMax API key is required. Pass it via the 'api_key' argument "
            "or set the MINIMAX_API_KEY environment variable."
        )

    resolved_url = url or MINIMAX_API_BASE_URL

    # Apply temperature clamping for MiniMax (must be in (0.0, 1.0])
    config = dict(model_config_dict or {})
    if "temperature" in config:
        temp = config["temperature"]
        if temp <= 0.0:
            config["temperature"] = 0.01
        elif temp > 1.0:
            config["temperature"] = 1.0

    return ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
        model_type=model_type,
        api_key=resolved_key,
        url=resolved_url,
        model_config_dict=config or None,
        **kwargs,
    )
