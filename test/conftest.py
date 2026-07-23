# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
import os
import sys

import pytest

# Add the project root directory to sys.path
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, root_path)


@pytest.fixture
def llm_test_model():
    """Shared LLM model fixture for agent tests.

    Skips the test when ``OPENAI_API_KEY`` is not configured so forks and
    local environments without credentials can still run the rest of the
    suite.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set; skipping LLM-backed test")

    from camel.models import ModelFactory
    from camel.types import ModelPlatformType, ModelType

    return ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_5_MINI,
    )
