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
import json

import pytest

from oasis.social_agent.agents_generator import generate_tiktok_agent_graph


@pytest.mark.asyncio
async def test_generate_tiktok_agent_graph_sets_username_and_name(
    tmp_path,
    monkeypatch,
):
    profile_path = tmp_path / "tiktok_agents.json"
    profile_path.write_text(
        json.dumps([
            {
                "username": "creator_001",
                "name": "Creator One",
                "bio": "Short bio",
                "persona": "Detailed personality description",
                "interested_topics": ["dance", "music"],
            }
        ]),
        encoding="utf-8",
    )

    class DummySocialAgent:

        def __init__(self, agent_id, user_info, **kwargs):
            self.social_agent_id = agent_id
            self.user_info = user_info

    monkeypatch.setattr(
        "oasis.social_agent.agents_generator.SocialAgent",
        DummySocialAgent,
    )

    agent_graph = await generate_tiktok_agent_graph(str(profile_path))
    agent = agent_graph.get_agent(0)

    assert agent.user_info.user_name == "creator_001"
    assert agent.user_info.name == "Creator One"
    assert "[interests: dance, music]" in agent.user_info.description
