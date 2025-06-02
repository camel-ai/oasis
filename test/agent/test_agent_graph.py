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
import os.path as osp

import pytest

from oasis.social_agent.agent import SocialAgent
from oasis.social_agent.agent_graph import AgentGraph
from oasis.social_platform.channel import Channel
from oasis.social_platform.config import Neo4jConfig, UserInfo


def neo4j_vars_set() -> bool:
    # Check if all required neo4j env variables are set
    return (os.getenv("NEO4J_URI") is not None
            and os.getenv("NEO4J_USERNAME") is not None
            and os.getenv("NEO4J_PASSWORD") is not None)


def test_agent_graph(tmp_path):
    twitter_channel = Channel()
    graph = AgentGraph()
    agent_0 = SocialAgent(
        agent_id=0,
        user_info=UserInfo(name="0"),
        channel=twitter_channel,
    )
    agent_1 = SocialAgent(
        agent_id=1,
        user_info=UserInfo(name="1"),
        channel=twitter_channel,
    )
    agent_2 = SocialAgent(
        agent_id=2,
        user_info=UserInfo(name="2"),
        channel=twitter_channel,
    )

    graph.add_agent(agent_0)
    graph.add_agent(agent_1)
    graph.add_agent(agent_2)
    assert graph.get_num_nodes() == 3
    assert len(graph.agent_mappings) == 3
    graph.add_edge(0, 1)
    graph.add_edge(0, 2)
    assert graph.get_num_edges() == 2
    edges = graph.get_edges()
    assert len(edges) == 2
    assert edges[0] == (0, 1)
    assert edges[1] == (0, 2)

    img_path = osp.join(tmp_path, "img.pdf")
    graph.visualize(img_path)
    assert osp.exists(img_path)

    agents = graph.get_agents()
    assert len(agents) == 3
    assert agents[0][0] == 0
    assert id(agents[0][1]) == id(agent_0)
    assert agents[1][0] == 1
    assert id(agents[1][1]) == id(agent_1)
    assert agents[2][0] == 2
    assert id(agents[2][1]) == id(agent_2)

    graph.remove_edge(0, 1)
    assert graph.get_num_edges() == 1

    graph.remove_agent(agent_0)
    assert len(graph.agent_mappings) == 2
    assert graph.get_num_nodes() == 2
    assert graph.get_num_edges() == 0

    graph.reset()
    assert len(graph.agent_mappings) == 0
    assert graph.get_num_nodes() == 0
    assert graph.get_num_edges() == 0


@pytest.mark.skipif(
    not neo4j_vars_set(),
    reason="One or more neo4j env variables are not set",
)
def test_agent_neo4j_graph():
    channel = Channel()
    graph = AgentGraph(
        backend="neo4j",
        neo4j_config=Neo4jConfig(
            uri=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
        ),
    )
    agent_0 = SocialAgent(
        agent_id=0,
        user_info=UserInfo(name="0", ),
        channel=channel,
    )
    agent_1 = SocialAgent(
        agent_id=1,
        user_info=UserInfo(name="1", ),
        channel=channel,
    )
    agent_2 = SocialAgent(
        agent_id=2,
        user_info=UserInfo(name="2", ),
        channel=channel,
    )
    graph.add_agent(agent_0)
    graph.add_agent(agent_1)
    graph.add_agent(agent_2)

    assert graph.get_num_nodes() == 3
    assert len(graph.agent_mappings) == 3
    graph.add_edge(0, 1)
    graph.add_edge(0, 2)
    assert graph.get_num_edges() == 2
    edges = graph.get_edges()
    assert len(edges) == 2
    assert edges[0] == (0, 1)
    assert edges[1] == (0, 2)

    agents = graph.get_agents()
    assert len(agents) == 3
    assert agents[0][0] == 0
    assert id(agents[0][1]) == id(agent_0)
    assert agents[1][0] == 1
    assert id(agents[1][1]) == id(agent_1)
    assert agents[2][0] == 2
    assert id(agents[2][1]) == id(agent_2)

    graph.remove_edge(0, 1)
    assert graph.get_num_edges() == 1

    graph.remove_agent(agent_0)
    assert len(graph.agent_mappings) == 2
    assert graph.get_num_nodes() == 2
    assert graph.get_num_edges() == 0

    graph.reset()
    assert len(graph.agent_mappings) == 0
    assert graph.get_num_nodes() == 0
    assert graph.get_num_edges() == 0
