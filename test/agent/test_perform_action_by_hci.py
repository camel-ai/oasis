import asyncio
import os
import os.path as osp

import pytest

from social_simulation.social_agent.agent import SocialAgent
from social_simulation.social_agent.agents_generator import \
    generate_controllable_agents
from social_simulation.social_platform.channel import Channel
from social_simulation.social_platform.config import UserInfo
from social_simulation.social_platform.platform import Platform

parent_folder = osp.dirname(osp.abspath(__file__))
test_db_filepath = osp.join(parent_folder, "test_hci.db")


# 定义一个fixture来初始化数据库和Twitter实例
@pytest.fixture
def setup_twitter():
    # 测试前确保test.db不存在
    if os.path.exists(test_db_filepath):
        os.remove(test_db_filepath)


@pytest.mark.asyncio
async def test_perform_action_by_hci():
    channel = Channel()

    agent_graph, _ = await generate_controllable_agents(channel, 1)
    test_agent = agent_graph.get_agent(0)
    assert test_agent.user_info.is_controllable is True

