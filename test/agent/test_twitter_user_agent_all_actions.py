import asyncio
import os
import os.path as osp
import random

import pytest

from social_simulation.social_agent.agent import SocialAgent
from social_simulation.social_platform.channel import Channel
from social_simulation.social_platform.config import UserInfo
from social_simulation.social_platform.platform import Platform
from social_simulation.social_platform.typing import ActionType

parent_folder = osp.dirname(osp.abspath(__file__))
test_db_filepath = osp.join(parent_folder, "test_actions.db")


# 定义一个fixture来初始化数据库和Twitter实例
@pytest.fixture
def setup_twitter():
    # 测试前确保test.db不存在
    if os.path.exists(test_db_filepath):
        os.remove(test_db_filepath)


@pytest.mark.asyncio
async def test_agents_actions(setup_twitter):
    agents = []
    channel = Channel()
    infra = Platform(test_db_filepath, channel)
    task = asyncio.create_task(infra.running())

    # 创建并注册用户
    for i in range(3):
        real_name = "name" + str(i)
        description = "No description."
        # profile = {"some_key": "some_value"}  # 根据实际需要配置profile
        profile = {
            'nodes': [],  # Relationships with other agents
            'edges': [],  # Relationship details
            'other_info': {
                "user_profile": "Nothing",
                "mbti": "INTJ",
                "activity_level": ["off_line"] * 24,
                "activity_level_frequency": [3] * 24,
                "active_threshold": [0.1] * 24
            },
        }
        user_info = UserInfo(name=real_name,
                             description=description,
                             profile=profile)
        agent = SocialAgent(i, user_info, channel)
        return_message = await agent.env.action.sign_up(
            f"user{i}0101", f"User{i}", "A bio.")
        assert return_message["success"] is True
        agents.append(agent)

    # 发送推文
    for agent in agents:
        for _ in range(4):
            return_message = \
                await agent.env.action.create_post(
                    f"hello from {agent.agent_id}",
                )
            await asyncio.sleep(random.uniform(0, 0.1))
            assert return_message["success"] is True

    await channel.write_to_receive_queue(
        (None, None, ActionType.UPDATE_REC_TABLE))

    # 看推荐系统返回post
    action_agent = agents[2]
    return_message = await action_agent.env.action.refresh()
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    return_message = await action_agent.env.action.like_post(1)
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    return_message = await action_agent.env.action.unlike(1)
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    return_message = await action_agent.env.action.dislike_post(1)
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    return_message = await action_agent.env.action.undo_dislike(1)
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    return_message = \
        await action_agent.env.action.search_posts('hello')
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    return_message = await action_agent.env.action.search_user('2')
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    return_message = await action_agent.env.action.follow(1)
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    return_message = await action_agent.env.action.unfollow(1)
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    return_message = await action_agent.env.action.mute(1)
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    return_message = await action_agent.env.action.unmute(1)
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    # 看最热post
    return_message = await action_agent.env.action.trend()
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    # repos
    return_message = await action_agent.env.action.repost(1)
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    return_message = await action_agent.env.action.create_comment(
        1, 'Test comment')
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    return_message = await action_agent.env.action.like_comment(1)
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    return_message = await action_agent.env.action.unlike_comment(1)
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    return_message = await action_agent.env.action.dislike_comment(1)
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    return_message = await (action_agent.env.action.undo_dislike_comment(1))
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    return_message = await (action_agent.env.action.do_nothing())
    print(return_message)
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    await channel.write_to_receive_queue((None, None, ActionType.EXIT))
    await task
