# File: ./test/infra/test_twitter_user_agent_all_actions.py
import asyncio
import os
import os.path as osp
import random

import pytest

from social_agent.agent import TwitterUserAgent
from twitter.channel import TwitterChannel
from twitter.config import UserInfo
from twitter.twitter import Twitter
from twitter.typing import ActionType

parent_folder = osp.dirname(osp.abspath(__file__))
test_db_filepath = osp.join(parent_folder, "test.db")


# 定义一个fixture来初始化数据库和Twitter实例
@pytest.fixture
def setup_twitter():
    # 测试前确保test.db不存在
    if os.path.exists(test_db_filepath):
        os.remove(test_db_filepath)


@pytest.mark.asyncio
async def test_agents_tweeting(setup_twitter):
    agents = []
    channel = TwitterChannel()
    infra = Twitter(test_db_filepath, channel)
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
        agent = TwitterUserAgent(i, user_info, channel)
        return_message = await agent.env.twitter_action.action_sign_up(
            f"user{i}0101", f"User{i}", "A bio.")
        assert return_message["success"] is True
        agents.append(agent)

    # 发送推文
    for agent in agents:
        for _ in range(4):
            return_message = \
                await agent.env.twitter_action.action_create_tweet(
                    f"hello from {agent.agent_id}",
                )
            await asyncio.sleep(random.uniform(0, 0.1))
            assert return_message["success"] is True

    await channel.write_to_receive_queue((None, None, ActionType.UPDATE_REC))

    # 看推荐系统返回tweet
    action_agent = agents[2]
    return_message = await action_agent.env.twitter_action.action_refresh()
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    return_message = await action_agent.env.twitter_action.action_like(1)
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    return_message = await action_agent.env.twitter_action.action_unlike(1)
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    return_message = \
        await action_agent.env.twitter_action.action_search_tweets('hello')
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    return_message = await action_agent.env.twitter_action.action_search_user(
        '2')
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    return_message = await action_agent.env.twitter_action.action_follow(1)
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    return_message = await action_agent.env.twitter_action.action_unfollow(1)
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    return_message = await action_agent.env.twitter_action.action_mute(1)
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    return_message = await action_agent.env.twitter_action.action_unmute(1)
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    # 看最热tweet
    return_message = await action_agent.env.twitter_action.action_trend()
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    # retweet
    return_message = await action_agent.env.twitter_action.action_retweet(1)
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    await channel.write_to_receive_queue((None, None, ActionType.EXIT))
    await task
