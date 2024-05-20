# File: ./test/test_infra/test_TwitterUserAgent_all_actions.py
import asyncio
import random
import pytest
import os

import os.path as osp
from social_agent.twitterUserAgent import TwitterUserAgent
from twitter.twitter import Twitter
from twitter.channel import Twitter_Channel
from test.show_db import print_db_contents
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
    channel = Twitter_Channel()
    infra = Twitter(test_db_filepath, channel)
    task = asyncio.create_task(infra.running())

    # 创建并注册用户
    for i in range(3):
        real_name = "name" + str(i)
        description = "No description."
        profile = {"some_key": "some_value"}  # 根据实际需要配置profile
        agent = TwitterUserAgent(i, real_name, description, profile, channel)
        return_message = await agent.action_sign_up(
            f"user{i}0101", f"User{i}", "A bio.")
        assert return_message["success"] is True
        agents.append(agent)

    # 发送推文
    for agent in agents:
        for _ in range(4):
            return_message = await agent.action_create_tweet(
                f"hello from {agent.agent_id}")
            await asyncio.sleep(random.uniform(0, 0.1))
            assert return_message["success"] is True

    await channel.write_to_receive_queue((None, None, ActionType.UPDATE_REC))

    # 看推荐系统返回tweet
    # print_db_contents(test_db_filepath)
    action_agent = agents[2]
    return_message = await action_agent.action_refresh()
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    return_message = await action_agent.action_like(1)
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    return_message = await action_agent.action_unlike(1)
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    return_message = await action_agent.action_search_tweets('hello')
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    return_message = await action_agent.action_search_user('2')
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    return_message = await action_agent.action_follow(1)
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    return_message = await action_agent.action_unfollow(1)
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    return_message = await action_agent.action_mute(1)
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    return_message = await action_agent.action_unmute(1)
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    # 看最热tweet
    return_message = await action_agent.action_trend()
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    await channel.write_to_receive_queue((None, None, "exit"))
    await task
