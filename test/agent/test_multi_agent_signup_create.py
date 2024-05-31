# File: ./test/infra/test_multi_agent_signup_create.py
import asyncio
import os
import os.path as osp
import random
import sqlite3

import pytest

from social_agent.agent import TwitterUserAgent
from twitter.channel import Twitter_Channel
from twitter.config import UserInfo
from twitter.twitter import Twitter

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
    N = 5  # 代理（用户）数量
    M = 3  # 每个用户要发送的推文数量

    agents = []
    channel = Twitter_Channel()
    infra = Twitter(test_db_filepath, channel)
    task = asyncio.create_task(infra.running())

    # 创建并注册用户
    for i in range(N):
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
        await agent.env.twitter_action.action_sign_up(f"user{i}0101",
                                                      f"User{i}", "A bio.")
        agents.append(agent)

    # 发送推文
    for agent in agents:
        for _ in range(M):
            await agent.env.twitter_action.action_create_tweet(
                f"hello from {agent.agent_id}")
            await asyncio.sleep(random.uniform(0, 0.1))

    await channel.write_to_receive_queue((None, None, "exit"))
    await task

    # 验证数据库中是否正确插入了数据
    conn = sqlite3.connect(test_db_filepath)
    cursor = conn.cursor()

    # 验证用户(user)表是否正确插入了数据
    cursor.execute("SELECT * FROM user")
    users = cursor.fetchall()
    assert len(users) == N, ("The number of users in the database"
                             "should match n")

    # 验证推文(tweet)表是否正确插入了数据
    cursor.execute("SELECT * FROM tweet")
    tweets = cursor.fetchall()
    assert len(tweets) == M * N, (
        "The number of tweets should match the expected value.")
    cursor.close()
    conn.close()
