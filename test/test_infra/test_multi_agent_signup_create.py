# File: ./test/test_infra/test_multi_agent_signup_create.py
import asyncio
import random
import pytest
import os
import sqlite3

import os.path as osp
from social_agent.twitterUserAgent import TwitterUserAgent
from twitter.twitter import Twitter
from twitter.channel import Twitter_Channel

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
        profile = {"some_key": "some_value"}  # 根据实际需要配置profile
        agent = TwitterUserAgent(i, real_name, description, profile, channel)
        await agent.action_sign_up(f"user{i}0101", f"User{i}", "A bio.")
        agents.append(agent)

    # 发送推文
    for agent in agents:
        for _ in range(M):
            await agent.action_create_tweet(f"hello from {agent.agent_id}")
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
