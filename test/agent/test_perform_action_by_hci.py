import asyncio
import os
import os.path as osp

import pytest

from social_agent.agent import TwitterUserAgent
from social_agent.agents_generator import generate_controllable_agents
from twitter.channel import Twitter_Channel
from twitter.config import UserInfo
from twitter.twitter import Twitter

parent_folder = osp.dirname(osp.abspath(__file__))
test_db_filepath = osp.join(parent_folder, "mock_twitter.db")
if osp.exists(test_db_filepath):
    os.remove(test_db_filepath)


@pytest.mark.asyncio
async def test_perform_action_by_hci(monkeypatch):
    channel = Twitter_Channel()

    infra = Twitter(test_db_filepath, channel, rec_update_time=1)
    task = asyncio.create_task(infra.running())

    inputs = iter(["Alice", "Ali", "a student"])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))

    agent_graph, _ = await generate_controllable_agents(channel, 1)
    test_agent = agent_graph.get_agent(0)
    assert test_agent.user_info.is_controllable is True

    user_info = UserInfo(profile={'other_info': {'user_profile': 'None'}})
    operated_agent = TwitterUserAgent(1, user_info, channel)
    await operated_agent.env.twitter_action.action_sign_up(
        "user0101", "User", "A bio.")

    param_lst = [
        'hello world', '2', '2', '1', '1', 'hello', 'user', None, None, '2',
        '2', '1'
    ]

    for i in range(12):
        inputs = iter([i, param_lst[i]]) if param_lst[i] else iter([i])
        monkeypatch.setattr('builtins.input', lambda _: next(inputs))
        result = await test_agent.perform_action_by_hci()
        assert result['success'] is True

    await channel.write_to_receive_queue((None, None, "exit"))
    await task
