import asyncio
import json
import os
from datetime import datetime

from colorama import Back

from social_simulation.clock.clock import Clock
from social_simulation.social_agent.agents_generator import (
    gen_control_agents_with_data, generate_reddit_agents)
from social_simulation.social_platform.channel import Channel
from social_simulation.social_platform.platform import Platform
from social_simulation.social_platform.typing import ActionType
from social_simulation.testing.show_db import print_db_contents

# 每轮在沙盒世界创建的帖子和评论数
round_post_num = 20


async def running(num_timestep):
    curr_dir = os.path.dirname(__file__)
    main_dir = os.path.dirname(curr_dir)
    data_dir = os.path.join(main_dir, "data")

    test_db_filepath = os.path.join(data_dir, "mock_reddit.db")
    if os.path.exists(test_db_filepath):
        os.remove(test_db_filepath)

    # 将sandbox启动时间设为当前时刻
    start_time = datetime.now()
    # 将sandbox时间放大系数设为60，即系统运行1秒相当于现实世界60秒
    sandbox_clock = Clock(k=60)

    channel = Channel()

    # 如果不传入start_time或sandbox_clock，默认start_time为实例化Twitter的时间，sandbox_clock的K=60
    infra = Platform(test_db_filepath,
                     channel,
                     sandbox_clock,
                     start_time,
                     allow_self_rating=False,
                     show_score=True,
                     recsys_type='reddit',
                     rec_update_time=40)

    task = asyncio.create_task(infra.running())

    agent_graph, agent_user_id_mapping = await gen_control_agents_with_data(
        channel, 2)

    json_path = os.path.join(data_dir, "reddit", "filter_user_results.json")
    agent_graph = await generate_reddit_agents(json_path, channel, agent_graph,
                                               agent_user_id_mapping)

    json_path = os.path.join(data_dir, "reddit", "RS-RC-pairs.json")
    with open(json_path, 'r') as file:
        rs_rc_pairs = json.load(file)

    len_rs_rc_pairs = len(rs_rc_pairs)

    for timestep in range(num_timestep):
        print(Back.GREEN + f"timestep:{timestep}" + Back.RESET)
        # 0号agent是发帖发评论机器人
        post_agent = agent_graph.get_agent(0)
        # 1号机器人是点赞点踩机器人
        rate_agent = agent_graph.get_agent(1)

        for i in range(round_post_num):
            rs_rc_index = i + timestep * round_post_num
            if rs_rc_index >= len_rs_rc_pairs:
                break
            else:
                response = await post_agent.perform_action_by_data(
                    'create_tweet',
                    content=rs_rc_pairs[rs_rc_index]["submission"])
                tweet_id = response['tweet_id']
                response = await post_agent.perform_action_by_data(
                    'create_comment',
                    tweet_id=tweet_id,
                    content=rs_rc_pairs[rs_rc_index]["comment"])
                comment_id = response['comment_id']

                if rs_rc_pairs[rs_rc_index]["group"] == 'up':
                    await rate_agent.perform_action_by_data(
                        'like_comment', comment_id)
                elif rs_rc_pairs[rs_rc_index]["group"] == 'down':
                    await rate_agent.perform_action_by_data(
                        'dislike_comment', comment_id)
                elif rs_rc_pairs[rs_rc_index]["group"] == 'control':
                    pass
                else:
                    raise ValueError("Unsupported value of 'group'")

        for _, node_data in agent_graph.get_agents():
            agent = node_data['agent']
            if agent.user_info.is_controllable is False:
                await agent.perform_action_by_llm()

    await channel.write_to_receive_queue((None, None, ActionType.EXIT))
    await task

    print_db_contents(test_db_filepath)


if __name__ == "__main__":
    asyncio.run(running(num_timestep=3))
