import asyncio
import os
import random
from datetime import datetime

from colorama import Back

from clock.clock import clock
from social_agent.agents_generator import generate_agents
from twitter.channel import TwitterChannel
from twitter.twitter import Twitter
from twitter.typing import ActionType


async def running(num_timestep):

    test_db_filepath = "./data/mock_twitter.db"
    if os.path.exists(test_db_filepath):
        os.remove(test_db_filepath)

    # 将sandbox启动时间设为当前时刻
    start_time = datetime.now()
    # 将sandbox时间放大系数设为60，即系统运行1秒相当于现实世界60秒
    sandbox_clock = clock(K=60)

    channel = TwitterChannel()

    # 如果不传入start_time或sandbox_clock，默认start_time为实例化Twitter的时间，sandbox_clock的K=60
    infra = Twitter(test_db_filepath, channel, sandbox_clock, start_time)

    task = asyncio.create_task(infra.running())

    agent_graph = await generate_agents("./data/user_all_id_time.csv", channel)
    '''
    一个比较简单的思路为：
    为每种活跃状态设置一个对应阈值
    比如
        threshold_dict = {"off_line":0.0, "busy":0.3, "normal":0.6, "active":1}
    simulation里每过去一个时间步
    每个agent会在执行动作前生成一个0~1的随机值
    并与这个时间步所在小时agent对应的活跃阈值进行对比
    得到该agent在这个时间步下是否要被触发

    也就是用激活概率实现了yuxian提到的对于设置agent在一段时间内的激活次数作为活跃程度表现
    从激活不激活来看优点是比较容易实现, 而且不激活就可以减少对于LLM的调用, 能够降低成本
    缺点就是目前的激活概率阈值比较简单, 就是靠初始化的数据得到的,
    后续在running的时候没有其他变量的影响的话
    这种方法的可解释性就全部来自于初始化爬到的数据了.
    但是可以先靠这个实现一下
    '''
    start_hour = 1  # 涉及到时间戳
    simulation_time_hour = start_hour  # 涉及到时间戳

    for timestep in range(num_timestep):
        print(Back.GREEN + f"timestep:{timestep}" + Back.RESET)

        # 这里的 0.2 就是指的 1个timestep对应12分钟， 涉及到时间戳
        simulation_time_hour = start_hour + 0.2 * timestep

        for node_id, node_data in agent_graph.get_agents():
            agent = node_data['agent']
            # 得到该agent激活概率
            agent_ac_prob = random.random()
            # 得到该agent激活阈值
            threshold = agent.user_info.profile['other_info'][
                'active_threshold'][int(simulation_time_hour % 24)]
            # 比较激活概率和阈值
            if agent_ac_prob < threshold:
                await agent.perform_action_by_llm()

    await channel.write_to_receive_queue((None, None, ActionType.EXIT))
    await task


if __name__ == "__main__":
    asyncio.run(running(num_timestep=3))
