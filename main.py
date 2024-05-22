import asyncio

from social_agent.agents_generator import generate_agents
from twitter.channel import Twitter_Channel
from twitter.twitter import Twitter
from twitter.typing import ActionType
from colorama import Back, Fore

import pandas as pd
import random
import os


async def running(num_timestep):
    os.environ["OPENAI_API_KEY"] = "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"
    test_db_filepath = "./data/mock_twitter.db"
    if os.path.exists(test_db_filepath):
        os.remove(test_db_filepath)

    channel = Twitter_Channel()
    infra = Twitter(test_db_filepath, channel)
    task = asyncio.create_task(infra.running())

    agent_graph = await generate_agents("./data/user_all_id_time.csv", channel)

    '''
    一个比较简单的思路为：为每种活跃状态设置一个对应阈值 比如  threshold_dict = {"off_line":0.0, "busy":0.3, "normal":0.6, "active":1} 
    simulation里每过去一个小时新产生一个[random.random() for _ in range(len(agents))] 的 0~1 的随机值列表
    并与 [ threshold_dict[agent.profile['other_info']['activity_level'][simulation_time_hour]] for agent in agents.values()]进行比较, 得到要被触发的agent列表
    也就是用激活概率实现了yuxian提到的对于设置agent在一段时间内的激活次数作为活跃程度表现
    从激活不激活来看优点是比较容易实现, 而且不激活就可以减少对于LLM的调用, 能够降低成本
    缺点就是目前的激活概率阈值比较简单, 就是靠初始化的数据得到的, 后续在running的时候没有其他变量的影响的话这种方法的可解释性就全部来自于初始化爬到的数据了.
    但是可以先靠这个实现一下
    '''

    threshold_dict = {"off_line":0.1, "busy":0.3, "normal":0.6, "active":1}    # 后续写到agent generator里
    start_hour = 1  # 涉及到时间戳
    simulation_time_hour = start_hour   # 涉及到时间戳
 
    # 根据开始时间所在小时得到每个agent此时的active level 对应的阈值
    # 应该在perform action时读取
    threshold_list_by_time = [ threshold_dict[node_data['agent'].user_info.profile['other_info']['activity_level'][int(simulation_time_hour%24)]]  for _, node_data in agent_graph.get_agents()]
    agent_count = len(agent_graph.get_agents())

    # 用来记录用户活跃时间段
    user_active_state_dict = {agent_id:[0]*24 for agent_id in range(agent_count)}

    for timestep in range(num_timestep):
        print(Back.GREEN + f"timestep:{timestep}" + Back.RESET)
        simulation_time_hour = start_hour + 0.2*timestep   # 这里的1就是指的 1个timestep对应12分钟， 涉及到时间戳

        # 得到当前时间步下的随机值序列
        active_probs = [random.random() for _ in range(agent_count)]   # 同样放在时间戳系统中
        # 得到当前时间步下各agent的活跃状态对应的阈值形成的阈值序列， 放在时间戳系统中更新
        threshold_list_by_time = [ threshold_dict[node_data['agent'].user_info.profile['other_info']['activity_level'][int(simulation_time_hour%24)]]  for _, node_data in agent_graph.get_agents()]

        # 比较前两步得到的序列得到该时间步下的应激活 agent，是否应该激活 放在perform action中判断
        active_agent_id_list = [0 if p > t else 1 for p, t in zip(active_probs, threshold_list_by_time)]

        for node_id, node_data in agent_graph.get_agents():
            agent = node_data['agent']
            # 该激活的激活
            if active_agent_id_list[node_id] == 1:        
                # 记录激活时间
                user_active_state_dict[node_id][int(simulation_time_hour%24)] += 1
                await agent.perform_action_by_llm()
            # 不该激活的print提示
            else:
                print(Back.BLUE + f"agent{node_id}: I am {agent.user_info.profile['other_info']['activity_level'][int(simulation_time_hour%24)]} now, don't really feel like checking Twitter." + Back.RESET)

    # 记录用户的模拟活跃次数
    user_active_state_df = pd.DataFrame(user_active_state_dict)
    user_active_state_df.to_csv(f"user_active_state_{num_timestep}_timestep.csv")


    await channel.write_to_receive_queue((None, None, ActionType.EXIT))
    await task


if __name__ == "__main__":
    asyncio.run(running(num_timestep=5))