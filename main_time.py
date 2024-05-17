
import asyncio
import os

from social_agent.agents_generator import generate_agents
from twitter.channel import Twitter_Channel
from twitter.twitter import Twitter

import random
from datetime import datetime
from colorama import Back, Fore
import time
import pandas as pd

def rank_elements(lst):
    # 对列表进行排序，并使用enumerate()函数获取索引和值
    sorted_lst = sorted(enumerate(lst), key=lambda x: x[1], reverse=True)
    ranks = [0] * len(lst)  # 创建一个与列表长度相同的初始等级列表

    for i, (index, value) in enumerate(sorted_lst):
        ranks[index] = i

    return ranks

async def running(num_timestep):
    test_db_filepath = "./db/test.db"

    channel = Twitter_Channel()
    infra = Twitter(test_db_filepath, channel)
    task = asyncio.create_task(infra.running())

    agents = await generate_agents("./data/user_all_id.csv", channel)
    
    # 一个time step对应simulation里面的一个小时，后续看情况改动
    '''
    一个比较简单的思路为：为每种活跃状态设置一个对应阈值 比如  threshold_dict = {"off_line":0.0, "busy":0.3, "normal":0.6, "active":1} 
    simulation里每过去一个小时新产生一个[random.random() for _ in range(len(agents))] 的 0~1 的随机值列表
    并与 [ threshold_dict[agent.profile['other_info']['activity_level'][simulation_time_hour]] for agent in agents.values()]进行比较, 得到要被触发的agent列表
    也就是用激活概率实现了yuxian提到的对于设置agent在一段时间内的激活次数作为活跃程度表现
    从激活不激活来看优点是比较容易实现, 而且不激活就可以减少对于LLM的调用, 能够降低成本
    缺点就是目前的激活概率阈值比较简单, 就是靠初始化的数据得到的, 后续在running的时候没有其他变量的影响的话这种方法的可解释性就全部来自于初始化爬到的数据了.
    但是可以先靠这个实现一下
    '''
    threshold_dict = {"off_line":0.0, "busy":0.3, "normal":0.6, "active":1} 

    # 暂时是取当前系统时间戳的小时作为起始时间，后需要做 谣言传播 的话需要将其设置为 source twitter 的 create 时间
    current_time = datetime.now()
    
    start_hour = current_time.hour
    simulation_time_hour = start_hour

    # 根据开始时间所在小时得到每个agent此时的active level 对应的阈值
    threshold_list_by_time = [ threshold_dict[agent.profile['other_info']['activity_level'][simulation_time_hour%24]]  for agent in agents.values()]

    # 得到每个agent在这个小时的活跃次数，用这个活跃次数决定后续在同一批中哪个agent被先调用
    active_fre_by_time = [agent.profile['other_info']['activity_level_frequency'][simulation_time_hour%24] for agent in agents.values()]
    fre_order_list = rank_elements(active_fre_by_time)

    user_active_state_dict = {agent_id:[0]*24 for agent_id in range(len(agents))}

    for timestep in range(num_timestep):
        print(Back.GREEN + f"timestep:{timestep}" + Back.RESET)
        simulation_time_hour = start_hour + 0.2*timestep   # 这里的1就是指的 1个timestep对应12分钟

        # 得到当前时间步下的随机值序列
        active_probs = [random.random() for _ in range(len(agents))]
        # 得到当前时间步下各agent的活跃状态对应的阈值形成的阈值序列
        threshold_list_by_time = [ threshold_dict[agent.profile['other_info']['activity_level'][int(simulation_time_hour%24)]]  for agent in agents.values()]

        # 比较前两步得到的序列得到该时间步下的应激活agent
        active_agent_index_list = [0 if p > t else 1 for p, t in zip(active_probs, threshold_list_by_time)]
        # for index, agent in enumerate(agents.values()):
        
        active_fre_by_time = [agent.profile['other_info']['activity_level_frequency'][int(simulation_time_hour%24)] for agent in agents.values()]
        fre_order_list = rank_elements(active_fre_by_time)

        # 对于这一批应激活的agent按照他们在这个时间步所处小时的活跃频率调整激活先后顺序
        for index in fre_order_list:
            agent = list(agents.values())[index]
            if active_agent_index_list[index] == 1:
                # print(Fore.RED + f"agent{index}: I am active now" + Fore.RESET)
                user_active_state_dict[index][int(simulation_time_hour%24)] += 1
                await agent.perform_action_by_llm()
            else:
                print(Back.BLUE + f"agent{index}: I am {agent.profile['other_info']['activity_level'][int(simulation_time_hour%24)]} now, don't really feel like checking Twitter." + Back.RESET)
    
    # 记录用户的模拟活跃次数
    user_active_state_df = pd.DataFrame(user_active_state_dict)
    user_active_state_df.to_csv(f"user_active_state_{num_timestep}_timestep.csv")
    

    await channel.write_to_receive_queue((None, None, "exit"))
    await task


if __name__ == "__main__":
    asyncio.run(running(num_timestep=5))
