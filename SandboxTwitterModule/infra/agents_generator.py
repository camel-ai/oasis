# File: SandboxTwitterModule/infra/agents_generator.py
import ast
import random
import networkx as nx
import pandas as pd
from SandboxTwitterModule.infra.Agent.twitterUserAgent import TwitterUserAgent


class AgentsGenerator:
    def __init__(self, agent_info_path):
        """
        Initializes the AgentsGenerator with predefined MBTI types and activity levels.
        
        Args:
            agent_info_path (str): The file path to the agent information CSV file.
        """
        self.mbti_types = ["INTJ", "ENTP", "INFJ", "ENFP"]
        self.activities = ["High", "Medium", "Low"]
        self.agent_info_path = agent_info_path  # Path to the agent information file.
    
    def generate_agents(self):
        """
        Generates and returns a dictionary of agents from the agent information CSV file.
        Each agent is added to the database and their respective profiles are updated.

        Returns:
            dict: A dictionary of agent IDs mapped to their respective agent class instances.
        """
        agent_info = pd.read_csv(self.agent_info_path)
        agents_dict = {}
        for i in range(len(agent_info)):
            # Instantiate an agent
            agent = TwitterUserAgent(
                i, 
                agent_info['username'][i], 
                None
            )
            
            # Sign up agent and add their information to the database
            agent.action_sign_up(
                agent_info['username'][i], 
                agent_info['name'][i], 
                agent_info['description']
            )
            
            # Update agent profile with additional information
            agent.profile['other_info']['user_profile'] = agent_info['user_char'][i]
            # Randomly assign an MBTI type (temporary, subject to change)
            agent.profile['other_info']['mbti'] = random.choice(self.mbti_types)
            # Randomly assign an activity level (temporary, subject to change)
            agent.profile['other_info']['activity_level'] = random.choice(self.activities)
            
            # Add user relationships if any
            if agent_info['following_agentid_list'][i] != "0":
                following_id_list = ast.literal_eval(agent_info['following_agentid_list'][i])
                for _agent_id in following_id_list:
                    agent.action_follow(_agent_id)

            agents_dict[i] = agent

        return agents_dict






# class AgentsGenerator:
#     def __init__(self, agent_info_path):
#         self.mbti_types = ["INTJ", "ENTP", "INFJ", "ENFP"]
#         self.activities = ["High", "Medium", "Low"]

#         self.agent_info_path = agent_info_path # agent info file path
    
#     def generate_agents(self):
#         """
#         生成agents
#             1. 创建Agent实例，将这些实例通过字典的方式返回
#             2. 将agent的信息传到数据库里面。
#         参数：无
#         输出：字典 {agent_id: agent class}
#         """
#         agent_info = pd.read_csv(self.agent_info_path)
#         agents_dict = {}
#         for i in range(len(agent_info)):
#             # 实例化agent
#             agent = TwitterUserAgent(i, agent_info['username'][i], None)
            
#             # 将用户信息写入数据库（通过signup来实现）
#             agent.action_sign_up(agent_info['username'][i], agent_info['name'][i], agent_info['description'])
            
#             # 将其他信息写入agent的profile
#             agent.profile['other_info']['user_profile'] = agent_info['user_char'][i]
#             # TODO mbti现在没有用，先随机选可以去掉
#             agent.profile['other_info']['mbti'] = random.choice(self.mbti_types)
#             # TODO 将用户的时序发推频率输入进来，列表的形式和时间有关系。这里先初始化。
#             agent.profile['other_info']['activity_level'] = random.choice(self.activities)
            
#             # 添加用户关系列表
#             if agent_info['following_agentid_list'][i] != "0" :
#                 following_id_list = ast.literal_eval(agent_info['following_agentid_list'][i])
#                 for _agent_id in following_id_list:
#                     agent.action_follow(_agent_id)

#                 agents_dict[i] = agent
#         return agents_dict



    # def generate_agents(self, G, count):
    #     for node_id, node_data in G.nodes(data=True):
    #         agent = node_data['agent']
    #         agent.profile['other_info']['mbti'] = random.choice(["INTJ", "ENTP", "INFJ", "ENFP"])
    #         # agent.profile['other_info']['activity_level'] = random.choice(["High", "Medium", "Low"])
    #         # 添加随机的关注关系
    #         self.create_following_relations(G, count)
    #     return G


# 现在先不管 agents 的 real-world interaction
'''
    def create_following_relations(self, G, count):
        return
        for node in G.nodes():
            num_following = random.randint(1, max(1, count // 10))  # 假设最多关注10%的其他用户
            followings = random.sample(list(G.nodes()), num_following)
            followings = [f for f in followings if f != node]  # 确保不自关注
            for target in followings:
                G.add_edge(node, target)
'''