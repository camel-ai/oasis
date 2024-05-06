# File: SandboxTwitterModule/infra/agents_graph.py
import networkx as nx
import matplotlib.pyplot as plt
import os, sys

from .agents_generator import AgentsGenerator as AgentsGen
from .Agent.twitterUserAgent import TwitterUserAgent as tuAgent

from ..core.decorators import function_call_logger

def id_to_name(agent_id):
    name = ''
    while len(name) < 3:
        remainder = agent_id % 26
        name = chr(97 + remainder) + name
        agent_id //= 26
    return name


class homoAgentsGraph:
    def __init__(self):
        self.count = self.get_agent_count() # 获取 Agents 的数量
        if self.count > 0:  # 确保 Agent 数量 > 0
            self.graph = self.build_agents_graph(self.count) # 创建 agentsgraph 并个性化 agent_count 个 tuAgent
            self.print_all_agents_info()  # 输出所有 tuAgents 的信息

    @function_call_logger
    def get_agents(self):
        """返回图中所有 agent 的信息，用于更新 DB 实例。"""
        return [(node_data['agent'], node_id) for node_id, node_data in self.graph.nodes(data=True)]


    # 获取 AgentsGraph 中 agent 的个数
    @function_call_logger
    def get_agent_count(self):
        agent_count_str = input("Please input agent_count: ")
        try:
            agent_count = int(agent_count_str)
            if agent_count <= 0:
                raise ValueError("Agent count must be a positive integer.")
                sys.exit("[Force_exit by <STM/infra/agents_graph.py>, AgentCountInputError_1]")
            return agent_count
        except ValueError:
            print('Input error, please enter a valid integer.')
            sys.exit("[Force_exit by <STM/infra/agents_graph.py>, AgentCountInputError_2]")
            return -1
        except KeyboardInterrupt:
            print('\nOperation interrupted by user.')
            sys.exit("[Force_exit by <STM/infra/agents_graph.py>, AgentCountInputError_3]")
            return -1


    # 初始化一个同构图并添加 agent_count 个 tuAgent
    @function_call_logger
    def init_homo_graph(self, agent_count):
        G = nx.DiGraph()  # 使用有向图
        
        # 为每个 tuAgent 生成一个节点，并将 tuAgent 实例作为属性添加到节点
        for agent_id in range(agent_count):
            name = id_to_name(agent_id)  # 使用自定义函数生成名字
            agent = tuAgent(agent_id, name)  # 创建 Agent 实例, 初始时不设置 DB
            G.add_node(agent_id, agent=agent) # 将创建的 Agent 实例添加到图的节点中

        return G

    @function_call_logger
    def build_agents_graph(self, agent_count):
        G = self.init_homo_graph(agent_count)
        tem_generator = AgentsGen()
        return (tem_generator.generate_agents(G, agent_count))


    # 暂时用于简单的检查 homoGraph是否成功
    @function_call_logger
    def print_all_agents_info(self):
        print("--------- All tuagents information: -----------")
        for node_id, node_data in self.graph.nodes(data=True):
            agent = node_data['agent']            
            print(f"tuAgent ID: {node_id}, Name: {agent.real_name}, MBTI: {agent.profile['other_info']['mbti']}, Activity Level: {agent.profile['other_info']['activity_level']}")

        print("------- All tuagents information ended ---------")




