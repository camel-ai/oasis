# File: SandboxTwitterModule/infra/agents_generator.py
import random
import networkx as nx

class AgentsGenerator:
    def __init__(self):
        self.mbti_types = ["INTJ", "ENTP", "INFJ", "ENFP"]
        self.activities = ["High", "Medium", "Low"]
    
    def generate_agents(self, G, count):
        for node_id, node_data in G.nodes(data=True):
            agent = node_data['agent']
            agent.profile['other_info']['mbti'] = random.choice(["INTJ", "ENTP", "INFJ", "ENFP"])
            # agent.profile['other_info']['activity_level'] = random.choice(["High", "Medium", "Low"])
            # 添加随机的关注关系
            self.create_following_relations(G, count)
        return G


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