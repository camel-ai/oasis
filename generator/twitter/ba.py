import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# 参数设置
# num_nodes = 997 # 总节点数
# num_edges = 7 # 每次新加节点时连接到的原节点数

# # 生成Barabási-Albert (BA) 模型的无标度网络
# ba_network = nx.barabasi_albert_graph(num_nodes, num_edges)

# # print(ba_network.edges)
# # # 打印生成的节点和边
# # print(f"生成的节点数: {ba_network.number_of_nodes()}")
# print(f"生成的边数: {ba_network.number_of_edges()}")
import random

# 假设有 n 个 agents
n = 997

# 设置边的数量
num_edges = 7000

# 生成随机的边，确保 follower != followee
edges = set()  # 用 set 来避免重复边

while len(edges) < num_edges:
    follower = random.randint(0, n - 1)  # 随机选择 follower
    followee = random.randint(0, n - 1)  # 随机选择 followee
    if follower != followee:  # 确保不是自己关注自己
        edges.add((follower, followee))

# 将 set 转为 list
edges = list(edges)

# 打印前几条边
print(edges[:10])  # 打印前10条边，作为示例
df = pd.read_csv('./1k_0.2.csv')

# 清空每个agent的following list
df['agent_following_agentid_list'] = [[] for _ in range(len(df))]

# 遍历edges，根据edges来更新关注关系
for follower, followee in edges:
    df.at[follower, 'agent_following_agentid_list'].append(followee)

df.to_csv("./1k_random.csv")