# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# print(ba_network.edges)
# # Print the generated nodes and edges
# print(f"Number of nodes generated: {ba_network.number_of_nodes()}")
# print(f"Number of edges generated: {ba_network.number_of_edges()}")
import random

import pandas as pd

# Parameter settings
# num_nodes = 997 # Total number of nodes
# num_edges = 7 # Number of original nodes each new node is connected to

# # Generate a scale-free network using the Barabási-Albert (BA) model
# ba_network = nx.barabasi_albert_graph(num_nodes, num_edges)

# Assuming there are n agents
n = 997

# Setting the number of edges
num_edges = 7000

# Generating random edges, ensuring follower != followee
edges = set()  # Use a set to avoid duplicate edges

while len(edges) < num_edges:
    follower = random.randint(0, n - 1)  # Randomly select a follower
    followee = random.randint(0, n - 1)  # Randomly select a followee
    if follower != followee:  # Ensure it's not following oneself
        edges.add((follower, followee))

# Convert the set to a list
edges = list(edges)

# Print the first few edges
print(edges[:10])  # Print the first 10 edges as an example
df = pd.read_csv('./1k_0.2.csv')

# Clear each agent's following list
df['agent_following_agentid_list'] = [[] for _ in range(len(df))]

# Iterate through edges to update the following relationships based on edges
for follower, followee in edges:
    df.at[follower, 'agent_following_agentid_list'].append(followee)

df.to_csv("./1k_random.csv")
