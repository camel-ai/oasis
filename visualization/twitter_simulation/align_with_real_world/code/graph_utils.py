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
import matplotlib.pyplot as plt
import networkx as nx


# Using NetworkX's dfs_tree() method to get the depth-first search tree, to
# find the depth of the graph
def get_dpeth(G: nx.Graph, source=0):
    # source = 0
    dfs_tree = nx.dfs_tree(G, source=source)
    # Get the depth of the tree
    max_depth = max(
        nx.single_source_shortest_path_length(dfs_tree,
                                              source=source).values())
    # print(f"The maximum depth of the graph is {max_depth}")
    return max_depth


# Get subgraph by time
def get_subgraph_by_time(G: nx.Graph, time_threshold=10):
    # Assuming we want to extract the tweets sent in the first time_threshold
    # seconds
    filtered_nodes = []
    for node, attr in G.nodes(data=True):
        try:
            if attr["timestamp"] <= time_threshold:
                filtered_nodes.append(node)
        except Exception:
            # print(f"node {node} does not exist")
            pass
    # Extract the subgraph using `subgraph()` method
    subG = G.subgraph(filtered_nodes)

    return subG


# Visualization of the graph
def hierarchy_pos(G,
                  root=None,
                  width=1.0,
                  vert_gap=0.2,
                  vert_loc=0,
                  xcenter=0.5):
    """
    Compute the positions of all nodes in the tree starting from a given root
    node position
    """
    pos = _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)
    return pos


def _hierarchy_pos(
    G,
    root,
    width=1.0,
    vert_gap=0.2,
    vert_loc=0,
    xcenter=0.5,
    pos=None,
    parent=None,
    parsed=None,
):
    if pos is None:
        pos = {root: (xcenter, vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)

    if parsed is None:
        parsed = {root}
    else:
        parsed.add(root)

    neighbors = list(G.neighbors(root))
    if not isinstance(G, nx.DiGraph) and parent is not None:
        neighbors.remove(
            parent)  # This ensures a directed graph is treated as such

    if len(neighbors) != 0:
        dx = width / len(neighbors)  # Horizontal space allocated for each node
        nextx = xcenter - width / 2 - dx / 2
        for neighbor in neighbors:
            nextx += dx
            pos = _hierarchy_pos(
                G,
                neighbor,
                width=dx,
                vert_gap=vert_gap,
                vert_loc=vert_loc - vert_gap,
                xcenter=nextx,
                pos=pos,
                parent=root,
                parsed=parsed,
            )
    return pos


def plot_graph_like_tree(G, root):
    # Get the positions of the nodes in the tree
    pos = hierarchy_pos(G, root)

    # Plot the graph
    plt.figure(figsize=(12, 8))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=3000,
        node_color="lightblue",
        font_size=10,
        font_weight="bold",
        arrows=True,
    )
    plt.title("Retweet Tree")
    plt.show()
