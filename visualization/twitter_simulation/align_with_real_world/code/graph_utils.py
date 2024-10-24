import pickle
import networkx as nx
import matplotlib.pyplot as plt


# 使用 NetworkX 的 dfs_tree() 方法获取深度优先搜索树，得到图的深度
def get_dpeth(G: nx.Graph, source=0):
    # source = 0
    dfs_tree = nx.dfs_tree(G, source=source)
    # 获取树的深度
    max_depth = max(nx.single_source_shortest_path_length(dfs_tree, source=source).values())
    # print(f"图的最大深度为 {max_depth}")
    return max_depth

# 根据时间获取子图
def get_subgraph_by_time(G: nx.Graph, time_threshold = 10):
    
    # 假设我们要提取前 time_threshod 秒发的推
    filtered_nodes = []
    for node, attr in G.nodes(data=True):
        try:
            if attr['timestamp'] <= time_threshold:
                filtered_nodes.append(node)
        except:
            # print(f"node {node} dose not exist")
            pass
    # 使用 `subgraph()` 方法提取子图
    subG = G.subgraph(filtered_nodes)

    return subG

# 图的可视化
def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """
    从一个给定的根节点位置计算树的所有节点位置
    """
    pos = _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)
    return pos

def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None, parsed=None):
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
        neighbors.remove(parent)
    
    if len(neighbors) != 0:
        dx = width / len(neighbors) 
        nextx = xcenter - width/2 - dx/2
        for neighbor in neighbors:
            nextx += dx
            pos = _hierarchy_pos(G, neighbor, width=dx, vert_gap=vert_gap, vert_loc=vert_loc-vert_gap, xcenter=nextx, pos=pos, parent=root, parsed=parsed)
    return pos

def plot_graph_like_tree(G, root):
    # 获取树的节点位置
    pos = hierarchy_pos(G, root)

    # 绘制图
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", font_size=10, font_weight="bold", arrows=True)
    plt.title("Retweet Tree")
    plt.show()

