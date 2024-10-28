import sqlite3

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from graph_utils import get_dpeth, get_subgraph_by_time, plot_graph_like_tree


class prop_graph:

    def __init__(self, source_post_content, db_path="", viz=False):
        self.source_post_content = source_post_content  # 传播源头推特内容
        self.db_path = db_path  # 模拟结束后获得的db文件路径
        self.viz = viz  # 是否可视化出来
        self.post_exist = False  # 确定模拟是否成功运行，如果是空db的话这里就是False

    def build_graph(self):
        # 创建到SQLite数据库的连接
        conn = sqlite3.connect(self.db_path)

        # 执行SQL查询并将结果加载到DataFrame中
        query = "SELECT * FROM post"
        df = pd.read_sql(query, conn)
        # import pdb; pdb.set_trace()
        # 关闭数据库连接
        conn.close()

        all_reposts_and_time = []

        # 收集repost数据
        for i in range(len(df)):
            content = df.loc[i]["content"]
            # 数据有点encoding上的问题，加上[0:10]避开
            if self.post_exist is False and self.source_post_content[
                    0:10] in content:
                self.post_exist = True
                self.root_id = df.loc[i]["user_id"]
            if "repost from" in content and (self.source_post_content[0:10]
                                             in content):
                repost_history = content.split(". original_post: ")[:-1]
                repost_time = df.loc[i]["created_at"]
                all_reposts_and_time.append((repost_history, repost_time))

        # 建图
        # 给定的数据
        data = all_reposts_and_time
        # 获取起始时间
        # start_time =  df.loc[df["content"]==
        # self.source_post_content]["created_at"].item()
        start_time = 0
        # now start time is int, represent minutes
        # start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S.%f')

        # 创建一个有向图
        self.G = nx.DiGraph()

        first_flag = 1
        # 从数据中提取边并添加到图中
        for reposts, timestamp in data:
            # timestamp = datetime.strptime(
            #     timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
            time_diff = timestamp - start_time
            for repost in reposts:
                repost_info = repost.split(" repost from ")
                user = repost_info[0]
                original_user = repost_info[1]

                if first_flag:
                    self.root_id = original_user  # 获取source_post对应的的根节点
                    first_flag = 0
                    # 为根节点添加 timestamp 属性，值为0
                    if original_user not in self.G:
                        self.G.add_node(original_user, timestamp=0)

                # 为其他节点添加 timestamp 属性，值为 time_diff, 单位为分钟
                if user not in self.G:
                    self.G.add_node(user, timestamp=time_diff)
                    # print(f"user {user}, timestamp:{time_diff}")

                self.G.add_edge(original_user, user)

        # 获取传播开始与终止时间戳
        self.start_timestamp = 0
        timestamps = nx.get_node_attributes(self.G, "timestamp")
        try:
            self.end_timestamp = max(timestamps.values()) + 3
        except Exception as e:
            print(self.source_post_content)
            print(f"ERROR: {e}, may cause by empty repost path")
            print(f"the simulation db is empty: {not self.post_exist}")
            print("len of repost path:", len(all_reposts_and_time))

        # 统计传播图深度，规模，最大宽度max_breadth，传播结构 total_structural_virality
        self.total_depth = get_dpeth(self.G, source=self.root_id)
        self.total_scale = self.G.number_of_nodes()
        self.total_max_breadth = 0
        last_breadth_list = [1]
        for depth in range(self.total_depth):
            breadth = len(
                list(
                    nx.bfs_tree(
                        self.G, source=self.root_id, depth_limit=depth +
                        1).nodes())) - sum(last_breadth_list)
            last_breadth_list.append(breadth)
            if breadth > self.total_max_breadth:
                self.total_max_breadth = breadth

        undirect_G = self.G.to_undirected()
        self.total_structural_virality = nx.average_shortest_path_length(
            undirect_G)

    def viz_graph(self, time_threshold=10000):
        # 图的可视化，可选择只看前time_threshold秒的传播图
        subG = get_subgraph_by_time(self.G, time_threshold)
        plot_graph_like_tree(subG, self.root_id)

    def plot_depth_time(self, separate_ratio: float = 1):
        """
        整个传播过程
        前separate_ratio的过程的数据详细刻画
        之后的数据粗略刻画
        default to 1
        当传播时间很长时使用该参数, 可设置为0.01
        """
        # 计算深度-时间信息
        depth_list = []
        self.d_t_list = list(
            range(int(self.start_timestamp), int(self.end_timestamp),
                  1))  # 按正常间隔为1算的 time list, depth-time的信息要足够详细
        depth = 0
        for t in self.d_t_list:
            if depth < self.total_depth:
                try:
                    sub_g = get_subgraph_by_time(self.G, time_threshold=t)
                    depth = get_dpeth(sub_g, source=self.root_id)
                except Exception:
                    import pdb

                    pdb.set_trace()
            depth_list.append(depth)
        self.depth_list = depth_list

        if self.viz:
            # 使用plot()函数绘制折线图
            _, ax = plt.subplots()
            ax.plot(self.d_t_list, self.depth_list)

            # 添加标题和标签
            plt.title("propagation depth-time")
            plt.xlabel("time/minute")
            plt.ylabel("depth")

            # 显示图形
            plt.show()
        else:
            return self.d_t_list, self.depth_list

    def plot_scale_time(self, separate_ratio: float = 1.0):
        """
        整个传播过程的前separate_ratio*T的过程之间的数据详细刻画
        之后的数据粗略刻画
        default to 1
        当传播时间很长时使用该参数, 可设置为0.1
        """
        self.node_nums = []
        separate_point = int(
            int(self.start_timestamp) + separate_ratio *
            (int(self.end_timestamp) - int(self.start_timestamp))
        )  # start_time到separate point之间的数据详细刻画，separate point到end_time的数据粗略刻画

        self.s_t_list = list(
            range(
                int(self.start_timestamp), separate_point,
                1))  # + list(range(separate_point, int(self.end_time), 1000))
        for t in self.s_t_list:
            try:
                sub_g = get_subgraph_by_time(self.G, time_threshold=t)
                node_num = sub_g.number_of_nodes()
            except Exception:
                import pdb

                pdb.set_trace()

            self.node_nums.append(node_num)

        if self.viz:
            # 使用plot()函数绘制折线图
            _, ax = plt.subplots()
            ax.plot(self.s_t_list, self.node_nums)
            # 设置x轴的对数缩放
            # ax.set_xscale('log')

            # 设置x轴的刻度位置
            # ax.set_xticks([1, 10, 100, 1000, 10000])

            # 设置x轴的刻度标签
            # ax.set_xticklabels(['1', '10', '100', '1k', '10k'])

            # 添加标题和标签
            plt.title("propagation scale-time")
            plt.xlabel("time/minute")
            plt.ylabel("scale")

            # 显示图形
            plt.show()
        else:
            return self.s_t_list, self.node_nums

    def plot_max_breadth_time(self, interval=1):
        self.max_breadth_list = []

        self.b_t_list = list(
            range(int(self.start_timestamp), int(self.end_timestamp),
                  interval))
        for t in self.b_t_list:
            try:
                sub_g = get_subgraph_by_time(self.G, time_threshold=t)
            except Exception:
                import pdb

                pdb.set_trace()
            max_depth = self.depth_list[t - self.b_t_list[0]]
            max_breadth = 0
            last_breadth_list = [1]
            for depth in range(max_depth):
                breadth = len(
                    list(
                        nx.bfs_tree(
                            sub_g, source=self.root_id, depth_limit=depth +
                            1).nodes())) - sum(last_breadth_list)
                last_breadth_list.append(breadth)
                if breadth > max_breadth:
                    max_breadth = breadth
            self.max_breadth_list.append(max_breadth)

        if self.viz:
            # 使用plot()函数绘制折线图
            _, ax = plt.subplots()
            ax.plot(self.b_t_list, self.max_breadth_list)

            # 添加标题和标签
            plt.title("propagation max breadth-time")
            plt.xlabel("time/minute")
            plt.ylabel("max breadth")

            # 显示图形
            plt.show()
        else:
            return self.b_t_list, self.max_breadth_list

    def plot_structural_virality_time(self, interval=1):
        self.sv_list = []
        self.sv_t_list = list(
            range(int(self.start_timestamp), int(self.end_timestamp),
                  interval))

        for t in self.sv_t_list:
            try:
                sub_g = get_subgraph_by_time(self.G, time_threshold=t)
            except Exception:
                import pdb

                pdb.set_trace()
            sub_g = sub_g.to_undirected()
            sv = nx.average_shortest_path_length(sub_g)
            self.sv_list.append(sv)

        if self.viz:
            # 使用plot()函数绘制折线图
            _, ax = plt.subplots()
            ax.plot(self.sv_t_list, self.sv_list)

            # 添加标题和标签
            plt.title("propagation structural virality -time")
            plt.xlabel("time/minute")
            plt.ylabel("structural virality")

            # 显示图形
            plt.show()
        else:
            return self.sv_t_list, self.sv_list
