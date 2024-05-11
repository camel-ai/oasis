# File: SandboxTwitterModule/sandbox_twitter.py
import time
import asyncio

from SandboxTwitterModule.core.twitter_controller import TwitterController as TCtrl
from SandboxTwitterModule.infra.twitter_hci import TwitterHCI as THci 
from SandboxTwitterModule.infra.agents_graph import homoAgentsGraph
from SandboxTwitterModule.infra.twitter_infra import TwitterInfra as TInfra

from SandboxTwitterModule.core.decorators import function_call_logger


class SandboxTwitter:
    def __init__(self):
        # 这是 agents 的 homoGraph, 至少不是一个 complete heteroGraph,
        # 主要包含了 (暂时)接受用户输入的 agent_count, 初始化 agent_count 个对应的 twitterUserAgent
        # 使用 networkx 保存它们
        # （暂时）运行在初始化的时候，给 twitterUserAgent 增加一些相互之间的关系，即 edge<twitterUserAgent-i, twitterUserAgent-j>

        # 先初始化 AgentsGraph，此时不提供 DB 实例
        self.AgentsGraph = homoAgentsGraph()

        # 初始化 TwitterInfra 其构造函数会 初始化 DB 模块, twitter-recsys 模块，以及其他twitte必须有的组件
        self.infra = TInfra()
        print("SandboxTwitter 实例已创建。")

    # @function_call_logger
    # def update_signal(self, new_signal):
    #     self.signal = new_signal
    #     print("-----[SandboxTwitter_udpate_signal:]Signal updated to:", self.signal)

    # to-do for run()
    # 为了增加可维护性和降低耦合度：
    # 1、核心逻辑分层：保持 SandboxTwitter 类负责调度和协调，但将实际的业务逻辑（如用户交互模拟、数据库操作）放在其他模块或类中
    # 2、 还是 让每个 agent自行注册，自行登陆，自行操作吧

    @function_call_logger
    def run(self):
        print("Starting simulation...")
        agent = self.AgentsGraph.get_agents()
        agent.perform_action_by_llm()
