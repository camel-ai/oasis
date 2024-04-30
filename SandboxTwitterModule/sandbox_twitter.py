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
        # 初始化 TwitterController，其构造函数中会自动将信号从 END 更新为 PRE_START
        self.controller = TCtrl() 
        self.signal = self.controller.signal # 从controller中获取当前的signal

        # 初始化 TwitterHCI 其构造函数会 获取一定的配置输入
        self.hci = THci(self.update_signal)



        # 这是 agents 的 homoGraph, 至少不是一个 complete heteroGraph, 
        # 主要包含了 (暂时)接受用户输入的 agent_count, 初始化 agent_count 个对应的 twitterUserAgent 
        # 使用 networkx 保存它们
        # （暂时）运行在初始化的时候，给 twitterUserAgent 增加一些相互之间的关系，即 edge<twitterUserAgent-i, twitterUserAgent-j>
        
        # 先初始化 AgentsGraph，此时不提供 DB 实例
        self.AgentsGraph = homoAgentsGraph() 

        # 初始化 TwitterInfra 其构造函数会 初始化 DB 模块, twitter-recsys 模块，以及其他twitte必须有的组件
        self.infra = TInfra()

        # 更新所有 twitterUserAgent 的 twitterDB实例 和 twitterAPI
        
        self.update_agents_db()  

        print("SandboxTwitter 实例已创建。")

    @function_call_logger
    def update_agents_db(self):
        """更新所有 TwitterUserAgent 实例，设置 TwitterDB 实例和 TwitterAPI。"""
        for agent, node_id in self.AgentsGraph.get_agents():
            agent.set_db(self.infra.db)  # 为每个代理设置数据库实例
    

    
    # to-do for run() 
    # 为了增加可维护性和降低耦合度：
    # 1、核心逻辑分层：保持 SandboxTwitter 类负责调度和协调，但将实际的业务逻辑（如用户交互模拟、数据库操作）放在其他模块或类中
    # 2、 还是 让每个 agent自行注册，自行登陆，自行操作吧
    


    @function_call_logger
    def run(self):  
        print("Starting simulation...")
        asyncio.run(self.RunSimula())

        
    @function_call_logger
    def update_signal(self, new_signal):
        self.signal = new_signal
        print("-----[SandboxTwitter_udpate_signal:]Signal updated to:", self.signal)




    @function_call_logger
    async def RunSimula(self):
        tasks = []
        for agent_data in self.AgentsGraph.get_agents():
            agent = TwitterUserAgent(self.infra.api, agent_data['id'], agent_data['name'])
            tasks.append(agent.signup("securepassword123")) # 注册
            tasks.append(agent.login("securepassword123")) # 登陆
            tasks.append(agent.perform_random_action()) # 执行随机动作
            tasks.append(agent.refresh_home()) # 刷新主页
            tasks.append(agent.logout()) # 登出

        # 并发运行所有任务
        await asyncio.gather(*tasks)




