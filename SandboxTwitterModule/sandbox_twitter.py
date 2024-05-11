# File: SandboxTwitterModule/sandbox_twitter.py
import time
import asyncio


from SandboxTwitterModule.infra.agents_graph import homoAgentsGraph
from SandboxTwitterModule.infra import TwitterUserAgent
from SandboxTwitterModule.infra import Twitter
from SandboxTwitterModule.infra import ActionType
from SandboxTwitterModule.infra import Twitter_Channel

from SandboxTwitterModule.core.decorators import function_call_logger


class SandboxTwitter:
    def __init__(self):
        # 这是 agents 的 homoGraph, 至少不是一个 complete heteroGraph,
        # 主要包含了 (暂时)接受用户输入的 agent_count, 初始化 agent_count 个对应的 twitterUserAgent
        # 使用 networkx 保存它们
        # （暂时）运行在初始化的时候，给 twitterUserAgent 增加一些相互之间的关系，
        # 即 edge<twitterUserAgent-i, twitterUserAgent-j>

        # 先初始化 AgentsGraph，此时不提供 DB 实例
        # self.AgentsGraph = homoAgentsGraph()

        self.mock_twitter_file_path = (
            'SandboxTwitterModule/infra/twitter_infra_core/'
            'mock_twitter.db'
        )

        self.infra = Twitter(self.mock_twitter_file_path)
        self.channel = Twitter_Channel()

        print("mock twitter info server 实例已创建。")
        print("SandboxTwitter 实例已创建。")

    @function_call_logger
    async def run(self):
        print("Starting simulation...")
        # 启动twitter.running在后台执行
        task = asyncio.create_task(self.infra.running(self.channel))

        # # 发送一个动作到Twitter实例并等待响应
        # message_id = await self.channel.write_to_receive_queue((1, ("alice0101", "Alice", "A girl."), "sign_up"))

        # response = await self.channel.read_from_send_queue(message_id)
        # await self.channel.write_to_receive_queue((None, None, "exit"))
        # # 发送一个动作到Twitter实例并等待响应

        # print(f"Received response: {response}")

        test_agent = TwitterUserAgent(1, 'Alice', self.channel)
        await test_agent.action_sign_up("alice0101", "Alice", "A girl.")
        await test_agent.action_create_tweet("hello world")
        # 发送退出信号以优雅地结束running方法

        await self.channel.write_to_receive_queue((None, None, "exit"))

        await task

        # 现在取出的agent还不是TwitterUserAgent实例，以下暂时为伪代码
        '''
        agent = self.AgentsGraph.get_agents()
        agent.perform_action()
        '''
