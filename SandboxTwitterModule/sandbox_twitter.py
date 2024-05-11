# File: SandboxTwitterModule/sandbox_twitter.py
import time
import asyncio
import random

from SandboxTwitterModule.infra.agents_graph import homoAgentsGraph
from SandboxTwitterModule.infra import Twitter
from SandboxTwitterModule.infra import Twitter_Channel
from SandboxTwitterModule.core.decorators import function_call_logger

class SandboxTwitter:
    def __init__(self):
        self.mock_twitter_file_path = (
            'SandboxTwitterModule/infra/twitter_infra_core/mock_twitter.db'
        )
        self.channel = Twitter_Channel()
        self.infra = Twitter(self.mock_twitter_file_path, self.channel)
        self.agents_graph = homoAgentsGraph(self.channel)  # Initialize the agents graph, ensuring the channel is passed

        print("Mock twitter info server instance created.")
        print("SandboxTwitter instance created.")

    @function_call_logger
    async def run(self):
        print("Starting simulation...")
        # Start the Twitter running in the background
        task = asyncio.create_task(self.infra.running())

        # Simulate agent activities
        self.agents_graph.simulate_agent_activities(num_activities=10)

<<<<<<< HEAD
        # response = await self.channel.read_from_send_queue(message_id)
        # await self.channel.write_to_receive_queue((None, None, "exit"))
        # # 发送一个动作到Twitter实例并等待响应

        # print(f"Received response: {response}")

        test_agent = TwitterUserAgent(1, 'Alice', self.channel)
        await test_agent.action_sign_up("alice0101", "Alice", "A girl.")
        await test_agent.action_create_tweet("hello world")

        # 发送退出信号以结束running方法
=======
        # Wait for all tasks to complete
        await asyncio.gather(*[task for task in asyncio.all_tasks() if task is not asyncio.current_task()])

        # Send an exit signal to gracefully end the `running` method
>>>>>>> c4bedce75bb9ab6f97abe3e6526a43fab6b513e5
        await self.channel.write_to_receive_queue((None, None, "exit"))
        await task
        print("Simulation completed.")
