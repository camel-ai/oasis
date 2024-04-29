import asyncio
import json
from time import sleep

class AgentThread:
    def __init__(self, agentmanager):
        self.agentmanager = agentmanager
        self.name = "Agent"

    async def RunAgent(self):
        print("AgentThread Start.")
        tasks = [agent.running() for agent in self.agentmanager.get_agents()]
        await asyncio.gather(*tasks)

    async def RunSocial(self, channel):
        print("SocialThread Start.")
        async def recive(channel):
            while True:
                data = await channel.recive_from('Twitter')
                try:
                    id, content = data
                    agent = self.agentmanager.get_agent(int(id))
                    await agent.process_input(content)

                except:
                    pass

        async def send(channel):
            while True:
                tasks = [agent.process_output(channel) for agent in self.agentmanager.get_agents()]
                await asyncio.gather(*tasks)

        while True:
            try:
                await asyncio.gather(recive(channel), send(channel))
            except:
                print("Waiting for connection...")
                sleep(3)
                continue