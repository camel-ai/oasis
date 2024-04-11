import asyncio
import json
from time import sleep

class ThreadManager:
    def __init__(self, agentmanager):
        self.agentmanager = agentmanager

    async def RunAgent(self):
        print("AgentThread Start.")
        tasks = [agent._arun() for agent in self.agentmanager.get_agents()]
        await asyncio.gather(*tasks)

    async def RunSocial(self, host='127.0.0.1', port=8000):
        print("SocialThread Start.")
        async def recive(reader):
            while True:
                data = await reader.readline()
                data = data.decode()
                try:
                    message = json.loads(data)
                    id, content = message["id"], message["content"]
                    agent = self.agentmanager.get_agent(int(id))
                    await agent.process_input(content)
                    
                except:
                    pass


        async def send(writer):
            while True:
                tasks = [agent.process_output(writer) for agent in self.agentmanager.get_agents()]
                await asyncio.gather(*tasks)
        
        while True:
            try:
                reader, writer = await asyncio.open_connection(host, port)  
                print(f'Connected to {host}:{port}')
                await asyncio.gather(recive(reader), send(writer))
            except:
                print("Waiting for connection...")
                sleep(3)
                continue
        