import asyncio
from info_server.core.twitter import Twitter
from recsys.code.recsys import RecSys
from social_agent.agent_manager.AgentManager import AgentManager
from controller.AgentThread import AgentThread
from communication.channel import Management
from generator.agent_generator import generate_agents

async def main():
    generate_agents()
    agentmanager = AgentManager()
    agentmanager.creat()
    agent_thread = AgentThread(agentmanager)
    twitter_thread = Twitter()
    recsys_thread = RecSys()
    main_manager = Management()
    twitter_channel = main_manager.regester_chanel("Twitter")
    agent_channel = main_manager.regester_chanel("Agent Group")
    recsys_channel = main_manager.regester_chanel("Recsys")
    main_manager.regester_thread(agent_thread.RunAgent)
    main_manager.regester_thread(agent_thread.RunSocial, channel=agent_channel)
    main_manager.regester_thread(twitter_thread.running, channel=twitter_channel)
    main_manager.regester_thread(recsys_thread.running, channel=recsys_channel)
    main_manager.run()


if __name__ == "__main__":
    asyncio.run(main())