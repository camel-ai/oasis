from .agent_manager import AgentManager

def main():
    agentmanager = AgentManager()
    agentmanager.creat(1000)
    agentmanager.run()

if __name__ == "__main__":
    main()