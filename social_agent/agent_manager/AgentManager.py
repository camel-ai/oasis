import sqlite3
from social_agent.agent import Agent
from generator.create_database import get_db_path

def timedelay(amount):
    import random

    weights = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.37]
    values = [0.5, 1, 2, 3, 5, 8, 11]

    delays = random.choices(values, weights, k=amount)
    return delays

class AgentManager:
    def __init__(self):
        self.agents = []

    def creat(self, amount=1000):
        self.amount = amount
        self.delays = timedelay(self.amount)
        db_path = get_db_path()
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM user")
        users = cursor.fetchall()
        for user_id, user_name, name, bio, created_at, num_followings, num_followers in users:
            agent = Agent(user_id, self.delays[int(user_id)%7])
            self.add_agent(agent)

    def add_agent(self, agent):
        self.agents.append(agent)

    def remove_agent(self, agent):
        self.agents.remove(agent)

    def get_agents(self):
        return self.agents

    def get_agent(self, agent_id):
        for agent in self.agents:
            if agent.get_id() == agent_id:
                return agent
        return None