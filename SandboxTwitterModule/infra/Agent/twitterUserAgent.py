# File: SandboxTwitterModule/infra/Agent/twitterUserAgent.py
import networkx as nx
import random
import asyncio

from ..twitter_api import TwitterAPI

class TwitterUserAgent:
    def __init__(self, agent_id, real_name):
        self.db = None
        self.api = None
        self.agent_id = agent_id
        self.real_name = real_name
        self.profile = {
            'nodes': [],  # Relationships with other agents
            'edges': [],  # Relationship details
            'other_info': {'mbti': None},
        }
        self.memory = {'system_one': [], 'system_two': []}
        self.home_content = []

    def set_db(self, db):
        """Set the database instance and initialize the API."""
        self.db = db
        self.api = TwitterAPI(db)
        db_info = self.db.get_db_info()  # 假设这个方法返回数据库的位置或名称
        print(f"Database [{db_info}] set for agent {self.agent_id}")


    async def signup(self, password):
        """Sign up the agent with a unique username based on their real name."""
        username = self.real_name.lower().replace(" ", "")
        result = await self.api.signup(username, password, self.real_name, "A generated user profile description")
        if result['status'] == 'success':
            print(f"Agent {self.agent_id} signed up successfully with username {username}")
        else:
            print(f"Signup failed for Agent {self.agent_id}: {result['message']}")

    async def login(self, password):
        """Login the agent using their username."""
        username = self.real_name.lower().replace(" ", "")
        result = await self.api.login(username, password)
        if result['status'] == 'success':
            print(f"Agent {self.agent_id} logged in successfully")
        else:
            print(f"Login failed for Agent {self.agent_id}: {result['message']}")



    async def logout(self):
        print("Agent logged out and sleeping...")
        await asyncio.sleep(10)  # 模拟休眠


    async def tweet(self):
        """Post a tweet from the agent."""
        content = f"Hello world from {self.real_name}!"
        result = await self.api.tweet(self.agent_id, content)
        print(f"Tweeted: {result.get('tweet_id', 'Failed to tweet')}")

    async def like_tweet(self):
        """Like a tweet, assuming tweet IDs are known or tracked elsewhere."""
        tweet_id = random.randint(1, 100)
        result = await self.api.like_tweet(self.agent_id, tweet_id)
        print("Liked a tweet" if result['status'] == 'success' else "Failed to like a tweet")

    async def retweet(self):
        """Retweet a tweet."""
        original_tweet_id = random.randint(1, 100)
        result = await self.api.retweet(self.agent_id, original_tweet_id)
        print("Retweeted" if result['status'] == 'success' else "Failed to retweet")

    async def refresh_home(self):
        """Refresh home content by fetching new recommended tweets."""
        result = await self.api.home_refresh(self.agent_id)
        if result['status'] == 'success':
            self.home_content = result['tweets']
            print(f"Home refreshed with new tweets: {self.home_content}")
        else:
            print("Failed to refresh home")

    def think(self):
        """Simulate the agent's thinking process."""
        thoughts = "Considering what to do next..."
        print(f"Agent {self.agent_id} thinks: {thoughts}")

    async def perform_random_action(self):
        """Randomly perform one of the agent's actions."""
        actions = [self.tweet, self.like_tweet, self.retweet, self.refresh_home]
        action = random.choice(actions)
        await action()



