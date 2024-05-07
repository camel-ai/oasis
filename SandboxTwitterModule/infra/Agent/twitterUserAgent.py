# File: SandboxTwitterModule/infra/Agent/twitterUserAgent.py
import networkx as nx
import random
import asyncio
from datetime import datetime
import logging

from ..twitter_api import TwitterAPI

class TwitterUserAgent:
    def __init__(self, agent_id, real_name):
        self.db = None
        self.api = None
        self.user_id = None
        self.agent_id = agent_id
        self.real_name = real_name
        self.profile = {
            'nodes': [],  # Relationships with other agents
            'edges': [],  # Relationship details
            'other_info': {'mbti': None, 'activity_level': 'Medium'},
        }
        self.memory = {'system_one': [], 'system_two': []}
        self.home_content = []

    def set_db(self, db):
        """Set the database instance and initialize the API."""
        self.db = db
        self.api = TwitterAPI(db)
        db_info = self.db.get_db_info()  # 假设这个方法返回数据库的位置或名称
        print(f"[tuAgent.py] Database [{db_info}] set for agent {self.agent_id}")


    def think(self):
        """Simulate the agent's thinking process."""
        thoughts = "Considering what to do next..."
        print(f"Agent {self.agent_id} thinks: {thoughts}")


    async def signup(self, password):
        """Sign up the agent with a unique username based on their real name."""
        username = self.real_name.lower().replace(" ", "")
        description = "A generated user profile description"
        result = await self.api.signup(self.real_name, username, password, description)
        if result['status'] == 'success':
            print(f"Agent {self.agent_id} signed up successfully with username {username}")
        else:
            print(f"Signup failed for Agent {self.agent_id}: {result['message']}")



    async def login_by_realname(self, password):
        """Login the agent using their real name."""
        result = await self.api.login_by_realname(self.real_name, password)
        if result['status'] == 'success':
            user = self.db.get_user_by_real_name(self.real_name)
            self.db.update_user_login_state(user.id, True)
            print(f"Agent {self.agent_id} logged-in successfully using realname[{self.real_name}]")
        else:
            print(f"Login failed for Agent {self.agent_id}: {result['message']}")

    async def login_by_username(self, password):
        """Login the agent using their username."""
        username = self.real_name.lower().replace(" ", "")
        result = await self.api.login_by_username(username, password)
        if result['status'] == 'success':
            user = self.db.get_user_by_username(username)
            self.db.update_user_login_state(user.id, True)
            print(f"Agent {self.agent_id} logged-in successfully using username[{username}]")
        else:
            print(f"Login failed for Agent {self.agent_id}: {result['message']}")

    async def login(self, password, by_real_name, by_username):
        """Login the agent using their real_name or username based on flags."""
        if by_real_name and by_username:
            await self.login_by_realname(password)
        elif by_real_name:
            await self.login_by_realname(password)
        elif by_username:
            await self.login_by_username(password)
        else:
            raise ValueError("Invalid login parameters: Must specify at least one method (by_real_name or by_username)")



    async def logout(self):
        """Logout the agent using its real name."""
        if self.api and self.db:
            user = self.db.get_user_by_real_name(self.real_name)  # 先获取用户对象
            if user:
                result = await self.api.logout(user.id)  # 传递正确的用户 ID
                if result['status'] == 'success':
                    print(f"Agent {self.agent_id} logged-out successfully.")
                else:
                    print(f"Logout failed for Agent {self.agent_id}: {result['message']}")
            else:
                print("User not found. Cannot perform logout.")
        else:
            print("Database or API instance not set. Cannot perform [logout].")




    async def perform_random_action(self):
        """根据 agent 的个性和情境随机选择行为"""
        activities = ['tweet_random', 'follow_random_user', 'unfollow_random_user', 'choose_tweet_to_like']
        probabilities = [0.25, 0.25, 0.25, 0.25]  # 添加点赞行为的概率

        # 调整概率以反映代理的活跃度
        if self.profile['other_info']['activity_level'] == 'High':
            probabilities = [0.25, 0.25, 0.25, 0.25]  # 对于活跃用户，均等分配概率
        elif self.profile['other_info']['activity_level'] == 'Low':
            probabilities = [0.15, 0.25, 0.25, 0.35]  # 对于不太活跃的用户，增加点赞的概率，减少发推的概率
        
        chosen_activity = random.choices(activities, weights=probabilities, k=1)[0]
        await getattr(self, chosen_activity)()  # 调用选择的方法





    def generate_tweet_content(self):
        """生成推文内容"""
        return f"Hello twitter from {self.real_name} at {datetime.now()}"

    async def tweet_random(self):
        """Attempt to post a random tweet."""
        tweet_content = self.generate_tweet_content()
        try:
            user = self.db.get_user_by_real_name(self.real_name)  # Try to get the user object
            if user:
                result = await self.api.tweet(user.id, tweet_content)
                if result['status'] == 'success':
                    print(f"Tweet result for {self.real_name}: Tweet posted successfully with ID {result['tweet_id']}")
                else:
                    logging.error(f"Tweet failed for {self.real_name}: {result['message']}")
                    print(f"Tweet result for {self.real_name}: {result['message']}")
            else:
                logging.warning(f"User not found for {self.real_name}: Cannot perform tweet.")
                print("User not found. Cannot perform [tweet].")
        except Exception as e:
            logging.error(f"Error during tweeting by {self.real_name}: {str(e)}")
            print(f"An error occurred while trying to tweet: {str(e)}")


    async def fetch_user_id_by_real_name(self, real_name):
        """Fetch user_id from the database by real_name."""
        try:
            user_id = await self.api.fetch_user_id_by_real_name(real_name)
            return user_id
        except Exception as e:
            print(f"Error fetching user ID for {real_name}: {str(e)}")
            return None
    
    async def setup_user_id(self, real_name):
        """Setup the user_id for the agent based on the real_name."""
        try:
            user_id = await self.api.fetch_user_id_by_real_name(real_name)
            if user_id is not None:
                self.user_id = user_id
                print(f"[tuAgent.py setup_user_id]User ID for User(real_name={real_name}) set to {self.user_id}.")
            else:
                print(f"[tuAgent.py setup_user_id]Failed to fetch or set User ID for User(real_name={real_name}).")
        except Exception as e:
            print(f"[tuAgent.py setup_user_id]Error setting user ID for User(real_name={real_name}): {str(e)}")


    async def follow_random_user(self):
        """Randomly follow a user, ensuring the user exists."""
        if self.user_id is None:
            await self.setup_user_id(self.real_name)  # Assume self.real_name is already set
            if self.user_id is None:
                print("[tuAgent.py follow_random_user]Unable to perform follow operation: User ID is not set.")
                return
        users = await self.api.fetch_all_users()  # Assuming this is a method from API to get all users
        valid_users = [user for user in users if user.id != self.user_id] # 请注意！这里没有严格检查 != 左右两边的 user['id'] 和 self.user_id 的类型是否一致
        if valid_users:
            user_to_follow = random.choice(valid_users)
            result = await self.api.follow_user(self.user_id, user_to_follow.id)
            print(f"[tuAgent.py follow_random_user]{self.real_name} follow operation: {result['message']}")
        else:
            print("[tuAgent.py follow_random_user]No valid users available to follow.")



    async def unfollow_random_user(self):
        """Randomly select a followed user and unfollow them."""
        if self.user_id is None:
            await self.setup_user_id(self.real_name)
            if self.user_id is None:
                print("[tuAgent.py unfollow_random_user] Unable to perform unfollow operation: User ID is not set.")
                return

        following_ids = await self.api.get_following_ids(self.user_id)
        print(f"[tuAgent.py unfollow_random_user] following_ids={following_ids}")
        if not following_ids:
            print(f"[tuAgent.py unfollow_random_user] {self.real_name} is not following anyone.")
            return

        user_to_unfollow_id = random.choice(following_ids)
        await self.unfollow_user(self.user_id, user_to_unfollow_id)

    async def unfollow_user(self, follower_id, followee_id):
        """取消关注特定用户"""
        result = await self.api.unfollow_user(follower_id, followee_id)
        print(f"[tuAgent.py unfollow_user] Unfollow operation for User(id={follower_id}) from User(id={followee_id}): {result['message']}")




    async def choose_tweet_to_like(self):
        """获取所有可点赞的推文"""
        if self.user_id is None:
            await self.setup_user_id(self.real_name)

        tweets = await self.api.fetch_tweets_by_user(self.user_id)
        if tweets:
            await self.like_random_tweet(tweets)
        else:
            print(f"No tweets available to like for {self.real_name}.")

    async def like_random_tweet(self, tweets):
        """从获取的推文中随机选择一条进行点赞"""
        tweet_to_like = random.choice(tweets) if tweets else None
        await self.like_tweet(tweet_to_like)

    async def like_tweet(self, tweet):
        """点赞指定的推文"""
        if tweet:
            result = await self.api.like_tweet(self.user_id, tweet.tweet_id)
            print(f"Like operation for {self.real_name}: {result['message']}")
        else:
            print("No tweet provided for liking operation.")





























    # 每次登出以后，agent 会休息一段时间，t0 是根据 agent 的活跃度来设定的
    # def agent_sleep(self, t0): 
    #   print(f"Agent {self.agent_id} logged-out and sleeping for t0 seconds...")
    #   await asyncio.sleep(t0)  # Simulate delay after logging out



    '''
    
    async def retweet(self):
        """Retweet a tweet."""
        original_tweet_id = random.randint(1, 100)
        result = await self.api.retweet(self.agent_id, original_tweet_id)
        print("Retweeted" if result['status'] == 'success' else "Failed to retweet")

    async def refresh_home(self):
        """Refresh the agent's home timeline."""
        result = await self.api.home_refresh(self.agent_id)
        if result['status'] == 'success':
            self.home_content = result['tweets']
            print(f"Home timeline refreshed for {self.real_name}")
        else:
            print(f"Failed to refresh home timeline for {self.real_name}")
    '''
