# File: SandboxTwitterModule/infra/Agent/twitterUserAgent.py
<<<<<<< Updated upstream:SandboxTwitterModule/infra/Agent/twitterUserAgent.py
import networkx as nx
import random
import asyncio
from datetime import datetime
import logging

from ..twitter_api import TwitterAPI

class TwitterUserAgent:
    def __init__(self, agent_id, real_name):
=======
import asyncio
from camel.functions import OpenAIFunction
from typing import List
from camel.agents import ChatAgent
from camel.configs import FunctionCallingConfig
from camel.functions import OpenAIFunction
from camel.memories import ChatHistoryMemory, MemoryRecord

from camel.memories.context_creators.score_based import ScoreBasedContextCreator
from camel.messages import BaseMessage
from camel.models import ModelFactory, BaseModelBackend
from camel.types import ModelType, OpenAIBackendRole

from twitter.typing import ActionType

class TwitterUserAgent(ChatAgent):
    def __init__(self, agent_id, real_name, profile, channel):
        system_message = BaseMessage.make_assistant_message(
            role_name="system",
            content=f"You are a twitter user agent. Your profile is: {profile}, choose your action next step in the following list: tweet, follow, unfollow, like, unlike."
        )
        super().__init__(system_message)
>>>>>>> Stashed changes:social_agent/twitterUserAgent.py
        self.db = None
        self.api = None
        self.user_id = None
        self.agent_id = agent_id
        self.real_name = real_name
<<<<<<< Updated upstream:SandboxTwitterModule/infra/Agent/twitterUserAgent.py
        self.profile = {
            'nodes': [],  # Relationships with other agents
            'edges': [],  # Relationship details
            'other_info': {'mbti': None, 'activity_level': 'Medium'},
        }
=======
        self.profile = profile
>>>>>>> Stashed changes:social_agent/twitterUserAgent.py
        self.memory = {'system_one': [], 'system_two': []}
        self.home_content = []

<<<<<<< Updated upstream:SandboxTwitterModule/infra/Agent/twitterUserAgent.py
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
=======
    async def running(self):
        print(f"Agent {self.agent_id} is running")
        while True:
            await self.action_refresh()
            await asyncio.sleep(10)

    async def _perform_action(self, message, action_type):
        """根据传入的action_type和message执行action, 并得到返回值"""
        print(f"Sending message: ")
        message_id = await self.channel.write_to_receive_queue(
            (self.agent_id, message, action_type))

        response = await self.channel.read_from_send_queue(message_id)
        # 发送一个动作到Twitter实例并等待响应
        print(f"Received response: {response[2]}")
        return response[2]

    async def action_sign_up(self, user_name: str, name: str, bio: str):
        r"""Signs up a new user with the provided user name, name, and bio.

        This method prepares a user message comprising the user's details and
        invokes an asynchronous action to perform the sign-up process. On
        successful execution, it returns a dictionary indicating success along
        with the newly created user ID.

        Args:
            user_name (str): The username for the new user.
            name (str): The full name of the new user.
            bio (str): A brief biography of the new user.

        Returns:
            dict: A dictionary with two key-value pairs. The 'success' key
                maps to a boolean indicating whether the sign-up was
                successful, and 'user_id' key maps to the integer ID of the
                newly created user on success.

            Example of a successful return:
            {'success': True, 'user_id': 2}
        """
        user_message = (user_name, name, bio)
        return await self._perform_action(
            user_message, ActionType.SIGNUP.value)

    async def action_refresh(self):
        r"""Refreshes to get recommended tweets.

        This method invokes an asynchronous action to refresh and fetch
        recommended tweets. On successful execution, it returns a dictionary
        indicating success along with a list of tweets. Each tweet in the list
        contains details such as tweet ID, user ID, content, creation date,
        and the number of likes.

        Returns:
            dict: A dictionary with two key-value pairs. The 'success' key
                maps to a boolean indicating whether the refresh was
                uccessful. The 'tweets' key maps to a list of dictionaries,
                each representing a tweet with its details.

            Example of a successful return:
            {
                "success": True,
                "tweets": [
                    {
                        "tweet_id": 1,
                        "user_id": 23,
                        "content": "This is an example tweet content.",
                        "created_at": "2024-05-14T12:00:00Z",
                        "num_likes": 5
                    },
                    {
                        "tweet_id": 2,
                        "user_id": 42,
                        "content": "Another example tweet content.",
                        "created_at": "2024-05-14T12:05:00Z",
                        "num_likes": 15
                    }
                ]
            }
        """
        return await self._perform_action(
            None, ActionType.REFRESH.value)

    async def action_create_tweet(self, content: str):
        r"""Creates a new tweet with the given content.

        This method invokes an asynchronous action to create a new tweet based
        on the provided content. Upon successful execution, it returns a
        dictionary indicating success and the ID of the newly created tweet.

        Args:
            content (str): The content of the tweet to be created.

        Returns:
            dict: A dictionary with two key-value pairs. The 'success' key
                maps to a boolean indicating whether the tweet creation was
                successful. The 'tweet_id' key maps to the integer ID of the
                newly created tweet.

            Example of a successful return:
            {'success': True, 'tweet_id': 50}
        """
        return await self._perform_action(
            content, ActionType.CREATE_TWEET.value)

    async def action_like(self, tweet_id: int):
        r"""Creates a new like for a specified tweet.

        This method invokes an asynchronous action to create a new like for a
        tweet. It is identified by the given tweet ID. Upon successful
        execution, it returns a dictionary indicating success and the ID of
        the newly created like.

        Args:
            tweet_id (int): The ID of the tweet to be liked.

        Returns:
            dict: A dictionary with two key-value pairs. The 'success' key
                maps to a boolean indicating whether the like creation was
                successful. The 'like_id' key maps to the integer ID of the
                newly created like.

            Example of a successful return:
            {"success": True, "like_id": 123}

        Note:
            Attempting to like a tweet that the user has already liked will
            result in a failure.
        """
        return await self._perform_action(
            tweet_id, ActionType.LIKE.value)

    async def action_unlike(self, tweet_id: int):
        """Removes a like based on the tweet's ID.

        This method removes a like from the database, identified by the
        tweet's ID. It returns a dictionary indicating the success of the
        operation and the ID of the removed like.

        Args:
            tweet_id (int): The ID of the tweet to be unliked.

        Returns:
            dict: A dictionary with 'success' indicating if the removal was
                successful, and 'like_id' the ID of the removed like.

            Example of a successful return:
            {"success": True, "like_id": 123}

        Note:
            Attempting to remove a like for a tweet that the user has not
            previously liked will result in a failure.
        """
        return await self._perform_action(
            tweet_id, ActionType.UNLIKE.value)

    async def action_search_tweets(self, query: str):
        r"""searches tweets based on a given query.

        This method performs a search operation in the database for tweets
        that match the given query string. The search considers the
        tweet's content, tweet ID, and user ID. It returns a dictionary
        indicating the operation's uccess and, if successful, a list of
        tweets that match the query.

        Args:
            query (str): The search query string. The search is performed
                against the tweet's content, tweet ID, and user ID.

        Returns:
            dict: A dictionary with a 'success' key indicating the operation's
                success. On success, it includes a 'tweets' key with a list of
                dictionaries, each representing a tweet. On failure, it
                includes an 'error' message or a 'message' indicating no
                tweets were found.

            Example of a successful return:
            {
                "success": True,
                "tweets": [
                    {
                        "tweet_id": 1,
                        "user_id": 42,
                        "content": "Hello, world!",
                        "created_at": "2024-05-14T12:00:00Z",
                        "num_likes": 150
                    },
                    ...
                ]
            }
        """
        return await self._perform_action(
            query, ActionType.SEARCH_TWEET.value)

    async def action_search_user(self, query: str):
        r"""Searches users based on a given query.

        This asynchronous method performs a search operation in the database
        for users that match the given query string. The search considers the
        user's username, name, bio, and user ID. It returns a dictionary
        indicating the operation's success and, if successful, a list of users
        that match the query.

        Args:
            query (str): The search query string. The search is performed
                against the user's username, name, bio, and user ID.

        Returns:
            dict: A dictionary with a 'success' key indicating the operation's
                success. On success, it includes a 'users' key with a list of
                dictionaries, each representing a user. On failure, it includes
                an 'error' message or a 'message' indicating no users were
                found.

            Example of a successful return:
            {
                "success": True,
                "users": [
                    {
                        "user_id": 1,
                        "user_name": "exampleUser",
                        "name": "John Doe",
                        "bio": "This is an example bio",
                        "created_at": "2024-05-14T12:00:00Z",
                        "num_followings": 100,
                        "num_followers": 150
                    },
                    ...
                ]
            }
        """
        return await self._perform_action(
            query, ActionType.SEARCH_USER.value)

    async def action_follow(self, followee_id: int):
        r"""Follow a users.

        This method allows agent to follow another user (followee).
        It checks if the agent initiating the follow request has a
        corresponding user ID and if the follow relationship already exists.

        Args:
            followee_id (int): The user ID of the user to be followed.

        Returns:
            dict: A dictionary with a 'success' key indicating the operation's
                success. On success, it includes a 'follow_id' key with the ID
                of the newly created follow record. On failure, it includes an
                'error' message.

            Example of a successful return:
            {"success": True, "follow_id": 123}
        """
        return await self._perform_action(
            followee_id, ActionType.FOLLOW.value)

    async def action_unfollow(self, followee_id: int):
        r"""Unfollow a users.

        This method allows agent to unfollow another user (followee). It
        checks if the agent initiating the unfollow request has a
        corresponding user ID and if the follow relationship exists. If so,
        it removes the follow record from the database, updates the followers
        and followings count for both users, and logs the action.

        Args:
            followee_id (int): The user ID of the user to be unfollowed.

        Returns:
            dict: A dictionary with a 'success' key indicating the operation's
                success. On success, it includes a 'follow_id' key with the ID
                of the removed follow record. On failure, it includes an
                'error' message.

            Example of a successful return:
            {"success": True, "follow_id": 123}
        """
        return await self._perform_action(
            followee_id, ActionType.UNFOLLOW.value)

    async def action_mute(self, mutee_id: int):
        r"""Mutes a user.

        Allows agent to mute another user. Checks for an existing mute
        record before adding a new one to the database.

        Args:
            mutee_id (int): ID of the user to be muted.

        Returns:
            dict: On success, returns a dictionary with 'success': True and
                mute_id' of the new record. On failure, returns 'success':
                False and an 'error' message.

            Example of a successful return:
            {"success": True, "mutee_id": 123}
        """
        return await self._perform_action(
            mutee_id, ActionType.MUTE.value)

    async def action_unmute(self, mutee_id: int):
        r"""Unmutes a user.

        Allows agent to remove a mute on another user. Checks for an
        existing mute record before removing it from the database.

        Args:
            mutee_id (int): ID of the user to be unmuted.

        Returns:
            dict: On success, returns a dictionary with 'success': True and
                'mutee_id' of the unmuted record. On failure, returns
                'success': False and an 'error' message.

            Example of a successful return:
            {"success": True, "mutee_id": 123}
        """
        return await self._perform_action(
            mutee_id, ActionType.UNMUTE.value)

    async def action_trend(self):
        r"""Fetches the top trending tweets within a predefined time period.

        Retrieves the top K tweets with the most likes in the last specified
        number of days.

        Returns:
            dict: On success, returns a dictionary with 'success': True and a
                list of 'tweets', each tweet being a dictionary containing
                'tweet_id', 'user_id', 'content', 'created_at', and
                'num_likes'. On failure, returns 'success': False and an
                'error' message or a message indicating no trending tweets
                were found.

        Example of a successful return:
        {
            "success": True,
            "tweets": [
                {
                    "tweet_id": 123,
                    "user_id": 456,
                    "content": "Example tweet content",
                    "created_at": "2024-05-14T12:00:00",
                    "num_likes": 789
                },
                ...
            ]
        }
        """
        return await self._perform_action(
            None, ActionType.TREND.value)

>>>>>>> Stashed changes:social_agent/twitterUserAgent.py
