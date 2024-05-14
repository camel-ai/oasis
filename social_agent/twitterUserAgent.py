import asyncio
import json
import os
import time


from colorama import Fore

from camel.configs import FunctionCallingConfig
from camel.functions import OpenAIFunction
from camel.memories import ChatHistoryMemory, MemoryRecord

from camel.memories.context_creators.score_based import ScoreBasedContextCreator
from camel.messages import BaseMessage
from camel.models import ModelFactory, BaseModelBackend
from camel.types import ModelType, OpenAIBackendRole
from twitter.channel import Twitter_Channel

from twitter.typing import ActionType


class TwitterUserAgent:

    def __init__(self, agent_id, real_name, description, profile, channel, model_type=ModelType.GPT_3_5_TURBO):
        self.user_id = None
        self.agent_id = agent_id
        self.real_name = real_name
        self.description = description
        self.profile = profile

        self.channel = channel

        # tweet follow unfollow like unlike
        function_list = [OpenAIFunction(func) for func in
                         [self.action_create_tweet, self.action_follow, self.action_unfollow, self.action_like,
                          self.action_unlike, self.action_search_tweets, self.action_search_user, self.action_trend,
                          self.action_refresh, self.action_mute, self.action_unmute]]
        assistant_model_config = FunctionCallingConfig.from_openai_function_list(
            function_list=function_list,
            kwargs=dict(temperature=0.0),
        )
        self.model_backend: BaseModelBackend = ModelFactory.create(
            model_type, assistant_model_config.__dict__)
        self.model_token_limit = self.model_backend.token_limit
        context_creator = ScoreBasedContextCreator(
            self.model_backend.token_counter,
            self.model_token_limit,
        )
        self.memory = ChatHistoryMemory(
            context_creator, window_size=3)
        self.system_message = BaseMessage.make_assistant_message(
            role_name="User",
            content=f"You are a twitter user agent named {self.real_name}. Your profile is: {self.description}\n. choose your action next step in the following list: create_tweet, follow user, unfollow user, like tweet, unlike tweet, search tweets, search user, trend, refresh, mute, unmute."
        )
        system_record = MemoryRecord(self.system_message,
                                     OpenAIBackendRole.SYSTEM)
        self.memory.write_record(system_record)
        self.home_content = []

    async def perform_action_by_llm(self):
        # 1. get 5 random tweets
        tweets = await self.action_refresh()

        print(Fore.LIGHTBLUE_EX + f"Agent {self.agent_id} fetched tweets after refreshing: {tweets}" + Fore.RESET)
        # 2. get context form memory
        user_msg = BaseMessage.make_user_message(
            role_name="User",
            content="After refreshing, you see some tweets, you want to perform some actions based on these tweets: " + str(tweets),
        )

        self.memory.write_record(MemoryRecord(user_msg, OpenAIBackendRole.USER))

        openai_messages, num_tokens = self.memory.get_context()

        print(Fore.LIGHTCYAN_EX + f"Agent {self.agent_id} got context from memory: {openai_messages}" + Fore.RESET)

        # Obtain the model's response
        response = self.model_backend.run(openai_messages)

        # 3. use llm to choose action
        print(Fore.RED + "Response: " + str(response.choices[0].message) + Fore.RESET)

        # 4. perform action
        if response.choices[0].message.function_call:
            name = response.choices[0].message.function_call.name
            args = json.loads(response.choices[0].message.function_call.arguments)
            print(f"Agent {self.agent_id} is performing action: {name} with args: {args}")
            await getattr(self, name)(**args)

    async def _perform_action(self, message, action_type):
        """根据传入的action_type和message执行action, 并得到返回值"""
        message_id = await self.channel.write_to_receive_queue(
            (self.agent_id, message, action_type))

        response = await self.channel.read_from_send_queue(message_id)
        # 发送一个动作到Twitter实例并等待响应
        print(f"{action_type}\tReceived response: {response[2]}")
        return response[2]

    async def action_sign_up(self, user_name: str, name: str, bio: str):
        r"""Signs up a new user with the provided username, name, and bio.

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

        print(f"Agent {self.agent_id} is signing up with user_name: {user_name}, name: {name}, bio: {bio}")

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


async def running():

    test_db_filepath = "./test.db"

    channel = Twitter_Channel()
    infra = Twitter(test_db_filepath, channel)
    task = asyncio.create_task(infra.running())

    agent = TwitterUserAgent(1, "Alice", channel)
    await agent.action_sign_up("Alice", "Alice", "Alice is a girl.")
    await agent.action_create_tweet("I have a dream. Can you share your dream with me?")

    agent2 = TwitterUserAgent(2, "Bob", channel, "You love singing and you want to be a singer. You like creating "
                                                 "tweets about music.")
    await agent2.action_sign_up("Bob", "Bob", "BoB")
    await agent2.perform_action_by_llm()
    time.sleep(10)
    await agent2.perform_action_by_llm()
    time.sleep(10)
    await agent2.perform_action_by_llm()
    time.sleep(10)
    await channel.write_to_receive_queue((None, None, "exit"))
    await task


if __name__ == "__main__":
    asyncio.run(running())
