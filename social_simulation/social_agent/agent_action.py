from typing import Any

from camel.functions import OpenAIFunction

from social_simulation.social_platform.channel import Channel
from social_simulation.social_platform.typing import ActionType


class SocialAction:

    def __init__(self, agent_id: int, channel: Channel):
        self.agent_id = agent_id
        self.channel = channel

    def get_openai_function_list(self) -> list[OpenAIFunction]:
        return [
            OpenAIFunction(func) for func in [
                self.create_tweet, self.follow, self.unfollow, self.like,
                self.unlike, self.dislike, self.undo_dislike,
                self.search_tweets, self.search_user, self.trend, self.refresh,
                self.mute, self.unmute, self.retweet, self.create_comment,
                self.like_comment, self.unlike_comment, self.dislike_comment,
                self.undo_dislike_comment, self.do_nothing
            ]
        ]

    async def perform_action(self, message: Any, action_type: str):
        message_id = await self.channel.write_to_receive_queue(
            (self.agent_id, message, action_type))
        response = await self.channel.read_from_send_queue(message_id)
        return response[2]

    async def sign_up(self, user_name: str, name: str, bio: str):
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

        print(f"Agent {self.agent_id} is signing up with "
              f"user_name: {user_name}, name: {name}, bio: {bio}")
        user_message = (user_name, name, bio)
        return await self.perform_action(user_message, ActionType.SIGNUP.value)

    async def refresh(self):
        r"""Refreshes to get recommended tweets.

        This method invokes an asynchronous action to refresh and fetch
        recommended tweets. On successful execution, it returns a dictionary
        indicating success along with a list of tweets. Each tweet in the list
        contains details such as tweet ID, user ID, content, creation date,
        and the number of likes.

        Returns:
            dict: A dictionary with two key-value pairs. The 'success' key
                maps to a boolean indicating whether the refresh is
                successful. The 'tweets' key maps to a list of dictionaries,
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
        return await self.perform_action(None, ActionType.REFRESH.value)

    async def create_tweet(self, content: str):
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
        return await self.perform_action(content,
                                         ActionType.CREATE_TWEET.value)

    async def retweet(self, tweet_id: int):
        r"""Retweet a specified tweet.

        This method invokes an asynchronous action to Retweet a specified
        tweet. It is identified by the given tweet ID. Upon successful
        execution, it returns a dictionary indicating success and the ID of
        the newly created retweet.

        Args:
            tweet_id (int): The ID of the tweet to be retweet.

        Returns:
            dict: A dictionary with two key-value pairs. The 'success' key
                maps to a boolean indicating whether the Retweet creation was
                successful. The 'tweet_id' key maps to the integer ID of the
                newly created retweet.

            Example of a successful return:
            {"success": True, "tweet_id": 123}

        Note:
            Attempting to retweet a tweet that the user has already retweet
            will result in a failure.
        """
        return await self.perform_action(tweet_id, ActionType.RETWEET.value)

    async def like(self, tweet_id: int):
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
        return await self.perform_action(tweet_id, ActionType.LIKE.value)

    async def unlike(self, tweet_id: int):
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
        return await self.perform_action(tweet_id, ActionType.UNLIKE.value)

    async def dislike(self, tweet_id: int):
        r"""Creates a new dislike for a specified tweet.

        This method invokes an asynchronous action to create a new dislike for
        a tweet. It is identified by the given tweet ID. Upon successful
        execution, it returns a dictionary indicating success and the ID of
        the newly created dislike.

        Args:
            tweet_id (int): The ID of the tweet to be disliked.

        Returns:
            dict: A dictionary with two key-value pairs. The 'success' key
                maps to a boolean indicating whether the dislike creation was
                successful. The 'dislike_id' key maps to the integer ID of the
                newly created like.

            Example of a successful return:
            {"success": True, "dislike_id": 123}

        Note:
            Attempting to dislike a tweet that the user has already liked will
            result in a failure.
        """
        return await self.perform_action(tweet_id, ActionType.DISLIKE.value)

    async def undo_dislike(self, tweet_id: int):
        """Removes a dislike based on the tweet's ID.

        This method removes a dislike from the database, identified by the
        tweet's ID. It returns a dictionary indicating the success of the
        operation and the ID of the removed dislike.

        Args:
            tweet_id (int): The ID of the tweet to be unliked.

        Returns:
            dict: A dictionary with 'success' indicating if the removal was
                successful, and 'dislike_id' the ID of the removed like.

            Example of a successful return:
            {"success": True, "dislike_id": 123}

        Note:
            Attempting to remove a dislike for a tweet that the user has not
            previously liked will result in a failure.
        """
        return await self.perform_action(tweet_id,
                                         ActionType.UNDO_DISLIKE.value)

    async def search_tweets(self, query: str):
        r"""searches tweets based on a given query.

        This method performs a search operation in the database for tweets
        that match the given query string. The search considers the
        tweet's content, tweet ID, and user ID. It returns a dictionary
        indicating the operation's success and, if successful, a list of
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
        return await self.perform_action(query, ActionType.SEARCH_TWEET.value)

    async def search_user(self, query: str):
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
        return await self.perform_action(query, ActionType.SEARCH_USER.value)

    async def follow(self, followee_id: int):
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
        return await self.perform_action(followee_id, ActionType.FOLLOW.value)

    async def unfollow(self, followee_id: int):
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
        return await self.perform_action(followee_id,
                                         ActionType.UNFOLLOW.value)

    async def mute(self, mutee_id: int):
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
        return await self.perform_action(mutee_id, ActionType.MUTE.value)

    async def unmute(self, mutee_id: int):
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
        return await self.perform_action(mutee_id, ActionType.UNMUTE.value)

    async def trend(self):
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
        return await self.perform_action(None, ActionType.TREND.value)

    async def create_comment(self, tweet_id: int, content: str):
        r"""Creates a new comment for a specified tweet with the given content.

        This method creates a new comment based on the provided content and
        associates it with the given tweet ID. Upon successful execution, it
        returns a dictionary indicating success and the ID of the newly created
        comment.

        Args:
            tweet_id (int): The ID of the tweet to which the comment is to be
                added.
            content (str): The content of the comment to be created.

        Returns:
            dict: A dictionary with two key-value pairs. The 'success' key
                maps to a boolean indicating whether the comment creation was
                successful. The 'comment_id' key maps to the integer ID of the
                newly created comment.

            Example of a successful return:
                {'success': True, 'comment_id': 123}
        """
        comment_message = (tweet_id, content)
        return await self.perform_action(comment_message,
                                         ActionType.CREATE_COMMENT.value)

    async def like_comment(self, comment_id: int):
        r"""Creates a new like for a specified comment.

        This method invokes an action to create a new like for a comment,
        identified by the given comment ID. Upon successful execution, it
        returns a dictionary indicating success and the ID of the newly
        created like.

        Args:
            comment_id (int): The ID of the comment to be liked.

        Returns:
            dict: A dictionary with two key-value pairs. The 'success' key
                maps to a boolean indicating whether the like creation was
                successful. The 'like_id' key maps to the integer ID of the
                newly created like.

            Example of a successful return:
            {"success": True, "comment_like_id": 456}

        Note:
            Attempting to like a comment that the user has already liked will
            result in a failure.
        """
        return await self.perform_action(comment_id,
                                         ActionType.LIKE_COMMENT.value)

    async def unlike_comment(self, comment_id: int):
        """Removes a like based on the comment's ID.

        This method removes a like from the database, identified by the
        comment's ID. It returns a dictionary indicating the success of the
        operation and the ID of the removed like.

        Args:
            comment_id (int): The ID of the comment to be unliked.

        Returns:
            dict: A dictionary with 'success' indicating if the removal was
                successful, and 'like_id' the ID of the removed like.

            Example of a successful return:
            {"success": True, "like_id": 456}

        Note:
            Attempting to remove a like for a comment that the user has not
            previously liked will result in a failure.
        """
        return await self.perform_action(comment_id,
                                         ActionType.UNLIKE_COMMENT.value)

    async def dislike_comment(self, comment_id: int):
        r"""Creates a new dislike for a specified comment.

        This method invokes an action to create a new dislike for a
        comment, identified by the given comment ID. Upon successful execution,
        it returns a dictionary indicating success and the ID of the newly
        created dislike.

        Args:
            comment_id (int): The ID of the comment to be disliked.

        Returns:
            dict: A dictionary with two key-value pairs. The 'success' key
                maps to a boolean indicating whether the dislike creation was
                successful. The 'dislike_id' key maps to the integer ID of the
                newly created dislike.

            Example of a successful return:
            {"success": True, "comment_dislike_id": 456}

        Note:
            Attempting to dislike a comment that the user has already liked
            will result in a failure.
        """
        return await self.perform_action(comment_id,
                                         ActionType.DISLIKE_COMMENT.value)

    async def undo_dislike_comment(self, comment_id: int):
        """Removes a dislike based on the comment's ID.

        This method removes a dislike from the database, identified by the
        comment's ID. It returns a dictionary indicating the success of the
        operation and the ID of the removed dislike.

        Args:
            comment_id (int): The ID of the comment to have its dislike
                removed.

        Returns:
            dict: A dictionary with 'success' indicating if the removal was
                successful, and 'dislike_id' the ID of the removed dislike.

            Example of a successful return:
            {"success": True, "dislike_id": 456}

        Note:
            Attempting to remove a dislike for a comment that the user has not
            previously disliked will result in a failure.
        """
        return await self.perform_action(comment_id,
                                         ActionType.UNDO_DISLIKE_COMMENT.value)

    async def do_nothing(self):
        """Performs no action and returns nothing.

        Returns:
            dict: A dictionary with 'success' indicating if the removal was
                successful.

            Example of a successful return:
                {"success": True}
        """
        return await self.perform_action(None, ActionType.DO_NOTHING.value)
