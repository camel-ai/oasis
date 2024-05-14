# File: SandboxTwitterModule/infra/Agent/twitterUserAgent.py
from SandboxTwitterModule.infra import ActionType


class TwitterUserAgent:
    def __init__(self, agent_id, real_name, channel):
        self.db = None
        self.api = None
        self.user_id = None
        self.agent_id = agent_id
        self.real_name = real_name
        self.profile = {
            'nodes': [],  # Relationships with other agents
            'edges': [],  # Relationship details
            'other_info': {'mbti': None, 'activity_level': 'Medium', "user_profile": None},
        }
        self.memory = {'system_one': [], 'system_two': []}
        self.home_content = []
        self.channel = channel

    async def _perform_action(self, message, action_type):
        """根据传入的action_type和message执行action, 并得到返回值"""
        message_id = await self.channel.write_to_receive_queue(
            (self.agent_id, message, action_type))

        response = await self.channel.read_from_send_queue(message_id)
        # 发送一个动作到Twitter实例并等待响应
        print(f"Received response: {response[2]}")
        return response[2]

    async def action_sign_up(self, user_name: str, name: str, bio: str):
        '''
        加上给camel生成openai schema的信息
        输出example:
        {'success': True, 'user_id': 2}
        '''
        user_message = (user_name, name, bio)
        return await self._perform_action(
            user_message, ActionType.SIGNUP.value)

    async def action_refresh(self):
        '''刷新得到推荐的tweet
        加上给camel生成openai schema的信息
        输出example:
        {"success": True, "tweets": tweets}
        其中：
        tweets = list[{
            "tweet_id": tweet_id,
            "user_id": user_id,
            "content": content,
            "created_at": created_at,
            "num_likes": num_likes
        }]
        '''
        return await self._perform_action(
            None, ActionType.REFRESH.value)

    async def action_create_tweet(self, content: str):
        '''加上给camel生成openai schema的信息
        输出example:
        {'success': True, 'tweet_id': 50}
        '''
        return await self._perform_action(
            content, ActionType.CREATE_TWEET.value)

    async def action_like(self, tweet_id: int):
        '''
        加上给camel生成openai schema的信息
        输出example:
        {"success": True, "like_id": like_id}
        '''
        return await self._perform_action(
            tweet_id, ActionType.LIKE.value)

    async def action_unlike(self, tweet_id: int):
        '''
        加上给camel生成openai schema的信息
        输出example:
        {"success": True, "like_id": like_id}
        这里的like_id是取关的like表的表项id
        '''
        return await self._perform_action(
            tweet_id, ActionType.UNLIKE.value)

    async def action_search_tweets(self, query: str):
        '''
        加上给camel生成openai schema的信息
        输出example:
        {"success": True, "tweets": tweets}
        其中：
        tweets = list[{
            "tweet_id": tweet_id,
            "user_id": user_id,
            "content": content,
            "created_at": created_at,
            "num_likes": num_likes
        }]
        '''
        return await self._perform_action(
            query, ActionType.SEARCH_TWEET.value)

    async def action_search_user(self, query: str):
        '''
        加上给camel生成openai schema的信息
        输出example:
        {"success": True, "users": users}
        其中：
        users = list[{
            "user_id": user_id,
            "user_name": user_name,
            "name": name,
            "bio": bio,
            "created_at": created_at,
            "num_followings": num_followings,
            "num_followers": num_followers
        }]
        '''
        return await self._perform_action(
            query, ActionType.SEARCH_USER.value)

    async def action_follow(self, followee_id: int):
        '''
        加上给camel生成openai schema的信息
        输出example:
        {"success": True, "follow_id": follow_id}
        这里的follow_id是follow表的表项id
        '''
        return await self._perform_action(
            followee_id, ActionType.FOLLOW.value)

    async def action_unfollow(self, followee_id: int):
        '''
        加上给camel生成openai schema的信息
        输出example:
        {"success": True, "follow_id": follow_id}
        这里的follow_id是follow表取关的表项id
        '''
        return await self._perform_action(
            followee_id, ActionType.UNFOLLOW.value)

    async def action_mute(self, mutee_id: int):
        '''
        加上给camel生成openai schema的信息
        输出example:
        {"success": True, "mute_id": mute_id}
        这里的mute_id是mute表的表项id
        '''
        return await self._perform_action(
            mutee_id, ActionType.MUTE.value)

    async def action_unmute(self, mutee_id: int):
        '''
        加上给camel生成openai schema的信息
        输出example:
        {"success": True, "mute_id": mute_id}
        这里的mute_id是mute表的删除的表项id
        '''
        return await self._perform_action(
            mutee_id, ActionType.UNMUTE.value)

    async def action_trend(self):
        '''看热搜推文
        加上给camel生成openai schema的信息
        输出example:
        {"success": True, "tweets": tweets}
        其中：
        tweets = list[{
            "tweet_id": tweet_id,
            "user_id": user_id,
            "content": content,
            "created_at": created_at,
            "num_likes": num_likes
        }]
        '''
        return await self._perform_action(
            None, ActionType.TREND.value)
