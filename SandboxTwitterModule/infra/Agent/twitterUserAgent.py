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
            'other_info': {'mbti': None, 'activity_level': 'Medium'},
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
        print(f"Received response: {response}")
        return response[2]

    async def action_sign_up(self, user_name, name, bio):
        '''
        加上给camel生成openai schema的信息
        输出example:
        {'success': True, 'user_id': 2}
        '''
        user_message = (user_name, name, bio)
        return await self._perform_action(
            user_message, ActionType.SIGNUP.value)

    async def action_create_tweet(self, content):
        '''加上给camel生成openai schema的信息
        输出example:
        {'success': True, 'tweet_id': 50}
        '''
        return await self._perform_action(
            content, ActionType.CREATE_TWEET.value)
