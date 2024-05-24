import json

from camel.configs import FunctionCallingConfig
from camel.memories import ChatHistoryMemory, MemoryRecord
from camel.memories.context_creators.score_based import \
    ScoreBasedContextCreator
from camel.messages import BaseMessage
from camel.models import BaseModelBackend, ModelFactory
from camel.types import ModelType, OpenAIBackendRole

from social_agent.agent_action import TwitterAction
from twitter.channel import Twitter_Channel
from twitter.config import UserInfo


class TwitterUserAgent:

    def __init__(
        self,
        agent_id: int,
        user_info: UserInfo,
        channel: Twitter_Channel,
        model_type: ModelType = ModelType.GPT_3_5_TURBO,
    ):
        self.agent_id = agent_id
        self.user_info = user_info
        self.channel = channel
        self.twitter_action = TwitterAction(agent_id, channel)
        model_config = FunctionCallingConfig.from_openai_function_list(
            function_list=self.twitter_action.get_openai_function_list(),
            kwargs=dict(temperature=0.0),
        )
        self.model_backend: BaseModelBackend = ModelFactory.create(
            model_type, model_config.__dict__)
        self.model_token_limit = self.model_backend.token_limit
        context_creator = ScoreBasedContextCreator(
            self.model_backend.token_counter,
            self.model_token_limit,
        )
        self.memory = ChatHistoryMemory(context_creator, window_size=3)
        self.system_message = BaseMessage.make_assistant_message(
            role_name="User",
            content=self.user_info.to_system_message(),
        )
        system_record = MemoryRecord(self.system_message,
                                     OpenAIBackendRole.SYSTEM)
        self.memory.write_record(system_record)
        self.home_content = []

    async def perform_action_by_llm(self):
        # Get 5 random tweets:
        tweets = await self.twitter_action.action_refresh()
        # Get context form memory:
        user_msg = BaseMessage.make_user_message(
            role_name="User",
            content=(f"Choose to perform any twitter action based "
                     f"on existing tweets: {tweets['tweets']}"))

        self.memory.write_record(MemoryRecord(user_msg,
                                              OpenAIBackendRole.USER))

        openai_messages, num_tokens = self.memory.get_context()
        response = self.model_backend.run(openai_messages)

        # Perform twitter action:
        if response.choices[0].message.function_call:
            action_name = response.choices[0].message.function_call.name
            args = json.loads(
                response.choices[0].message.function_call.arguments)
            print(f"Agent {self.agent_id} is performing "
                  f"twitter action: {action_name} with args: {args}")
            await getattr(self.twitter_action, action_name)(**args)
