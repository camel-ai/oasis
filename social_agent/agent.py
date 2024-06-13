import inspect
import json

from camel.configs import FunctionCallingConfig
from camel.memories import ChatHistoryMemory, MemoryRecord
from camel.memories.context_creators.score_based import \
    ScoreBasedContextCreator
from camel.messages import BaseMessage
from camel.models import BaseModelBackend, ModelFactory
from camel.types import ModelType, OpenAIBackendRole

from social_agent.agent_action import TwitterAction
from social_agent.agent_environment import TwitterEnvironment
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
        self.env = TwitterEnvironment(TwitterAction(agent_id, channel))
        model_config = FunctionCallingConfig.from_openai_function_list(
            function_list=self.env.twitter_action.get_openai_function_list(),
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
        env_prompt = await self.env.to_text_prompt()
        user_msg = BaseMessage.make_user_message(
            role_name="User",
            content=(
                f"Please perform twitter actions after observing the twitter "
                f"environments. Notice that don't limit your actions for "
                f"example to just like the tweets. "
                f"Here is your twitter environment: {env_prompt}"),
        )

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
            await getattr(self.env.twitter_action, action_name)(**args)

    async def perform_action_by_hci(self):
        print('Please choose one function to perform:')
        function_list = self.env.twitter_action.get_openai_function_list()
        for i in range(len(function_list)):
            print(f"{i}.", function_list[i].func.__name__, end=', ')
        print()

        selection = int(input("Enter your choice: "))
        if not 0 <= selection < len(function_list):
            print("Invalid input. Please enter a number.")
            return

        func = function_list[selection].func

        # 使用inspect获取函数的参数列表
        params = inspect.signature(func).parameters
        args = []
        for param in params.values():
            while True:
                try:
                    value = input(f"Enter value for {param.name}: ")
                    # 假设所有参数都是整数，根据需要可以调整
                    args.append(value)
                    break  # 成功获取有效输入，跳出循环
                except ValueError:
                    print("Invalid input, please enter an integer.")

        # 调用函数并传入用户输入的参数
        result = await func(*args)
        # print(result)
        return result
