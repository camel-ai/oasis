from __future__ import annotations

import inspect

from camel.agents import ChatAgent
from camel.configs import ChatGPTConfig
from camel.messages import BaseMessage
from camel.models import OpenAIModel
from camel.types import ModelType

from social_simulation.social_agent.agent_action import SocialAction
from social_simulation.social_agent.agent_environment import SocialEnvironment
from social_simulation.social_platform.channel import Channel
from social_simulation.social_platform.config import UserInfo


class SocialAgent:

    def __init__(
        self,
        agent_id: int,
        user_info: UserInfo,
        channel: Channel,
        model_type: ModelType = ModelType.GPT_3_5_TURBO,
    ):
        self.agent_id = agent_id
        self.user_info = user_info
        self.channel = channel
        self.env = SocialEnvironment(SocialAction(agent_id, channel))
        model_config = ChatGPTConfig(
            tools=self.env.action.get_openai_function_list(),
            temperature=0.0,
        )
        model = OpenAIModel(model_config_dict=model_config.__dict__,
                            model_type=model_type)
        self.system_message = BaseMessage.make_assistant_message(
            role_name="User",
            content=self.user_info.to_system_message(),
        )
        self.chat_agent = ChatAgent(
            system_message=self.system_message,
            model=model,
            tools=self.env.action.get_openai_function_list(),
        )

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
        await self.chat_agent.step_async(user_msg)
        record = self.chat_agent.memory.retrieve()[-1].memory_record
        print(record.message.result)

    async def perform_action_by_hci(self):
        print('Please choose one function to perform:')
        function_list = self.env.action.get_openai_function_list()
        for i in range(len(function_list)):
            print(f"{i}.", function_list[i].func.__name__, end=', ')
        print()

        selection = int(input("Enter your choice: "))
        if not 0 <= selection < len(function_list):
            print("Invalid input. Please enter a number.")
            return

        func = function_list[selection].func

        params = inspect.signature(func).parameters
        args = []
        for param in params.values():
            while True:
                try:
                    value = input(f"Enter value for {param.name}: ")
                    args.append(value)
                    break
                except ValueError:
                    print("Invalid input, please enter an integer.")

        # 调用函数并传入用户输入的参数
        result = await func(*args)
        return result

    async def perform_action_by_data(self, func_name, *args, **kwargs):
        function_list = self.env.action.get_openai_function_list()
        for i in range(len(function_list)):
            if function_list[i].func.__name__ == func_name:
                func = function_list[i].func
                result = await func(*args, **kwargs)
                print(result)
                return result
        raise ValueError(f"Function {func_name} not found in the list.")
