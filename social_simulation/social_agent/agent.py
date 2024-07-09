from __future__ import annotations

import inspect
import json

from camel.configs import ChatGPTConfig, OpenSourceConfig
from camel.memories import (ChatHistoryMemory, MemoryRecord,
                            ScoreBasedContextCreator)
from camel.messages import BaseMessage
from camel.models import BaseModelBackend, ModelFactory
from camel.types import ModelType, OpenAIBackendRole
from colorama import Fore, Style

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
        model_path:
        str = "/mnt/hwfile/trustai/models/Meta-Llama-3-8B-Instruct",  # noqa
        server_url: str = "http://10.140.0.144:8000/v1",
        stop_tokens: list[str] = None,
        model_type: ModelType = ModelType.LLAMA_3,
        temperature: float = 0.0,
    ):
        self.agent_id = agent_id
        self.user_info = user_info
        self.channel = channel
        self.env = SocialEnvironment(SocialAction(agent_id, channel))
        self.system_message = BaseMessage.make_assistant_message(
            role_name="User",
            content=self.user_info.to_system_message(),
        )

        if model_type.is_open_source:
            self.has_function_call = False
            api_params = ChatGPTConfig(
                temperature=temperature,
                stop=stop_tokens,
            )
            model_config = OpenSourceConfig(
                model_path=model_path,
                server_url=server_url,
                api_params=api_params,
            )
        else:
            self.has_function_call = True
            model_config = ChatGPTConfig(
                temperature=temperature,
                # tools=self.env.action.get_openai_function_list(),
            )
        self.model_backend: BaseModelBackend = ModelFactory.create(
            model_type, model_config.__dict__)
        self.model_token_limit = self.model_backend.token_limit
        context_creator = ScoreBasedContextCreator(
            self.model_backend.token_counter,
            self.model_token_limit,
        )
        self.memory = ChatHistoryMemory(context_creator, window_size=5)
        self.system_message = BaseMessage.make_assistant_message(
            role_name="system",
            content=self.user_info.to_system_message()  # system prompt
        )
        self.home_content = []
        print(Fore.RED + f"{agent_id}: model type {model_type}" + Fore.RESET)

    async def perform_action_by_llm(self):
        # Get 5 random tweets:
        env_prompt = await self.env.to_text_prompt()
        user_msg = BaseMessage.make_user_message(
            role_name="User",
            content=(
                f"Please perform social media actions after observing the "
                f"platform environments. Notice that don't limit your actions "
                f"for example to just like the posts. "
                f"Here is your social media environment: {env_prompt}"),
        )
        self.memory.write_record(
            MemoryRecord(
                user_msg,
                OpenAIBackendRole.USER,
            ))

        openai_messages, num_tokens = self.memory.get_context()
        content = ""

        if self.has_function_call:
            response = self.model_backend.run(openai_messages)
            if response.choices[0].message.function_call:
                action_name = response.choices[0].message.function_call.name
                args = json.loads(
                    response.choices[0].message.function_call.arguments)
                print(f"Agent {self.agent_id} is performing "
                      f"twitter action: {action_name} with args: {args}")
                await getattr(self.env.action, action_name)(**args)

        else:
            retry = 5
            exec_functions = []

            while retry > 0:
                start_message = openai_messages[0]
                if start_message["role"] != self.system_message.role_name:
                    openai_messages = [{
                        "role": self.system_message.role_name,
                        "content": self.system_message.content
                    }] + openai_messages

                response = self.model_backend.run(openai_messages)
                content = response.choices[0].message.content
                try:
                    content_json = json.loads(content)
                    functions = content_json['functions']
                    reason = content_json['reason']
                    print(f"Agent {self.agent_id} choose "
                          f"{functions} \nbecause: {reason}.")
                    for function in functions:
                        name = function['name']
                        arguments = function['arguments']
                        print(f"Agent {self.agent_id} is performing "
                              f"twitter action: {name} with args: {arguments}")
                        exec_functions.append({
                            'name': name,
                            'arguments': arguments
                        })
                    break
                except Exception as e:
                    print(Fore.LIGHTRED_EX + f"Agent {self.agent_id}, time " +
                          Style.BRIGHT + str(retry) + Style.RESET_ALL +
                          f"\nError: {e} when parsing response:{content}\n" +
                          Fore.RESET + "=" * 20 + "\n")
                    print(Fore.LIGHTBLUE_EX + "For DEBUG, Messages:",
                          openai_messages, "\n" + Fore.RESET + "=" * 20 + "\n")
                    exec_functions = []
                    retry -= 1
            for function in exec_functions:
                try:
                    await getattr(self.env.action,
                                  function['name'])(**function['arguments'])
                except Exception as e:
                    print(Fore.LIGHTRED_EX + f"Agent {self.agent_id}, time " +
                          Style.BRIGHT + str(retry) + Style.RESET_ALL +
                          f"\nError: {e} when performing twitter action:" +
                          f" {function['name']} with " +
                          f"args: {function['arguments']}\n" + Fore.RESET +
                          "=" * 20 + "\n")

        agent_msg = BaseMessage.make_assistant_message(role_name="Assistant",
                                                       content=content)
        self.memory.write_record(
            MemoryRecord(agent_msg, OpenAIBackendRole.ASSISTANT))

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
