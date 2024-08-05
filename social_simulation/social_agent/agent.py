from __future__ import annotations

from datetime import datetime
from genericpath import isfile
import inspect
import json
import os
from typing import TYPE_CHECKING, Any
import logging
from camel.memories import (ChatHistoryMemory, MemoryRecord,
                            ScoreBasedContextCreator)
from camel.messages import BaseMessage
from camel.utils import OpenAITokenCounter
from camel.types import ModelType, OpenAIBackendRole
from colorama import Fore, Style

from social_simulation.social_agent.agent_action import SocialAction
from social_simulation.social_agent.agent_environment import SocialEnvironment
from social_simulation.social_platform import Channel
from social_simulation.social_platform.config import UserInfo

if TYPE_CHECKING:
    from social_simulation.social_agent import AgentGraph

agent_log = logging.getLogger(name='social.agent')
agent_log.setLevel('DEBUG')
now = datetime.now()
file_handler = logging.FileHandler(f'./log/social.agent-{str(now)}.log')
file_handler.setLevel('DEBUG')
file_handler.setFormatter(logging.Formatter('%(levelname)s - %(asctime)s - %(name)s - %(message)s'))
agent_log.addHandler(file_handler)


class SocialAgent:

    def __init__(
        self,
        agent_id: int,
        user_info: UserInfo,
        twitter_channel: Channel,
        inference_channel: Channel = None,
        model_type: ModelType = ModelType.LLAMA_3,
        agent_graph: "AgentGraph" = None,
    ):
        self.agent_id = agent_id
        self.user_info = user_info
        self.twitter_channel = twitter_channel
        self.inference_channel = inference_channel
        self.env = SocialEnvironment(SocialAction(agent_id, twitter_channel))
        self.system_message = BaseMessage.make_assistant_message(
            role_name="User",
            content=self.user_info.to_system_message(),
        )
        self.model_type = model_type
        self.has_function_call = False
        context_creator = ScoreBasedContextCreator(
            OpenAITokenCounter(ModelType.GPT_3_5_TURBO),
            4096,
        )
        self.memory = ChatHistoryMemory(context_creator, window_size=5)
        self.system_message = BaseMessage.make_assistant_message(
            role_name="system",
            content=self.user_info.to_system_message()  # system prompt
        )
        self.agent_graph = agent_graph

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
        if not openai_messages:
            openai_messages = [{
                        "role": self.system_message.role_name,
                        "content": self.system_message.content
                    }] + user_msg.to_openai_message()
        agent_log.info(f"Agent {self.agent_id} is running with prompt: {openai_messages}")

        if self.has_function_call:
            response = self.model_backend.run(openai_messages)
            agent_log.info(f"Agent {self.agent_id} response: {response}")
            if response.choices[0].message.function_call:
                action_name = response.choices[0].message.function_call.name
                args = json.loads(
                    response.choices[0].message.function_call.arguments)
                print(f"Agent {self.agent_id} is performing "
                      f"twitter action: {action_name} with args: {args}")
                await getattr(self.env.action, action_name)(**args)
                self.perform_agent_graph_action(action_name, args)

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

                message_id = await self.inference_channel.write_to_receive_queue(
                    openai_messages)
                message_id, content = await self.inference_channel.read_from_send_queue(message_id)
                agent_log.info(f"Agent {self.agent_id} receve response: {content}")

                try:
                    content_json = json.loads(content)
                    functions = content_json['functions']
                    reason = content_json['reason']
                    # print(f"Agent {self.agent_id} choose "
                    #       f"{functions} \nbecause: {reason}.")
                    for function in functions:
                        name = function['name']
                        arguments = function['arguments']
                        # print(f"Agent {self.agent_id} is performing "
                        #       f"twitter action: {name} with args: {arguments}")
                        exec_functions.append({
                            'name': name,
                            'arguments': arguments
                        })
                        self.perform_agent_graph_action(name, arguments)
                    break
                except Exception as e:
                    # print(Fore.LIGHTRED_EX + f"Agent {self.agent_id}, time " +
                    #       Style.BRIGHT + str(retry) + Style.RESET_ALL +
                    #       f"\nError: {e} when parsing response:{content}\n" +
                    #       Fore.RESET + "=" * 20 + "\n")
                    # print(Fore.LIGHTBLUE_EX + "For DEBUG, Messages:",
                    #       openai_messages, "\n" + Fore.RESET + "=" * 20 + "\n")
                    agent_log.error(f"Agent {self.agent_id} error: {e}")
                    exec_functions = []
                    retry -= 1
            for function in exec_functions:
                try:
                    await getattr(self.env.action,
                                  function['name'])(**function['arguments'])
                except Exception as e:
                    # print(Fore.LIGHTRED_EX + f"Agent {self.agent_id}, time " +
                    #       Style.BRIGHT + str(retry) + Style.RESET_ALL +
                    #       f"\nError: {e} when performing twitter action:" +
                    #       f" {function['name']} with " +
                    #       f"args: {function['arguments']}\n" + Fore.RESET +
                    #       "=" * 20 + "\n")
                    agent_log.error(f"Agent {self.agent_id} error: {e}")

        if retry == 0:
            content = "No response." 
        agent_msg = BaseMessage.make_assistant_message(role_name="Assistant",
                                                       content=content)
        self.memory.write_record(
            MemoryRecord(agent_msg, OpenAIBackendRole.ASSISTANT))

    async def perform_action_by_hci(self) -> Any:
        print('Please choose one function to perform:')
        function_list = self.env.action.get_openai_function_list()
        for i in range(len(function_list)):
            # print(f"{i}.", function_list[i].func.__name__, end=', ')
            agent_log.info(f"Agent {self.agent_id} function: {function_list[i].func.__name__}")
        # print()

        selection = int(input("Enter your choice: "))
        if not 0 <= selection < len(function_list):
            # print("Invalid input. Please enter a number.")
            agent_log.error(f"Agent {self.agent_id} invalid input.")
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
                    # print("Invalid input, please enter an integer.")
                    agent_log.error("Invalid input, please enter an integer.")

        result = await func(*args)
        return result

    async def perform_action_by_data(self, func_name, *args, **kwargs) -> Any:
        function_list = self.env.action.get_openai_function_list()
        for i in range(len(function_list)):
            if function_list[i].func.__name__ == func_name:
                func = function_list[i].func
                result = await func(*args, **kwargs)
                # print(result)
                agent_log.info(f"Agent {self.agent_id}: {result}")
                return result
        raise ValueError(f"Function {func_name} not found in the list.")

    def perform_agent_graph_action(
        self,
        action_name: str,
        arguments: dict[str, Any],
    ):
        r"""Remove edge if action is unfollow or add edge
        if action is follow to the agent graph.
        """
        if "unfollow" in action_name:
            followee_id: int | None = arguments.get("followee_id", None)
            if followee_id is None:
                return
            self.agent_graph.remove_edge(self.agent_id, followee_id)
            agent_log.info(f"Agent {self.agent_id} unfollowed {followee_id}")
        elif "follow" in action_name:
            followee_id: int | None = arguments.get("followee_id", None)
            if followee_id is None:
                return
            self.agent_graph.add_edge(self.agent_id, followee_id)
            agent_log.info(f"Agent {self.agent_id} followed {followee_id}")

    def __str__(self) -> str:
        return (f"{self.__class__.__name__}(agent_id={self.agent_id}, "
                f"model_type={self.model_type.value})")
