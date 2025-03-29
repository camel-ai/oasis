# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
from __future__ import annotations

import inspect
import logging
import sys
from datetime import datetime
from typing import TYPE_CHECKING, Any

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

from oasis.social_agent.agent_action import SocialAction
from oasis.social_agent.agent_environment import SocialEnvironment
from oasis.social_platform import Channel
from oasis.social_platform.config import UserInfo

if TYPE_CHECKING:
    from oasis.social_agent import AgentGraph

if "sphinx" not in sys.modules:
    agent_log = logging.getLogger(name="social.agent")
    agent_log.setLevel("DEBUG")

    if not agent_log.handlers:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_handler = logging.FileHandler(
            f"./log/social.agent-{str(now)}.log")
        file_handler.setLevel("DEBUG")
        file_handler.setFormatter(
            logging.Formatter(
                "%(levelname)s - %(asctime)s - %(name)s - %(message)s"))
        agent_log.addHandler(file_handler)


class SocialAgent(ChatAgent):
    r"""Social Agent."""

    def __init__(
        self,
        agent_id: int,
        user_info: UserInfo,
        twitter_channel: Channel,
        inference_channel: Channel = None,  # TODO: will deprecate
        model_type: str = "gpt-4o-mini",
        agent_graph: "AgentGraph" = None,
        action_space_prompt: str = None,  # TODO: will deprecate
        is_openai_model: bool = False,
    ):
        self.social_agent_id = agent_id
        self.user_info = user_info
        self.twitter_channel = twitter_channel
        self.infe_channel = inference_channel
        self.env = SocialEnvironment(SocialAction(agent_id, twitter_channel))
        self.model_type = model_type
        self.is_openai_model = is_openai_model
        if self.is_openai_model:
            self.model_backend = ModelFactory.create(
                model_platform=ModelPlatformType.OPENAI,
                model_type=ModelType(model_type),
            )
            self.model_backend.model_config_dict['temperature'] = 0.6
        else:
            self.model_backend = ModelFactory.create(
                model_platform=ModelPlatformType.VLLM,
                model_type=ModelType(model_type),
                url="http://localhost:8000/v1",  # TODO: change to server url
                model_config_dict={"temperature":
                                   0.5},  # TODO: Need to be customized
            )
        # context_creator = ScoreBasedContextCreator(
        #     OpenAITokenCounter(ModelType.GPT_3_5_TURBO),
        #     4096,
        # )
        # self.memory = ChatHistoryMemory(context_creator, window_size=5)
        system_message = BaseMessage.make_assistant_message(
            role_name="system",
            content=self.user_info.to_system_message(),  # system prompt
        )

        super().__init__(system_message=system_message,
                         model=self.model_backend,
                         tools=self.env.action.get_openai_function_list())
        self.agent_graph = agent_graph
        self.test_prompt = (
            "\n"
            "Helen is a successful writer who usually writes popular western "
            "novels. Now, she has an idea for a new novel that could really "
            "make a big impact. If it works out, it could greatly "
            "improve her career. But if it fails, she will have spent "
            "a lot of time and effort for nothing.\n"
            "\n"
            "What do you think Helen should do?")

    async def perform_action_by_llm(self):
        # Get posts:
        env_prompt = await self.env.to_text_prompt()
        user_msg = BaseMessage.make_user_message(
            role_name="User",
            # content=(
            #     f"Please perform social media actions after observing the "
            #     f"platform environments. Notice that don't limit your "
            #     f"actions for example to just like the posts. "
            #     f"Here is your social media environment: {env_prompt}"),
            content=(
                f"Please perform social media actions after observing the "
                f"platform environments. "
                f"Here is your social media environment: {env_prompt}"),
        )
        try:
            agent_log.info(
                f"Agent {self.social_agent_id} observing environment: "
                f"{env_prompt}")
            response = await self.astep(user_msg)
            for tool_call in response.info['tool_calls']:
                action_name = tool_call.tool_name
                args = tool_call.args
                agent_log.info(f"Agent {self.social_agent_id} performed "
                               f"action: {action_name} with args: {args}")
                self.perform_agent_graph_action(action_name, args)
        except Exception as e:
            agent_log.error(f"Agent {self.social_agent_id} error: {e}")

    async def perform_test(self):
        """
        doing test for all agents.
        TODO: rewrite the function according to the ChatAgent.
        """
        # user conduct test to agent
        _ = BaseMessage.make_user_message(role_name="User",
                                          content=("You are a twitter user."))
        # TODO error occurs
        # self.memory.write_record(MemoryRecord(user_msg,
        #                                       OpenAIBackendRole.USER))

        openai_messages, num_tokens = self.memory.get_context()

        openai_messages = ([{
            "role":
            self.system_message.role_name,
            "content":
            self.system_message.content.split("# RESPONSE FORMAT")[0],
        }] + openai_messages + [{
            "role": "user",
            "content": self.test_prompt
        }])
        agent_log.info(f"Agent {self.social_agent_id}: {openai_messages}")

        message_id = await self.infe_channel.write_to_receive_queue(
            openai_messages)
        message_id, content = await self.infe_channel.read_from_send_queue(
            message_id)
        agent_log.info(
            f"Agent {self.social_agent_id} receive response: {content}")
        return {
            "user_id": self.social_agent_id,
            "prompt": openai_messages,
            "content": content
        }

    async def perform_action_by_hci(self) -> Any:
        print("Please choose one function to perform:")
        function_list = self.env.action.get_openai_function_list()
        for i in range(len(function_list)):
            agent_log.info(f"Agent {self.social_agent_id} function: "
                           f"{function_list[i].func.__name__}")

        selection = int(input("Enter your choice: "))
        if not 0 <= selection < len(function_list):
            agent_log.error(f"Agent {self.social_agent_id} invalid input.")
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
                    agent_log.error("Invalid input, please enter an integer.")

        result = await func(*args)
        return result

    async def perform_action_by_data(self, func_name, *args, **kwargs) -> Any:
        function_list = self.env.action.get_openai_function_list()
        for i in range(len(function_list)):
            if function_list[i].func.__name__ == func_name:
                func = function_list[i].func
                result = await func(*args, **kwargs)
                agent_log.info(f"Agent {self.social_agent_id}: {result}")
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
            self.agent_graph.remove_edge(self.social_agent_id, followee_id)
            agent_log.info(
                f"Agent {self.social_agent_id} unfollowed Agent {followee_id}")
        elif "follow" in action_name:
            followee_id: int | None = arguments.get("followee_id", None)
            if followee_id is None:
                return
            self.agent_graph.add_edge(self.social_agent_id, followee_id)
            agent_log.info(
                f"Agent {self.social_agent_id} followed Agent {followee_id}")

    def __str__(self) -> str:
        return (f"{self.__class__.__name__}(agent_id={self.social_agent_id}, "
                f"model_type={self.model_type.value})")
