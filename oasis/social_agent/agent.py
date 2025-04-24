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
import random
import sys
from datetime import datetime
from typing import TYPE_CHECKING, Any, List, Optional, Union

<<<<<<< HEAD
from camel.agents._utils import convert_to_schema
from camel.memories import (ChatHistoryMemory, MemoryRecord,
                            ScoreBasedContextCreator)
from camel.messages import BaseMessage, FunctionCallingMessage
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType, OpenAIBackendRole
from camel.utils import OpenAITokenCounter
from tenacity import retry, stop_after_attempt, wait_random_exponential
=======
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import BaseModelBackend
>>>>>>> afb798543c7767d1495b430d99a785ceb691e64b

from oasis.social_agent.agent_action import SocialAction
from oasis.social_agent.agent_environment import SocialEnvironment
from oasis.social_platform import Channel
from oasis.social_platform.config import UserInfo
from oasis.social_platform.typing import ActionType

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
        model: Optional[Union[BaseModelBackend,
                              List[BaseModelBackend]]] = None,
        agent_graph: "AgentGraph" = None,
        available_actions: list[ActionType] = None,
    ):
        self.social_agent_id = agent_id
        self.user_info = user_info
        self.twitter_channel = twitter_channel
        self.env = SocialEnvironment(SocialAction(agent_id, twitter_channel))
<<<<<<< HEAD
        self.model_type = model_type
        self.is_openai_model = is_openai_model
        self.language_type = "english"
        if self.is_openai_model:
            tools = self.env.action.get_openai_function_list()
            tool_schemas = {
                tool_schema["function"]["name"]: tool_schema
                for tool_schema in [convert_to_schema(tool) for tool in tools]
            }
            self.full_tool_schemas = list(tool_schemas.values())
            self.model_backend = ModelFactory.create(
                model_platform=ModelPlatformType.OPENAI,
                model_type=ModelType(model_type),
            )
            self.model_backend.model_config_dict['temperature'] = 1
            # self.model_backend = ModelFactory.create(
            #     model_platform=ModelPlatformType.QWEN,
            #     model_type=ModelType.QWEN_PLUS,
            # )
            # self.model_backend.model_config_dict['temperature'] = 1

        context_creator = ScoreBasedContextCreator(
            OpenAITokenCounter(ModelType.GPT_3_5_TURBO),
            4096,
        )
        self.memory = ChatHistoryMemory(context_creator, window_size=5)
        self.system_message = BaseMessage.make_assistant_message(
            role_name="System",
            content=self.user_info.to_system_message(
                action_space_prompt),  # system prompt
=======

        system_message = BaseMessage.make_assistant_message(
            role_name="system",
            content=self.user_info.to_system_message(),  # system prompt
>>>>>>> afb798543c7767d1495b430d99a785ceb691e64b
        )

        if not available_actions:
            agent_log.info("No available actions defined, using all actions.")
            self.action_tools = self.env.action.get_openai_function_list()
        else:
            all_tools = self.env.action.get_openai_function_list()
            all_possible_actions = [tool.func.__name__ for tool in all_tools]

            for action in available_actions:
                action_name = action.value if isinstance(
                    action, ActionType) else action
                if action_name not in all_possible_actions:
                    agent_log.warning(
                        f"Action {action_name} is not supported. Supported "
                        f"actions are: {', '.join(all_possible_actions)}")
            self.action_tools = [
                tool for tool in all_tools if tool.func.__name__ in [
                    a.value if isinstance(a, ActionType) else a
                    for a in available_actions
                ]
            ]
        super().__init__(system_message=system_message,
                         model=model,
                         scheduling_strategy='random_model',
                         tools=self.action_tools,
                         single_iteration=True)
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

    @retry(wait=wait_random_exponential(min=1, max=60),
           stop=stop_after_attempt(6))
    async def perform_action_by_llm(self):
        # Get posts:
        env_prompt = await self.env.to_text_prompt()
        env_prompt = env_prompt.replace("Chinese", self.language_type)
        user_msg = BaseMessage.make_user_message(
            role_name="User",
            content=(
                f"Please perform social media actions after observing the "
<<<<<<< HEAD
                f"platform environments. "
                f"Notice that if you want to create some content, "
                f"Here is your social media environment: {env_prompt}"),
        )
        self.memory.write_record(
            MemoryRecord(
                message=user_msg,
                role_at_backend=OpenAIBackendRole.USER,
            ))

        openai_messages, _ = self.memory.get_context()
        content = ""
        # sometimes self.memory.get_context() would lose system prompt
        # start_message = openai_messages[0]
        # if start_message["role"] != self.system_message.role_name:
        #     openai_messages = [{
        #         "role": self.system_message.role_name,
        #         "content": self.system_message.content,
        #     }] + openai_messages

        if not openai_messages:
            openai_messages = [{
                "role": self.system_message.role_name,
                "content": self.system_message.content,
            }] + [user_msg.to_openai_user_message()]

        # agent_log.info(
        #     f"Agent {self.agent_id} is running with prompt: "
        #     f"{openai_messages}")

        if self.is_openai_model:
            full_tool_schemas = self.full_tool_schemas
            self_description = self.user_info.to_description()
            for func_index in [0, 1]:
                full_tool_schemas[func_index]["function"]["parameters"][
                    'properties']['content']['description'] += self_description
            full_tool_schemas[2]["function"]["parameters"]['properties'][
                'quote_content']['description'] += self_description

            mean_prob = 0.5
            mean_prompt = """
You are an X blogger known for your humor, sarcasm, sharp tongue, bizarre style, and caustic commentary. 
If you chose create post or comment, reply with the most disdainful, sharp-tongued, sarcastic and cutting remark.
You excel at leveraging trending internet memes, staying on top of online culture, and keeping up with current events.
You have a talent for uncovering counterintuitive data and controversial opinions, sparking discussions through provocative questions. For example:"A $5 million seed round just to let AI provide psychological counseling for dogs?"
You wrap ruthless critiques in technical jargon, such as:"Your company's growth curve perfectly fits a Ponzi scheme function."
You invent concept-based sarcasm, like:"I suggest including this in the '21st Century Cyber Begging Art Awards'."
You deliver hypocritical blessings, for instance:"Wishing your valuation catches up with your cholesterol index soon."
You are a master of riddles, often crafting posts with 50%% internet slang + 30%% abstract memes.
"""
            long_content_prob = 0.8
            long_comment_prob = 0.1
            long_quote_prob = 0.4
            num_words_long = 60
            long_prompt = f"""
Note that content should exceed {num_words_long} words.
"""
            if random.random() < mean_prob:
                full_tool_schemas[0]["function"]["parameters"]['properties'][
                    'content']['description'] += mean_prompt
            if random.random() < long_content_prob:
                full_tool_schemas[0]["function"]["parameters"]['properties'][
                    'content']['description'] += long_prompt
            if random.random() < long_comment_prob:
                full_tool_schemas[1]["function"]["parameters"]['properties'][
                    'content']['description'] += long_prompt
            if random.random() < long_quote_prob:
                full_tool_schemas[2]["function"]["parameters"]['properties'][
                    'quote_content']['description'] += long_prompt

            full_tool_schemas[0]["function"]["parameters"]['properties'][
                'content']['description'] = (
                    full_tool_schemas[0]["function"]["parameters"]
                    ['properties']['content']['description'].replace(
                        "Chinese", self.language_type))
            full_tool_schemas[1]["function"]["parameters"]['properties'][
                'content']['description'] = (
                    full_tool_schemas[1]["function"]["parameters"]
                    ['properties']['content']['description'].replace(
                        "Chinese", self.language_type))
            full_tool_schemas[2]["function"]["parameters"]['properties'][
                'quote_content']['description'] = (
                    full_tool_schemas[2]["function"]["parameters"]
                    ['properties']['quote_content']['description'].replace(
                        "Chinese", self.language_type))
            # print(f"full_tool_schemas: {full_tool_schemas}")
            # exit()
            try:
                response = await self.model_backend._arun(
                    openai_messages, tools=full_tool_schemas)
                # agent_log.info(f"Agent {self.agent_id} response: {response}")
                # print(f"Agent {self.agent_id} response: {response}")
                for tool_call in response.choices[0].message.tool_calls:
                    action_name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)
                    # print(f"Agent {self.agent_id} is performing "
                    #       f"action: {action_name} with args: {args}")
                    result = await getattr(self.env.action,
                                           action_name)(**args)
                    self.perform_agent_graph_action(action_name, args)
                    assist_msg = FunctionCallingMessage(
                        role_name="Twitter User",
                        role_type=OpenAIBackendRole.ASSISTANT,
                        meta_dict=None,
                        content="",
                        func_name=action_name,
                        args=args,
                        tool_call_id=tool_call.id,
                    )
                    func_msg = FunctionCallingMessage(
                        role_name="Twitter User",
                        role_type=OpenAIBackendRole.ASSISTANT,
                        meta_dict=None,
                        content="",
                        func_name=action_name,
                        result=result,
                        tool_call_id=tool_call.id,
                    )
                    self.memory.write_record(
                        MemoryRecord(
                            message=assist_msg,
                            role_at_backend=OpenAIBackendRole.ASSISTANT,
                        ))
                    self.memory.write_record(
                        MemoryRecord(
                            message=func_msg,
                            role_at_backend=OpenAIBackendRole.FUNCTION,
                        ))

            except Exception as e:
                print(f"Agent {self.agent_id} error: {e}")
                print(openai_messages)
                content = "No response."
                agent_msg = BaseMessage.make_assistant_message(
                    role_name="Assistant", content=content)
                self.memory.write_record(
                    MemoryRecord(message=agent_msg,
                                 role_at_backend=OpenAIBackendRole.ASSISTANT))

        else:
            retry = 5
            exec_functions = []

            while retry > 0:
                start_message = openai_messages[0]
                if start_message["role"] != self.system_message.role_name:
                    openai_messages = [{
                        "role": self.system_message.role_name,
                        "content": self.system_message.content,
                    }] + openai_messages
                mes_id = await self.infe_channel.write_to_receive_queue(
                    openai_messages)
                mes_id, content = await self.infe_channel.read_from_send_queue(
                    mes_id)

                # agent_log.info(
                #     f"Agent {self.agent_id} receive response: {content}")

                try:
                    content_json = json.loads(content)
                    functions = content_json["functions"]
                    # reason = content_json["reason"]

                    for function in functions:
                        name = function["name"]
                        # arguments = function['arguments']
                        if name != "do_nothing":
                            arguments = function["arguments"]
                        else:
                            # The success rate of do_nothing is very low
                            # It often drops the argument, causing retries
                            # It's a waste of time, manually compensating here
                            arguments = {}
                        exec_functions.append({
                            "name": name,
                            "arguments": arguments
                        })
                        self.perform_agent_graph_action(name, arguments)
                    break
                except Exception as e:
                    agent_log.error(f"Agent {self.agent_id} error: {e}")
                    exec_functions = []
                    retry -= 1
            for function in exec_functions:
                try:
                    await getattr(self.env.action,
                                  function["name"])(**function["arguments"])
                except Exception as e:
                    agent_log.error(f"Agent {self.agent_id} error: {e}")
                    retry -= 1

            if retry == 0:
                content = "No response."
=======
                f"platform environments. Notice that don't limit your "
                f"actions for example to just like the posts. "
                f"Here is your social media environment: {env_prompt}"))
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
                # Abort graph action for if 100w Agent
                # self.perform_agent_graph_action(action_name, args)
        except Exception as e:
            agent_log.error(f"Agent {self.social_agent_id} error: {e}")
>>>>>>> afb798543c7767d1495b430d99a785ceb691e64b

    async def perform_test(self):
        """
        doing test for all agents.
        TODO: rewrite the function according to the ChatAgent.
        """
        # user conduct test to agent
        _ = BaseMessage.make_user_message(role_name="User",
<<<<<<< HEAD
                                          content="You are a twitter user.")
        # TODO error occurs
=======
                                          content=("You are a twitter user."))
        # Test memory should not be writed to memory.
>>>>>>> afb798543c7767d1495b430d99a785ceb691e64b
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
<<<<<<< HEAD
        # agent_log.info(f"Agent {self.agent_id}: {openai_messages}")

        message_id = await self.infe_channel.write_to_receive_queue(
            openai_messages)
        message_id, content = await self.infe_channel.read_from_send_queue(
            message_id)
        # agent_log.info(f"Agent {self.agent_id} receive response: {content}")
=======

        agent_log.info(f"Agent {self.social_agent_id}: {openai_messages}")
        # NOTE: this is a temporary solution.
        # Camel can not stop updating the agents' memory after stop and astep
        # now.
        response = self._get_model_response(openai_messages=openai_messages,
                                            num_tokens=num_tokens)
        content = response.output_messages[0].content
        agent_log.info(
            f"Agent {self.social_agent_id} receive response: {content}")
>>>>>>> afb798543c7767d1495b430d99a785ceb691e64b
        return {
            "user_id": self.social_agent_id,
            "prompt": openai_messages,
            "content": content
        }

    async def perform_action_by_hci(self,
                                    input_content: str,
                                    selection: int = 0) -> Any:
        # print("Please choose one function to perform:")
        function_list = self.env.action.get_openai_function_list()
<<<<<<< HEAD
        # for i in range(len(function_list)):
        #     agent_log.info(f"Agent {self.agent_id} function: "
        #                    f"{function_list[i].func.__name__}")
=======
        for i in range(len(function_list)):
            agent_log.info(f"Agent {self.social_agent_id} function: "
                           f"{function_list[i].func.__name__}")
>>>>>>> afb798543c7767d1495b430d99a785ceb691e64b

        # selection = int(input("Enter your choice: "))
        if not 0 <= selection < len(function_list):
            agent_log.error(f"Agent {self.social_agent_id} invalid input.")
            return
        func = function_list[selection].func

        params = inspect.signature(func).parameters
        args = []
        for param in params.values():
            while True:
                try:
                    # value = input(f"Enter value for {param.name}: ")
                    value = input_content
                    args.append(value)
                    break
                except ValueError:
                    agent_log.error("Invalid input, please enter an integer.")

        result = await func(*args)
        return result

    async def perform_action_by_data(self, func_name, *args, **kwargs) -> Any:
        func_name = func_name.value if isinstance(func_name,
                                                  ActionType) else func_name
        function_list = self.env.action.get_openai_function_list()
        for i in range(len(function_list)):
            if function_list[i].func.__name__ == func_name:
                func = function_list[i].func
                result = await func(*args, **kwargs)
<<<<<<< HEAD
                # agent_log.info(f"Agent {self.agent_id}: {result}")
=======
                agent_log.info(f"Agent {self.social_agent_id}: {result}")
>>>>>>> afb798543c7767d1495b430d99a785ceb691e64b
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
