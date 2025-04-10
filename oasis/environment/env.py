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
import asyncio
import logging
from typing import List, Optional, Union

from camel.models import BaseModelBackend

from oasis.environment.env_action import EnvAction, SingleAction
from oasis.social_agent.agents_generator import (generate_agents,
                                                 generate_reddit_agents)
from oasis.social_platform.channel import Channel
from oasis.social_platform.platform import Platform
from oasis.social_platform.typing import (ActionType, DefaultPlatformType,
                                          RecsysType)

env_log = logging.getLogger("oasis.env")
env_log.setLevel("DEBUG")


class OasisEnv:

    def __init__(
        self,
        platform: Union[DefaultPlatformType, Platform],
        database_path: str,
        agent_profile_path: str,
        agent_models: Optional[Union[BaseModelBackend,
                                     List[BaseModelBackend]]] = None,
        available_actions: list[ActionType] = None,
    ) -> None:
        r"""Init the oasis environment.

        Args:
            platform: The platform type to use. Including
                `DefaultPlatformType.TWITTER` or `DefaultPlatformType.REDDIT`.
                Or you can pass a custom `Platform` instance.
            database_path: The path to create a sqlite3 database. The file
                extension must be `.db` such as `twitter_simulation.db`.
            agent_profile_path: The path to the agent profile. Make sure the
                data format is align with the `platform`.
            agent_models: The model backend to use for all agents to generate
                responses. (default: :obj:`ModelPlatformType.DEFAULT` with
                `ModelType.DEFAULT`)
            available_actions: The actions to use for the agents. Choose from
                `ActionType`.
        """
        self.agent_profile_path = agent_profile_path
        self.agent_models = agent_models
        self.available_actions = available_actions
        if isinstance(platform, DefaultPlatformType):
            self.platform = platform
            if platform == DefaultPlatformType.TWITTER:
                self.channel = Channel()
                self.platform = Platform(
                    db_path=database_path,
                    channel=self.channel,
                    recsys_type="twhin-bert",
                    refresh_rec_post_count=2,
                    max_rec_post_len=2,
                    following_post_count=3,
                )
                self.platform_type = DefaultPlatformType.TWITTER
            elif platform == DefaultPlatformType.REDDIT:
                self.channel = Channel()
                self.platform = Platform(
                    db_path=database_path,
                    channel=self.channel,
                    recsys_type="reddit",
                    allow_self_rating=True,
                    show_score=True,
                    max_rec_post_len=100,
                    refresh_rec_post_count=5,
                )
                self.platform_type = DefaultPlatformType.REDDIT
            else:
                raise ValueError(f"Invalid platform: {platform}. Only "
                                 "DefaultPlatformType.TWITTER or "
                                 "DefaultPlatformType.REDDIT are supported.")
        elif isinstance(platform, Platform):
            self.platform = platform
            self.channel = platform.channel
            if platform.recsys_type == RecsysType.REDDIT:
                self.platform_type = DefaultPlatformType.REDDIT
            else:
                self.platform_type = DefaultPlatformType.TWITTER
        else:
            raise ValueError(
                f"Invalid platform: {platform}. You should pass a "
                "DefaultPlatformType or a Platform instance.")

    async def reset(self) -> None:
        self.platform_task = asyncio.create_task(self.platform.running())
        if self.platform_type == DefaultPlatformType.TWITTER:
            self.agent_graph = await generate_agents(
                agent_info_path=self.agent_profile_path,
                twitter_channel=self.channel,
                model=self.agent_models,
                recsys_type=RecsysType.TWHIN_BERT,
                available_actions=self.available_actions,
                twitter=self.platform,
            )
        elif self.platform_type == DefaultPlatformType.REDDIT:
            self.agent_graph = await generate_reddit_agents(
                agent_info_path=self.agent_profile_path,
                twitter_channel=self.channel,
                model=self.agent_models,
                available_actions=self.available_actions,
            )

    async def _perform_control_action(self, action: SingleAction) -> None:
        control_agent = self.agent_graph.get_agent(action.agent_id)
        await control_agent.perform_action_by_data(action.action,
                                                   **action.args)

    async def step(self, action: EnvAction) -> None:
        # Control some agents to perform actions
        control_tasks = [
            self._perform_control_action(single_action)
            for single_action in action.intervention
        ]
        await asyncio.gather(*control_tasks)
        env_log.info("performed control actions.")

        # Update the recommendation system
        await self.platform.update_rec_table()
        env_log.info("update rec table.")

        # Some llm agents perform actions
        if not action.activate_agents:
            env_log.warning(
                "activate_agents is None, default to activate all agents.")
            activate_agents = [
                agent.social_agent_id
                for agent in self.agent_graph.get_agents()
            ]
        else:
            activate_agents = action.activate_agents

        llm_tasks = []
        for agent_id in activate_agents:
            agent = self.agent_graph.get_agent(agent_id)
            llm_tasks.append(agent.perform_action_by_llm())

        await asyncio.gather(*llm_tasks)
        env_log.info("performed llm actions.")
        # Update the clock
        if self.platform_type == DefaultPlatformType.TWITTER:
            self.platform.clock.time_step += 1

    async def close(self) -> None:
        await self.channel.write_to_receive_queue(
            (None, None, ActionType.EXIT))
        await self.platform_task
        env_log.info("Simulation finished! Please check the results in the "
                     f"database: {self.platform.db_path}")
