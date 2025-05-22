# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
import asyncio
import logging
import os
from datetime import datetime
from typing import List, Optional, Union

from camel.models import BaseModelBackend

from oasis.environment.env_action import EnvAction, SingleAction
from oasis.social_agent.agents_generator import (generate_agents,
                                                 generate_reddit_agents)
from oasis.social_platform.channel import Channel
from oasis.social_platform.platform import Platform
from oasis.social_platform.typing import (ActionType, DefaultPlatformType,
                                          RecsysType)

# Create log directory if it doesn't exist
log_dir = "./log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Configure logger
env_log = logging.getLogger("oasis.env")
env_log.setLevel("INFO")

# Add file handler to save logs to file
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
file_handler = logging.FileHandler(f"{log_dir}/oasis-{current_time}.log",
                                   encoding="utf-8")
file_handler.setLevel("INFO")
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
env_log.addHandler(file_handler)


class OasisEnv:

    def __init__(
        self,
        platform: Union[DefaultPlatformType, Platform],
        agent_profile_path: str,
        database_path: str = None,
        agent_models: Optional[Union[BaseModelBackend,
                                     List[BaseModelBackend]]] = None,
        available_actions: list[ActionType] = None,
        semaphore: int = 128,
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
        # Use a semaphore to limit the number of concurrent requests
        self.llm_semaphore = asyncio.Semaphore(semaphore)
        if isinstance(platform, DefaultPlatformType):
            if database_path is None:
                raise ValueError(
                    "database_path is required for DefaultPlatformType")
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
            if database_path != platform.db_path:
                env_log.warning("database_path is not the same as the "
                                "platform.db_path, using the platform.db_path")
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
        r"""Start the platform and sign up the agents.
        """
        self.platform_task = asyncio.create_task(self.platform.running())
        if self.platform_type == DefaultPlatformType.TWITTER:
            self.agent_graph = await generate_agents(
                agent_info_path=self.agent_profile_path,
                twitter_channel=self.channel,
                model=self.agent_models,
                recsys_type=RecsysType.TWHIN,
                start_time=self.platform.sandbox_clock.time_step,
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
        r"""Perform a control action.

        Args:
            action(SingleAction): The action to perform.
        """
        control_agent = self.agent_graph.get_agent(action.agent_id)
        
        # For regular actions
        await control_agent.perform_action_by_data(action.action, **action.args)
        
        # If this was an interview action, perform the actual interview and update the database
        if action.action == ActionType.INTERVIEW:
            # Extract interview prompt from args
            interview_prompt = action.args.get("prompt", "")
            if not interview_prompt:
                env_log.warning(f"Empty interview prompt for agent {action.agent_id}")
                return
                
            # Perform the interview to get actual response
            result = await self._perform_interview_action(control_agent, interview_prompt)
            
            # Get the response content
            response = result.get("content", "")
            
            try:
                # Update the database with the actual response
                # First, fetch the latest trace record for this interview
                query = """SELECT rowid, info FROM trace 
                           WHERE user_id = ? AND action = ? 
                           ORDER BY created_at DESC LIMIT 1"""
                self.platform.pl_utils._execute_db_command(query, (action.agent_id, ActionType.INTERVIEW.value))
                last_interview_data = self.platform.db_cursor.fetchone()
                
                if last_interview_data:
                    rowid, info_json = last_interview_data
                    import json
                    info = json.loads(info_json)
                    info["response"] = response
                    
                    # Update the trace record with the actual response
                    update_query = "UPDATE trace SET info = ? WHERE rowid = ?"
                    self.platform.pl_utils._execute_db_command(
                        update_query, 
                        (json.dumps(info), rowid),
                        commit=True
                    )
                    
                    env_log.info(f"Updated interview result for agent {action.agent_id}")
                else:
                    # If we couldn't find the trace record, create a new one with the response
                    interview_info = {
                        "prompt": interview_prompt,
                        "response": response
                    }
                    
                    if self.platform.recsys_type == RecsysType.REDDIT:
                        current_time = self.platform.sandbox_clock.time_transfer(
                            datetime.now(), self.platform.start_time)
                    else:
                        current_time = self.platform.sandbox_clock.get_time_step()
                        
                    self.platform.pl_utils._record_trace(
                        user_id=action.agent_id,
                        action_type=ActionType.INTERVIEW.value,
                        action_info=interview_info,
                        current_time=current_time
                    )
                    
                    env_log.info(f"Created new interview record for agent {action.agent_id}")
            except Exception as e:
                env_log.error(f"Error updating interview result: {str(e)}")
                
            return

    async def _perform_llm_action(self, agent):
        r"""Send the request to the llm model and execute the action.
        """
        async with self.llm_semaphore:
            return await agent.perform_action_by_llm()

    async def _perform_interview_action(self, agent, interview_prompt: str):
        r"""Send the request to the llm model and execute the interview.
        """
        async with self.llm_semaphore:
            return await agent.perform_interview(interview_prompt)

    async def step(self, action: EnvAction) -> None:
        r"""Perform some control actions, update the recommendation system,
        and let some llm agents perform actions.

        Args:
            action(EnvAction): The activate agents and control actions to
                perform.
        """
        # Control some agents to perform actions
        if action.intervention:
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
                agent_id for agent_id, _ in self.agent_graph.get_agents()
            ]
        else:
            activate_agents = action.activate_agents

        llm_tasks = []
        for agent_id in activate_agents:
            agent = self.agent_graph.get_agent(agent_id)
            llm_tasks.append(self._perform_llm_action(agent))

        await asyncio.gather(*llm_tasks)
        env_log.info("performed llm actions.")
        # Update the clock
        if self.platform_type == DefaultPlatformType.TWITTER:
            self.platform.sandbox_clock.time_step += 1

    async def close(self) -> None:
        r"""Stop the platform and close the environment.
        """
        await self.channel.write_to_receive_queue(
            (None, None, ActionType.EXIT))
        await self.platform_task
        env_log.info("Simulation finished! Please check the results in the "
                     f"database: {self.platform.db_path}. Note that the trace "
                     "table stored all the actions of the agents.")
