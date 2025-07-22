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
import json
import logging
import os
from datetime import datetime
from typing import List, Union, Dict, Any

from oasis.environment.env_action import LLMAction, ManualAction
from oasis.social_agent.agent import SocialAgent
from oasis.social_agent.agent_graph import AgentGraph
from oasis.social_agent.agents_generator import generate_custom_agents
from oasis.social_platform.channel import Channel
from oasis.social_platform.platform import Platform
from oasis.social_platform.typing import (ActionType, DefaultPlatformType,
                                          RecsysType)
from oasis.user_profile_agent.agent import PreferenceAgent

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
        agent_graph: AgentGraph,
        platform: Union[DefaultPlatformType, Platform],
        database_path: str = None,
        semaphore: int = 128,
    ) -> None:
        r"""Init the oasis environment.

        Args:
            agent_graph: The AgentGraph to use in the simulation.
            platform: The platform type to use. Including
                `DefaultPlatformType.TWITTER` or `DefaultPlatformType.REDDIT`.
                Or you can pass a custom `Platform` instance.
            database_path: The path to create a sqlite3 database. The file
                extension must be `.db` such as `twitter_simulation.db`.
        """
        # Initialize the agent graph
        self.agent_graph = agent_graph
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
        
        # Initialize user profile management
        self.user_profiles: Dict[int, Dict[str, Any]] = {}  # 存储用户画像数据
        self.user_profile_agent: PreferenceAgent = None  # 用户画像分析代理
        self.profile_update_enabled: bool = True  # 是否启用动态画像更新
        self.user_profile_file: str = "user_profile.json"  # 用户画像存储文件

    async def reset(self) -> None:
        r"""Start the platform and sign up the agents."""
        self.platform_task = asyncio.create_task(self.platform.running())
        self.agent_graph = await generate_custom_agents(
            channel=self.channel, agent_graph=self.agent_graph)
        
        # Initialize user profile agent and load existing profiles
        await self._initialize_user_profile_system()
    
    async def _initialize_user_profile_system(self) -> None:
        """初始化用户画像系统"""
        try:
            # 初始化用户画像分析代理
            # 这里需要传入模型，可以从第一个代理获取模型配置
            agents_list = self.agent_graph.get_agents()
            if agents_list:
                first_agent = agents_list[0][1]  # get_agents() 返回 [(agent_id, agent), ...]
                self.user_profile_agent = PreferenceAgent(model=first_agent.model_backend)
                env_log.info("User profile agent initialized successfully")
            
            # 加载已有的用户画像数据
            await self._load_user_profiles()
            env_log.info("User profile system initialized")
        except Exception as e:
            env_log.error(f"Failed to initialize user profile system: {e}")
            self.profile_update_enabled = False
    
    async def _load_user_profiles(self) -> None:
        """从文件加载已有的用户画像数据"""
        try:
            if os.path.exists(self.user_profile_file):
                with open(self.user_profile_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 兼容新旧格式
                if isinstance(data, dict) and "user_profiles" in data:
                    # 新格式：包含时间戳的数据
                    raw_profiles = data["user_profiles"]
                    # 确保所有 key 都是字符串类型
                    self.user_profiles = {str(k): v for k, v in raw_profiles.items()}
                    self._profile_update_count = data.get("update_count", 0)
                    env_log.info(f"Loaded {len(self.user_profiles)} user profiles from {self.user_profile_file} (last updated: {data.get('last_updated', 'unknown')}, update #{self._profile_update_count})")
                    env_log.info(f"Profile keys loaded: {list(self.user_profiles.keys())}")
                else:
                    # 旧格式：直接的用户画像数据
                    # 确保所有 key 都是字符串类型
                    self.user_profiles = {str(k): v for k, v in data.items()}
                    self._profile_update_count = 0
                    env_log.info(f"Loaded {len(self.user_profiles)} user profiles from {self.user_profile_file} (legacy format)")
                    env_log.info(f"Profile keys loaded: {list(self.user_profiles.keys())}")
            else:
                self.user_profiles = {}
                self._profile_update_count = 0
                env_log.info("No existing user profile file found, starting with empty profiles")
        except Exception as e:
            env_log.error(f"Failed to load user profiles: {e}")
            self.user_profiles = {}
            self._profile_update_count = 0
    
    async def _save_user_profiles(self) -> None:
        """保存用户画像数据到文件"""
        try:
            # 添加更新时间字段
            from datetime import datetime
            
            profile_data_with_timestamp = {
                "user_profiles": self.user_profiles,
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "update_count": getattr(self, '_profile_update_count', 0) + 1
            }
            
            # 记录更新次数
            self._profile_update_count = profile_data_with_timestamp["update_count"]
            
            with open(self.user_profile_file, 'w', encoding='utf-8') as f:
                json.dump(profile_data_with_timestamp, f, ensure_ascii=False, indent=2)
            env_log.info(f"Saved {len(self.user_profiles)} user profiles to {self.user_profile_file} at {profile_data_with_timestamp['last_updated']} (update #{profile_data_with_timestamp['update_count']})")
        except Exception as e:
            env_log.error(f"Failed to save user profiles: {e}")
    
    async def _collect_user_history(self, agent_id: int) -> Dict[str, Any]:
        """收集指定用户的历史行为数据"""
        try:
            # 使用环境中的数据库路径
            import sqlite3
            
            db_path = self.platform.db_path
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # 收集用户的帖子和评论历史
            user_history = {
                "user_id": agent_id,
                "actions": {
                    "create_post": [],
                    "create_comment": [],
                    "like_post": [],
                    "sign_up": [],
                    "refresh": [],
                    "do_nothing": []
                },
                "time_distribution": {
                    "morning": 0,
                    "afternoon": 0, 
                    "evening": 0,
                    "night": 0
                }
            }
            
            # 查询用户的帖子
            try:
                cursor.execute("SELECT content FROM post WHERE agent_id = ?", (agent_id,))
                posts = cursor.fetchall()
                user_history["actions"]["create_post"] = [post[0] for post in posts if post[0]]
                env_log.info(f"Found {len(posts)} posts for agent {agent_id}")
            except Exception as e:
                env_log.warning(f"Failed to query posts for agent {agent_id}: {e}")
            
            # 查询用户的评论
            try:
                cursor.execute("SELECT content FROM comment WHERE agent_id = ?", (agent_id,))
                comments = cursor.fetchall()
                user_history["actions"]["create_comment"] = [comment[0] for comment in comments if comment[0]]
                env_log.info(f"Found {len(comments)} comments for agent {agent_id}")
            except Exception as e:
                env_log.warning(f"Failed to query comments for agent {agent_id}: {e}")
            
            # 查询用户的操作历史（从 trace 表）
            try:
                cursor.execute("""
                    SELECT action_type, COUNT(*) as count 
                    FROM trace 
                    WHERE agent_id = ? 
                    GROUP BY action_type
                """, (agent_id,))
                action_counts = cursor.fetchall()
                for action_type, count in action_counts:
                    if action_type in user_history["actions"]:
                        # 这里只记录数量，不记录具体内容
                        pass
                env_log.info(f"Found {len(action_counts)} action types for agent {agent_id}")
            except Exception as e:
                env_log.warning(f"Failed to query action history for agent {agent_id}: {e}")
            
            conn.close()
            env_log.info(f"Successfully collected history for agent {agent_id}")
            return user_history
            
        except Exception as e:
            env_log.error(f"Failed to collect user history for agent {agent_id}: {e}")
            return {
                "user_id": agent_id,
                "actions": {"create_post": [], "create_comment": []},
                "time_distribution": {"morning": 0, "afternoon": 0, "evening": 0, "night": 0}
            }
    
    async def _update_user_profiles(self) -> None:
        """动态更新所有用户的画像数据"""
        if not self.profile_update_enabled:
            env_log.info("User profile update is disabled")
            return
            
        if not self.user_profile_agent:
            env_log.error("User profile agent is not initialized")
            return
        
        try:
            env_log.info("Starting dynamic user profile update...")
            
            # 获取所有代理的信息
            agents_list = self.agent_graph.get_agents()
            env_log.info(f"Found {len(agents_list)} agents to update profiles for")
            
            agent_profile_dic = {
                agent_id: agent.user_info.profile
                for agent_id, agent in agents_list
            }
            
            # 为每个代理更新用户画像
            updated_count = 0
            for agent_id, agent in agents_list:
                try:
                    # 确保 agent_id 是字符串类型，保持一致性
                    agent_id_str = str(agent_id)
                    env_log.info(f"Processing profile update for agent {agent_id_str} (original: {agent_id}, type: {type(agent_id)})")
                    
                    # 收集用户历史数据
                    user_history = await self._collect_user_history(agent_id)
                    env_log.info(f"Collected history for agent {agent_id_str}: {len(user_history.get('actions', {}).get('create_post', []))} posts, {len(user_history.get('actions', {}).get('create_comment', []))} comments")
                    
                    # 获取之前的用户画像作为 previous_profile
                    previous_profile = self.user_profiles.get(agent_id_str, None)
                    env_log.info(f"Previous profile exists for agent {agent_id_str}: {previous_profile is not None}")
                    
                    # 调用 PreferenceAgent 进行画像分析
                    env_log.info(f"Calling PreferenceAgent.analyse for agent {agent_id_str}")
                    new_user_profile = await self.user_profile_agent.analyse(
                        (user_history, self.user_profiles.get(agent_id_str, {}), agent_profile_dic.get(agent_id)),
                        previous_profile=previous_profile
                    )
                    
                    # 检查返回的结果
                    if new_user_profile is None:
                        env_log.warning(f"PreferenceAgent returned None for agent {agent_id_str}")
                        # 保持原有的画像数据，不覆盖为 None
                        continue
                    
                    # 更新用户画像数据（使用字符串类型的 agent_id）
                    env_log.info(f"Updating profile for agent {agent_id_str}. Current profiles: {list(self.user_profiles.keys())}")
                    self.user_profiles[agent_id_str] = new_user_profile
                    updated_count += 1
                    
                    env_log.info(f"Successfully updated user profile for agent {agent_id_str}. Total profiles now: {list(self.user_profiles.keys())}")
                    
                except Exception as e:
                    env_log.error(f"Failed to update profile for agent {agent_id}: {e}")
                    import traceback
                    env_log.error(f"Traceback: {traceback.format_exc()}")
                    # 在失败时，保持原有的画像数据不变
                    continue
            
            # 保存更新后的用户画像数据
            await self._save_user_profiles()
            env_log.info(f"Saved {len(self.user_profiles)} user profiles to {self.user_profile_file}")
            
            if updated_count > 0:
                env_log.info(f"Successfully updated {updated_count} user profiles")
            else:
                env_log.warning("No user profiles were successfully updated")
            
        except Exception as e:
            env_log.error(f"Failed to update user profiles: {e}")
            import traceback
            env_log.error(f"Traceback: {traceback.format_exc()}")

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

    async def step(
        self, actions: dict[SocialAgent, Union[ManualAction, LLMAction,
                                               List[Union[ManualAction,
                                                          LLMAction]]]]
    ) -> None:
        r"""Update the recommendation system and perform the actions.

        Args:
            actions(dict[SocialAgent, Union[ManualAction, LLMAction,
                List[Union[ManualAction, LLMAction]]]]): The actions to
                perform, including the manual(pre-defined) actions and llm
                actions.
        Returns:
            None
        """
        # Update the recommendation system
        await self.platform.update_rec_table()
        env_log.info("update rec table.")

        # Create tasks for both manual and LLM actions
        tasks = []
        for agent, action in actions.items():
            if isinstance(action, list):
                for single_action in action:
                    if isinstance(single_action, ManualAction):
                        if single_action.action_type == ActionType.INTERVIEW:
                            # Use the agent's perform_interview method for
                            # interview actions
                            interview_prompt = single_action.action_args.get(
                                "prompt", "")
                            tasks.append(
                                self._perform_interview_action(
                                    agent, interview_prompt))
                        else:
                            tasks.append(
                                agent.perform_action_by_data(
                                    single_action.action_type,
                                    **single_action.action_args))
                    elif isinstance(single_action, LLMAction):
                        tasks.append(self._perform_llm_action(agent))
            else:
                if isinstance(action, ManualAction):
                    if action.action_type == ActionType.INTERVIEW:
                        # Use the agent's perform_interview method for
                        # interview actions
                        interview_prompt = action.action_args.get("prompt", "")
                        tasks.append(
                            self._perform_interview_action(
                                agent, interview_prompt))
                    else:
                        tasks.append(
                            agent.perform_action_by_data(
                                action.action_type, **action.action_args))
                elif isinstance(action, LLMAction):
                    tasks.append(self._perform_llm_action(agent))

        # Execute all tasks concurrently
        await asyncio.gather(*tasks)
        env_log.info("performed all actions.")
        
        # 在所有操作执行完成后，动态更新用户画像
        await self._update_user_profiles()
        
        # # Control some agents to perform actions
        # Update the clock
        if self.platform_type == DefaultPlatformType.TWITTER:
            self.platform.sandbox_clock.time_step += 1
    
    def enable_user_profile_update(self, enabled: bool = True) -> None:
        """启用或禁用动态用户画像更新"""
        self.profile_update_enabled = enabled
        env_log.info(f"User profile update {'enabled' if enabled else 'disabled'}")
    
    def set_user_profile_file(self, file_path: str) -> None:
        """设置用户画像存储文件路径"""
        self.user_profile_file = file_path
        env_log.info(f"User profile file set to: {file_path}")
    
    def get_user_profile(self, agent_id: int) -> Dict[str, Any]:
        """获取指定代理的用户画像数据"""
        return self.user_profiles.get(agent_id, {})
    
    def get_all_user_profiles(self) -> Dict[int, Dict[str, Any]]:
        """获取所有用户的画像数据"""
        return self.user_profiles.copy()

    async def close(self) -> None:
        r"""Stop the platform and close the environment.
        """
        await self.channel.write_to_receive_queue(
            (None, None, ActionType.EXIT))
        await self.platform_task
        env_log.info("Simulation finished! Please check the results in the "
                     f"database: {self.platform.db_path}. Note that the trace "
                     "table stored all the actions of the agents.")
