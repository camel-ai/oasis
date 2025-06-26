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

import json
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime
from string import Template
from typing import Any, Dict, List

from oasis.social_agent.agent_action import SocialAction
from oasis.social_platform.database import get_db_path


class Environment(ABC):

    @abstractmethod
    def to_text_prompt(self) -> str:
        r"""Convert the environment to text prompt."""
        raise NotImplementedError


class SocialEnvironment(Environment):
    followers_env_template = Template("I have $num_followers followers.")
    follows_env_template = Template("I have $num_follows follows.")

    posts_env_template = Template(
        "After refreshing, you see some posts $posts")

    groups_env_template = Template(
        "And there are many group chat channels $all_groups\n"
        "And You are already in some groups $joined_groups\n"
        "You receive some messages from them $messages\n"
        "You can join the groups you are interested, "
        "leave the groups you already in, send messages to the group "
        "you already in.\n"
        "You must make sure you can only send messages to the group you "
        "are already in")
    env_template = Template(
        "$groups_env\n"
        "$posts_env\npick one you want to perform action that best "
        "reflects your current inclination based on your profile and "
        "posts content. Do not limit your action in just `like` to like posts")

    def __init__(self, action: SocialAction):
        self.action = action

    def _filter_posts_by_strategy(self, posts: List[Dict[str,
                                                         Any]], max_posts: int,
                                  strategy: str) -> List[Dict[str, Any]]:
        """
        Filter posts based on the specified strategy to limit memory usage.

        Args:
            posts: List of post dictionaries
            max_posts: Maximum number of posts to keep
            strategy: Filtering strategy ("recency", "popularity", or "mixed")

        Returns:
            Filtered list of posts
        """
        if len(posts) <= max_posts:
            return posts

        if strategy == "recency":
            # Sort by creation time (most recent first)
            sorted_posts = sorted(posts,
                                  key=lambda x: x.get('created_at', ''),
                                  reverse=True)
            return sorted_posts[:max_posts]

        elif strategy == "popularity":
            # Sort by engagement metrics (likes + dislikes + shares)
            def get_engagement_score(post):
                likes = post.get('num_likes', 0) or 0
                dislikes = post.get('num_dislikes', 0) or 0
                shares = post.get('num_shares', 0) or 0
                return likes + dislikes + shares

            sorted_posts = sorted(posts,
                                  key=get_engagement_score,
                                  reverse=True)
            return sorted_posts[:max_posts]

        elif strategy == "mixed":
            # Combined score: recency + popularity
            def get_mixed_score(post):
                likes = post.get('num_likes', 0) or 0
                dislikes = post.get('num_dislikes', 0) or 0
                shares = post.get('num_shares', 0) or 0
                engagement = likes + dislikes + shares

                # Simple time-based score (more recent = higher score)
                try:
                    created_at = post.get('created_at', '')
                    if isinstance(created_at, str):
                        post_time = datetime.fromisoformat(
                            created_at.replace('Z', '+00:00'))
                    else:
                        post_time = created_at
                    current_time = datetime.now()
                    # Give higher score to more recent posts
                    time_diff_hours = max(
                        1, (current_time -
                            post_time.replace(tzinfo=None)).total_seconds() /
                        3600)
                    recency_score = 1 / time_diff_hours
                except:
                    recency_score = 0

                # Normalize and combine scores
                return engagement * 0.7 + recency_score * 100 * 0.3

            sorted_posts = sorted(posts, key=get_mixed_score, reverse=True)
            return sorted_posts[:max_posts]

        else:
            # Default: return first max_posts items
            return posts[:max_posts]

    async def get_posts_env(self, user_info=None) -> str:
        posts = await self.action.refresh(user_info)
        # TODO: Replace posts json format string to other formats
        if posts["success"]:
            posts_list = posts["posts"]

            # Apply post filtering if enabled and user_info is provided
            if (user_info and hasattr(user_info, 'enable_post_filtering')
                    and user_info.enable_post_filtering):

                max_posts = getattr(user_info, 'max_posts_in_memory', 10)
                strategy = getattr(user_info, 'post_filter_strategy',
                                   'recency')

                # Filter posts to prevent context overflow
                posts_list = self._filter_posts_by_strategy(
                    posts_list, max_posts, strategy)

            posts_env = json.dumps(posts_list, indent=4)
            posts_env = self.posts_env_template.substitute(posts=posts_env)
        else:
            posts_env = "After refreshing, there are no existing posts."
        return posts_env

    async def get_followers_env(self) -> str:
        # TODO: Implement followers env
        agent_id = self.action.agent_id
        db_path = get_db_path()
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT num_followers FROM user WHERE agent_id = ?",
                           (agent_id, ))
            result = cursor.fetchone()
            num_followers = result[0] if result else 0
            conn.close()
        except Exception:
            num_followers = 0
        return self.followers_env_template.substitute(
            {"num_followers": num_followers})

    async def get_follows_env(self) -> str:
        # TODO: Implement follows env
        agent_id = self.action.agent_id
        try:
            db_path = get_db_path()
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT num_followings FROM user WHERE agent_id = ?",
                (agent_id, ))
            result = cursor.fetchone()
            num_followings = result[0] if result else 0
            conn.close()
        except Exception:
            num_followings = 0
        return self.follows_env_template.substitute(
            {"num_follows": num_followings})

    async def get_group_env(self) -> str:
        groups = await self.action.listen_from_group()
        if groups["success"]:
            all_groups = json.dumps(groups["all_groups"])
            joined_groups = json.dumps(groups["joined_groups"])
            messages = json.dumps(groups["messages"])
            groups_env = self.groups_env_template.substitute(
                all_groups=all_groups,
                joined_groups=joined_groups,
                messages=messages,
            )
        else:
            groups_env = "No groups."
        return groups_env

    async def to_text_prompt(
        self,
        include_posts: bool = True,
        include_followers: bool = True,
        include_follows: bool = True,
        user_info=None,
    ) -> str:
        followers_env = (await self.get_followers_env()
                         if include_follows else "No followers.")
        follows_env = (await self.get_follows_env()
                       if include_followers else "No follows.")
        posts_env = await self.get_posts_env(user_info
                                             ) if include_posts else ""

        return self.env_template.substitute(
            followers_env=followers_env,
            follows_env=follows_env,
            posts_env=posts_env,
            groups_env=await self.get_group_env(),
        )
