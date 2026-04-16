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
from string import Template

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
    short_video_posts_env_template = Template(
        "After refreshing your short-video feed, you see these videos $posts")
    livestream_env_template = Template(
        "There are active livestream rooms right now $streams")

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
        "$followers_env\n"
        "$follows_env\n"
        "$groups_env\n"
        "$livestream_env\n"
        "$posts_env\npick one you want to perform action that best "
        "reflects your current inclination based on your profile and "
        "posts content. Do not limit your action in just `like` to like posts")
    short_video_env_template = Template(
        "$followers_env\n"
        "$follows_env\n"
        "$livestream_env\n"
        "$posts_env\npick one action that best reflects how you would behave "
        "in a short-video platform. Consider actions such as watch_video, "
        "share_video, follow, not_interested, duet, stitch, and livestream "
        "interactions when they fit your interests and the feed context.")

    def __init__(self, action: SocialAction):
        self.action = action

    def _is_short_video_posts(self, posts: list[dict]) -> bool:
        return any(
            post.get("content_format") == "short_video"
            or "traffic_pool_level" in post
            or "avg_watch_ratio" in post for post in posts)

    async def get_posts_env(self) -> tuple[str, bool]:
        posts = await self.action.refresh()
        # TODO: Replace posts json format string to other formats
        if posts["success"]:
            post_list = posts["posts"]
            posts_env = json.dumps(post_list, indent=4)
            is_short_video = self._is_short_video_posts(post_list)
            template = (self.short_video_posts_env_template
                        if is_short_video else self.posts_env_template)
            posts_env = template.substitute(posts=posts_env)
        else:
            posts_env = "After refreshing, there are no existing posts."
            is_short_video = False
        return posts_env, is_short_video

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

    async def get_livestream_env(self, include_livestreams: bool) -> str:
        if not include_livestreams:
            return ""
        try:
            db_path = get_db_path()
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name='livestream'"
            )
            if cursor.fetchone() is None:
                conn.close()
                return ""
            cursor.execute(
                "SELECT stream_id, host_id, current_viewers, peak_viewers, "
                "total_comments, total_gifts_value FROM livestream "
                "WHERE status = 'live' ORDER BY current_viewers DESC LIMIT 3"
            )
            live_streams = [{
                "stream_id": row[0],
                "host_id": row[1],
                "current_viewers": row[2],
                "peak_viewers": row[3],
                "total_comments": row[4],
                "total_gifts_value": row[5],
            } for row in cursor.fetchall()]
            conn.close()
            if not live_streams:
                return "There are no active livestream rooms right now."
            return self.livestream_env_template.substitute(
                streams=json.dumps(live_streams))
        except Exception:
            return ""

    async def to_text_prompt(
        self,
        include_posts: bool = True,
        include_followers: bool = True,
        include_follows: bool = True,
    ) -> str:
        followers_env = (await self.get_followers_env()
                         if include_follows else "No followers.")
        follows_env = (await self.get_follows_env()
                       if include_followers else "No follows.")
        if include_posts:
            posts_env, is_short_video = await self.get_posts_env()
        else:
            posts_env, is_short_video = "", False
        livestream_env = await self.get_livestream_env(is_short_video)

        template = (self.short_video_env_template
                    if is_short_video else self.env_template)
        return template.substitute(
            followers_env=followers_env,
            follows_env=follows_env,
            posts_env=posts_env,
            groups_env=await self.get_group_env(),
            livestream_env=livestream_env,
        )
