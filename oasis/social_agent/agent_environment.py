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
from io import BytesIO
from string import Template
from typing import List, Union, Tuple
from urllib.parse import urlparse

import requests
from PIL import Image
from camel.utils.commons import logger

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
        "After refreshing, you see some posts."
        "post is represented by image,The text of post is in image.info")

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

    async def get_posts_env(self) -> Tuple[str, List[Image.Image]]:
        posts = await self.action.refresh()
        if posts["success"]:
            posts_env = json.dumps(posts["posts"], indent=4)
            posts = json.loads(posts_env)
            images = []
            for post in posts:
                if post.get("image_path"):
                    image = self._load_image(post["image_path"])
                    image.info["post_content"] = post.get("content", "")
                    images.append(image)
                    del post["image_path"]
            posts_env = json.dumps(posts, indent=4)
            posts_env = self.posts_env_template.substitute(posts=posts_env)
        else:
            posts_env = "After refreshing, there are no existing posts."
            images = []
        return posts_env, images

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
    ) -> Tuple[str, List[Image.Image]]:
        followers_env = (await self.get_followers_env()
                         if include_follows else "No followers.")
        follows_env = (await self.get_follows_env()
                       if include_followers else "No follows.")
        posts_env, images = await self.get_posts_env() if include_posts else ("", [])

        return self.env_template.substitute(
            followers_env=followers_env,
            follows_env=follows_env,
            posts_env=posts_env,
            groups_env=await self.get_group_env(),
        ), images

    def _load_image(self, image_path: str) -> Image.Image:
        r"""Loads an image from either local path or URL.

        Args:
            image_path (str): Local path or URL to image.

        Returns:
            Image.Image: Loaded PIL Image object.

        Raises:
            ValueError: For invalid paths/URLs or unreadable images.
            requests.exceptions.RequestException: For URL fetch failures.
        """
        parsed = urlparse(image_path)

        if parsed.scheme in ("http", "https"):
            try:
                response = requests.get(image_path, timeout=15)
                response.raise_for_status()
                return Image.open(BytesIO(response.content))
            except requests.exceptions.RequestException as e:
                logger.error(f"URL fetch failed: {e}")
                raise
        else:
            logger.debug(f"Loading local image: {image_path}")
            try:
                return Image.open(image_path)
            except Exception as e:
                logger.error(f"Image loading failed: {e}")
                raise ValueError(f"Invalid image file: {e}")