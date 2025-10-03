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
    followers_env_template = Template("You currently have $num_followers followers.")
    follows_env_template = Template("You are following $num_follows accounts.")

    posts_env_template = Template(
        "After refreshing, you see the following posts. "
        "Each post may contain both text content and images."
    )

    groups_env_template = Template(
        "Available Groups: $all_groups\n"
        "Groups You Joined: $joined_groups\n"
        "Recent Messages: $messages\n"
        "Actions allowed: join groups you're interested in, leave groups you're in, "
        "send messages to groups you're a member of."
    )

    # 全面重构主环境模板，采用层次分明的结构
    env_template = Template("""
# Social Media Environment

## Profile Information
$followers_info
$follows_info

## Content Feed
$posts_info

## Groups
$groups_info

## Task Instructions
Please analyze the provided social media environment and perform appropriate actions.
Your actions should reflect your interests and engagement patterns.
You are not limited to any specific type of action.

IMPORTANT: Each post's text content is closely related to its corresponding images. When analyzing posts, please consider both the text and images as a cohesive unit. The images provide visual context that complements the text content.
""")

    def __init__(self, action: SocialAction):
        self.action = action

    async def get_posts_env(self) -> Tuple[str, List[Image.Image]]:
        posts = await self.action.refresh()
        if posts["success"]:
            posts_env = json.dumps(posts["posts"], indent=4)
            posts = json.loads(posts_env)
            images = []

            post_descriptions = []
            for i, post in enumerate(posts, 1):
                post_content = post.get("content", "")
                post_description = f"Post {i}: {post_content}"
                post_descriptions.append(post_description)

                if post.get("image_path"):
                    image = self._load_image(post["image_path"])
                    image.info["post_content"] = post_content
                    image.info["post_id"] = str(i)
                    images.append(image)
                    del post["image_path"]

            # 使用更详细的帖子描述替代简单的JSON
            posts_env = "\n\n".join(post_descriptions)
            if posts_env:
                posts_env = self.posts_env_template.substitute() + "\n" + posts_env
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
                           (agent_id,))
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
                (agent_id,))
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
                         if include_followers else "")
        follows_env = (await self.get_follows_env()
                       if include_follows else "")
        posts_env, images = await self.get_posts_env() if include_posts else ("", [])

        followers_info = followers_env if followers_env else "Not showing follower information."
        follows_info = follows_env if follows_env else "Not showing following information."
        posts_info = posts_env if posts_env else "No posts available."
        groups_info = await self.get_group_env()

        return self.env_template.substitute(
            followers_info=followers_info,
            follows_info=follows_info,
            posts_info=posts_info,
            groups_info=groups_info,
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
