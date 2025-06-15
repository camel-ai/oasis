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
from abc import ABC, abstractmethod
from string import Template

from oasis.social_agent.agent_action import SocialAction


class Environment(ABC):

    @abstractmethod
    def to_text_prompt(self) -> str:
        r"""Convert the environment to text prompt."""
        raise NotImplementedError


class SocialEnvironment(Environment):
    r"""
    Class for translating raw platform data into prompts. The templated 
    prompts can be customized with specific parameters such as number of 
    followers and number of all groups.
    """
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
        r"""Initialize the social environment.
        
        Args:
            action (SocialAction): Pre-configured social action instance that
                                  handles actual platform interactions.
        """
        self.action = action

    async def get_posts_env(self) -> str:
        r"""Fetch the latest posts and formats them. Then generate the post
        description.

        Returns:
            str: Formatted post feed description. If success, fill in the
            latest information into template prompt. If refresh fails, show the 
            fail message.
        """
        posts = await self.action.refresh()
        # TODO: Replace posts json format string to other formats
        if posts["success"]:
            posts_env = json.dumps(posts["posts"], indent=4)
            posts_env = self.posts_env_template.substitute(posts=posts_env)
        else:
            posts_env = "After refreshing, there are no existing posts."
        return posts_env

    async def get_followers_env(self) -> str:
        r"""Fetch the number of followers and generate followers description. 

        Returns:
            str: Fill in the latest information into template prompt. 
            Example: "I have 40 followers."
        """
        # TODO: Implement followers env
        return self.followers_env_template.substitute(num_followers=0)

    async def get_follows_env(self) -> str:
        r"""Fetch the number of follows and generate follows description. 

        Returns:
            str: Fill in the latest information into template prompt. 
            Example: "I have 50 follows."
        """
        # TODO: Implement follows env
        return self.follows_env_template.substitute(num_follows=0)

    async def get_group_env(self) -> str:
        r"""Fetch group information (e.g. all_groups) and generate group 
        interaction environment description.

        Returns:
            str: If group exist, Fill in the latest information into template prompt.  
            If it doesn't, return "No groups"
        """
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
        include_followers: bool = False,
        include_follows: bool = False,
    ) -> str:
        r"""Generate social environment prompt from selected components.
        
        Args:
            include_posts (bool): Whether to include post feed. 
            include_followers (bool): Whether to include follower count. 
            include_follows (bool): Whether to include follows count. 
        """
        followers_env = (await self.get_followers_env()
                         if include_follows else "No followers.")
        follows_env = (await self.get_follows_env()
                       if include_followers else "No follows.")
        posts_env = await self.get_posts_env() if include_posts else ""

        return self.env_template.substitute(
            followers_env=followers_env,
            follows_env=follows_env,
            posts_env=posts_env,
            groups_env=await self.get_group_env(),
        )
