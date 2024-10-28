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
    followers_env_template = Template("I have $num_followers followers.")
    follows_env_template = Template("I have $num_follows follows.")

    posts_env_template = Template(
        "After refreshing, you see some posts $posts")
    env_template = Template(
        "$posts_env\npick one you want to perform action that best "
        "reflects your current inclination based on your profile and "
        "posts content. Do not limit your action in just `like` to like posts")

    def __init__(self, action: SocialAction):
        self.action = action

    async def get_posts_env(self) -> str:
        posts = await self.action.refresh()
        # TODO: Replace posts json format string to other formats
        if posts["success"]:
            posts_env = json.dumps(posts["posts"], indent=4)
            posts_env = self.posts_env_template.substitute(posts=posts_env)
        else:
            posts_env = "After refreshing, there are no existing posts."
        return posts_env

    async def get_followers_env(self) -> str:
        # TODO: Implement followers env
        return self.followers_env_template.substitute(num_followers=0)

    async def get_follows_env(self) -> str:
        # TODO: Implement follows env
        return self.follows_env_template.substitute(num_follows=0)

    async def to_text_prompt(
        self,
        include_posts: bool = True,
        include_followers: bool = False,
        include_follows: bool = False,
    ) -> str:
        followers_env = (await self.get_followers_env()
                         if include_follows else "No followers.")
        follows_env = (await self.get_follows_env()
                       if include_followers else "No follows.")
        posts_env = await self.get_posts_env() if include_posts else ""

        return self.env_template.substitute(
            followers_env=followers_env,
            follows_env=follows_env,
            posts_env=posts_env,
        )
