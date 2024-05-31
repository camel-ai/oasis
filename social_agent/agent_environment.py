import json
from abc import ABC, abstractmethod
from string import Template

from social_agent.agent_action import TwitterAction


class Environment(ABC):

    @abstractmethod
    def to_text_prompt(self) -> str:
        r"""Convert the environment to text prompt.
        """
        raise NotImplementedError


class TwitterEnvironment(Environment):
    followers_env_template = Template("I have $num_followers followers.")
    follows_env_template = Template("I have $num_follows follows.")
    tweets_env_template = Template(
        "Here are the tweets available after refreshing:"
        "\n$tweets.")
    env_template = Template("Followers: $followers_env\n"
                            "Follows: $follows_env\n"
                            "Tweets: $tweets_env")

    def __init__(self, twitter_action: TwitterAction):
        self.twitter_action = twitter_action

    async def get_tweets_env(self) -> str:
        tweets = await self.twitter_action.action_refresh()
        # TODO: Replace tweets json format string to other formats
        if tweets["success"]:
            tweets_env = json.dumps(tweets["tweets"], indent=4)
            tweets_env = self.tweets_env_template.substitute(tweets=tweets_env)
        else:
            tweets_env = "After refreshing, there are no existing tweets."
        return tweets_env

    async def get_followers_env(self) -> str:
        # TODO: Implement followers env
        return self.followers_env_template.substitute(num_followers=0)

    async def get_follows_env(self) -> str:
        # TODO: Implement follows env
        return self.follows_env_template.substitute(num_follows=0)

    async def to_text_prompt(self) -> str:
        followers_env = await self.get_followers_env()
        follows_env = await self.get_follows_env()
        tweets_env = await self.get_tweets_env()

        return self.env_template.substitute(
            followers_env=followers_env,
            follows_env=follows_env,
            tweets_env=tweets_env,
        )
