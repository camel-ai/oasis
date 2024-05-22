from typing import List

from camel.functions import OpenAIFunction

from social_agent.agent import ActionSpace


def test_transfer_to_openai_function():
    # 如果这部分不会通过camel的openAIFunction函数raise error，
    # 即正确生成了openai schema，并可以被camel Chatagent调用。
    ACTION_FUNCS: List[OpenAIFunction] = [
        OpenAIFunction(func) for func in [
            ActionSpace.action_sign_up,
            ActionSpace.action_refresh,
            ActionSpace.action_create_tweet,
            ActionSpace.action_like,
            ActionSpace.action_unlike,
            ActionSpace.action_search_tweets,
            ActionSpace.action_search_user,
            ActionSpace.action_follow,
            ActionSpace.action_unfollow,
            ActionSpace.action_mute,
            ActionSpace.action_unmute,
            ActionSpace.action_trend,
        ]
    ]
    assert ACTION_FUNCS is not None
