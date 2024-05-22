from typing import List

from camel.functions import OpenAIFunction

from social_agent.agent import TwitterAction


def test_transfer_to_openai_function():
    # 如果这部分不会通过camel的openAIFunction函数raise error，
    # 即正确生成了openai schema，并可以被camel Chatagent调用。
    ACTION_FUNCS: List[OpenAIFunction] = [
        OpenAIFunction(func) for func in [
            TwitterAction.action_sign_up,
            TwitterAction.action_refresh,
            TwitterAction.action_create_tweet,
            TwitterAction.action_like,
            TwitterAction.action_unlike,
            TwitterAction.action_search_tweets,
            TwitterAction.action_search_user,
            TwitterAction.action_follow,
            TwitterAction.action_unfollow,
            TwitterAction.action_mute,
            TwitterAction.action_unmute,
            TwitterAction.action_trend,
        ]
    ]
    assert ACTION_FUNCS is not None
