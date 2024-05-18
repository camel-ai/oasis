# File: ./test/test_agent/test_action_docstring.py

from camel.functions import OpenAIFunction
from social_agent.twitterUserAgent import TwitterUserAgent
from typing import List


def test_transfer_to_openai_function():
    # 如果这部分不会通过camel的openAIFunction函数raise error，
    # 即正确生成了openai schema，并可以被camel Chatagent调用。
    ACTION_FUNCS: List[OpenAIFunction] = [
        OpenAIFunction(func)
        for func in [
            TwitterUserAgent.action_sign_up,
            TwitterUserAgent.action_refresh,
            TwitterUserAgent.action_create_tweet,
            TwitterUserAgent.action_like,
            TwitterUserAgent.action_unlike,
            TwitterUserAgent.action_search_tweets,
            TwitterUserAgent.action_search_user,
            TwitterUserAgent.action_follow,
            TwitterUserAgent.action_unfollow,
            TwitterUserAgent.action_mute,
            TwitterUserAgent.action_unmute,
            TwitterUserAgent.action_trend,
        ]
    ]
    assert ACTION_FUNCS is not None
