from typing import List

from camel.functions import OpenAIFunction

from social_agent.agent import TwitterAction


def test_transfer_to_openai_function():
    # 如果这部分不会通过camel的openAIFunction函数raise error，
    # 即正确生成了openai schema，并可以被camel Chatagent调用。
    ACTION_FUNCS: List[OpenAIFunction] = [
        OpenAIFunction(func) for func in [
            TwitterAction.sign_up, TwitterAction.refresh, TwitterAction.
            create_tweet, TwitterAction.like, TwitterAction.unlike,
            TwitterAction.dislike, TwitterAction.undo_dislike,
            TwitterAction.search_tweets, TwitterAction.search_user,
            TwitterAction.follow, TwitterAction.unfollow, TwitterAction.mute,
            TwitterAction.unmute, TwitterAction.trend, TwitterAction.retweet,
            TwitterAction.create_comment, TwitterAction.like_comment,
            TwitterAction.unlike_comment, TwitterAction.dislike_comment,
            TwitterAction.undo_dislike_comment
        ]
    ]
    assert ACTION_FUNCS is not None
