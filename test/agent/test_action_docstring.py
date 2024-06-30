from typing import List

from camel.functions import OpenAIFunction

from social_simulation.social_agent.agent import SocialAction


def test_transfer_to_openai_function():
    action_funcs: List[OpenAIFunction] = [
        OpenAIFunction(func) for func in [
            SocialAction.sign_up, SocialAction.refresh,
            SocialAction.create_tweet, SocialAction.like, SocialAction.unlike,
            SocialAction.dislike, SocialAction.undo_dislike,
            SocialAction.search_tweets, SocialAction.search_user,
            SocialAction.follow, SocialAction.unfollow, SocialAction.mute,
            SocialAction.unmute, SocialAction.trend, SocialAction.retweet,
            SocialAction.create_comment, SocialAction.like_comment,
            SocialAction.unlike_comment, SocialAction.dislike_comment,
            SocialAction.undo_dislike_comment, SocialAction.do_nothing
        ]
    ]
    assert action_funcs is not None
