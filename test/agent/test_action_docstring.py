from typing import List

from camel.toolkits import OpenAIFunction

from social_simulation.social_agent.agent import SocialAction


def test_transfer_to_openai_function():
    action_funcs: List[OpenAIFunction] = [
        OpenAIFunction(func) for func in [
            SocialAction.sign_up,
            SocialAction.refresh,
            SocialAction.create_post,
            SocialAction.like_post,
            SocialAction.unlike_post,
            SocialAction.dislike_post,
            SocialAction.undo_dislike_post,
            SocialAction.search_posts,
            SocialAction.search_user,
            SocialAction.follow,
            SocialAction.unfollow,
            SocialAction.mute,
            SocialAction.unmute,
            SocialAction.trend,
            SocialAction.repost,
            SocialAction.create_comment,
            SocialAction.like_comment,
            SocialAction.unlike_comment,
            SocialAction.dislike_comment,
            SocialAction.undo_dislike_comment,
            SocialAction.do_nothing,
        ]
    ]
    assert action_funcs is not None
