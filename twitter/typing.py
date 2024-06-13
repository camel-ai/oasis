# # File: SandboxTwitterModule/twitter_infra_core/typing.py
from enum import Enum


class ActionType(Enum):
    EXIT = "exit"
    REFRESH = "refresh"
    SEARCH_USER = "search_user"
    SEARCH_TWEET = "search_tweets"
    CREATE_TWEET = "create_tweet"
    LIKE = "like"
    UNLIKE = "unlike"
    DISLIKE = "dislike"
    UNDO_DISLIKE = "undo_dislike"
    FOLLOW = "follow"
    UNFOLLOW = "unfollow"
    MUTE = "mute"
    UNMUTE = "unmute"
    TREND = "trend"
    SIGNUP = "sign_up"
    RETWEET = "retweet"
    UPDATE_REC = "update_rec"
