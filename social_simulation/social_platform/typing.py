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
    CREATE_COMMENT = "create_comment"
    LIKE_COMMENT = "like_comment"
    UNLIKE_COMMENT = "unlike_comment"
    DISLIKE_COMMENT = "dislike_comment"
    UNDO_DISLIKE_COMMENT = "undo_dislike_comment"
    DO_NOTHING = "do_nothing"


class RecsysType(Enum):
    TWITTER = "twitter"
    TWHIN = "twhin-bert"
    REDDIT = "reddit"
    RANDOM = "random"
