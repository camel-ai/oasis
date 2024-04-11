from enum import Enum

class PlatformAction(Enum):
    FOLLOW = "follow"
    UNFOLLOW = "unfollow"
    POST = "post"
    REPLY = "reply"
    LIKE = "like"
    UNLIKE = "unlike"
    SHARE = "share"
    DELETE = "delete"

class PlatformMessage(Enum):
    FOLLOW = {"name": str, "id": int}
    UNFOLLOW = {"name": str, "id": int}
    POST = {"content": str}
    REPLY = {"content": str, "reply_to": int}
    LIKE = {"id": int}
    UNLIKE = {"id": int}
    SHARE = {"id": int}
    DELETE = {"id": int}


class PlatformMessage:
    sender_name: str
    sender_id: int
    type: PlatformAction
    content: PlatformMessage

