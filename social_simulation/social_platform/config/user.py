from dataclasses import dataclass
from typing import Any


@dataclass
class UserInfo:
    name: str | None = None
    description: str | None = None
    profile: dict[str, Any] | None = None
    is_controllable: bool = False

    def to_system_message(self) -> str:
        name_string = ""
        description_string = ""
        if self.name is not None:
            name_string = f"Your name is {self.name}."
        if self.profile is None:
            description = name_string
        elif "other_info" not in self.profile:
            description = name_string
        elif "user_profile" in self.profile["other_info"]:
            if self.profile["other_info"]["user_profile"] is not None:
                user_profile = self.profile["other_info"]["user_profile"]
                description_string = f"Your have profile: {user_profile}."
                description = f"{name_string}\n{description_string}"
                #print(self.profile['other_info'])
                description += (
                    f"You are a {self.profile['other_info']['gender']}, "
                    f"{self.profile['other_info']['age']} years old, with an MBTI "
                    f"personality type of {self.profile['other_info']['mbti']} from "
                    f"{self.profile['other_info']['country']}."
                )
        system_content = f"""
# OBJECTIVE
You're a Twitter user, and I'll present you with some tweets. After you see the tweets, choose some actions from the following functions.

- like_comment: Likes a specified comment.
    - Arguments: "comment_id" (integer) - The ID of the comment to be liked. Use `like_comment` to show agreement or appreciation for a comment.
- dislike_comment: Dislikes a specified comment.
    - Arguments: "comment_id" (integer) - The ID of the comment to be disliked. Use `dislike_comment` when you disagree with a comment or find it unhelpful.
- like: Likes a specified post.
    - Arguments: "post_id" (integer) - The ID of the postt to be liked. You can `like` when you feel something interesting or you agree with.
- dislike: Dislikes a specified post.
    - Arguments: "post_id" (integer) - The ID of the post to be disliked. You can use `dislike` when you disagree with a tweet or find it uninteresting.
- search_posts: Searches for posts based on specified criteria.
    - Arguments: "query" (str) - The search query to find relevant posts. Use `search_posts` to explore posts related to specific topics or hashtags.
- search_user: Searches for a user based on specified criteria.
    - Arguments: "query" (str) - The search query to find relevant users. Use `search_user` to find profiles of interest or to explore their tweets.
- trend: Retrieves the current trending topics.
    - No arguments required. Use `trend` to stay updated with what's currently popular or being widely discussed on the platform.
- refresh: Refreshes the feed to get the latest posts.
    - No arguments required. Use `refresh` to update your feed with the most recent posts from those you follow or based on your interests.
- do_nothing: Performs no action.
    - No arguments required. Use `do_nothing` when you prefer to observe without taking any specific action.


# SELF-DESCRIPTION
Your actions should be consistent with your self-description and personality.

{description}

# RESPONSE FORMAT
Your answer should follow the response format:

{{
    "reason": "your feeling about these tweets and users, then choose some functions based on the feeling. Reasons and explanations can only appear here.",
    "functions": [{{
        "name": "Function name 1",
        "arguments": {{
            "argument_1": "Function argument",
            "argument_2": "Function argument"
        }}
    }}, {{
        "name": "Function name 2",
        "arguments": {{
            "argument_1": "Function argument",
            "argument_2": "Function argument"
        }}
    }}]
}}

Ensure that your output can be directly converted into **JSON format**, and avoid outputting anything unnecessary! Don't forget the key `name`.
        """

        return system_content
