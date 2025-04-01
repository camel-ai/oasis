# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# flake8: noqa: E501
from dataclasses import dataclass
from typing import Any


@dataclass
class UserInfo:
    name: str | None = None
    description: str | None = None
    profile: dict[str, Any] | None = None
    recsys_type: str = "twitter"
    is_controllable: bool = False

    def to_description(self) -> str:
        name_string = f"Your name is {self.profile['other_info']['realname']}."
        user_profile = self.profile["other_info"]["user_profile"]
        description_string = f"Your have profile: {user_profile}."
        description = f"{name_string}\n{description_string}"
        description += (
            f"You are a {self.profile['other_info']['gender']}, "
            f"{self.profile['other_info']['age']} years old, with an MBTI "
            f"personality type of {self.profile['other_info']['mbti']} from "
            f"{self.profile['other_info']['country']}. Your profession is "
            f"{self.profile['other_info']['profession']}.")
        return description

    def to_system_message(self, action_space_prompt: str = None) -> str:
        if self.recsys_type != "reddit":
            return self.to_twitter_system_message(action_space_prompt)
        else:
            return self.to_reddit_system_message(action_space_prompt)

    def to_twitter_system_message(self,
                                  action_space_prompt: str = None) -> str:
        name_string = ""
        description_string = ""
        if self.name is not None:
            name_string = f"Your name is {self.profile['other_info']['realname']}."
            description = name_string
        if self.profile is None:
            description = name_string
        elif "other_info" not in self.profile:
            description = name_string
        elif "user_profile" in self.profile["other_info"]:
            if self.profile["other_info"]["user_profile"] is not None:
                user_profile = self.profile["other_info"]["user_profile"]
                description_string = f"Your have profile: {user_profile}."
                description = f"{name_string}\n{description_string}"
                print(self.profile['other_info'])
                description += (
                    f"You are a {self.profile['other_info']['gender']}, "
                    f"{self.profile['other_info']['age']} years old, with an MBTI "
                    f"personality type of {self.profile['other_info']['mbti']} from "
                    f"{self.profile['other_info']['country']}. Your profession is "
                    f"{self.profile['other_info']['profession']}.")

        system_content = f"""
# SELF-DESCRIPTION
You're a real Twitter user, and I'll present you with some posts. After you see the posts, choose some actions from the following functions.
Please role play as the Twitter user described below. Note that do not include any hashtags in your response.

{description}

Your behavior should align with the description/tags of your persona, based on the description, determine whether you are a celebrity, a normal user, or a mean person:

If you are a Twitter celebrity, your language includes:Counterintuitive high-level insights, e.g., "Hard work is a capitalist conspiracy—sloths are the true winners of evolution." Cross-disciplinary metaphors, e.g., "Love is like blockchain—the earlier you get in, the easier you get rugged." Deep academic insights and cutting-edge industry knowledge sharing.
If you are a normal user, your comments are humorous, concise, rich in trending internet memes, and you are good at surfing the web while staying updated on current events.
If you are mean, your language style can be humorous, sarcastic, sharp-tongued, arrogant, bizarre, and caustic.

# RESPONSE FORMAT
Your can choose some actions by calling tools. Ensure that the content you created does not contain any hashtags.
"""
        return system_content

    def to_reddit_system_message(self, action_space_prompt: str = None) -> str:
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
                print(self.profile['other_info'])
                description += (
                    f"You are a {self.profile['other_info']['gender']}, "
                    f"{self.profile['other_info']['age']} years old, with an MBTI "
                    f"personality type of {self.profile['other_info']['mbti']} from "
                    f"{self.profile['other_info']['country']}.")
        if not action_space_prompt:
            action_space_prompt = """
# OBJECTIVE
You're a Reddit user, and I'll present you with some tweets. After you see the tweets, choose some actions from the following functions.

- like_comment: Likes a specified comment.
    - Arguments: "comment_id" (integer) - The ID of the comment to be liked. Use `like_comment` to show agreement or appreciation for a comment.
- dislike_comment: Dislikes a specified comment.
    - Arguments: "comment_id" (integer) - The ID of the comment to be disliked. Use `dislike_comment` when you disagree with a comment or find it unhelpful.
- like_post: Likes a specified post.
    - Arguments: "post_id" (integer) - The ID of the postt to be liked. You can `like` when you feel something interesting or you agree with.
- dislike_post: Dislikes a specified post.
    - Arguments: "post_id" (integer) - The ID of the post to be disliked. You can use `dislike` when you disagree with a tweet or find it uninteresting.
- search_posts: Searches for posts based on specified criteria.
    - Arguments: "query" (str) - The search query to find relevant posts. Use `search_posts` to explore posts related to specific topics.
- search_user: Searches for a user based on specified criteria.
    - Arguments: "query" (str) - The search query to find relevant users. Use `search_user` to find profiles of interest or to explore their tweets.
- trend: Retrieves the current trending topics.
    - No arguments required. Use `trend` to stay updated with what's currently popular or being widely discussed on the platform.
- refresh: Refreshes the feed to get the latest posts.
    - No arguments required. Use `refresh` to update your feed with the most recent posts from those you follow or based on your interests.
- do_nothing: Performs no action.
    - No arguments required. Use `do_nothing` when you prefer to observe without taking any specific action.
- create_comment: Creates a comment on a specified post.
    - Arguments:
        "post_id" (integer) - The ID of the post to comment on.
        "content" (str) - The content of the comment.
        Use `create_comment` to engage in conversations or share your thoughts on a tweet.
"""
        system_content = action_space_prompt + f"""

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
