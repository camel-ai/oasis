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
from typing import List

from camel.toolkits import FunctionTool

from oasis.social_agent.agent import SocialAction


def test_transfer_to_openai_function():
    action_funcs: List[FunctionTool] = [
        FunctionTool(func) for func in [
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
            SocialAction.purchase_product,
            SocialAction.report_post,
            SocialAction.join_group,
            SocialAction.leave_group,
            SocialAction.listen_from_group,
            SocialAction.send_to_group,
            SocialAction.create_group,
        ]
    ]
    assert action_funcs is not None
