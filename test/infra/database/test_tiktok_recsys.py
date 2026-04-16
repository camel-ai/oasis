# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
import os
import os.path as osp
import sqlite3

import pytest

from oasis.social_platform.platform import Platform

parent_folder = osp.dirname(osp.abspath(__file__))
test_db_filepath = osp.join(parent_folder, "test_tiktok_recsys.db")


@pytest.mark.asyncio
async def test_tiktok_recommendation_uses_watch_completion_and_negative_feedback():
    conn = None
    try:
        if os.path.exists(test_db_filepath):
            os.remove(test_db_filepath)

        platform = Platform(
            test_db_filepath,
            recsys_type="tiktok",
            refresh_rec_post_count=2,
            max_rec_post_len=2,
        )
        platform.tiktok_recsys_params = {
            "promote_percentile": 0.5,
            "demote_percentile": 0.5,
            "max_pool_level": 3,
        }

        await platform.sign_up(0, ("dance_creator", "Dance Creator",
                                   "Dance creator [interests: dance]"))
        await platform.sign_up(1, ("tech_creator", "Tech Creator",
                                   "Tech creator [interests: tech]"))
        await platform.sign_up(2, ("viewer", "Viewer",
                                   "Viewer [interests: dance]"))

        first_upload = await platform.upload_video(
            0, ("Dance challenge", 15, "dance", '["dance"]', 0.8, 0.9))
        second_upload = await platform.upload_video(
            1, ("Tech explainer", 15, "tech", '["tech"]', 0.8, 0.9))

        first_post_id = first_upload["post_id"]
        second_post_id = second_upload["post_id"]

        await platform.watch_video(2, (first_post_id, 1.0))
        await platform.share_video(2, first_post_id)
        await platform.watch_video(2, (second_post_id, 0.1))
        await platform.not_interested(2, second_post_id)

        await platform.update_rec_table()

        conn = sqlite3.connect(test_db_filepath)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT post_id, traffic_pool_level FROM video ORDER BY post_id")
        traffic_levels = dict(cursor.fetchall())
        assert traffic_levels[first_post_id] == 2
        assert traffic_levels[second_post_id] == 0

        cursor.execute(
            "SELECT post_id FROM rec WHERE user_id = ? ORDER BY post_id",
            (2,),
        )
        recommended_posts = [row[0] for row in cursor.fetchall()]
        assert first_post_id in recommended_posts
        assert second_post_id not in recommended_posts

    finally:
        if conn is not None:
            conn.close()
        if os.path.exists(test_db_filepath):
            os.remove(test_db_filepath)
