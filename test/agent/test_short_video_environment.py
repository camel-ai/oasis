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

from oasis.social_agent.agent_environment import SocialEnvironment

parent_folder = osp.dirname(osp.abspath(__file__))
test_db_filepath = osp.join(parent_folder, "test_short_video_env.db")


class MockShortVideoAction:

    agent_id = 1

    async def refresh(self):
        return {
            "success": True,
            "posts": [{
                "post_id": 1,
                "content": "Dance challenge",
                "traffic_pool_level": 2,
                "view_count": 12,
                "avg_watch_ratio": 0.85,
                "content_format": "short_video",
            }]
        }

    async def listen_from_group(self):
        return {"success": False}


@pytest.mark.asyncio
async def test_short_video_environment_prompt_includes_feed_and_livestreams(
    monkeypatch,
):
    try:
        if os.path.exists(test_db_filepath):
            os.remove(test_db_filepath)

        conn = sqlite3.connect(test_db_filepath)
        cursor = conn.cursor()
        cursor.execute(
            "CREATE TABLE livestream ("
            "stream_id INTEGER PRIMARY KEY, host_id INTEGER, status TEXT, "
            "start_time DATETIME, end_time DATETIME, current_viewers INTEGER, "
            "peak_viewers INTEGER, total_viewers INTEGER, total_comments INTEGER, "
            "total_likes INTEGER, total_gifts_value REAL)"
        )
        cursor.execute(
            "INSERT INTO livestream VALUES "
            "(1, 7, 'live', 0, NULL, 15, 20, 30, 4, 0, 12.5)"
        )
        conn.commit()
        conn.close()

        monkeypatch.setenv("OASIS_DB_PATH", test_db_filepath)

        env = SocialEnvironment(MockShortVideoAction())
        prompt = await env.to_text_prompt()

        assert "short-video feed" in prompt
        assert "watch_video" in prompt
        assert "active livestream rooms" in prompt
        assert "traffic_pool_level" in prompt
        assert "avg_watch_ratio" in prompt

    finally:
        if os.path.exists(test_db_filepath):
            os.remove(test_db_filepath)
