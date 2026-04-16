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
"""Tests for TikTok platform actions: upload_video, watch_video,
share_video, not_interested, start/enter/exit_livestream, send_gift."""

import os
import os.path as osp
import sqlite3

import pytest

from oasis.social_platform.platform import Platform
from oasis.social_platform.typing import ActionType

parent_folder = osp.dirname(osp.abspath(__file__))
test_db_filepath = osp.join(parent_folder, "test_tiktok.db")


class MockTikTokChannel:
    """Mock channel that exercises core TikTok actions."""

    def __init__(self):
        self.call_count = 0
        self.messages = []

    async def receive_from(self):
        actions = [
            # Sign up two users
            ("id_0", (0, ("creator1", "Creator One", "A TikTok creator"),
                      "sign_up")),
            ("id_1", (1, ("viewer1", "Viewer One", "A TikTok viewer"),
                      "sign_up")),
            # Creator uploads a video
            ("id_2", (0, ("Dance video #trending", 15, "dance", '["dance"]',
                          0.8, 0.9), "upload_video")),
            # Viewer watches the video (80% completion)
            ("id_3", (1, (1, 0.8), "watch_video")),
            # Viewer shares the video
            ("id_4", (1, 1, "share_video")),
            # Viewer marks not interested on a different context
            ("id_5", (1, 1, "not_interested")),
            # Creator starts a livestream
            ("id_6", (0, None, "start_livestream")),
            # Viewer enters the livestream
            ("id_7", (1, 1, "enter_livestream")),
            # Viewer sends a gift
            ("id_8", (1, (1, 50.0), "send_gift")),
            # Viewer exits the livestream
            ("id_9", (1, 1, "exit_livestream")),
            # Creator ends the livestream
            ("id_10", (0, 1, "end_livestream")),
            # Exit
            ("id_exit", (None, None, "exit")),
        ]
        if self.call_count < len(actions):
            action = actions[self.call_count]
            self.call_count += 1
            return action
        return ("id_exit", (None, None, "exit"))

    async def send_to(self, message):
        self.messages.append(message)


@pytest.fixture
def setup_tiktok_platform():
    if os.path.exists(test_db_filepath):
        os.remove(test_db_filepath)
    mock_channel = MockTikTokChannel()
    instance = Platform(test_db_filepath, mock_channel,
                        recsys_type="tiktok")
    return instance


@pytest.mark.asyncio
async def test_tiktok_upload_and_watch(setup_tiktok_platform):
    try:
        platform = setup_tiktok_platform
        await platform.running()

        conn = sqlite3.connect(test_db_filepath)
        cursor = conn.cursor()

        # Verify users created
        cursor.execute("SELECT * FROM user")
        users = cursor.fetchall()
        assert len(users) == 2

        # Verify video uploaded (post + video table)
        cursor.execute("SELECT * FROM post")
        posts = cursor.fetchall()
        assert len(posts) == 1
        assert posts[0][1] == 0  # creator user_id

        cursor.execute("SELECT * FROM video")
        videos = cursor.fetchall()
        assert len(videos) == 1
        # Schema: post_id=0, duration=1, category=2, topic_tags=3,
        # quality_score=4, hook_strength=5, traffic_pool_level=6,
        # pool_enter_time=7, total_impressions=8, view_count=9,
        # total_watch_ratio=10, share_count=11, negative_count=12
        assert videos[0][1] == 15  # duration_seconds
        assert videos[0][2] == "dance"  # category
        assert videos[0][6] == 1  # traffic_pool_level = 1

        # Verify watch recorded
        cursor.execute(
            "SELECT * FROM video WHERE post_id = 1")
        video = cursor.fetchone()
        assert video[9] == 1  # view_count
        assert video[10] == 0.8  # total_watch_ratio

        # Verify share recorded
        cursor.execute("SELECT num_shares FROM post WHERE post_id = 1")
        shares = cursor.fetchone()[0]
        assert shares == 1

        # Verify not_interested recorded
        cursor.execute(
            "SELECT negative_count FROM video WHERE post_id = 1")
        negative = cursor.fetchone()[0]
        assert negative == 1

        # Verify livestream created and ended
        cursor.execute("SELECT * FROM livestream")
        streams = cursor.fetchall()
        assert len(streams) == 1
        assert streams[0][1] == 0  # host_id
        assert streams[0][2] == "ended"  # status
        assert streams[0][6] == 1  # total_viewers

        # Verify gift recorded
        cursor.execute(
            "SELECT total_gifts_value FROM livestream WHERE stream_id = 1")
        gifts = cursor.fetchone()[0]
        assert gifts == 50.0

        # Verify viewer session recorded with exit
        cursor.execute("SELECT * FROM livestream_viewer")
        viewers = cursor.fetchall()
        assert len(viewers) == 1
        assert viewers[0][2] == 1  # viewer_id
        assert viewers[0][4] is not None  # exit_time set

        # Verify traces recorded for all TikTok actions
        cursor.execute("SELECT DISTINCT action FROM trace")
        actions = {row[0] for row in cursor.fetchall()}
        expected = {
            "sign_up", "upload_video", "watch_video", "share_video",
            "not_interested", "start_livestream", "enter_livestream",
            "send_gift", "exit_livestream", "end_livestream",
        }
        assert expected.issubset(actions), (
            f"Missing traces: {expected - actions}")

    finally:
        conn.close()
        if os.path.exists(test_db_filepath):
            os.remove(test_db_filepath)
