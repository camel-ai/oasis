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
import os
import os.path as osp

import pytest

from oasis import (get_short_video_observability_report,
                   get_short_video_time_series_report)
from oasis.social_platform.platform import Platform

parent_folder = osp.dirname(osp.abspath(__file__))
test_db_filepath = osp.join(parent_folder, "test_short_video_metrics.db")


@pytest.mark.asyncio
async def test_short_video_observability_report_summarizes_simulation():
    try:
        if os.path.exists(test_db_filepath):
            os.remove(test_db_filepath)

        platform = Platform(test_db_filepath, recsys_type="tiktok")
        await platform.sign_up(0, ("creator1", "Creator One",
                                   "Creator [interests: dance]"))
        await platform.sign_up(1, ("viewer1", "Viewer One",
                                   "Viewer [interests: dance]"))

        upload_result = await platform.upload_video(
            0, ("Dance video", 15, "dance", '["dance"]', 0.9, 0.8))
        post_id = upload_result["post_id"]

        await platform.watch_video(1, (post_id, 0.75))
        await platform.share_video(1, post_id)
        await platform.create_comment(1, (post_id, "Great dance video"))

        stream = await platform.start_livestream(0)
        await platform.enter_livestream(1, stream["stream_id"])
        await platform.send_gift(1, (stream["stream_id"], 20.0))
        await platform.exit_livestream(1, stream["stream_id"])
        await platform.end_livestream(0, stream["stream_id"])

        report = get_short_video_observability_report(test_db_filepath)

        assert report["summary"]["total_videos"] == 1
        assert report["summary"]["total_views"] == 1
        assert report["summary"]["total_comments"] == 1
        assert report["summary"]["total_shares"] == 1
        assert report["summary"]["avg_watch_ratio"] == 0.75
        assert report["summary"]["retention_3s_rate"] == 1.0
        assert report["summary"]["negative_feedback_rate"] == 0.0
        assert report["summary"]["creator_coverage"] == 1.0

        assert len(report["creators"]) == 1
        assert report["creators"][0]["user_name"] == "creator1"
        assert report["creators"][0]["total_views"] == 1

        assert len(report["top_videos"]) == 1
        assert report["top_videos"][0]["post_id"] == post_id
        assert report["top_videos"][0]["comment_count"] == 1
        assert report["top_videos"][0]["avg_watch_ratio"] == 0.75

        assert len(report["livestreams"]) == 1
        assert report["livestreams"][0]["total_comments"] == 0
        assert report["livestreams"][0]["total_gifts_value"] == 20.0
        assert report["livestreams"][0]["avg_interactions"] >= 1.0

        time_series = get_short_video_time_series_report(test_db_filepath)
        assert len(time_series["video_time_series"]) == 1
        assert time_series["video_time_series"][0]["uploaded_videos"] == 1
        assert time_series["video_time_series"][0]["cumulative_views"] == 1
        assert len(time_series["creator_growth"]) == 1
        assert time_series["creator_growth"][0]["creator_id"] == 0
        assert len(time_series["livestream_time_series"]) == 1
        assert (time_series["livestream_time_series"][0]
                ["livestreams_started"] == 1)
        assert len(time_series["viewer_retention_time_series"]) == 1
        assert (time_series["viewer_retention_time_series"][0]
                ["viewer_sessions"]) == 1

    finally:
        if os.path.exists(test_db_filepath):
            os.remove(test_db_filepath)
