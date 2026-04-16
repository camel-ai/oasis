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
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
import tempfile

_MPL_CONFIG_DIR = Path(tempfile.gettempdir()) / "oasis-mpl-cache"
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oasis.analysis.short_video_metrics import (  # noqa: E402
    get_short_video_observability_report,
    get_short_video_time_series_report,
)


def _format_table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "_No data available._\n"
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(str(row.get(col, "")) for col in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, divider] + body) + "\n"


def generate_markdown_report(db_path: str) -> str:
    observability = get_short_video_observability_report(db_path)
    time_series = get_short_video_time_series_report(db_path)

    summary = observability["summary"]
    creators = observability["creators"][:5]
    top_videos = observability["top_videos"][:5]
    livestreams = observability["livestreams"][:5]
    latest_video_step = (
        time_series["video_time_series"][-1]
        if time_series["video_time_series"] else {}
    )

    return f"""# Short-Video Simulation Report

## Summary

- Total videos: {summary["total_videos"]}
- Total creators: {summary["total_creators"]}
- Total views: {summary["total_views"]}
- Total comments: {summary["total_comments"]}
- Total shares: {summary["total_shares"]}
- Total negative feedback: {summary["total_negative_feedback"]}
- Average watch ratio: {summary["avg_watch_ratio"]:.3f}
- Average traffic pool level: {summary["avg_traffic_pool_level"]:.3f}

## Top Creators

{_format_table(creators, ["user_id", "user_name", "uploaded_videos", "total_views", "total_shares", "avg_watch_ratio"])}
## Top Videos

{_format_table(top_videos, ["post_id", "creator_id", "category", "view_count", "avg_watch_ratio", "share_count", "negative_count", "traffic_pool_level"])}
## Livestream Overview

{_format_table(livestreams, ["stream_id", "host_id", "status", "peak_viewers", "total_comments", "total_gifts_value", "avg_stay_seconds", "avg_interactions"])}
## Latest Feed Step Snapshot

{_format_table([latest_video_step] if latest_video_step else [], ["step", "uploaded_videos", "cumulative_views", "cumulative_shares", "cumulative_negative_feedback", "avg_watch_ratio", "avg_traffic_pool_level"])}
"""


def main(db_path: str, output_path: str) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(generate_markdown_report(db_path), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a markdown report for a short-video simulation.")
    parser.add_argument("db_path", help="Path to the simulation SQLite database")
    parser.add_argument(
        "--output",
        default="visualization/short_video_simulation/output/report.md",
        help="Markdown output path",
    )
    args = parser.parse_args()
    main(args.db_path, args.output)
