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
import sys
import tempfile
from pathlib import Path

_MPL_CONFIG_DIR = Path(tempfile.gettempdir()) / "oasis-mpl-cache"
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oasis.analysis.short_video_metrics import (  # noqa: E402
    get_short_video_observability_report, get_short_video_time_series_report)


def _plot_video_metrics(time_series: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(time_series["step"], time_series["cumulative_views"])
    axes[0, 0].set_title("Cumulative Views")
    axes[0, 0].set_xlabel("Step")

    axes[0, 1].plot(time_series["step"], time_series["avg_watch_ratio"])
    axes[0, 1].set_title("Average Watch Ratio")
    axes[0, 1].set_xlabel("Step")

    axes[1, 0].plot(time_series["step"], time_series["avg_traffic_pool_level"])
    axes[1, 0].set_title("Average Traffic Pool Level")
    axes[1, 0].set_xlabel("Step")

    axes[1, 1].plot(
        time_series["step"],
        time_series["cumulative_negative_feedback"],
    )
    axes[1, 1].set_title("Cumulative Negative Feedback")
    axes[1, 1].set_xlabel("Step")

    fig.tight_layout()
    fig.savefig(output_dir / "short_video_feed_metrics.png",
                dpi=300,
                bbox_inches="tight")
    plt.close(fig)


def _plot_livestream_metrics(livestreams: pd.DataFrame,
                             retention: pd.DataFrame,
                             output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if not livestreams.empty:
        axes[0].plot(livestreams["step"], livestreams["avg_peak_viewers"])
        axes[0].set_title("Average Livestream Peak Viewers")
        axes[0].set_xlabel("Step")
    else:
        axes[0].text(0.5, 0.5, "No livestream data", ha="center", va="center")
        axes[0].set_axis_off()

    if not retention.empty:
        axes[1].plot(retention["step"], retention["avg_stay_seconds"])
        axes[1].set_title("Average Viewer Stay Seconds")
        axes[1].set_xlabel("Step")
    else:
        axes[1].text(0.5, 0.5, "No retention data", ha="center", va="center")
        axes[1].set_axis_off()

    fig.tight_layout()
    fig.savefig(output_dir / "short_video_livestream_metrics.png",
                dpi=300,
                bbox_inches="tight")
    plt.close(fig)


def main(db_path: str, output_dir: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    observability = get_short_video_observability_report(db_path)
    time_series = get_short_video_time_series_report(db_path)

    pd.DataFrame(observability["creators"]).to_csv(
        output_path / "creator_metrics.csv", index=False)
    pd.DataFrame(observability["top_videos"]).to_csv(
        output_path / "top_video_metrics.csv", index=False)
    pd.DataFrame(time_series["video_time_series"]).to_csv(
        output_path / "video_time_series.csv", index=False)
    pd.DataFrame(time_series["viewer_retention_time_series"]).to_csv(
        output_path / "viewer_retention_time_series.csv", index=False)

    video_df = pd.DataFrame(time_series["video_time_series"])
    livestream_df = pd.DataFrame(time_series["livestream_time_series"])
    retention_df = pd.DataFrame(time_series["viewer_retention_time_series"])

    if not video_df.empty:
        _plot_video_metrics(video_df, output_path)
    _plot_livestream_metrics(livestream_df, retention_df, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot short-video simulation observability metrics.")
    parser.add_argument("db_path",
                        help="Path to the simulation SQLite database")
    parser.add_argument(
        "--output-dir",
        default="visualization/short_video_simulation/output",
        help="Directory for CSV and PNG outputs",
    )
    args = parser.parse_args()
    main(args.db_path, args.output_dir)
