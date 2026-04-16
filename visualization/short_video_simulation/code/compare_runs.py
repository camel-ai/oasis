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

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oasis.analysis.short_video_metrics import get_short_video_time_series_report  # noqa: E402


def _parse_run(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Each run must be in the form label=/path/to/db.db")
    label, db_path = value.split("=", 1)
    return label, db_path


def main(runs: list[tuple[str, str]], output_dir: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    combined_rows = []
    for label, db_path in runs:
        report = get_short_video_time_series_report(db_path)
        for row in report["video_time_series"]:
            combined_rows.append({
                "label": label,
                **row,
            })

    combined_df = pd.DataFrame(combined_rows)
    combined_df.to_csv(output_path / "comparison_time_series.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metrics = [
        ("cumulative_views", "Cumulative Views"),
        ("avg_watch_ratio", "Average Watch Ratio"),
        ("avg_traffic_pool_level", "Avg Traffic Pool Level"),
    ]
    for ax, (metric, title) in zip(axes, metrics):
        for label, group in combined_df.groupby("label"):
            ax.plot(group["step"], group[metric], marker="o", label=label)
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_path / "comparison_metrics.png",
                dpi=300,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare multiple short-video simulation runs.")
    parser.add_argument(
        "runs",
        nargs="+",
        type=_parse_run,
        help="Runs in the form label=/path/to/db.db",
    )
    parser.add_argument(
        "--output-dir",
        default="visualization/short_video_simulation/output/comparison",
        help="Directory for comparison CSV and PNG outputs",
    )
    args = parser.parse_args()
    main(args.runs, args.output_dir)
