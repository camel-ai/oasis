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
from __future__ import annotations

import argparse
import asyncio
import csv
import itertools
from pathlib import Path
from typing import Any

from run_experiment import main as run_experiment_main


def _parse_summary_csv(path: Path) -> dict[str, dict[str, float]]:
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out: dict[str, dict[str, float]] = {}
    for row in rows:
        metric = row["metric"]
        out[metric] = {
            "paired_diff_mean": float(row["paired_diff_mean"]),
            "paired_diff_ci95_low": float(row["paired_diff_ci95_low"]),
            "paired_diff_ci95_high": float(row["paired_diff_ci95_high"]),
            "paired_signflip_pvalue": float(row["paired_signflip_pvalue"]),
            "baseline_mean": float(row["baseline_mean"]),
            "treatment_mean": float(row["treatment_mean"]),
        }
    return out


def _config_id(config: dict[str, Any]) -> str:
    return (
        f"wj{config['watch_jitter']:.2f}"
        f"_bf{config['behavior_flip_prob']:.2f}"
        f"_tn{config['treatment_extra_neg_prob']:.2f}"
    )


def _write_batch_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
) -> None:
    csv_path = output_dir / "batch_config_summary.csv"
    fieldnames = [
        "config_id",
        "n_runs",
        "seed_start",
        "strategy_set",
        "creator_multiplier",
        "viewer_multiplier",
        "video_multiplier",
        "random_negative_prob",
        "watch_jitter",
        "behavior_flip_prob",
        "treatment_extra_neg_prob",
        "comedy_pool_diff_mean",
        "comedy_pool_ci95_low",
        "comedy_pool_ci95_high",
        "comedy_pool_pvalue",
        "comedy_negative_diff_mean",
        "comedy_negative_ci95_low",
        "comedy_negative_ci95_high",
        "comedy_negative_pvalue",
        "watch_ratio_diff_mean",
        "watch_ratio_pvalue",
        "shares_diff_mean",
        "shares_pvalue",
        "condition_output_dir",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ranked = sorted(rows, key=lambda r: (r["comedy_pool_pvalue"], r["comedy_pool_diff_mean"]))
    report_lines = [
        "# Batch Experiment Summary",
        "",
        f"- Config count: {len(rows)}",
        "",
        "## Top Configs by comedy pool suppression significance",
        "",
        "| rank | config_id | comedy_pool_diff_mean | comedy_pool_pvalue | comedy_negative_diff_mean | watch_ratio_diff_mean |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for i, row in enumerate(ranked[:10], start=1):
        report_lines.append(
            f"| {i} | {row['config_id']} | {row['comedy_pool_diff_mean']:.4f} | "
            f"{row['comedy_pool_pvalue']:.6f} | {row['comedy_negative_diff_mean']:.4f} | "
            f"{row['watch_ratio_diff_mean']:.6f} |"
        )
    report_lines.append("")
    report_lines.append(f"- Full CSV: `{csv_path}`")
    (output_dir / "batch_report.md").write_text("\n".join(report_lines), encoding="utf-8")


async def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    watch_jitter_values = [float(x) for x in args.watch_jitter_values.split(",")]
    behavior_flip_values = [float(x) for x in args.behavior_flip_values.split(",")]
    treatment_neg_values = [float(x) for x in args.treatment_extra_neg_values.split(",")]

    configs = [
        {
            "watch_jitter": wj,
            "behavior_flip_prob": bf,
            "treatment_extra_neg_prob": tn,
        }
        for (wj, bf, tn) in itertools.product(
            watch_jitter_values,
            behavior_flip_values,
            treatment_neg_values,
        )
    ]

    if args.quick:
        configs = configs[:2]

    rows: list[dict[str, Any]] = []
    for idx, config in enumerate(configs):
        cid = _config_id(config)
        run_out = output_dir / cid
        run_out.mkdir(parents=True, exist_ok=True)

        await run_experiment_main(
            output_dir=str(run_out),
            n_runs=args.n_runs,
            seed_start=args.seed_start + idx * 10000,
            watch_jitter=config["watch_jitter"],
            behavior_flip_prob=config["behavior_flip_prob"],
            treatment_extra_neg_prob=config["treatment_extra_neg_prob"],
            save_all_dbs=args.save_all_dbs,
            strategy_set=args.strategy_set,
            random_negative_prob=args.random_negative_prob,
            creator_multiplier=args.creator_multiplier,
            viewer_multiplier=args.viewer_multiplier,
            video_multiplier=args.video_multiplier,
        )

        summary = _parse_summary_csv(run_out / "multirun_summary.csv")
        comedy_pool = summary["comedy_traffic_pool_level"]
        comedy_neg = summary["comedy_negative_count"]
        watch_ratio = summary["avg_watch_ratio"]
        shares = summary["total_shares"]

        rows.append({
            "config_id": cid,
            "n_runs": args.n_runs,
            "seed_start": args.seed_start + idx * 10000,
            "strategy_set": args.strategy_set,
            "creator_multiplier": args.creator_multiplier,
            "viewer_multiplier": args.viewer_multiplier,
            "video_multiplier": args.video_multiplier,
            "random_negative_prob": args.random_negative_prob,
            "watch_jitter": config["watch_jitter"],
            "behavior_flip_prob": config["behavior_flip_prob"],
            "treatment_extra_neg_prob": config["treatment_extra_neg_prob"],
            "comedy_pool_diff_mean": comedy_pool["paired_diff_mean"],
            "comedy_pool_ci95_low": comedy_pool["paired_diff_ci95_low"],
            "comedy_pool_ci95_high": comedy_pool["paired_diff_ci95_high"],
            "comedy_pool_pvalue": comedy_pool["paired_signflip_pvalue"],
            "comedy_negative_diff_mean": comedy_neg["paired_diff_mean"],
            "comedy_negative_ci95_low": comedy_neg["paired_diff_ci95_low"],
            "comedy_negative_ci95_high": comedy_neg["paired_diff_ci95_high"],
            "comedy_negative_pvalue": comedy_neg["paired_signflip_pvalue"],
            "watch_ratio_diff_mean": watch_ratio["paired_diff_mean"],
            "watch_ratio_pvalue": watch_ratio["paired_signflip_pvalue"],
            "shares_diff_mean": shares["paired_diff_mean"],
            "shares_pvalue": shares["paired_signflip_pvalue"],
            "condition_output_dir": str(run_out),
        })

    _write_batch_outputs(output_dir, rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a batch matrix of short-video negative-feedback experiments and aggregate results.",
    )
    parser.add_argument(
        "--output-dir",
        default="examples/experiment/short_video_negative_feedback/batch_output",
        help="Directory for per-config outputs and batch summary files.",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=30,
        help="Paired baseline/treatment runs per config.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=2026,
        help="Base seed; each config receives a non-overlapping seed block.",
    )
    parser.add_argument(
        "--watch-jitter-values",
        default="0.04,0.08,0.12",
        help="Comma-separated watch_jitter values for grid search.",
    )
    parser.add_argument(
        "--behavior-flip-values",
        default="0.02,0.05",
        help="Comma-separated behavior_flip_prob values for grid search.",
    )
    parser.add_argument(
        "--treatment-extra-neg-values",
        default="0.70,0.85,1.00",
        help="Comma-separated treatment_extra_neg_prob values for grid search.",
    )
    parser.add_argument(
        "--save-all-dbs",
        action="store_true",
        help="Keep all per-run databases for each config.",
    )
    parser.add_argument(
        "--strategy-set",
        choices=["baseline_treatment", "four_arm"],
        default="baseline_treatment",
        help="Experiment arms for each config.",
    )
    parser.add_argument(
        "--random-negative-prob",
        type=float,
        default=0.35,
        help="Per-watch random negative-feedback probability for random arm.",
    )
    parser.add_argument(
        "--creator-multiplier",
        type=int,
        default=1,
        help="Scale factor applied to creator population.",
    )
    parser.add_argument(
        "--viewer-multiplier",
        type=int,
        default=1,
        help="Scale factor applied to viewer population.",
    )
    parser.add_argument(
        "--video-multiplier",
        type=int,
        default=1,
        help="Scale factor applied to video count.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only the first 2 configs for fast smoke validation.",
    )
    cli_args = parser.parse_args()
    asyncio.run(main(cli_args))
