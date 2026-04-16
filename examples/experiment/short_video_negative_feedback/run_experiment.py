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
import logging
import math
import os
from pathlib import Path
import random
import sqlite3
import statistics
import sys
import tempfile
from typing import Any

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
)
from oasis.social_platform.platform import Platform  # noqa: E402


RECSYS_DEFAULT_PARAMS = {
    "score_weights": {
        "completion_rate": 0.40,
        "like_rate": 0.10,
        "comment_rate": 0.10,
        "share_rate": 0.20,
        "negative_rate": 0.20,
    },
    "promote_percentile": 0.25,
    "demote_percentile": 0.75,
    "max_pool_level": 5,
    "rec_mix": {
        "interest": 0.80,
        "following": 0.10,
        "explore": 0.10,
    },
    "decay_half_life": 72,
}

RECSYS_NEGATIVE_SENSITIVE_PARAMS = {
    "score_weights": {
        "completion_rate": 0.25,
        "like_rate": 0.08,
        "comment_rate": 0.07,
        "share_rate": 0.15,
        "negative_rate": 0.45,
    },
    "promote_percentile": 0.20,
    "demote_percentile": 0.55,
    "max_pool_level": 5,
    "rec_mix": {
        "interest": 0.80,
        "following": 0.10,
        "explore": 0.10,
    },
    "decay_half_life": 72,
}

RECSYS_BALANCED_SENSITIVE_PARAMS = {
    "score_weights": {
        "completion_rate": 0.30,
        "like_rate": 0.10,
        "comment_rate": 0.10,
        "share_rate": 0.18,
        "negative_rate": 0.32,
    },
    "promote_percentile": 0.22,
    "demote_percentile": 0.68,
    "max_pool_level": 5,
    "rec_mix": {
        "interest": 0.80,
        "following": 0.10,
        "explore": 0.10,
    },
    "decay_half_life": 72,
}

RECSYS_PROFILES = {
    "default": RECSYS_DEFAULT_PARAMS,
    "balanced_sensitive": RECSYS_BALANCED_SENSITIVE_PARAMS,
    "negative_sensitive": RECSYS_NEGATIVE_SENSITIVE_PARAMS,
}

CREATORS = [
    (0, "dance_lab", "Dance Lab", "Dance creator [interests: dance, music]"),
    (1, "tech_loop", "Tech Loop", "Tech creator [interests: tech, gadgets]"),
    (2, "prank_box", "Prank Box", "Prank creator [interests: comedy, prank]"),
]

VIEWERS = [
    (10, "viewer_dance_1", "Viewer Dance 1", "Viewer [interests: dance]"),
    (11, "viewer_dance_2", "Viewer Dance 2", "Viewer [interests: dance]"),
    (12, "viewer_tech_1", "Viewer Tech 1", "Viewer [interests: tech]"),
    (13, "viewer_tech_2", "Viewer Tech 2", "Viewer [interests: tech]"),
    (14, "viewer_fun_1", "Viewer Fun 1", "Viewer [interests: comedy]"),
    (15, "viewer_fun_2", "Viewer Fun 2", "Viewer [interests: comedy]"),
]

VIDEO_BLUEPRINTS = [
    {
        "creator_id": 0,
        "content": "Dance challenge with a strong opening hook",
        "duration_seconds": 18,
        "category": "dance",
        "topic_tags": '["dance", "challenge"]',
        "quality_score": 0.92,
        "hook_strength": 0.91,
        "viewer_plan": [
            (10, 0.98, True, True, False),
            (11, 0.93, True, True, False),
            (12, 0.78, False, False, False),
            (14, 0.81, False, False, False),
        ],
    },
    {
        "creator_id": 1,
        "content": "Quick gadget explainers with clean edits",
        "duration_seconds": 22,
        "category": "tech",
        "topic_tags": '["tech", "gadgets"]',
        "quality_score": 0.86,
        "hook_strength": 0.82,
        "viewer_plan": [
            (12, 0.94, True, True, False),
            (13, 0.91, True, False, False),
            (10, 0.61, False, False, False),
            (11, 0.58, False, False, False),
        ],
    },
    {
        "creator_id": 2,
        "content": "Polarizing prank bait with uneven payoff",
        "duration_seconds": 24,
        "category": "comedy",
        "topic_tags": '["comedy", "prank"]',
        "quality_score": 0.35,
        "hook_strength": 0.30,
        "viewer_plan": [
            (14, 0.62, True, True, False),
            (15, 0.55, True, True, False),
            (10, 0.41, False, False, True),
            (12, 0.36, False, False, False),
        ],
    },
    {
        "creator_id": 1,
        "content": "Quiet desk setup vlog with a niche audience",
        "duration_seconds": 20,
        "category": "lifestyle",
        "topic_tags": '["lifestyle", "vlog"]',
        "quality_score": 0.58,
        "hook_strength": 0.45,
        "viewer_plan": [
            (11, 0.46, True, False, False),
            (13, 0.44, False, True, False),
            (14, 0.40, False, False, False),
            (15, 0.42, False, False, False),
        ],
    },
]

BASE_FOLLOW_EDGES = [
    (10, 0),
    (11, 0),
    (12, 1),
    (13, 1),
    (14, 2),
]

FeedbackMode = str
VALID_FEEDBACK_MODES = {"baseline", "treatment", "none", "random"}


def _build_scaled_scenario(
    creator_multiplier: int,
    viewer_multiplier: int,
    video_multiplier: int,
) -> dict[str, Any]:
    if creator_multiplier < 1 or viewer_multiplier < 1 or video_multiplier < 1:
        raise ValueError("Scale multipliers must be >= 1.")

    creator_map: dict[int, list[int]] = {}
    creators: list[tuple[int, str, str, str]] = []
    for idx in range(creator_multiplier):
        for base_id, username, display_name, bio in CREATORS:
            uid = base_id + idx * 1000
            creator_map.setdefault(base_id, []).append(uid)
            creators.append((
                uid,
                username if idx == 0 else f"{username}_{idx + 1}",
                display_name if idx == 0 else f"{display_name} {idx + 1}",
                bio,
            ))

    viewer_map: dict[int, list[int]] = {}
    viewers: list[tuple[int, str, str, str]] = []
    for idx in range(viewer_multiplier):
        for base_id, username, display_name, bio in VIEWERS:
            uid = base_id + idx * 1000
            viewer_map.setdefault(base_id, []).append(uid)
            viewers.append((
                uid,
                username if idx == 0 else f"{username}_{idx + 1}",
                display_name if idx == 0 else f"{display_name} {idx + 1}",
                bio,
            ))

    video_blueprints: list[dict[str, Any]] = []
    for v_idx in range(video_multiplier):
        for base_blueprint in VIDEO_BLUEPRINTS:
            creator_base_id = int(base_blueprint["creator_id"])
            creator_clone_ids = creator_map[creator_base_id]
            creator_id = creator_clone_ids[v_idx % len(creator_clone_ids)]
            viewer_plan: list[tuple[int, float, bool, bool, bool]] = []
            for base_viewer_id, watch_ratio, should_share, should_comment, dislikes in base_blueprint[
                    "viewer_plan"]:
                for clone_id in viewer_map[base_viewer_id]:
                    viewer_plan.append(
                        (clone_id, watch_ratio, should_share, should_comment, dislikes))

            content = str(base_blueprint["content"])
            if video_multiplier > 1:
                content = f"{content} [variant {v_idx + 1}]"
            video_blueprints.append({
                "creator_id": creator_id,
                "content": content,
                "duration_seconds": base_blueprint["duration_seconds"],
                "category": base_blueprint["category"],
                "topic_tags": base_blueprint["topic_tags"],
                "quality_score": base_blueprint["quality_score"],
                "hook_strength": base_blueprint["hook_strength"],
                "viewer_plan": viewer_plan,
            })

    follow_edges: list[tuple[int, int]] = []
    for base_viewer_id, base_creator_id in BASE_FOLLOW_EDGES:
        for viewer_id in viewer_map[base_viewer_id]:
            for creator_id in creator_map[base_creator_id]:
                follow_edges.append((viewer_id, creator_id))

    return {
        "creators": creators,
        "viewers": viewers,
        "video_blueprints": video_blueprints,
        "follow_edges": follow_edges,
    }


def _resolve_feedback_mode(
    heavy_negative_feedback: bool,
    feedback_mode: FeedbackMode | None,
) -> FeedbackMode:
    if feedback_mode is None:
        return "treatment" if heavy_negative_feedback else "baseline"
    if feedback_mode not in VALID_FEEDBACK_MODES:
        raise ValueError(
            f"Invalid feedback mode: {feedback_mode}. "
            f"Valid modes: {sorted(VALID_FEEDBACK_MODES)}")
    return feedback_mode


def _advance_step(platform: Platform) -> None:
    platform.sandbox_clock.time_step += 1


async def _sign_up_users(
    platform: Platform,
    creators: list[tuple[int, str, str, str]],
    viewers: list[tuple[int, str, str, str]],
) -> None:
    for user in creators + viewers:
        result = await platform.sign_up(user[0], user[1:])
        if not result["success"]:
            raise RuntimeError(f"Failed to sign up user {user[0]}: {result}")


async def _upload_videos(
    platform: Platform,
    video_blueprints: list[dict[str, Any]],
) -> list[int]:
    post_ids = []
    for blueprint in video_blueprints:
        result = await platform.upload_video(
            blueprint["creator_id"],
            (
                blueprint["content"],
                blueprint["duration_seconds"],
                blueprint["category"],
                blueprint["topic_tags"],
                blueprint["quality_score"],
                blueprint["hook_strength"],
            ),
        )
        if not result["success"]:
            raise RuntimeError(f"Failed to upload video: {result}")
        post_ids.append(result["post_id"])
        _advance_step(platform)
    return post_ids


async def _seed_social_graph(
    platform: Platform,
    follow_edges: list[tuple[int, int]],
) -> None:
    for follower_id, followee_id in follow_edges:
        result = await platform.follow(follower_id, followee_id)
        if not result["success"]:
            raise RuntimeError(f"Failed to follow: {result}")
    _advance_step(platform)


async def _simulate_round(
    platform: Platform,
    video_blueprints: list[dict[str, Any]],
    post_ids: list[int],
    feedback_mode: FeedbackMode,
    rng: random.Random | None = None,
    watch_jitter: float = 0.0,
    behavior_flip_prob: float = 0.0,
    treatment_extra_neg_prob: float = 1.0,
    random_negative_prob: float = 0.35,
) -> None:
    for blueprint, post_id in zip(video_blueprints, post_ids):
        negative_feedback_viewers: set[int] = set()
        for viewer_id, watch_ratio, should_share, should_comment, dislikes in (
                blueprint["viewer_plan"]):
            sampled_watch_ratio = watch_ratio
            if rng is not None and watch_jitter > 0:
                sampled_watch_ratio = max(
                    0.05,
                    min(0.99, watch_ratio + rng.uniform(-watch_jitter, watch_jitter)),
                )

            sampled_share = should_share
            sampled_comment = should_comment
            sampled_dislike = dislikes
            if rng is not None and behavior_flip_prob > 0:
                if rng.random() < behavior_flip_prob:
                    sampled_share = not sampled_share
                if rng.random() < behavior_flip_prob:
                    sampled_comment = not sampled_comment
                if rng.random() < behavior_flip_prob:
                    sampled_dislike = not sampled_dislike

            result = await platform.watch_video(
                viewer_id,
                (post_id, sampled_watch_ratio),
            )
            if not result["success"]:
                raise RuntimeError(f"Failed to watch video: {result}")

            if sampled_share:
                result = await platform.share_video(viewer_id, post_id)
                if not result["success"]:
                    raise RuntimeError(f"Failed to share video: {result}")

            if sampled_comment:
                result = await platform.create_comment(
                    viewer_id,
                    (post_id, "Loved this clip, watched all the way through."),
                )
                if not result["success"]:
                    raise RuntimeError(f"Failed to comment: {result}")

            should_negative = sampled_dislike
            if feedback_mode == "none":
                should_negative = False
            elif feedback_mode == "random":
                random_source = rng or random.Random(2026 + post_id * 17 + viewer_id)
                should_negative = random_source.random() < random_negative_prob

            if should_negative:
                result = await platform.not_interested(viewer_id, post_id)
                if not result["success"]:
                    raise RuntimeError(f"Failed to record negative feedback: {result}")
                negative_feedback_viewers.add(viewer_id)

        if feedback_mode == "treatment" and blueprint["category"] == "comedy":
            unique_viewers = []
            for viewer_id, *_ in blueprint["viewer_plan"]:
                if viewer_id not in unique_viewers:
                    unique_viewers.append(viewer_id)
            for extra_viewer_id in unique_viewers[:2]:
                if extra_viewer_id in negative_feedback_viewers:
                    continue
                if (rng is not None and treatment_extra_neg_prob < 1
                        and rng.random() > treatment_extra_neg_prob):
                    continue
                result = await platform.not_interested(extra_viewer_id, post_id)
                if not result["success"]:
                    raise RuntimeError(
                        f"Failed to record treatment negative feedback: {result}")
                negative_feedback_viewers.add(extra_viewer_id)

        _advance_step(platform)


async def _simulate_livestream(
    platform: Platform,
    creators: list[tuple[int, str, str, str]],
    viewers: list[tuple[int, str, str, str]],
) -> None:
    host_id = creators[0][0]
    audience = [v[0] for v in viewers[:3]]

    result = await platform.start_livestream(host_id)
    if not result["success"]:
        raise RuntimeError(f"Failed to start livestream: {result}")
    stream_id = result["stream_id"]
    _advance_step(platform)

    for viewer_id in audience:
        result = await platform.enter_livestream(viewer_id, stream_id)
        if not result["success"]:
            raise RuntimeError(f"Failed to enter livestream: {result}")

    result = await platform.livestream_comment(audience[0], (stream_id, "The live choreography is great."))
    if not result["success"]:
        raise RuntimeError(f"Failed to comment in livestream: {result}")

    result = await platform.send_gift(audience[1], (stream_id, 5))
    if not result["success"]:
        raise RuntimeError(f"Failed to send gift: {result}")
    _advance_step(platform)

    for viewer_id in audience:
        result = await platform.exit_livestream(viewer_id, stream_id)
        if not result["success"]:
            raise RuntimeError(f"Failed to exit livestream: {result}")

    result = await platform.end_livestream(host_id, stream_id)
    if not result["success"]:
        raise RuntimeError(f"Failed to end livestream: {result}")
    _advance_step(platform)


def _fetch_video_snapshot(db_path: str) -> list[dict[str, int | float | str]]:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT
                v.post_id,
                p.user_id,
                v.category,
                v.view_count,
                v.share_count,
                v.negative_count,
                v.traffic_pool_level
            FROM video v
            JOIN post p ON p.post_id = v.post_id
            ORDER BY v.post_id
            """
        )
        rows = cursor.fetchall()
        return [
            {
                "post_id": row[0],
                "creator_id": row[1],
                "category": row[2],
                "view_count": row[3],
                "share_count": row[4],
                "negative_count": row[5],
                "traffic_pool_level": row[6],
            }
            for row in rows
        ]
    finally:
        conn.close()


async def run_condition(
    db_path: str,
    heavy_negative_feedback: bool,
    feedback_mode: FeedbackMode | None = None,
    scenario: dict[str, Any] | None = None,
    seed: int | None = None,
    watch_jitter: float = 0.0,
    behavior_flip_prob: float = 0.0,
    treatment_extra_neg_prob: float = 1.0,
    random_negative_prob: float = 0.35,
    recsys_params: dict[str, Any] | None = None,
) -> dict[str, object]:
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    if db_file.exists():
        db_file.unlink()

    platform = Platform(
        str(db_file),
        recsys_type="tiktok",
        refresh_rec_post_count=4,
        max_rec_post_len=6,
    )
    platform.tiktok_recsys_params = recsys_params or RECSYS_DEFAULT_PARAMS
    mode = _resolve_feedback_mode(heavy_negative_feedback, feedback_mode)
    scenario_data = scenario or _build_scaled_scenario(
        creator_multiplier=1,
        viewer_multiplier=1,
        video_multiplier=1,
    )

    rng = None
    if seed is not None and (watch_jitter > 0 or behavior_flip_prob > 0
                             or treatment_extra_neg_prob < 1
                             or mode == "random"):
        rng = random.Random(seed)
    elif mode == "random":
        rng = random.Random(2026)

    try:
        await _sign_up_users(
            platform,
            creators=scenario_data["creators"],
            viewers=scenario_data["viewers"],
        )
        _advance_step(platform)
        await _seed_social_graph(platform, follow_edges=scenario_data["follow_edges"])
        post_ids = await _upload_videos(
            platform,
            video_blueprints=scenario_data["video_blueprints"],
        )

        await _simulate_round(
            platform,
            scenario_data["video_blueprints"],
            post_ids,
            mode,
            rng=rng,
            watch_jitter=watch_jitter,
            behavior_flip_prob=behavior_flip_prob,
            treatment_extra_neg_prob=treatment_extra_neg_prob,
            random_negative_prob=random_negative_prob,
        )
        await platform.update_rec_table()
        _advance_step(platform)

        await _simulate_livestream(
            platform,
            creators=scenario_data["creators"],
            viewers=scenario_data["viewers"],
        )
    finally:
        platform.db_cursor.close()
        platform.db.close()

    return {
        "db_path": str(db_file),
        "observability": get_short_video_observability_report(str(db_file)),
        "videos": _fetch_video_snapshot(str(db_file)),
        "seed": seed,
        "heavy_negative_feedback": heavy_negative_feedback,
        "feedback_mode": mode,
    }


def _extract_condition_metrics(result: dict[str, object]) -> dict[str, float | int]:
    summary = result["observability"]["summary"]
    comedy = next(row for row in result["videos"] if row["category"] == "comedy")
    lifestyle = next(row for row in result["videos"] if row["category"] == "lifestyle")
    return {
        "total_views": int(summary["total_views"]),
        "total_shares": int(summary["total_shares"]),
        "total_negative_feedback": int(summary["total_negative_feedback"]),
        "negative_feedback_rate": float(summary["negative_feedback_rate"]),
        "avg_watch_ratio": float(summary["avg_watch_ratio"]),
        "retention_3s_rate": float(summary["retention_3s_rate"]),
        "creator_coverage": float(summary["creator_coverage"]),
        "avg_traffic_pool_level": float(summary["avg_traffic_pool_level"]),
        "comedy_negative_count": int(comedy["negative_count"]),
        "comedy_traffic_pool_level": int(comedy["traffic_pool_level"]),
        "lifestyle_traffic_pool_level": int(lifestyle["traffic_pool_level"]),
    }


def _mean_std_ci95(values: list[float]) -> dict[str, float]:
    n = len(values)
    mean = statistics.mean(values)
    std = statistics.stdev(values) if n > 1 else 0.0
    ci_half_width = 1.96 * std / math.sqrt(n) if n > 1 else 0.0
    return {
        "n": float(n),
        "mean": mean,
        "std": std,
        "ci95_low": mean - ci_half_width,
        "ci95_high": mean + ci_half_width,
    }


def _paired_sign_flip_pvalue(
    diffs: list[float],
    rng_seed: int,
    max_samples: int = 20000,
) -> float:
    observed = abs(statistics.mean(diffs))
    n = len(diffs)
    if n == 0:
        return 1.0

    if n <= 16:
        # Exhaustive randomization test under the paired sign-flip null.
        total = 1 << n
        extreme = 0
        for mask in range(total):
            signed = []
            for i, d in enumerate(diffs):
                sign = -1.0 if ((mask >> i) & 1) else 1.0
                signed.append(sign * d)
            if abs(statistics.mean(signed)) >= observed:
                extreme += 1
        return extreme / total

    rng = random.Random(rng_seed)
    extreme = 0
    for _ in range(max_samples):
        signed = [(d if rng.random() < 0.5 else -d) for d in diffs]
        if abs(statistics.mean(signed)) >= observed:
            extreme += 1
    return (extreme + 1) / (max_samples + 1)


def _write_multirun_outputs(
    output_dir: Path,
    run_rows: list[dict[str, Any]],
    metric_names: list[str],
    n_runs: int,
) -> None:
    csv_path = output_dir / "multirun_metrics.csv"
    fieldnames = ["run_id", "seed", "condition", *metric_names]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(run_rows)

    baseline_rows = [r for r in run_rows if r["condition"] == "baseline"]
    treatment_rows = [r for r in run_rows if r["condition"] == "treatment"]

    summary_csv = output_dir / "multirun_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "metric",
                "baseline_mean",
                "baseline_std",
                "baseline_ci95_low",
                "baseline_ci95_high",
                "treatment_mean",
                "treatment_std",
                "treatment_ci95_low",
                "treatment_ci95_high",
                "paired_diff_mean",
                "paired_diff_std",
                "paired_diff_ci95_low",
                "paired_diff_ci95_high",
                "paired_signflip_pvalue",
            ],
        )
        writer.writeheader()
        for metric in metric_names:
            baseline_values = [float(r[metric]) for r in baseline_rows]
            treatment_values = [float(r[metric]) for r in treatment_rows]
            baseline_stats = _mean_std_ci95(baseline_values)
            treatment_stats = _mean_std_ci95(treatment_values)
            diffs = [treatment_values[i] - baseline_values[i] for i in range(n_runs)]
            diff_stats = _mean_std_ci95(diffs)
            p_value = _paired_sign_flip_pvalue(diffs, rng_seed=2026 + len(metric))
            writer.writerow({
                "metric": metric,
                "baseline_mean": f"{baseline_stats['mean']:.6f}",
                "baseline_std": f"{baseline_stats['std']:.6f}",
                "baseline_ci95_low": f"{baseline_stats['ci95_low']:.6f}",
                "baseline_ci95_high": f"{baseline_stats['ci95_high']:.6f}",
                "treatment_mean": f"{treatment_stats['mean']:.6f}",
                "treatment_std": f"{treatment_stats['std']:.6f}",
                "treatment_ci95_low": f"{treatment_stats['ci95_low']:.6f}",
                "treatment_ci95_high": f"{treatment_stats['ci95_high']:.6f}",
                "paired_diff_mean": f"{diff_stats['mean']:.6f}",
                "paired_diff_std": f"{diff_stats['std']:.6f}",
                "paired_diff_ci95_low": f"{diff_stats['ci95_low']:.6f}",
                "paired_diff_ci95_high": f"{diff_stats['ci95_high']:.6f}",
                "paired_signflip_pvalue": f"{p_value:.6f}",
            })

    report_lines = [
        "# Short-Video Negative Feedback Multi-Run Summary",
        "",
        f"- Runs per condition: {n_runs}",
        "- Pairing: each baseline run is paired with treatment run at the same seed.",
        "- Significance: paired randomization (sign-flip) test, two-sided.",
        "",
        "## Output Files",
        "",
        f"- Per-run metrics: `{csv_path}`",
        f"- Aggregate summary: `{summary_csv}`",
        "",
        "## Notes",
        "",
        "- This report is intended to increase experimental robustness beyond a single deterministic run.",
        "- Keep the deterministic single-run artifacts for figure-level qualitative interpretation.",
        "",
    ]
    (output_dir / "MULTIRUN_REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")


def _write_strategy_multirun_outputs(
    output_dir: Path,
    run_rows: list[dict[str, Any]],
    metric_names: list[str],
    n_runs: int,
    baseline_condition: str = "baseline",
) -> None:
    metrics_csv = output_dir / "strategy_multirun_metrics.csv"
    fieldnames = ["run_id", "seed", "condition", *metric_names]
    with metrics_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(run_rows)

    conditions = sorted({str(r["condition"]) for r in run_rows})
    summary_csv = output_dir / "strategy_multirun_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "condition",
                "metric",
                "mean",
                "std",
                "ci95_low",
                "ci95_high",
            ],
        )
        writer.writeheader()
        for condition in conditions:
            rows = [r for r in run_rows if r["condition"] == condition]
            for metric in metric_names:
                values = [float(r[metric]) for r in rows]
                stats = _mean_std_ci95(values)
                writer.writerow({
                    "condition": condition,
                    "metric": metric,
                    "mean": f"{stats['mean']:.6f}",
                    "std": f"{stats['std']:.6f}",
                    "ci95_low": f"{stats['ci95_low']:.6f}",
                    "ci95_high": f"{stats['ci95_high']:.6f}",
                })

    if baseline_condition not in conditions:
        return

    pairwise_csv = output_dir / "strategy_pairwise_vs_baseline.csv"
    with pairwise_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "condition",
                "metric",
                "paired_diff_mean",
                "paired_diff_std",
                "paired_diff_ci95_low",
                "paired_diff_ci95_high",
                "paired_signflip_pvalue",
            ],
        )
        writer.writeheader()
        baseline_by_run = {
            int(r["run_id"]): r
            for r in run_rows if r["condition"] == baseline_condition
        }
        for condition in conditions:
            if condition == baseline_condition:
                continue
            rows = [r for r in run_rows if r["condition"] == condition]
            for metric in metric_names:
                diffs: list[float] = []
                for row in rows:
                    rid = int(row["run_id"])
                    if rid not in baseline_by_run:
                        continue
                    diffs.append(float(row[metric]) - float(baseline_by_run[rid][metric]))
                if len(diffs) != n_runs:
                    continue
                stats = _mean_std_ci95(diffs)
                p_value = _paired_sign_flip_pvalue(
                    diffs, rng_seed=2039 + len(metric) + len(condition))
                writer.writerow({
                    "condition": condition,
                    "metric": metric,
                    "paired_diff_mean": f"{stats['mean']:.6f}",
                    "paired_diff_std": f"{stats['std']:.6f}",
                    "paired_diff_ci95_low": f"{stats['ci95_low']:.6f}",
                    "paired_diff_ci95_high": f"{stats['ci95_high']:.6f}",
                    "paired_signflip_pvalue": f"{p_value:.6f}",
                })


def _render_condition_summary(name: str, result: dict[str, object]) -> list[str]:
    summary = result["observability"]["summary"]
    video_rows = result["videos"]
    lines = [
        f"## {name}",
        "",
        f"- Database: `{result['db_path']}`",
        f"- Total views: {summary['total_views']}",
        f"- Total shares: {summary['total_shares']}",
        f"- Total negative feedback: {summary['total_negative_feedback']}",
        f"- Negative feedback rate: {summary['negative_feedback_rate']:.4f}",
        f"- Average watch ratio: {summary['avg_watch_ratio']:.3f}",
        f"- Retention 3s rate: {summary['retention_3s_rate']:.4f}",
        f"- Creator coverage: {summary['creator_coverage']:.4f}",
        f"- Average traffic pool level: {summary['avg_traffic_pool_level']:.3f}",
        "",
        "| post_id | category | view_count | share_count | negative_count | traffic_pool_level |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in video_rows:
        lines.append(
            f"| {row['post_id']} | {row['category']} | {row['view_count']} | "
            f"{row['share_count']} | {row['negative_count']} | "
            f"{row['traffic_pool_level']} |"
        )
    lines.append("")
    return lines


def _write_readme(
    output_dir: Path,
    baseline_result: dict[str, object],
    treatment_result: dict[str, object],
) -> None:
    baseline_bad_video = next(
        row for row in baseline_result["videos"] if row["category"] == "comedy")
    treatment_bad_video = next(
        row for row in treatment_result["videos"] if row["category"] == "comedy")

    lines = [
        "# Short-Video Negative Feedback Experiment",
        "",
        "This experiment is a deterministic template for a paper-style short-video study in OASIS.",
        "It compares two otherwise identical TikTok-style runs:",
        "",
        "- `baseline`: weak negative feedback against a low-quality comedy video",
        "- `treatment`: stronger `not_interested` signals against that same low-quality video",
        "",
        "The intended research question is whether explicit negative feedback suppresses reach in a traffic-pool recommender.",
        "",
    ]
    lines.extend(_render_condition_summary("Baseline", baseline_result))
    lines.extend(_render_condition_summary("Treatment", treatment_result))
    lines.extend([
        "## Key Contrast",
        "",
        f"- Baseline low-quality comedy video traffic pool level: {baseline_bad_video['traffic_pool_level']}",
        f"- Treatment low-quality comedy video traffic pool level: {treatment_bad_video['traffic_pool_level']}",
        f"- Baseline low-quality comedy video negative feedback: {baseline_bad_video['negative_count']}",
        f"- Treatment low-quality comedy video negative feedback: {treatment_bad_video['negative_count']}",
        "",
        "## Suggested Analysis Commands",
        "",
        "```bash",
        f"python visualization/short_video_simulation/code/generate_report.py {baseline_result['db_path']} --output {output_dir / 'baseline_report.md'}",
        f"python visualization/short_video_simulation/code/generate_report.py {treatment_result['db_path']} --output {output_dir / 'treatment_report.md'}",
        "python visualization/short_video_simulation/code/compare_runs.py "
        f"baseline={baseline_result['db_path']} "
        f"treatment={treatment_result['db_path']} "
        f"--output-dir {output_dir / 'comparison'}",
        "```",
        "",
    ])
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


async def main(
    output_dir: str,
    n_runs: int,
    seed_start: int,
    watch_jitter: float,
    behavior_flip_prob: float,
    treatment_extra_neg_prob: float,
    save_all_dbs: bool,
    strategy_set: str,
    random_negative_prob: float,
    creator_multiplier: int,
    viewer_multiplier: int,
    video_multiplier: int,
    recsys_profile: str,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    scenario = _build_scaled_scenario(
        creator_multiplier=creator_multiplier,
        viewer_multiplier=viewer_multiplier,
        video_multiplier=video_multiplier,
    )
    recsys_params = RECSYS_PROFILES[recsys_profile]

    baseline_result = await run_condition(
        str(output_path / "negative_feedback_baseline.db"),
        heavy_negative_feedback=False,
        feedback_mode="baseline",
        scenario=scenario,
        recsys_params=recsys_params,
    )
    treatment_result = await run_condition(
        str(output_path / "negative_feedback_treatment.db"),
        heavy_negative_feedback=True,
        feedback_mode="treatment",
        scenario=scenario,
        recsys_params=recsys_params,
    )
    _write_readme(output_path, baseline_result, treatment_result)

    extra_single_results: dict[str, dict[str, object]] = {}
    if strategy_set == "four_arm":
        none_result = await run_condition(
            str(output_path / "negative_feedback_none.db"),
            heavy_negative_feedback=False,
            feedback_mode="none",
            scenario=scenario,
            recsys_params=recsys_params,
        )
        random_result = await run_condition(
            str(output_path / "negative_feedback_random.db"),
            heavy_negative_feedback=False,
            feedback_mode="random",
            scenario=scenario,
            seed=seed_start,
            random_negative_prob=random_negative_prob,
            recsys_params=recsys_params,
        )
        extra_single_results = {
            "none": none_result,
            "random": random_result,
        }
        lines = ["", "## Additional Single-Run Controls", ""]
        for condition_name, result in extra_single_results.items():
            summary = result["observability"]["summary"]
            comedy = next(row for row in result["videos"] if row["category"] == "comedy")
            lines.extend([
                f"### {condition_name}",
                "",
                f"- Database: `{result['db_path']}`",
                f"- Total views: {summary['total_views']}",
                f"- Total negative feedback: {summary['total_negative_feedback']}",
                f"- Negative feedback rate: {summary['negative_feedback_rate']:.4f}",
                f"- Retention 3s rate: {summary['retention_3s_rate']:.4f}",
                f"- Creator coverage: {summary['creator_coverage']:.4f}",
                f"- Comedy negative_count: {comedy['negative_count']}",
                f"- Comedy traffic_pool_level: {comedy['traffic_pool_level']}",
                "",
            ])
        with (output_path / "README.md").open("a", encoding="utf-8") as f:
            f.write("\n".join(lines))

    if n_runs <= 1:
        return

    run_rows: list[dict[str, Any]] = []
    conditions: list[FeedbackMode] = ["baseline", "treatment"]
    if strategy_set == "four_arm":
        conditions.extend(["none", "random"])
    metric_names = [
        "total_views",
        "total_shares",
        "total_negative_feedback",
        "negative_feedback_rate",
        "avg_watch_ratio",
        "retention_3s_rate",
        "creator_coverage",
        "avg_traffic_pool_level",
        "comedy_negative_count",
        "comedy_traffic_pool_level",
        "lifestyle_traffic_pool_level",
    ]

    run_db_dir = output_path / "multirun_dbs"
    if save_all_dbs:
        run_db_dir.mkdir(parents=True, exist_ok=True)

    for run_id in range(n_runs):
        run_seed = seed_start + run_id
        for condition in conditions:
            if save_all_dbs:
                db_file = run_db_dir / f"run_{run_id:03d}_{condition}.db"
            else:
                db_file = output_path / f".tmp_run_{run_id:03d}_{condition}.db"

            result = await run_condition(
                str(db_file),
                heavy_negative_feedback=(condition == "treatment"),
                feedback_mode=condition,
                scenario=scenario,
                seed=run_seed,
                watch_jitter=watch_jitter,
                behavior_flip_prob=behavior_flip_prob,
                treatment_extra_neg_prob=treatment_extra_neg_prob,
                random_negative_prob=random_negative_prob,
                recsys_params=recsys_params,
            )
            metrics = _extract_condition_metrics(result)
            run_rows.append({
                "run_id": run_id,
                "seed": run_seed,
                "condition": condition,
                **metrics,
            })
            if not save_all_dbs:
                Path(db_file).unlink(missing_ok=True)

    paired_rows = [r for r in run_rows if r["condition"] in ("baseline", "treatment")]
    _write_multirun_outputs(
        output_dir=output_path,
        run_rows=paired_rows,
        metric_names=metric_names,
        n_runs=n_runs,
    )
    if strategy_set == "four_arm":
        _write_strategy_multirun_outputs(
            output_dir=output_path,
            run_rows=run_rows,
            metric_names=metric_names,
            n_runs=n_runs,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a deterministic short-video negative-feedback experiment.",
    )
    parser.add_argument(
        "--output-dir",
        default="examples/experiment/short_video_negative_feedback/output",
        help="Directory where baseline/treatment DBs and README are written.",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=1,
        help="Number of paired baseline/treatment runs for statistical summary.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=2026,
        help="Start seed for multi-run experiments; each run increments by 1.",
    )
    parser.add_argument(
        "--watch-jitter",
        type=float,
        default=0.0,
        help="Uniform perturbation radius for watch ratio in each run (0 to 0.4).",
    )
    parser.add_argument(
        "--behavior-flip-prob",
        type=float,
        default=0.0,
        help="Probability of flipping share/comment/dislike boolean decisions.",
    )
    parser.add_argument(
        "--treatment-extra-neg-prob",
        type=float,
        default=1.0,
        help="Probability of applying each extra treatment negative feedback event.",
    )
    parser.add_argument(
        "--save-all-dbs",
        action="store_true",
        help="When running multi-run experiments, keep all per-run DB files.",
    )
    parser.add_argument(
        "--strategy-set",
        choices=["baseline_treatment", "four_arm"],
        default="baseline_treatment",
        help="Experiment arms: baseline/treatment only, or add none/random controls.",
    )
    parser.add_argument(
        "--random-negative-prob",
        type=float,
        default=0.35,
        help="Per-watch negative-feedback probability when strategy mode is random.",
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
        help="Scale factor applied to uploaded video count.",
    )
    parser.add_argument(
        "--recsys-profile",
        choices=sorted(RECSYS_PROFILES.keys()),
        default="default",
        help="Recommendation sensitivity profile for short-video traffic-pool scoring.",
    )
    args = parser.parse_args()
    asyncio.run(
        main(
            output_dir=args.output_dir,
            n_runs=args.n_runs,
            seed_start=args.seed_start,
            watch_jitter=args.watch_jitter,
            behavior_flip_prob=args.behavior_flip_prob,
            treatment_extra_neg_prob=args.treatment_extra_neg_prob,
            save_all_dbs=args.save_all_dbs,
            strategy_set=args.strategy_set,
            random_negative_prob=args.random_negative_prob,
            creator_multiplier=args.creator_multiplier,
            viewer_multiplier=args.viewer_multiplier,
            video_multiplier=args.video_multiplier,
            recsys_profile=args.recsys_profile,
        ))
