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

import json
import sqlite3
from typing import Any


def _table_exists(cursor: sqlite3.Cursor, table_name: str) -> bool:
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name = ?",
        (table_name,),
    )
    return cursor.fetchone() is not None


def _rows_to_dicts(cursor: sqlite3.Cursor, query: str,
                   params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    cursor.execute(query, params)
    columns = [description[0] for description in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def _compute_retention_3s_rate(
    cursor: sqlite3.Cursor,
    duration_by_post_id: dict[int, int],
) -> float:
    """Compute the fraction of watch events whose watched seconds >= 3."""
    if not _table_exists(cursor, "trace"):
        return 0.0

    cursor.execute("SELECT info FROM trace WHERE action = 'watch_video'")
    watch_rows = cursor.fetchall()
    if not watch_rows:
        return 0.0

    retained = 0
    valid_watch_events = 0
    for (info_str,) in watch_rows:
        try:
            info = json.loads(info_str) if info_str else {}
        except (TypeError, json.JSONDecodeError):
            continue

        post_id = info.get("post_id")
        watch_ratio = info.get("watch_ratio")
        if not isinstance(post_id, int) or not isinstance(watch_ratio,
                                                           (int, float)):
            continue

        duration = duration_by_post_id.get(post_id)
        if not isinstance(duration, int) or duration <= 0:
            continue

        valid_watch_events += 1
        if float(watch_ratio) * duration >= 3.0:
            retained += 1

    if valid_watch_events == 0:
        return 0.0
    return retained / valid_watch_events


def get_short_video_observability_report(db_path: str) -> dict[str, Any]:
    r"""Generate a short-video observability report from a simulation DB.

    The report summarizes creator growth, feed performance, and livestream
    activity so researchers can inspect short-video simulations without
    writing SQL manually.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        if not _table_exists(cursor, "video"):
            raise ValueError("The database does not contain short-video tables.")

        summary_query = """
            SELECT
                COUNT(*) AS total_videos,
                COALESCE(SUM(view_count), 0) AS total_views,
                COALESCE(SUM(share_count), 0) AS total_shares,
                COALESCE(SUM(negative_count), 0) AS total_negative_feedback,
                COALESCE(AVG(traffic_pool_level), 0.0) AS avg_traffic_pool_level,
                COALESCE(SUM(total_watch_ratio), 0.0) AS total_watch_ratio,
                COALESCE(SUM(view_count), 0) AS total_view_rows
            FROM video
        """
        cursor.execute(summary_query)
        (total_videos, total_views, total_shares, total_negative_feedback,
         avg_traffic_pool_level, total_watch_ratio,
         total_view_rows) = cursor.fetchone()

        cursor.execute("SELECT COUNT(DISTINCT user_id) FROM post")
        total_creators = cursor.fetchone()[0] or 0
        cursor.execute(
            "SELECT COUNT(DISTINCT p.user_id) "
            "FROM video v JOIN post p ON p.post_id = v.post_id "
            "WHERE v.view_count > 0")
        creators_with_views = cursor.fetchone()[0] or 0

        cursor.execute("SELECT COUNT(*) FROM comment")
        total_comments = cursor.fetchone()[0] or 0

        cursor.execute("SELECT post_id, duration_seconds FROM video")
        duration_by_post_id = {
            int(post_id): int(duration_seconds)
            for post_id, duration_seconds in cursor.fetchall()
            if isinstance(post_id, int) and isinstance(duration_seconds, int)
        }
        retention_3s_rate = _compute_retention_3s_rate(
            cursor, duration_by_post_id)

        summary = {
            "total_videos": total_videos or 0,
            "total_creators": total_creators,
            "total_views": total_views or 0,
            "total_comments": total_comments,
            "total_shares": total_shares or 0,
            "total_negative_feedback": total_negative_feedback or 0,
            "negative_feedback_rate": (
                (total_negative_feedback or 0) / total_views
                if total_views else 0.0
            ),
            "avg_watch_ratio": (
                (total_watch_ratio or 0.0) / total_view_rows
                if total_view_rows else 0.0
            ),
            "retention_3s_rate": retention_3s_rate,
            "creator_coverage": (
                creators_with_views / total_creators if total_creators else 0.0
            ),
            "avg_traffic_pool_level": avg_traffic_pool_level or 0.0,
        }

        creator_query = """
            SELECT
                p.user_id,
                u.user_name,
                u.name,
                u.num_followers,
                COUNT(v.post_id) AS uploaded_videos,
                COALESCE(SUM(v.view_count), 0) AS total_views,
                COALESCE(SUM(v.share_count), 0) AS total_shares,
                COALESCE(SUM(v.negative_count), 0) AS total_negative_feedback,
                COALESCE(SUM(v.total_watch_ratio), 0.0) AS total_watch_ratio,
                COALESCE(SUM(v.view_count), 0) AS total_view_rows
            FROM video v
            JOIN post p ON p.post_id = v.post_id
            LEFT JOIN user u ON u.user_id = p.user_id
            GROUP BY p.user_id, u.user_name, u.name, u.num_followers
            ORDER BY total_views DESC, uploaded_videos DESC
        """
        creator_rows = _rows_to_dicts(cursor, creator_query)
        creators = []
        for row in creator_rows:
            creators.append({
                "user_id": row["user_id"],
                "user_name": row["user_name"],
                "name": row["name"],
                "num_followers": row["num_followers"],
                "uploaded_videos": row["uploaded_videos"],
                "total_views": row["total_views"],
                "total_shares": row["total_shares"],
                "total_negative_feedback": row["total_negative_feedback"],
                "avg_watch_ratio": (
                    row["total_watch_ratio"] / row["total_view_rows"]
                    if row["total_view_rows"] else 0.0
                ),
            })

        comment_counts_query = """
            SELECT post_id, COUNT(*) AS comment_count
            FROM comment
            GROUP BY post_id
        """
        comment_counts = {
            row["post_id"]: row["comment_count"]
            for row in _rows_to_dicts(cursor, comment_counts_query)
        }

        video_query = """
            SELECT
                v.post_id,
                p.user_id AS creator_id,
                p.content,
                v.category,
                v.topic_tags,
                v.traffic_pool_level,
                v.view_count,
                v.total_watch_ratio,
                v.share_count,
                v.negative_count
            FROM video v
            JOIN post p ON p.post_id = v.post_id
            ORDER BY v.view_count DESC, v.share_count DESC, v.post_id ASC
        """
        video_rows = _rows_to_dicts(cursor, video_query)
        top_videos = []
        for row in video_rows:
            try:
                topic_tags = json.loads(row["topic_tags"]) if row["topic_tags"] else []
            except (TypeError, json.JSONDecodeError):
                topic_tags = []
            top_videos.append({
                "post_id": row["post_id"],
                "creator_id": row["creator_id"],
                "content": row["content"],
                "category": row["category"],
                "topic_tags": topic_tags,
                "traffic_pool_level": row["traffic_pool_level"],
                "view_count": row["view_count"],
                "avg_watch_ratio": (
                    row["total_watch_ratio"] / row["view_count"]
                    if row["view_count"] else 0.0
                ),
                "share_count": row["share_count"],
                "negative_count": row["negative_count"],
                "comment_count": comment_counts.get(row["post_id"], 0),
            })

        livestreams: list[dict[str, Any]] = []
        if _table_exists(cursor, "livestream") and _table_exists(
                cursor, "livestream_viewer"):
            livestream_query = """
                SELECT
                    l.stream_id,
                    l.host_id,
                    l.status,
                    l.current_viewers,
                    l.peak_viewers,
                    l.total_viewers,
                    l.total_comments,
                    l.total_gifts_value,
                    COALESCE(AVG(lv.total_stay_seconds), 0.0) AS avg_stay_seconds,
                    COALESCE(AVG(lv.interactions), 0.0) AS avg_interactions
                FROM livestream l
                LEFT JOIN livestream_viewer lv ON lv.stream_id = l.stream_id
                GROUP BY
                    l.stream_id,
                    l.host_id,
                    l.status,
                    l.current_viewers,
                    l.peak_viewers,
                    l.total_viewers,
                    l.total_comments,
                    l.total_gifts_value
                ORDER BY l.peak_viewers DESC, l.stream_id ASC
            """
            livestreams = _rows_to_dicts(cursor, livestream_query)

        return {
            "summary": summary,
            "creators": creators,
            "top_videos": top_videos,
            "livestreams": livestreams,
        }
    finally:
        conn.close()


def get_short_video_time_series_report(db_path: str) -> dict[str, Any]:
    r"""Generate time-series observability metrics for short-video simulations.

    Groups core feed and livestream signals by simulation step so downstream
    analysis and visualization can track recommendation dynamics over time.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        if not _table_exists(cursor, "video"):
            raise ValueError("The database does not contain short-video tables.")

        video_time_series_query = """
            SELECT
                pool_enter_time AS step,
                COUNT(*) AS uploaded_videos,
                COALESCE(SUM(view_count), 0) AS cumulative_views,
                COALESCE(SUM(share_count), 0) AS cumulative_shares,
                COALESCE(SUM(negative_count), 0) AS cumulative_negative_feedback,
                COALESCE(AVG(traffic_pool_level), 0.0) AS avg_traffic_pool_level,
                COALESCE(SUM(total_watch_ratio), 0.0) AS total_watch_ratio,
                COALESCE(SUM(view_count), 0) AS total_view_rows
            FROM video
            GROUP BY pool_enter_time
            ORDER BY step
        """
        video_rows = _rows_to_dicts(cursor, video_time_series_query)
        video_time_series = []
        for row in video_rows:
            video_time_series.append({
                "step": row["step"],
                "uploaded_videos": row["uploaded_videos"],
                "cumulative_views": row["cumulative_views"],
                "cumulative_shares": row["cumulative_shares"],
                "cumulative_negative_feedback":
                row["cumulative_negative_feedback"],
                "avg_traffic_pool_level": row["avg_traffic_pool_level"],
                "avg_watch_ratio": (
                    row["total_watch_ratio"] / row["total_view_rows"]
                    if row["total_view_rows"] else 0.0
                ),
            })

        creator_time_series_query = """
            SELECT
                created_at AS step,
                user_id AS creator_id,
                COUNT(*) AS uploaded_videos
            FROM post
            GROUP BY created_at, user_id
            ORDER BY created_at, user_id
        """
        creator_rows = _rows_to_dicts(cursor, creator_time_series_query)
        creator_growth = creator_rows

        livestream_time_series: list[dict[str, Any]] = []
        if _table_exists(cursor, "livestream"):
            livestream_query = """
                SELECT
                    start_time AS step,
                    COUNT(*) AS livestreams_started,
                    COALESCE(AVG(peak_viewers), 0.0) AS avg_peak_viewers,
                    COALESCE(AVG(total_comments), 0.0) AS avg_total_comments,
                    COALESCE(AVG(total_gifts_value), 0.0) AS avg_total_gifts_value
                FROM livestream
                GROUP BY start_time
                ORDER BY step
            """
            livestream_time_series = _rows_to_dicts(cursor, livestream_query)

        retention_time_series: list[dict[str, Any]] = []
        if _table_exists(cursor, "livestream_viewer"):
            retention_query = """
                SELECT
                    enter_time AS step,
                    COUNT(*) AS viewer_sessions,
                    COALESCE(AVG(total_stay_seconds), 0.0) AS avg_stay_seconds,
                    COALESCE(AVG(interactions), 0.0) AS avg_interactions
                FROM livestream_viewer
                GROUP BY enter_time
                ORDER BY step
            """
            retention_time_series = _rows_to_dicts(cursor, retention_query)

        return {
            "video_time_series": video_time_series,
            "creator_growth": creator_growth,
            "livestream_time_series": livestream_time_series,
            "viewer_retention_time_series": retention_time_series,
        }
    finally:
        conn.close()
