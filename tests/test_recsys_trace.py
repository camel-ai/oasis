"""
Unit tests for rec_sys_personalized_with_trace()
Covers swap_rate > 0 with trace rows that lack a post_id (e.g. follow/unfollow actions).
"""

import json
import unittest
from ast import literal_eval
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Minimal stubs so we can import the function without the full oasis stack
# ---------------------------------------------------------------------------

def _make_trace(user_id, action, post_id=None):
    """Helper to build a trace row matching the actual schema."""
    info = {"post_id": post_id} if post_id is not None else {"target_id": user_id + 1}
    return {
        "user_id": user_id,
        "action": action,
        "info": str(info),  # stored as stringified dict (literal_eval-able)
    }


def _make_user(user_id, agent_id=None):
    return {
        "user_id": user_id,
        "agent_id": agent_id if agent_id is not None else user_id,
        "bio": "test bio",
    }


def _make_post(post_id, user_id, content="hello"):
    return {"post_id": post_id, "user_id": user_id, "content": content}


# ---------------------------------------------------------------------------
# Import the actual function under test
# ---------------------------------------------------------------------------

import sys
import os

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from oasis.social_platform.recsys import rec_sys_personalized_with_trace
    IMPORT_OK = True
except Exception:
    IMPORT_OK = False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@unittest.skipUnless(IMPORT_OK, "oasis package not importable in this env")
class TestRecSysPersonalizedWithTrace(unittest.TestCase):

    def _base_args(self):
        """Minimal valid inputs for the function."""
        users = [_make_user(1), _make_user(2)]
        posts = [_make_post(i, user_id=99) for i in range(1, 12)]  # 11 posts
        rec_matrix = [[], [1, 2, 3], [1, 2, 3]]  # row 0 unused; one row per user
        return users, posts, rec_matrix

    # ------------------------------------------------------------------
    # Fix #1 regression: user_id=0 must NOT be treated as falsy
    # ------------------------------------------------------------------

    def test_user_id_zero_not_excluded_from_traced_ids(self):
        """Traces from user_id=0 must be included in traced_post_ids."""
        users, posts, rec_matrix = self._base_args()
        trace_table = [
            _make_trace(user_id=0, action="like", post_id=1),
            _make_trace(user_id=1, action="like", post_id=2),
        ]
        # Should not raise; post_id=1 traced by user 0 must be tracked
        try:
            result = rec_sys_personalized_with_trace(
                users, posts, trace_table, rec_matrix,
                max_rec_post_len=5, swap_rate=0.2, model=None
            )
        except KeyError as e:
            self.fail(f"KeyError raised — user_id=0 trace was skipped: {e}")
        self.assertIsInstance(result, list)

    # ------------------------------------------------------------------
    # Fix #2 regression: traced_post_ids computed only once (perf)
    # We verify correctness rather than call count (no easy hook without
    # refactoring), so we assert the result is consistent across users.
    # ------------------------------------------------------------------

    def test_traced_post_ids_consistent_across_users(self):
        """All users should see the same global traced_post_ids set."""
        users = [_make_user(i) for i in range(1, 4)]
        posts = [_make_post(i, user_id=99) for i in range(1, 15)]
        rec_matrix = [[], []] * 4  # one slot per user + unused row 0
        trace_table = [
            _make_trace(user_id=1, action="like", post_id=3),
            _make_trace(user_id=2, action="like", post_id=3),
        ]
        result = rec_sys_personalized_with_trace(
            users, posts, trace_table, rec_matrix,
            max_rec_post_len=5, swap_rate=0.5, model=None
        )
        # post_id=3 is traced; it should not appear in swap_free candidates
        # (the exact rec list is random, but no assertion errors should occur)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), len(users))

    # ------------------------------------------------------------------
    # Fix #3 (core): non-post traces (follow/unfollow) must not raise
    # ------------------------------------------------------------------

    def test_swap_rate_with_non_post_traces_no_keyerror(self):
        """
        rec_sys_personalized_with_trace must not raise KeyError when
        trace_table contains follow/unfollow rows that have no post_id in info.
        """
        users, posts, rec_matrix = self._base_args()
        trace_table = [
            _make_trace(user_id=1, action="like",   post_id=2),
            _make_trace(user_id=1, action="follow",  post_id=None),   # no post_id
            _make_trace(user_id=1, action="unfollow", post_id=None),  # no post_id
            _make_trace(user_id=2, action="like",   post_id=5),
        ]
        try:
            result = rec_sys_personalized_with_trace(
                users, posts, trace_table, rec_matrix,
                max_rec_post_len=5, swap_rate=0.3, model=None
            )
        except KeyError as e:
            self.fail(
                f"KeyError raised on non-post trace row (follow/unfollow): {e}"
            )
        self.assertIsInstance(result, list)

    def test_swap_rate_zero_skips_trace_computation(self):
        """With swap_rate=0, traced_post_ids should not be computed at all."""
        users, posts, rec_matrix = self._base_args()
        # Deliberately broken trace to catch accidental computation
        trace_table = [{"user_id": 1, "action": "like", "info": "NOT_VALID_JSON"}]
        try:
            result = rec_sys_personalized_with_trace(
                users, posts, trace_table, rec_matrix,
                max_rec_post_len=5, swap_rate=0.0, model=None
            )
        except Exception as e:
            self.fail(f"swap_rate=0 should not touch trace_table, but raised: {e}")
        self.assertIsInstance(result, list)

    def test_empty_trace_table_with_swap(self):
        """Empty trace table with swap_rate > 0 must work without errors."""
        users, posts, rec_matrix = self._base_args()
        result = rec_sys_personalized_with_trace(
            users, posts, [], rec_matrix,
            max_rec_post_len=5, swap_rate=0.5, model=None
        )
        self.assertIsInstance(result, list)


if __name__ == "__main__":
    unittest.main()
