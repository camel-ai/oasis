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
from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np

from oasis.social_platform.recsys import (calculate_hot_score,
                                          calculate_like_similarity,
                                          coarse_filtering, get_like_post_id,
                                          normalize_similarity_adjustments,
                                          rec_sys_personalized,
                                          rec_sys_personalized_twh,
                                          rec_sys_random, rec_sys_reddit,
                                          reset_globals, swap_random_posts)


def _trace(user_id, action, post_id):
    return {
        "user_id": user_id,
        "action": action,
        "info": str({"post_id": post_id}),
    }


def test_calculate_hot_score_increases_after_45000_seconds():
    created_at = datetime(2024, 1, 1)

    earlier_score = calculate_hot_score(10, 2, created_at)
    later_score = calculate_hot_score(10, 2,
                                      created_at + timedelta(seconds=45000))

    assert later_score == earlier_score + 1


def test_calculate_hot_score_respects_vote_direction():
    reddit_epoch = datetime(2005, 12, 8, 7, 46, 43)

    scores = (
        calculate_hot_score(10, 0, reddit_epoch),
        calculate_hot_score(0, 10, reddit_epoch),
        calculate_hot_score(5, 5, reddit_epoch),
    )

    assert scores == (1.0, -1.0, 0.0)


def test_coarse_filtering_keeps_items_within_scale():
    elements, indices = coarse_filtering([10, 20, 30], 5)

    assert (elements, list(indices)) == ([10, 20, 30], [0, 1, 2])


def test_coarse_filtering_returns_items_at_sampled_indices():
    with patch("oasis.social_platform.recsys.random.sample",
               return_value=[4, 1, 3]):
        elements, indices = coarse_filtering([10, 20, 30, 40, 50], 3)

    assert (elements, indices) == ([50, 20, 40], [4, 1, 3])


def test_get_like_post_id_pads_with_most_recent_match():
    trace_table = [
        _trace(1, "like_post", 101),
        _trace(2, "like_post", 999),
        _trace(1, "unlike_post", 888),
        _trace(1, "like_post", 102),
    ]

    result = get_like_post_id(1, "like_post", trace_table)

    assert result == [101, 102, 102, 102, 102]


def test_get_like_post_id_returns_five_most_recent_matches():
    trace_table = [
        _trace(1, "like_post", post_id) for post_id in range(101, 108)
    ]

    result = get_like_post_id(1, "like_post", trace_table)

    assert result == [103, 104, 105, 106, 107]


def test_calculate_like_similarity_averages_liked_posts():
    liked_vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
    target_vectors = np.array([[1.0, 0.0]])

    result = calculate_like_similarity(liked_vectors, target_vectors)

    np.testing.assert_allclose(result, [0.5])


def test_normalize_similarity_adjustments_keeps_base_without_scores():
    result = normalize_similarity_adjustments([], 0.4, 0.9, 0.1)

    assert result == 0.4


def test_normalize_similarity_adjustments_scales_to_score_range():
    post_scores = [(1, 0.2), (2, 0.8)]

    result = normalize_similarity_adjustments(post_scores, 0.5, 0.9, 0.3)

    np.testing.assert_allclose(result, 0.68)


def test_swap_random_posts_keeps_recommendations_at_zero_percent():
    result = swap_random_posts([1, 2, 3], [4, 5, 6], swap_percent=0)

    assert result == [1, 2, 3]


def test_swap_random_posts_replaces_requested_fraction():
    result = swap_random_posts([1, 2, 3, 4], [5, 6, 7, 8],
                               swap_percent=0.5)

    assert len(result) == 4
    assert len(set(result) & {1, 2, 3, 4}) == 2
    assert len(set(result) & {5, 6, 7, 8}) == 2


def test_rec_sys_random_all_posts():
    # Test the scenario when the number of tweets is less than or equal to the
    # maximum recommendation length
    post_table = [{"post_id": "1"}, {"post_id": "2"}]
    rec_matrix = [[], []]
    max_rec_post_len = 2  # Maximum recommendation length set to 2

    expected = [["1", "2"], ["1", "2"]]
    result = rec_sys_random(post_table, rec_matrix, max_rec_post_len)
    assert result == expected


def test_rec_sys_reddit_all_posts():
    # Test the scenario when the number of tweets is less than or equal to the
    # maximum recommendation length
    post_table = [{"post_id": "1"}, {"post_id": "2"}]
    rec_matrix = [[], []]
    max_rec_post_len = 2  # Maximum recommendation length set to 2

    expected = [["1", "2"], ["1", "2"]]
    result = rec_sys_reddit(post_table, rec_matrix, max_rec_post_len)
    assert result == expected


def test_get_like_post_id_exactly_five():
    # A user with exactly 5 liked posts must get those 5 ids back (last-5,
    # padded when fewer). Regression: the len==5 boundary previously fell
    # through to the empty-case placeholder [0], discarding the likes.
    action = "like_post"
    trace_table = [
        _trace(1, action, post_id)
        for post_id in [101, 102, 103, 104, 105]
    ]

    assert get_like_post_id(
        1, action, trace_table
    ) == [101, 102, 103, 104, 105]


def test_rec_sys_personalized_all_posts():
    # Test the scenario when the number of tweets is less than or equal to the
    # maximum recommendation length
    user_table = [
        {
            "user_id": 0,
            "bio": "I like cats"
        },
        {
            "user_id": 1,
            "bio": "I like dogs"
        },
    ]
    post_table = [
        {
            "post_id": "1",
            "user_id": 2,
            "content": "I like dogs"
        },
        {
            "post_id": "2",
            "user_id": 3,
            "content": "I like cats"
        },
    ]
    trace_table = []
    rec_matrix = [[], []]
    max_rec_post_len = 2  # Maximum recommendation length set to 2

    expected = [["1", "2"], ["1", "2"]]
    result = rec_sys_personalized(user_table, post_table, trace_table,
                                  rec_matrix, max_rec_post_len)
    assert result == expected


def test_rec_sys_personalized_twhin():
    # Test the scenario when the number of tweets is less than or equal to the
    # maximum recommendation length
    user_table = [
        {
            "user_id": 0,
            "bio": "I like cats",
            "num_followers": 3
        },
        {
            "user_id": 1,
            "bio": "I like dogs",
            "num_followers": 5
        },
        {
            "user_id": 2,
            "bio": "",
            "num_followers": 5
        },
        {
            "user_id": 3,
            "bio": "",
            "num_followers": 5
        },
    ]
    post_table = [
        {
            "post_id": "1",
            "user_id": 2,
            "content": "I like dogs",
            "created_at": "0"
        },
        {
            "post_id": "2",
            "user_id": 3,
            "content": "I like cats",
            "created_at": "0"
        },
    ]
    trace_table = []
    rec_matrix = [[], [], [], []]
    max_rec_post_len = 2  # Maximum recommendation length set to 2
    latest_post_count = len(post_table)
    expected = [["1", "2"], ["1", "2"], ["1", "2"], ["1", "2"]]

    reset_globals()
    result = rec_sys_personalized_twh(user_table,
                                      post_table,
                                      latest_post_count,
                                      trace_table,
                                      rec_matrix,
                                      max_rec_post_len,
                                      current_time=1)
    assert result == expected


def test_rec_sys_random_sample_posts():
    # Test the scenario when the number of tweets is greater than the maximum
    # recommendation length
    post_table = [{"post_id": "1"}, {"post_id": "2"}, {"post_id": "3"}]
    rec_matrix = [[], []]  # Assuming two users
    max_rec_post_len = 2  # Maximum recommendation length set to 2

    result = rec_sys_random(post_table, rec_matrix, max_rec_post_len)
    # Validate that each user received 2 tweet IDs
    for rec in result:
        assert len(rec) == max_rec_post_len
        # Validate that the recommended tweet IDs are indeed from the original
        # list of tweet IDs
        for post_id in rec:
            assert post_id in ["1", "2", "3"]


def test_rec_sys_reddit_sample_posts():
    # Test the scenario when the number of tweets is greater than the maximum
    # recommendation length
    post_table = [
        {
            "post_id": "1",
            "num_likes": 100000,
            "num_dislikes": 25,
            "created_at": "2024-06-25 12:00:00.222000",
        },
        {
            "post_id": "2",
            "num_likes": 90,
            "num_dislikes": 30,
            "created_at": "2024-06-26 12:00:00.321009",
        },
        {
            "post_id": "3",
            "num_likes": 75,
            "num_dislikes": 50,
            "created_at": "2024-06-27 12:00:00.123009",
        },
        {
            "post_id": "4",
            "num_likes": 70,
            "num_dislikes": 50,
            "created_at": "2024-06-27 13:00:00.321009",
        },
    ]
    rec_matrix = [[], []]  # Assuming two users
    max_rec_post_len = 3  # Maximum recommendation length set to 3

    result = rec_sys_reddit(post_table, rec_matrix, max_rec_post_len)

    assert result == [["3", "4", "1"], ["3", "4", "1"]]


def test_rec_sys_personalized_sample_posts():
    # Test the scenario when the number of tweets is greater than the maximum
    # recommendation length
    user_table = [
        {
            "user_id": 0,
            "bio": "I like cats"
        },
        {
            "user_id": 1,
            "bio": "I like dogs"
        },
    ]
    post_table = [
        {
            "post_id": "1",
            "user_id": 2,
            "content": "I like dogs"
        },
        {
            "post_id": "2",
            "user_id": 3,
            "content": "I like cats"
        },
        {
            "post_id": "3",
            "user_id": 4,
            "content": "I like birds"
        },
    ]
    trace_table = []  # Not used in this test, but included for completeness
    rec_matrix = [[], []]  # Assuming two users
    max_rec_post_len = 2  # Maximum recommendation length set to 2

    result = rec_sys_personalized(user_table, post_table, trace_table,
                                  rec_matrix, max_rec_post_len)
    # Validate that each user received 2 tweet IDs
    for rec in result:
        assert len(rec) == max_rec_post_len
        # Validate that the recommended tweet IDs are indeed from the original
        # list of tweet IDs
        for post_id in rec:
            assert post_id in ["1", "2", "3"]

    # The personalized recommendation should be based on the user's bio
    for i in range(len(result)):
        if i == 0:
            assert result[i] == ["2", "1"]

        if i == 1:
            assert result[i] == ["1", "2"]


def test_rec_sys_personalized_twhin_sample_posts():
    # Test the scenario when the number of tweets is greater than the maximum
    # recommendation length
    user_table = [
        {
            "user_id": 0,
            "bio": "I like cats",
            "num_followers": 3
        },
        {
            "user_id": 1,
            "bio": "I like dogs",
            "num_followers": 3
        },
        {
            "user_id": 2,
            "bio": "",
            "num_followers": 3
        },
        {
            "user_id": 3,
            "bio": "",
            "num_followers": 3
        },
        {
            "user_id": 4,
            "bio": "",
            "num_followers": 3
        },
    ]
    post_table = [
        {
            "post_id": "1",
            "user_id": 2,
            "content": "I like dogs",
            "created_at": "0"
        },
        {
            "post_id": "2",
            "user_id": 3,
            "content": "I like cats",
            "created_at": "0"
        },
        {
            "post_id": "3",
            "user_id": 4,
            "content": "I like birds",
            "created_at": "0"
        },
    ]
    trace_table = []  # Not used in this test, but included for completeness
    rec_matrix = [[], [], [], [], []]  # Assuming five users
    max_rec_post_len = 2  # Maximum recommendation length set to 2
    latest_post_count = len(post_table)
    reset_globals()
    result = rec_sys_personalized_twh(user_table,
                                      post_table,
                                      latest_post_count,
                                      trace_table,
                                      rec_matrix,
                                      max_rec_post_len,
                                      current_time=1)
    # Validate that each user received 2 tweet IDs
    for rec in result:
        assert len(rec) == max_rec_post_len
        # Validate that the recommended tweet IDs are indeed from the original
        # list of tweet IDs
        for post_id in rec:
            assert post_id in ["1", "2", "3"]

    # The personalized recommendation should be based on the user's bio
    for i in range(len(result)):
        if i == 0:
            assert result[i] == ["2", "1"]

        if i == 1:
            assert result[i] == ["1", "2"]
