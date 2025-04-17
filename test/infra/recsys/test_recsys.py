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
from oasis.social_platform.recsys import (rec_sys_personalized,
                                          rec_sys_personalized_twh,
                                          rec_sys_random, rec_sys_reddit,
                                          reset_globals)


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
    # Validate that each user received 3 tweet IDs
    for rec in result:
        assert len(rec) == max_rec_post_len
        # Validate that the recommended tweet IDs are indeed from the original
        # list of tweet IDs
        for post_id in rec:
            assert post_id in ["3", "4", "1"]


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
