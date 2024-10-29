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
import sqlite3
from datetime import datetime

from oasis.social_platform.database import create_db

parent_folder = osp.dirname(osp.abspath(__file__))
test_db_filepath = osp.join(parent_folder, "test.db")


def test_multi_signup():
    if os.path.exists(test_db_filepath):
        os.remove(test_db_filepath)
    N = 100
    create_db(test_db_filepath)

    db = sqlite3.connect(test_db_filepath, check_same_thread=False)
    db_cursor = db.cursor()
    user_insert_query = (
        "INSERT INTO user (agent_id, user_name, name, bio, created_at,"
        " num_followings, num_followers) VALUES (?, ?, ?, ?, ?, ?, ?)")
    for i in range(N):
        db_cursor.execute(user_insert_query,
                          (i, i, i, i, datetime.now(), 0, 0))
        db.commit()

    db_cursor.execute("SELECT * FROM user")
    users = db_cursor.fetchall()
    assert len(users) == N
