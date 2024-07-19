import os
import os.path as osp
import sqlite3
from datetime import datetime

from social_simulation.social_platform.database import create_db

parent_folder = osp.dirname(osp.abspath(__file__))
test_db_filepath = osp.join(parent_folder, "test.db")


def test_multi_signup():
    if os.path.exists(test_db_filepath):
        os.remove(test_db_filepath)
    N = 100000
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
