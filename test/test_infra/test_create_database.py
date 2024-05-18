<<<<<<< Updated upstream
# File: test_create_database.py
=======
# File: ./test/test_infra/test_create_database.py
>>>>>>> Stashed changes
import datetime
import os
import os.path as osp
import sqlite3

import pytest

from SandboxTwitterModule.infra import create_db

parent_folder = osp.dirname(osp.abspath(__file__))
db_filepath = osp.join(parent_folder, "mock_twitter.db")


@pytest.fixture(scope="session", autouse=True)
def setup_session():
    # Create a mock database:
    create_db(db_filepath)
    yield
    if osp.exists(db_filepath):
        os.remove(db_filepath)


def test_user_operations():
    conn = sqlite3.connect(db_filepath)
    cursor = conn.cursor()
    # Insert a user
    cursor.execute(
        ("INSERT INTO user (agent_id, user_name, name, bio, created_at, "
         "num_followings, num_followers) VALUES (?, ?, ?, ?, ?, ?, ?)"),
        (2, 'testuser', 'Test User', 'A test user', '2024-04-21 22:02:42',
         0, 0))
    conn.commit()

    # Assert the user was inserted correctly
    cursor.execute("SELECT * FROM user WHERE user_name = 'testuser'")
    user = cursor.fetchone()
    assert user is not None
    assert user[1] == 2
    assert user[2] == 'testuser'
    assert user[3] == 'Test User'
    assert user[4] == 'A test user'
    assert user[5] == '2024-04-21 22:02:42'
    assert user[6] == 0
    assert user[7] == 0

    # Update the user
    cursor.execute("UPDATE user SET name = ? WHERE user_name = ?",
                   ('Updated User', 'testuser'))
    conn.commit()

    # Assert the user was updated correctly
    cursor.execute("SELECT * FROM user WHERE user_name = 'testuser'")
    user = cursor.fetchone()
    assert user[3] == 'Updated User'

    # Delete the user
    cursor.execute("DELETE FROM user WHERE user_name = 'testuser'")
    conn.commit()

    # Assert the user was deleted correctly
    cursor.execute("SELECT * FROM user WHERE user_name = 'testuser'")
    assert cursor.fetchone() is None


def test_tweet_operations():
    conn = sqlite3.connect(db_filepath)
    cursor = conn.cursor()
    # Insert a tweet:
    cursor.execute(
        ("INSERT INTO tweet (user_id, content, created_at, num_likes) "
         "VALUES (?, ?, ?, ?)"),
        (1, 'This is a test tweet', '2024-04-21 22:02:42', 0))
    conn.commit()

    # Assert the tweet was inserted correctly
    cursor.execute(
        "SELECT * FROM tweet WHERE content = 'This is a test tweet'")
    tweet = cursor.fetchone()
    assert tweet is not None
    assert tweet[1] == 1
    assert tweet[2] == 'This is a test tweet'
    assert tweet[3] == '2024-04-21 22:02:42'
    assert tweet[4] == 0

    # Update the tweet
    cursor.execute("UPDATE tweet SET content = ? WHERE content = ?",
                   ('Updated tweet', 'This is a test tweet'))
    conn.commit()

    # Assert the tweet was updated correctly
    cursor.execute("SELECT * FROM tweet WHERE content = 'Updated tweet'")
    tweet = cursor.fetchone()
    assert tweet[2] == 'Updated tweet'

    # Delete the tweet
    cursor.execute("DELETE FROM tweet WHERE content = 'Updated tweet'")
    conn.commit()

    # Assert the tweet was deleted correctly
    cursor.execute("SELECT * FROM tweet WHERE content = 'Updated tweet'")
    assert cursor.fetchone() is None


def test_follow_operations():
    conn = sqlite3.connect(db_filepath)
    cursor = conn.cursor()
    # Insert a follow relation:
    cursor.execute(
        ("INSERT INTO follow (follower_id, followee_id, created_at) "
         "VALUES (?, ?, ?)"), (1, 2, '2024-04-21 22:02:42'))
    conn.commit()

    # Assert the follow relation was inserted correctly
    cursor.execute(
        "SELECT * FROM follow WHERE follower_id = 1 AND followee_id = 2")
    follow = cursor.fetchone()
    assert follow is not None
    assert follow[1] == 1
    assert follow[2] == 2
    assert follow[3] == '2024-04-21 22:02:42'

    # Delete the follow relation
    cursor.execute(
        "DELETE FROM follow WHERE follower_id = 1 AND followee_id = 2")
    conn.commit()

    # Assert the follow relation was deleted correctly
    cursor.execute(
        "SELECT * FROM follow WHERE follower_id = 1 AND followee_id = 2")
    assert cursor.fetchone() is None


def test_mute_operations():
    conn = sqlite3.connect(db_filepath)
    cursor = conn.cursor()
    # Insert a mute relation
    cursor.execute(
        "INSERT INTO mute (muter_id, mutee_id, created_at) VALUES (?, ?, ?)",
        (1, 2, '2024-04-21 22:02:42'))
    conn.commit()

    # Assert the mute relation was inserted correctly
    cursor.execute("SELECT * FROM mute WHERE muter_id = 1 AND mutee_id = 2")
    mute = cursor.fetchone()
    assert mute is not None
    assert mute[1] == 1
    assert mute[2] == 2
    assert mute[3] == '2024-04-21 22:02:42'

    # Delete the mute relation
    cursor.execute("DELETE FROM mute WHERE muter_id = 1 AND mutee_id = 2")
    conn.commit()

    # Assert the mute relation was deleted correctly
    cursor.execute("SELECT * FROM mute WHERE muter_id = 1 AND mutee_id = 2")
    assert cursor.fetchone() is None


def test_like_operations():
    conn = sqlite3.connect(db_filepath)
    cursor = conn.cursor()
    # Insert a like relation
    cursor.execute(
        "INSERT INTO like (user_id, tweet_id, created_at) VALUES (?, ?, ?)",
        (1, 2, '2024-04-21 22:02:42'))
    conn.commit()

    # Assert the like relation was inserted correctly
    cursor.execute("SELECT * FROM like WHERE user_id = 1 AND tweet_id = 2")
    like = cursor.fetchone()
    assert like is not None
    assert like[1] == 1
    assert like[2] == 2
    assert like[3] == '2024-04-21 22:02:42'

    # Delete the like relation
    cursor.execute("DELETE FROM like WHERE user_id = 1 AND tweet_id = 2")
    conn.commit()

    # Assert the like relation was deleted correctly
    cursor.execute("SELECT * FROM like WHERE user_id = 1 AND tweet_id = 2")
    assert cursor.fetchone() is None


def test_trace_operations():
    conn = sqlite3.connect(db_filepath)
    cursor = conn.cursor()
    # Insert a trace
    created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    cursor.execute(("INSERT INTO trace (user_id, created_at, action, info) "
                    "VALUES (?, ?, ?, ?)"),
                   (1, created_at, 'test_action', 'test_info'))
    conn.commit()

    # Assert the trace was inserted correctly
    cursor.execute("SELECT * FROM trace WHERE user_id = 1 AND created_at = ?",
                   (created_at, ))
    trace = cursor.fetchone()
    assert trace is not None
    assert trace[0] == 1
    assert trace[1] == created_at
    assert trace[2] == 'test_action'
    assert trace[3] == 'test_info'

    # Delete the trace
    cursor.execute("DELETE FROM trace WHERE user_id = 1 AND created_at = ?",
                   (created_at, ))
    conn.commit()

    # Assert the trace was deleted correctly
    cursor.execute("SELECT * FROM trace WHERE user_id = 1 AND created_at = ?",
                   (created_at, ))
    assert cursor.fetchone() is None
