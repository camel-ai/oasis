# File: ./test/infra/test_create_database.py
import datetime
import os
import os.path as osp
import sqlite3

import pytest

from social_platform.database import (create_db, fetch_rec_table_as_matrix,
                                      fetch_table_from_db)

parent_folder = osp.dirname(osp.abspath(__file__))
db_filepath = osp.join(parent_folder, "test.db")


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
        (2, 'testuser', 'Test User', 'A test user', '2024-04-21 22:02:42', 0,
         0))
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

    cursor.execute(
        ("INSERT INTO user (agent_id, user_name, name, bio, created_at, "
         "num_followings, num_followers) VALUES (?, ?, ?, ?, ?, ?, ?)"),
        (1, 'testuser_2', 'Test User_2', 'Another user', '2024-05-21 22:02:42',
         0, 0))
    conn.commit()

    expected_result = [{
        'user_id': 1,
        'agent_id': 2,
        'user_name': 'testuser',
        'name': 'Updated User',
        'bio': 'A test user',
        'created_at': '2024-04-21 22:02:42',
        'num_followings': 0,
        'num_followers': 0
    }, {
        'user_id': 2,
        'agent_id': 1,
        'user_name': 'testuser_2',
        'name': 'Test User_2',
        'bio': 'Another user',
        'created_at': '2024-05-21 22:02:42',
        'num_followings': 0,
        'num_followers': 0
    }]

    actual_result = fetch_table_from_db(cursor, 'user')

    # 使用assert语句进行比较
    assert actual_result == expected_result, "The fetched data does not match."

    cursor.execute(
        ("INSERT INTO user (agent_id, user_name, name, bio, created_at, "
         "num_followings, num_followers) VALUES (?, ?, ?, ?, ?, ?, ?)"),
        (3, 'testuser_3', 'Test User_3', 'Third user', '2024-05-21 22:02:42',
         0, 0))
    conn.commit()
    # Delete the user
    cursor.execute("DELETE FROM user WHERE user_name = 'testuser_3'")
    conn.commit()

    # Assert the user was deleted correctly
    cursor.execute("SELECT * FROM user WHERE user_name = 'testuser_3'")
    assert cursor.fetchone() is None


def test_tweet_operations():
    conn = sqlite3.connect(db_filepath)
    cursor = conn.cursor()
    # Insert a tweet:
    cursor.execute(
        ("INSERT INTO tweet (user_id, content, created_at, num_likes, "
         "num_dislikes) VALUES (?, ?, ?, ?, ?)"),
        (1, 'This is a test tweet', '2024-04-21 22:02:42', 0, 1))
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
    assert tweet[5] == 1

    # Update the tweet
    cursor.execute("UPDATE tweet SET content = ? WHERE content = ?",
                   ('Updated tweet', 'This is a test tweet'))
    conn.commit()

    expected_result = [{
        'tweet_id': 1,
        'user_id': 1,
        'content': 'Updated tweet',
        'created_at': '2024-04-21 22:02:42',
        'num_likes': 0,
        'num_dislikes': 1,
    }]
    actual_result = fetch_table_from_db(cursor, 'tweet')

    # 使用assert语句进行比较
    assert actual_result == expected_result, "The fetched data does not match."

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


def test_dislike_operations():
    conn = sqlite3.connect(db_filepath)
    cursor = conn.cursor()
    # Insert a like relation
    cursor.execute(
        "INSERT INTO dislike (user_id, tweet_id, created_at) VALUES (?, ?, ?)",
        (1, 2, '2024-04-21 22:02:42'))
    conn.commit()

    # Assert the like relation was inserted correctly
    cursor.execute("SELECT * FROM dislike WHERE user_id = 1 AND tweet_id = 2")
    dislike = cursor.fetchone()
    assert dislike is not None
    assert dislike[1] == 1
    assert dislike[2] == 2
    assert dislike[3] == '2024-04-21 22:02:42'

    # Delete the like relation
    cursor.execute("DELETE FROM dislike WHERE user_id = 1 AND tweet_id = 2")
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

    expected_result = [{
        'user_id': 1,
        'created_at': created_at,
        'action': 'test_action',
        'info': 'test_info',
    }]

    actual_result = fetch_table_from_db(cursor, 'trace')
    assert actual_result == expected_result
    # Delete the trace
    cursor.execute("DELETE FROM trace WHERE user_id = 1 AND created_at = ?",
                   (created_at, ))
    conn.commit()

    # Assert the trace was deleted correctly
    cursor.execute("SELECT * FROM trace WHERE user_id = 1 AND created_at = ?",
                   (created_at, ))
    assert cursor.fetchone() is None


def test_rec_operations():
    conn = sqlite3.connect(db_filepath)
    cursor = conn.cursor()
    # Insert a trace
    cursor.execute(("INSERT INTO rec (user_id, tweet_id) "
                    "VALUES (?, ?)"), (2, 2))
    cursor.execute(("INSERT INTO rec (user_id, tweet_id) "
                    "VALUES (?, ?)"), (2, 3))
    cursor.execute(("INSERT INTO rec (user_id, tweet_id) "
                    "VALUES (?, ?)"), (1, 3))
    conn.commit()

    # Assert the rec was inserted correctly
    cursor.execute("SELECT * FROM rec WHERE user_id = ? AND tweet_id = ?",
                   (2, 2))
    record = cursor.fetchone()
    assert record is not None
    assert record[0] == 2
    assert record[1] == 2

    cursor.execute("SELECT * FROM rec WHERE user_id = ? AND tweet_id = ?",
                   (2, 3))
    record = cursor.fetchone()
    assert record is not None
    assert record[0] == 2
    assert record[1] == 3

    assert fetch_rec_table_as_matrix(cursor) == [None, [3], [2, 3]]
    # Delete the rec
    cursor.execute("DELETE FROM rec WHERE user_id = 2 AND tweet_id = 2")
    conn.commit()

    # Assert the rec was deleted correctly
    cursor.execute("SELECT * FROM rec WHERE user_id = 2 AND tweet_id = 2")
    assert cursor.fetchone() is None


def test_comment_operations():
    conn = sqlite3.connect(db_filepath)
    cursor = conn.cursor()

    # Insert a comment:
    cursor.execute(
        ("INSERT INTO comment (tweet_id, user_id, content, created_at) "
         "VALUES (?, ?, ?, ?)"),
        (1, 2, 'This is a test comment', '2024-04-21 22:05:00'))
    conn.commit()

    # Assert the comment was inserted correctly
    cursor.execute(
        "SELECT * FROM comment WHERE content = 'This is a test comment'")
    comment = cursor.fetchone()
    assert comment is not None, "Comment insertion failed."
    assert comment[1] == 1, "Tweet ID mismatch."
    assert comment[2] == 2, "User ID mismatch."
    assert comment[3] == 'This is a test comment', "Content mismatch."
    assert comment[4] == '2024-04-21 22:05:00', "Created at mismatch."
    assert comment[5] == 0, "Likes count mismatch."
    assert comment[6] == 0, "Dislikes count mismatch."

    # Update the comment
    cursor.execute("UPDATE comment SET content = ? WHERE content = ?",
                   ('Updated comment', 'This is a test comment'))
    conn.commit()

    expected_result = [{
        'comment_id': 1,
        'tweet_id': 1,
        'user_id': 2,
        'content': 'Updated comment',
        'created_at': '2024-04-21 22:05:00',
        'num_likes': 0,
        'num_dislikes': 0,
    }]
    actual_result = fetch_table_from_db(cursor, 'comment')

    # 使用assert语句进行比较
    assert actual_result == expected_result, "The fetched data does not match."

    # Assert the comment was updated correctly
    cursor.execute("SELECT * FROM comment WHERE content = 'Updated comment'")
    comment = cursor.fetchone()
    assert comment[3] == 'Updated comment', "Comment update failed."

    # Delete the comment
    cursor.execute("DELETE FROM comment WHERE content = 'Updated comment'")
    conn.commit()

    # Assert the comment was deleted correctly
    cursor.execute("SELECT * FROM comment WHERE content = 'Updated comment'")
    assert cursor.fetchone() is None, "Comment deletion failed."


def test_comment_like_operations():
    conn = sqlite3.connect(db_filepath)
    cursor = conn.cursor()

    # Insert a comment like relation
    cursor.execute(
        "INSERT INTO comment_like (user_id, comment_id, created_at) VALUES "
        "(?, ?, ?)", (1, 2, '2024-04-21 22:05:00'))
    conn.commit()

    # Assert the comment like relation was inserted correctly
    cursor.execute("SELECT * FROM comment_like WHERE user_id = 1 AND "
                   "comment_id = 2")
    comment_like = cursor.fetchone()
    assert comment_like is not None, "Comment like insertion failed."
    assert comment_like[1] == 1, "User ID mismatch."
    assert comment_like[2] == 2, "Comment ID mismatch."
    assert comment_like[3] == '2024-04-21 22:05:00', "Created at mismatch."

    # Delete the comment like relation
    cursor.execute("DELETE FROM comment_like WHERE user_id = 1 AND "
                   "comment_id = 2")
    conn.commit()

    # Assert the comment like relation was deleted correctly
    cursor.execute("SELECT * FROM comment_like WHERE user_id = 1 AND "
                   "comment_id = 2")
    assert cursor.fetchone() is None, "Comment like deletion failed."


def test_comment_dislike_operations():
    conn = sqlite3.connect(db_filepath)
    cursor = conn.cursor()

    # Insert a comment dislike relation
    cursor.execute(
        "INSERT INTO comment_dislike (user_id, comment_id, created_at) VALUES "
        "(?, ?, ?)", (1, 2, '2024-04-21 22:05:00'))
    conn.commit()

    # Assert the comment dislike relation was inserted correctly
    cursor.execute("SELECT * FROM comment_dislike WHERE user_id = 1 AND "
                   "comment_id = 2")
    comment_dislike = cursor.fetchone()
    assert comment_dislike is not None, "Comment dislike insertion failed."
    assert comment_dislike[1] == 1, "User ID mismatch."
    assert comment_dislike[2] == 2, "Comment ID mismatch."
    assert comment_dislike[3] == '2024-04-21 22:05:00', "Created at mismatch."

    # Delete the comment dislike relation
    cursor.execute("DELETE FROM comment_dislike WHERE user_id = 1 AND "
                   "comment_id = 2")
    conn.commit()

    # Assert the comment dislike relation was deleted correctly
    cursor.execute("SELECT * FROM comment_dislike WHERE user_id = 1 AND "
                   "comment_id = 2")
    assert cursor.fetchone() is None, "Comment dislike deletion failed."
