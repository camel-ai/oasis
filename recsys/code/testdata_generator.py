# File: recsys/code/testdata_generator.py

import sqlite3
import random

def initialize_test_db(db_file):
    """
    Initializes the test database with necessary tables and sample data for unit testing.

    Args:
        db_file (str): Path to the SQLite database file.
    """
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Drop tables if they already exist
    cursor.execute("DROP TABLE IF EXISTS user")
    cursor.execute("DROP TABLE IF EXISTS tweets")
    cursor.execute("DROP TABLE IF EXISTS rec")

    # Create user table
    cursor.execute("""
        CREATE TABLE user (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT
        )
    """)

    # Create tweets table
    cursor.execute("""
        CREATE TABLE tweets (
            tweet_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            content TEXT,
            FOREIGN KEY (user_id) REFERENCES user (user_id)
        )
    """)

    # Create recommendations table
    cursor.execute("""
        CREATE TABLE rec (
            user_id INTEGER,
            recommendations TEXT,
            FOREIGN KEY (user_id) REFERENCES user (user_id)
        )
    """)

    # Insert users
    for i in range(1, 101):  # Creating 100 users
        cursor.execute("INSERT INTO user (username) VALUES (?)", (f'user_{i}',))

    # Insert tweets for each user
    for user_id in range(1, 101):
        for _ in range(10):  # 10 tweets per user
            tweet_content = f"Sample tweet content for user {user_id} number {random.randint(1, 1000)}"
            cursor.execute("INSERT INTO tweets (user_id, content) VALUES (?, ?)", (user_id, tweet_content))

    conn.commit()
    conn.close()

# Usage example
db_test_file = 'testDB.db'
initialize_test_db(db_test_file)
