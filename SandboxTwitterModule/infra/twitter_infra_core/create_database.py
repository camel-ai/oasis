# File: SandboxTwitterModule/twitter_infra_core/create_database.py
from __future__ import annotations

import os
import os.path as osp
import sqlite3

SCHEMA_DIR = "schema"
DB_DIR = "db"
DB_NAME = "twitter.db"

USER_SCHEMA_SQL = "user.sql"
TWEET_SCHEMA_SQL = "tweet.sql"
FOLLOW_SCHEMA_SQL = "follow.sql"
MUTE_SCHEMA_SQL = "mute.sql"
LIKE_SCHEMA_SQL = "like.sql"
TRACE_SCHEMA_SQL = "trace.sql"

TABLE_NAMES = {"user", "tweet", "follow", "mute", "like", "trace"}


def get_db_path() -> str:
    curr_file_path = osp.abspath(__file__)
    parent_dir = osp.dirname(osp.dirname(curr_file_path))
    db_dir = osp.join(parent_dir, DB_DIR)
    os.makedirs(db_dir, exist_ok=True)
    db_path = osp.join(db_dir, DB_NAME)
    return db_path


def get_schema_dir_path() -> str:
    curr_file_path = osp.abspath(__file__)
    parent_dir = osp.dirname(osp.dirname(curr_file_path))
    schema_dir = osp.join(parent_dir, SCHEMA_DIR)
    return schema_dir


def create_db(db_path: str | None = None):
    r"""Create the database if it does not exist. A :obj:`twitter.db`
    file will be automatically created  in the :obj:`data` directory.
    """
    schema_dir = get_schema_dir_path()
    if db_path is None:
        db_path = get_db_path()

    # Connect to the database:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Read and execute the user table SQL script:
        user_sql_path = osp.join(schema_dir, USER_SCHEMA_SQL)
        with open(user_sql_path, 'r') as sql_file:
            user_sql_script = sql_file.read()
        cursor.executescript(user_sql_script)

        # Read and execute the tweet table SQL script:
        tweet_sql_path = osp.join(schema_dir, TWEET_SCHEMA_SQL)
        with open(tweet_sql_path, 'r') as sql_file:
            tweet_sql_script = sql_file.read()
        cursor.executescript(tweet_sql_script)

        # Read and execute the follow table SQL script:
        follow_sql_path = osp.join(schema_dir, FOLLOW_SCHEMA_SQL)
        with open(follow_sql_path, 'r') as sql_file:
            follow_sql_script = sql_file.read()
        cursor.executescript(follow_sql_script)

        # Read and execute the mute table SQL script:
        mute_sql_path = osp.join(schema_dir, MUTE_SCHEMA_SQL)
        with open(mute_sql_path, 'r') as sql_file:
            mute_sql_script = sql_file.read()
        cursor.executescript(mute_sql_script)

        # Read and execute the like table SQL script:
        like_sql_path = osp.join(schema_dir, LIKE_SCHEMA_SQL)
        with open(like_sql_path, 'r') as sql_file:
            like_sql_script = sql_file.read()
        cursor.executescript(like_sql_script)

        # Read and execute the trace table SQL script:
        trace_sql_path = osp.join(schema_dir, TRACE_SCHEMA_SQL)
        with open(trace_sql_path, 'r') as sql_file:
            trace_sql_script = sql_file.read()
        cursor.executescript(trace_sql_script)

        # Commit the changes:
        conn.commit()

        print("All tables created successfully.")
    except sqlite3.Error as e:
        print(f"An error occurred while creating tables: {e}")
    finally:
        # Close the database connection
        conn.close()


def print_db_tables_summary():
    # Connect to the SQLite database
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Retrieve a list of all tables in the database
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    # Print a summary of each table
    for table in tables:
        table_name = table[0]
        if table_name not in TABLE_NAMES:
            continue
        print(f"Table: {table_name}")

        # Retrieve the table schema
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        column_names = [column[1] for column in columns]
        print("- Columns:", column_names)

        # Retrieve and print foreign key information
        cursor.execute(f"PRAGMA foreign_key_list({table_name})")
        foreign_keys = cursor.fetchall()
        if foreign_keys:
            print("- Foreign Keys:")
            for fk in foreign_keys:
                print(f"    {fk[2]} references {fk[3]}({fk[4]}) on update "
                      f"{fk[5]} on delete {fk[6]}")
        else:
            print("  No foreign keys.")

        # Print the first few rows of the table
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 5;")
        rows = cursor.fetchall()
        for row in rows:
            print(row)
        print()  # Adds a newline for better readability between tables

    # Close the database connection
    conn.close()


if __name__ == "__main__":
    create_db()
    print_db_tables_summary()
