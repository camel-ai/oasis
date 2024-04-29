
import os.path as osp
import sqlite3

from create_database import create_db

# prepare agents_info_list
agents_info_list = [{
    "user_id": 1,
    "user_name": 'testuser',
    "name": 'Test User',
    "bio": 'This is a test user',
    "created_at": '2024-04-21 22:02:42',
    "num_followings": 0,
    "num_followers": 0
}]
# prepare tweet dict
tweet_info_dict = {
    "tweet_id": 101,
    "user_id": 1,
    "content": "This is a test tweet",
    "created_at": '2024-04-24 22:02:42',
    "num_likes": 1
 }

def generate_agents(agents_info_list: list):
    r"""Generate the agents useing agents_info.

    Args:
        agents_info_list (List[Dict]): list of agent info
    """
    db_path = create_db()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for index, item in enumerate(agents_info_list):
        # Insert agents info to USER database
        cursor.execute(
            ("INSERT INTO user (user_name, name, bio, created_at, "
            "num_followings, num_followers) VALUES (?, ?, ?, ?, ?, ?)"),
            (item["user_name"], item["name"], item["bio"],
            item["created_at"], item["num_followings"],
            item["num_followers"]))
        conn.commit()