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
import sqlite3
from datetime import datetime

from neo4j import GraphDatabase


# Neo4j configuration
class Neo4jConfig:

    def __init__(self, uri, username, password):
        self.uri = uri
        self.username = username
        self.password = password


neo4j_config = Neo4jConfig(
    uri=os.getenv('NEO4J_URI'),
    username=os.getenv('NEO4J_USERNAME'),
    password=os.getenv('NEO4J_PASSWORD'),
)


# Connect to Neo4j database
def connect_to_neo4j(config):
    return GraphDatabase.driver(config.uri,
                                auth=(config.username, config.password))


# Connect to SQLite database
def connect_to_sqlite(db_path):
    return sqlite3.connect(db_path)


# Format date time string
def format_datetime(dt_string):
    try:
        dt = datetime.strptime(dt_string, "%Y-%m-%d %H:%M:%S.%f")
        print(dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z")
        return dt.strftime(
            "%Y-%m-%dT%H:%M:%S.%f"
        )[:-3] + "Z"  # Keep milliseconds, remove excess decimal places
    except Exception:
        return int(dt_string)  # if dt_string is a integer


# Create user node
def create_user_node(tx, user_id, action_info, created_at):
    query = ("MERGE (u:User {user_id: $user_id}) "
             "SET u += $action_info, u.created_at = $created_at")
    tx.run(query,
           user_id=user_id,
           action_info=action_info,
           created_at=created_at)


# Create follow relationship
def create_follow_relationship(tx, user_id, follow_id, timestamp):
    query = ("MERGE (u1:User {user_id: $user_id}) "
             "MERGE (u2:User {user_id: $follow_id}) "
             "CREATE (u1)-[:FOLLOWS {timestamp: $timestamp}]->(u2)")
    tx.run(query, user_id=user_id, follow_id=follow_id, timestamp=timestamp)


# Main function
def main(sqlite_db_path):
    # Connect to databases
    neo4j_driver = connect_to_neo4j(neo4j_config)
    sqlite_conn = connect_to_sqlite(sqlite_db_path)
    sqlite_cursor = sqlite_conn.cursor()

    with neo4j_driver.session() as session:
        # Query the user table
        sqlite_cursor.execute(
            "SELECT user_id, user_name, name, bio, created_at FROM user ORDER "
            "BY created_at")
        for row in sqlite_cursor:
            user_id, user_name, name, bio, created_at = row
            info_dict = {"user_name": user_name, "name": name, "bio": bio}
            print('info_dict:\n', info_dict)
            session.execute_write(create_user_node, user_id, info_dict,
                                  created_at)

        sqlite_cursor.execute(
            "SELECT follower_id, followee_id, created_at FROM follow ORDER BY "
            "created_at")
        for row in sqlite_cursor:
            follower_id, followee_id, created_at = row
            print(f"follower_id:{follower_id}, followee_id:{followee_id}, "
                  f"created_at:{created_at}")
            session.execute_write(create_follow_relationship, follower_id,
                                  followee_id, created_at)

    # Close connections
    sqlite_conn.close()
    neo4j_driver.close()


if __name__ == "__main__":
    # Replace with your SQLite database path
    sqlite_db_path = "misinfo.db"
    main(sqlite_db_path)
