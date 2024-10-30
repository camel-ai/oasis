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
import json
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
    dt = datetime.strptime(dt_string, "%Y-%m-%d %H:%M:%S.%f")
    print(dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z")
    return dt.strftime(
        "%Y-%m-%dT%H:%M:%S.%f"
    )[:-3] + "Z"  # Keep milliseconds, remove excess decimal places


# Create user node
def create_user_node(tx, user_id, action_info, created_at):
    formatted_datetime = format_datetime(created_at)
    query = ("MERGE (u:User {user_id: $user_id}) "
             "SET u += $action_info, u.created_at = datetime($created_at)")
    tx.run(query,
           user_id=user_id,
           action_info=action_info,
           created_at=formatted_datetime)


# Create follow relationship
def create_follow_relationship(tx, user_id, follow_id, created_at, timestamp):
    formatted_datetime = format_datetime(created_at)
    query = ("MERGE (u1:User {user_id: $user_id}) "
             "MERGE (u2:User {user_id: $follow_id}) "
             "CREATE (u1)-[:FOLLOWS {created_at: datetime($created_at), "
             "timestamp: $timestamp}]->(u2)")
    tx.run(query,
           user_id=user_id,
           follow_id=follow_id,
           created_at=formatted_datetime,
           timestamp=timestamp)


# Main function
def main(sqlite_db_path):
    # Connect to databases
    neo4j_driver = connect_to_neo4j(neo4j_config)
    sqlite_conn = connect_to_sqlite(sqlite_db_path)
    sqlite_cursor = sqlite_conn.cursor()

    # Query the trace table
    sqlite_cursor.execute(
        "SELECT user_id, created_at, action, info FROM trace ORDER BY "
        "created_at")

    with neo4j_driver.session() as session:
        for row in sqlite_cursor:
            user_id, created_at, action, info = row
            info_dict = json.loads(info)

            if action == 'sign_up':
                session.execute_write(create_user_node, user_id, info_dict,
                                      created_at)
            elif action == 'follow':
                follow_id = int(info_dict['follow_id'])
                timestamp = int(info_dict['time_stamp']
                                )  # Extract timestamp from info_dict
                session.execute_write(create_follow_relationship, user_id,
                                      follow_id, created_at, timestamp)

    # Close connections
    sqlite_conn.close()
    neo4j_driver.close()


if __name__ == "__main__":
    # Replace with your SQLite database path
    sqlite_db_path = "all_360_follow.db"
    main(sqlite_db_path)
