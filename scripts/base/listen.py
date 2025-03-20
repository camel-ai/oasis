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
import subprocess

import redis as redis_client
from dotenv import load_dotenv

load_dotenv()

redis_url = os.getenv("REDIS_URL")

redis = redis_client.from_url(redis_url)
process_table: dict[str, subprocess.Popen] = {}
script_path = os.path.dirname(os.path.dirname(__file__))


def redis_publish(content_id, message: dict):
    redis.publish(f"predict_{content_id}", json.dumps(message))


def start_predict(predict_id, content):

    command = (f"python {script_path}/twitter_game/twitter_simulation.py "
               f"--db_path log/twitter_{predict_id}.db "
               f"--content '{content}'  "
               f"--content_id {predict_id}")

    proc = subprocess.Popen([command], env=os.environ.copy(), shell=True)
    process_table[predict_id] = proc


if __name__ == "__main__":
    pubsub = redis.pubsub()
    pubsub.subscribe("predict")
    try:
        for message in pubsub.listen():
            print(message)
            if message["type"] != "message":
                continue

            data = json.loads(message["data"])
            predict_id = data["predict_id"]
            if data["action"] == "start":
                start_predict(predict_id, data["content"])
            if data["action"] == "end" and predict_id in process_table:
                process_table.pop(predict_id).kill()
    except Exception as e:
        print(e)
    finally:
        pubsub.unsubscribe("predict")
        pubsub.close()
        redis.close()
        for process in process_table.values():
            process.kill()
