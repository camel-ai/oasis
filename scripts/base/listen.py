import json
import os
import redis as redis_client
import subprocess
from dotenv import load_dotenv

load_dotenv()

redis_url = os.getenv("REDIS_URL")

redis = redis_client.from_url(redis_url)
process_table: dict[str, subprocess.Popen] = {}
script_path = os.path.dirname(os.path.dirname(__file__))


def redis_publish(content_id, message: dict):
    redis.publish(f"predict_{content_id}", json.dumps(message))


def start_predict(predict_id, content):

    # command = f"python {script_path}/twitter_game/twitter_simulation.py --db_path log/twitter_{predict_id}.db --content '{content}'  --content_id {predict_id}"

    proc = subprocess.Popen(
        [
            "python",
            f"{script_path}/twitter_game/twitter_simulation.py",
            "--db_path",
            f"log/twitter_{predict_id}.db",
            "--content",
            f"{content}",
            "--content_id",
            f"{predict_id}",
        ],
        env=os.environ.copy(),
        shell=True,
    )
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
