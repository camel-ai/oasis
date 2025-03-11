import json
import os
import redis as redis_client
import subprocess
from dotenv import load_dotenv

load_dotenv()

redis = redis_client.from_url(os.getenv("REDIS_URL"))
process_table: dict[str, subprocess.Popen] = {}


def redis_publish(content_id, message: dict):
    # print(f'predict_{content_id}')
    redis.publish(f"predict_{content_id}", json.dumps(message))


def start_predict(predict_id, content):
    predict_id = 0
    content = "hello world"
    command = f"python ../scripts/twitter_simulation.py --db_path log/twitter_{predict_id}.db --content '{content}'  --content_id {predict_id}"

    proc = subprocess.Popen([command], env=os.environ.copy(), shell=True)
    process_table[predict_id] = proc


if __name__ == "__main__":
    pubsub = redis.pubsub()
    pubsub.subscribe("predict")
    try:
        for message in pubsub.listen():
            if message["type"] != "message":
                continue

            data = json.loads(message["data"])

    except Exception as e:
        print(e)
        pubsub.unsubscribe("predict")
        pubsub.close()
        redis.close()
        for process in process_table.values():
            process.kill()
