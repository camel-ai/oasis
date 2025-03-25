import json
import os
import redis as redis_client
from dotenv import load_dotenv

load_dotenv()

redis_url = os.getenv("REDIS_URL")

redis = redis_client.from_url(redis_url)


redis.publish(
    "predict",
    json.dumps(
        {"action": "start", "predict_id": 1000, "content": "this is the best product"}
    ),
)


# redis.publish("predict_new_1000", "Repeat this is best")
