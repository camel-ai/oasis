
import json
import redis as redis_client

redis = redis_client.from_url('redis://localhost:6379/0')


def redis_publish(content_id, message: dict):
    print(f'predict_{content_id}')
    redis.publish(f'predict_{content_id}', json.dumps(message))
