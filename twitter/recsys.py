'''注意需要在写入rec_matrix的时候判断是否超过max_rec_tweet_len'''
from typing import List, Dict, Any
import random
import time

import numpy as np
# init model
from sentence_transformers import SentenceTransformer

try:
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
except Exception as e:
    print(e)
    model = None

def rec_sys_random(
    user_table: List[Dict[str, Any]],
    tweet_table: List[Dict[str, Any]],
    trace_table: List[Dict[str, Any]],
    rec_matrix: List[List],
    max_rec_tweet_len: int
) -> List[List]:
    # 获取所有推文的ID
    tweet_ids = [tweet['tweet_id'] for tweet in tweet_table]

    if len(tweet_ids) <= max_rec_tweet_len:
        # 如果推文数量小于等于最大推荐数，每个用户获得所有推文ID
        new_rec_matrix = [tweet_ids] * (len(rec_matrix) - 1)
        new_rec_matrix = [None] + new_rec_matrix
    else:
        new_rec_matrix = [None]
        # 如果推文数量大于最大推荐数，每个用户随机获得指定数量的推文ID
        for _ in range(1, len(rec_matrix)):
            new_rec_matrix.append(random.sample(tweet_ids, max_rec_tweet_len))
    return new_rec_matrix

def rec_sys_personalized(
    user_table: List[Dict[str, Any]],
    tweet_table: List[Dict[str, Any]],
    trace_table: List[Dict[str, Any]],
    rec_matrix: List[List],
    max_rec_tweet_len: int
) -> List[List]:

    # 获取所有推文的ID
    tweet_ids = [tweet['tweet_id'] for tweet in tweet_table]

    if len(tweet_ids) <= max_rec_tweet_len:
        # 如果推文数量小于等于最大推荐数，每个用户获得所有推文ID
        new_rec_matrix = [tweet_ids] * (len(rec_matrix) - 1)
        new_rec_matrix = [None] + new_rec_matrix
    else:
        new_rec_matrix = [None]
        # 如果推文数量大于最大推荐数，每个用户随机获得personalized推文ID
        for idx in range(1, len(rec_matrix)):
            user_id = user_table[idx - 1]['user_id']
            user_bio = user_table[idx - 1]['bio']
            # filter out tweets that belong to the user
            available_tweet_contents = [(tweet['tweet_id'], tweet['content']) for tweet in tweet_table if
                                        tweet['user_id'] != user_id]
            # calculate similarity between user bio and tweet text
            tweet_scores = []
            for tweet_id, tweet_content in available_tweet_contents:
                if model is not None:
                    user_embedding = model.encode(user_bio)
                    tweet_embedding = model.encode(tweet_content)
                    similarity = np.dot(user_embedding, tweet_embedding) / (
                            np.linalg.norm(user_embedding) * np.linalg.norm(tweet_embedding))
                else:
                    similarity = random.random()

                tweet_scores.append((tweet_id, similarity))

            # sort tweets by similarity
            tweet_scores.sort(key=lambda x: x[1], reverse=True)

            # extract tweet ids
            rec_tweet_ids = [tweet_id for tweet_id, _ in tweet_scores[:max_rec_tweet_len]]
            new_rec_matrix.append(rec_tweet_ids)

    return new_rec_matrix



