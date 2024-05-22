'''注意需要在写入rec_matrix的时候判断是否超过max_rec_tweet_len'''
from typing import List, Dict, Any
import random
import time


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
