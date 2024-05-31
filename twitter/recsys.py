'''注意需要在写入rec_matrix的时候判断是否超过max_rec_tweet_len'''
import random
import time
from .typing import ActionType
import numpy as np
# init model
from sentence_transformers import SentenceTransformer
from typing import Any, Dict, List

try:
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
except Exception as e:
    print(e)
    model = None

def rec_sys_random(user_table: List[Dict[str,
                                         Any]], tweet_table: List[Dict[str,
                                                                       Any]],
                   trace_table: List[Dict[str, Any]], rec_matrix: List[List],
                   max_rec_tweet_len: int) -> List[List]:
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


def normalize_similarity_adjustments(tweet_scores, base_similarity, like_similarity, dislike_similarity):
    """ Normalize the adjustments to keep them in scale with overall similarities. """
    if len(tweet_scores) == 0:
        return base_similarity

    max_score = max(tweet_scores, key=lambda x: x[1])[1]
    min_score = min(tweet_scores, key=lambda x: x[1])[1]
    score_range = max_score - min_score
    adjustment = (like_similarity - dislike_similarity) * (score_range / 2)  # Scale adjustment to half of range
    return base_similarity + adjustment

def swap_random_tweets(rec_tweet_ids, tweet_ids, swap_percent=0.1):
    """ Swap a percentage of recommended tweets with random tweets. """
    num_to_swap = int(len(rec_tweet_ids) * swap_percent)
    tweets_to_swap = random.sample(tweet_ids, num_to_swap)
    indices_to_replace = random.sample(range(len(rec_tweet_ids)), num_to_swap)

    for idx, new_tweet in zip(indices_to_replace, tweets_to_swap):
        rec_tweet_ids[idx] = new_tweet

    return rec_tweet_ids


def get_trace_contents(user_id, action, tweet_table, trace_table):

    # Get tweet IDs from trace table for the given user and action
    trace_tweet_ids = [trace['tweet_id'] for trace in trace_table
                        if trace['user_id'] == user_id and trace['action'] == action]
    # Fetch tweet contents from tweet table where tweet IDs match those in the trace
    trace_contents = [tweet['content'] for tweet in tweet_table if tweet['tweet_id'] in trace_tweet_ids]
    return trace_contents

def rec_sys_personalized_with_trace(
    user_table: List[Dict[str, Any]],
    tweet_table: List[Dict[str, Any]],
    trace_table: List[Dict[str, Any]],
    rec_matrix: List[List],
    max_rec_tweet_len: int,
    swap_rate: float = 0.1,
) -> List[List]:
    """
    This version:
    1. If the number of tweets is less than or equal to the maximum recommended length, each user gets all tweet IDs

    2. Otherwise:
        - For each user, get a like-trace pool and dislike-trace pool from the trace table
        - For each user, calculate the similarity between the user's bio and the tweet text
        - Use the trace table to adjust the similarity score
        - Swap 10% of the recommended tweets with the random tweets
    """

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

            # filter out like-trace and dislike-trace
            like_trace_contents = get_trace_contents(user_id, ActionType.LIKE.value, tweet_table, trace_table)
            dislike_trace_contents = get_trace_contents(user_id, ActionType.UNLIKE.value, tweet_table, trace_table)
            # calculate similarity between user bio and tweet text
            tweet_scores = []
            for tweet_id, tweet_content in available_tweet_contents:
                if model is not None:
                    user_embedding = model.encode(user_bio)
                    tweet_embedding = model.encode(tweet_content)
                    base_similarity = np.dot(user_embedding, tweet_embedding) / (
                                np.linalg.norm(user_embedding) * np.linalg.norm(tweet_embedding))
                    tweet_scores.append((tweet_id, base_similarity))
                else:
                    tweet_scores.append((tweet_id, random.random()))

            new_tweet_scores = []
            # adjust similarity based on like and dislike traces
            for _tweet_id, _base_similarity in tweet_scores:
                _tweet_content = tweet_table[tweet_ids.index(_tweet_id)]['content']
                like_similarity = sum(np.dot(model.encode(_tweet_content), model.encode(like)) / (
                        np.linalg.norm(model.encode(_tweet_content)) * np.linalg.norm(model.encode(like))) for
                                      like in like_trace_contents) / len(
                    like_trace_contents) if like_trace_contents else 0
                dislike_similarity = sum(np.dot(model.encode(_tweet_content), model.encode(dislike)) / (
                        np.linalg.norm(model.encode(_tweet_content)) * np.linalg.norm(model.encode(dislike))) for
                                         dislike in dislike_trace_contents) / len(
                    dislike_trace_contents) if dislike_trace_contents else 0

                # Normalize and apply adjustments
                adjusted_similarity = normalize_similarity_adjustments(tweet_scores, _base_similarity,
                                                                       like_similarity, dislike_similarity)
                new_tweet_scores.append((_tweet_id, adjusted_similarity))

            # sort tweets by similarity
            new_tweet_scores.sort(key=lambda x: x[1], reverse=True)
            # extract tweet ids
            rec_tweet_ids = [tweet_id for tweet_id, _ in new_tweet_scores[:max_rec_tweet_len]]

            if swap_rate > 0:
                #swap the recommended tweets with random tweets
                swap_free_ids = [tweet_id for tweet_id in tweet_ids if tweet_id not in rec_tweet_ids and tweet_id not in
                                 [trace['tweet_id'] for trace in trace_table if trace['user_id']]]
                rec_tweet_ids = swap_random_tweets(rec_tweet_ids, swap_free_ids, swap_rate)

            new_rec_matrix.append(rec_tweet_ids)

    return new_rec_matrix


