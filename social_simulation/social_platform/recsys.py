'''注意需要在写入rec_matrix的时候判断是否超过max_rec_tweet_len'''
import os
import heapq
import random
from datetime import datetime
from math import log
from typing import Any, Dict, List
from .typing import ActionType, RecsysType
import numpy as np
from yaml import safe_load
import torch

# init model
with open("scripts/twitter.yaml", "r") as f:
    cfg = safe_load(f)
recsys_type = cfg.get("simulation").get("recsys_type")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if recsys_type == RecsysType.TWITTER:
    from sentence_transformers import SentenceTransformer
    try:
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    except Exception as e:
        print(e)
        model = None
elif recsys_type == RecsysType.TWHIN:
    from transformers import AutoTokenizer, AutoModel
    from .process_recsys_tweets import generate_tweet_vector
    from sklearn.metrics.pairwise import cosine_similarity
    try:
        twhin_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="/mnt/hwfile/trustai/zhangzaibin/twhin-bert-base" , model_max_length=512 )
        twhin_model = AutoModel.from_pretrained(pretrained_model_name_or_path="/mnt/hwfile/trustai/zhangzaibin/twhin-bert-base").to(device)
    except Exception as e:
        print(e)

def rec_sys_random(user_table: List[Dict[str,
                                         Any]], tweet_table: List[Dict[str,
                                                                       Any]],
                   trace_table: List[Dict[str, Any]], rec_matrix: List[List],
                   max_rec_tweet_len: int) -> List[List]:
    """
    Randomly recommend tweets to users.

    Args:
        user_table (List[Dict[str, Any]]): List of users.
        tweet_table (List[Dict[str, Any]]): List of tweets.
        trace_table (List[Dict[str, Any]]): List of user interactions.
        rec_matrix (List[List]): Existing recommendation matrix.
        max_rec_tweet_len (int): Maximum number of recommended tweets.

    Returns:
        List[List]: Updated recommendation matrix.
    """
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

def calculate_hot_score(num_likes: int, num_dislikes: int,
                        created_at: datetime) -> int:
    """
    Compute the hot score for a tweet.

    Args:
        num_likes (int): Number of likes.
        num_dislikes (int): Number of dislikes.
        created_at (datetime): Creation time of the tweet.

    Returns:
        int: Hot score of the tweet.

    Reference:
        https://medium.com/hacking-and-gonzo/how-reddit-ranking-algorithms-work-ef111e33d0d9
    """
    s = num_likes - num_dislikes
    order = log(max(abs(s), 1), 10)
    sign = 1 if s > 0 else -1 if s < 0 else 0

    # epoch_seconds
    epoch = datetime(1970, 1, 1)
    td = created_at - epoch
    epoch_seconds_result = td.days * 86400 + td.seconds + (
        float(td.microseconds) / 1e6)

    seconds = epoch_seconds_result - 1134028003
    return round(sign * order + seconds / 45000, 7)

def get_recommendations(user_index, cosine_similarities, items, top_n = 100):
    similarities = np.array(cosine_similarities[user_index])
    top_item_indices = similarities.argsort()[::-1][:top_n]
    recommended_items = [(list(items.keys())[i], similarities[i]) for i in top_item_indices]
    return recommended_items

def rec_sys_reddit(tweet_table: List[Dict[str, Any]], rec_matrix: List[List],
                   max_rec_tweet_len: int) -> List[List]:
    """
    Recommend tweets based on Reddit-like hot score.

    Args:
        tweet_table (List[Dict[str, Any]]): List of tweets.
        rec_matrix (List[List]): Existing recommendation matrix.
        max_rec_tweet_len (int): Maximum number of recommended tweets.

    Returns:
        List[List]: Updated recommendation matrix.
    """
    # 获取所有推文的ID
    tweet_ids = [tweet['tweet_id'] for tweet in tweet_table]

    if len(tweet_ids) <= max_rec_tweet_len:
        # 如果推文数量小于等于最大推荐数，每个用户获得所有推文ID
        new_rec_matrix = [tweet_ids] * (len(rec_matrix) - 1)
        new_rec_matrix = [None] + new_rec_matrix
    else:
        # 该推荐系统的时间复杂度是O(tweet_num * log max_rec_tweet_len)
        all_hot_score = []
        for tweet in tweet_table:
            created_at_dt = datetime.strptime(tweet['created_at'],
                                              "%Y-%m-%d %H:%M:%S.%f")
            hot_score = calculate_hot_score(tweet['num_likes'],
                                            tweet['num_dislikes'],
                                            created_at_dt)
            all_hot_score.append((hot_score, tweet['tweet_id']))
        # 排序
        # print(all_hot_score)
        top_tweets = heapq.nlargest(max_rec_tweet_len,
                                    all_hot_score,
                                    key=lambda x: x[0])
        top_tweet_ids = [tweet_id for _, tweet_id in top_tweets]

        new_rec_matrix = [None]
        # 如果推文数量大于最大推荐数，每个用户随机获得指定数量的推文ID
        new_rec_matrix = [top_tweet_ids] * (len(rec_matrix) - 1)
        new_rec_matrix = [None] + new_rec_matrix

    return new_rec_matrix


def rec_sys_personalized(user_table: List[Dict[str, Any]],
                         tweet_table: List[Dict[str, Any]],
                         trace_table: List[Dict[str,
                                                Any]], rec_matrix: List[List],
                         max_rec_tweet_len: int) -> List[List]:
    """
    Recommend tweets based on personalized similarity scores.

    Args:
        user_table (List[Dict[str, Any]]): List of users.
        tweet_table (List[Dict[str, Any]]): List of tweets.
        trace_table (List[Dict[str, Any]]): List of user interactions.
        rec_matrix (List[List]): Existing recommendation matrix.
        max_rec_tweet_len (int): Maximum number of recommended tweets.

    Returns:
        List[List]: Updated recommendation matrix.
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
            available_tweet_contents = [(tweet['tweet_id'], tweet['content'])
                                        for tweet in tweet_table
                                        if tweet['user_id'] != user_id]
            # calculate similarity between user bio and tweet text
            tweet_scores = []
            for tweet_id, tweet_content in available_tweet_contents:
                if model is not None:
                    user_embedding = model.encode(user_bio)
                    tweet_embedding = model.encode(tweet_content)
                    similarity = np.dot(user_embedding, tweet_embedding) / (
                        np.linalg.norm(user_embedding) *
                        np.linalg.norm(tweet_embedding))
                else:
                    similarity = random.random()

                tweet_scores.append((tweet_id, similarity))

            # sort tweets by similarity
            tweet_scores.sort(key=lambda x: x[1], reverse=True)

            # extract tweet ids
            rec_tweet_ids = [
                tweet_id for tweet_id, _ in tweet_scores[:max_rec_tweet_len]
            ]
            new_rec_matrix.append(rec_tweet_ids)

    return new_rec_matrix

def rec_sys_personalized_twh(
    user_table: List[Dict[str, Any]],
    tweet_table: List[Dict[str, Any]],
    trace_table: List[Dict[str, Any]],
    rec_matrix: List[List],
    max_rec_tweet_len: int,
) -> List[List]:
    # 获取所有推文的ID
    tweet_ids = [tweet['tweet_id'] for tweet in tweet_table]
    # 获取 id: content dict
    items = {tweet['tweet_id']: tweet['content'] for tweet in tweet_table}

    if len(tweet_ids) <= max_rec_tweet_len:
        # 如果推文数量小于等于最大推荐数，每个用户获得所有推文ID
        rec_matrix = [tweet_ids] * (len(rec_matrix) - 1)
        rec_ids_matrix = [None] + rec_matrix
        new_rec_matrix = []
        for index, rec_ids in enumerate(rec_ids_matrix[1:]):
            new_rec_matrix.append(rec_ids)
        new_rec_matrix = [None] + new_rec_matrix

    else: 
        new_rec_matrix = [None]
        # 如果推文数量大于最大推荐数，每个用户随机获得personalized推文ID
        user_profiles = [user['bio'] for user in user_table]
        user_profiles = [profile if profile is not None else 'This user does not have profile' for profile in user_profiles]
        
        # user_id - 1 = agent_id
        user_previous_tweet =  {tweet['user_id']-1: tweet['content'] for tweet in tweet_table} # 获取每个用户最近发的一条推特，根据最近发的推特进行推荐（这里面可以）
        # user_profiles.update(user_previous_tweet)
        # print(len(user_previous_tweet))
        for tweet_user_index in user_previous_tweet:
            try:
                user_profiles[tweet_user_index] = user_previous_tweet[tweet_user_index]
            except:
                print("update previous tweet failed")

        corpus = user_profiles + list(items.values())
        all_tweet_vector = generate_tweet_vector(twhin_model, twhin_tokenizer, corpus, batch_size=1000) 
        all_tweet_vector_list = [all_tweet_vector[i] for i in range(all_tweet_vector.size(0))] 
        user_vector = all_tweet_vector_list[:len(user_profiles)]
        item_vector = all_tweet_vector_list[len(user_profiles):]
        cosine_similarities = cosine_similarity(user_vector, item_vector)

        for user_index, profile in enumerate(user_profiles):
            rec_items = get_recommendations(user_index, cosine_similarities, items, top_n=max_rec_tweet_len)
            new_rec_ids = [item for item, _ in rec_items]
            new_rec_matrix.append(new_rec_ids)
                
    return new_rec_matrix

def normalize_similarity_adjustments(tweet_scores, base_similarity,
                                     like_similarity, dislike_similarity):
    """
    Normalize the adjustments to keep them in scale with overall similarities.

    Args:
        tweet_scores (list): List of tweet scores.
        base_similarity (float): Base similarity score.
        like_similarity (float): Similarity score for liked tweets.
        dislike_similarity (float): Similarity score for disliked tweets.

    Returns:
        float: Adjusted similarity score.
    """
    if len(tweet_scores) == 0:
        return base_similarity

    max_score = max(tweet_scores, key=lambda x: x[1])[1]
    min_score = min(tweet_scores, key=lambda x: x[1])[1]
    score_range = max_score - min_score
    adjustment = (like_similarity - dislike_similarity) * (score_range / 2)
    return base_similarity + adjustment


def swap_random_tweets(rec_tweet_ids, tweet_ids, swap_percent=0.1):
    """
    Swap a percentage of recommended tweets with random tweets.

    Args:
        rec_tweet_ids (list): List of recommended tweet IDs.
        tweet_ids (list): List of all tweet IDs.
        swap_percent (float): Percentage of tweets to swap.

    Returns:
        list: Updated list of recommended tweet IDs.
    """
    num_to_swap = int(len(rec_tweet_ids) * swap_percent)
    tweets_to_swap = random.sample(tweet_ids, num_to_swap)
    indices_to_replace = random.sample(range(len(rec_tweet_ids)), num_to_swap)

    for idx, new_tweet in zip(indices_to_replace, tweets_to_swap):
        rec_tweet_ids[idx] = new_tweet

    return rec_tweet_ids


def get_trace_contents(user_id, action, tweet_table, trace_table):
    """
    Get the contents of tweets that a user has interacted with.

    Args:
        user_id (str): ID of the user.
        action (str): Type of action (like or unlike).
        tweet_table (list): List of tweets.
        trace_table (list): List of user interactions.

    Returns:
        list: List of tweet contents.
    """
    # Get tweet IDs from trace table for the given user and action
    trace_tweet_ids = [
        trace['tweet_id'] for trace in trace_table
        if (trace['user_id'] == user_id and trace['action'] == action)
    ]
    # Fetch tweet contents from tweet table where tweet IDs match those in the
    # trace
    trace_contents = [
        tweet['content'] for tweet in tweet_table
        if tweet['tweet_id'] in trace_tweet_ids
    ]
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
    1. If the number of tweets is less than or equal to the maximum
        recommended length, each user gets all tweet IDs

    2. Otherwise:
        - For each user, get a like-trace pool and dislike-trace pool from the
            trace table
        - For each user, calculate the similarity between the user's bio and
            the tweet text
        - Use the trace table to adjust the similarity score
        - Swap 10% of the recommended tweets with the random tweets

    Personalized recommendation system that uses user interaction traces.

    Args:
        user_table (List[Dict[str, Any]]): List of users.
        tweet_table (List[Dict[str, Any]]): List of tweets.
        trace_table (List[Dict[str, Any]]): List of user interactions.
        rec_matrix (List[List]): Existing recommendation matrix.
        max_rec_tweet_len (int): Maximum number of recommended tweets.
        swap_rate (float): Percentage of tweets to swap for diversity.

    Returns:
        List[List]: Updated recommendation matrix.
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
            available_tweet_contents = [(tweet['tweet_id'], tweet['content'])
                                        for tweet in tweet_table
                                        if tweet['user_id'] != user_id]

            # filter out like-trace and dislike-trace
            like_trace_contents = get_trace_contents(user_id,
                                                     ActionType.LIKE.value,
                                                     tweet_table, trace_table)
            dislike_trace_contents = get_trace_contents(
                user_id, ActionType.UNLIKE.value, tweet_table, trace_table)
            # calculate similarity between user bio and tweet text
            tweet_scores = []
            for tweet_id, tweet_content in available_tweet_contents:
                if model is not None:
                    user_embedding = model.encode(user_bio)
                    tweet_embedding = model.encode(tweet_content)
                    base_similarity = np.dot(
                        user_embedding,
                        tweet_embedding) / (np.linalg.norm(user_embedding) *
                                            np.linalg.norm(tweet_embedding))
                    tweet_scores.append((tweet_id, base_similarity))
                else:
                    tweet_scores.append((tweet_id, random.random()))

            new_tweet_scores = []
            # adjust similarity based on like and dislike traces
            for _tweet_id, _base_similarity in tweet_scores:
                _tweet_content = tweet_table[tweet_ids.index(
                    _tweet_id)]['content']
                like_similarity = sum(
                    np.dot(model.encode(_tweet_content), model.encode(like)) /
                    (np.linalg.norm(model.encode(_tweet_content)) *
                     np.linalg.norm(model.encode(like)))
                    for like in like_trace_contents) / len(
                        like_trace_contents) if like_trace_contents else 0
                dislike_similarity = sum(
                    np.dot(model.encode(_tweet_content), model.encode(dislike))
                    / (np.linalg.norm(model.encode(_tweet_content)) *
                       np.linalg.norm(model.encode(dislike)))
                    for dislike in dislike_trace_contents) / len(
                        dislike_trace_contents
                    ) if dislike_trace_contents else 0

                # Normalize and apply adjustments
                adjusted_similarity = normalize_similarity_adjustments(
                    tweet_scores, _base_similarity, like_similarity,
                    dislike_similarity)
                new_tweet_scores.append((_tweet_id, adjusted_similarity))

            # sort tweets by similarity
            new_tweet_scores.sort(key=lambda x: x[1], reverse=True)
            # extract tweet ids
            rec_tweet_ids = [
                tweet_id
                for tweet_id, _ in new_tweet_scores[:max_rec_tweet_len]
            ]

            if swap_rate > 0:
                # swap the recommended tweets with random tweets
                swap_free_ids = [
                    tweet_id for tweet_id in tweet_ids
                    if tweet_id not in rec_tweet_ids and tweet_id not in [
                        trace['tweet_id']
                        for trace in trace_table if trace['user_id']
                    ]
                ]
                rec_tweet_ids = swap_random_tweets(rec_tweet_ids,
                                                   swap_free_ids, swap_rate)

            new_rec_matrix.append(rec_tweet_ids)

    return new_rec_matrix
