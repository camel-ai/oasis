'''注意需要在写入rec_matrix的时候判断是否超过max_rec_post_len'''
import heapq
import random
from datetime import datetime
from math import log
from typing import Any, Dict, List
from .typing import ActionType, RecsysType
import numpy as np
from yaml import safe_load
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from .process_recsys_tweets import generate_tweet_vector
from sklearn.metrics.pairwise import cosine_similarity


def load_model(model_name):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_name == 'paraphrase-MiniLM-L6-v2':
            return SentenceTransformer(model_name, device=device, cache_folder="./models")
        elif model_name == 'Twitter/twhin-bert-base':
            tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
            model = AutoModel.from_pretrained(model_name, cache_dir="./models").to(device)
            return tokenizer, model
        else:
            raise ValueError(f"Unknown model name: {model_name}")
    except Exception as e:
        raise Exception(f"Failed to load the model: {model_name}") from e

def get_recsys_model(config_path="scripts/twitter.yaml", recsys_type:str=None):
    if recsys_type == None:
        with open(config_path, "r") as f:
            cfg = safe_load(f)
        recsys_type = cfg.get("simulation").get("recsys_type")
    if recsys_type == RecsysType.TWITTER.value:
        model = load_model('paraphrase-MiniLM-L6-v2')
        return model
    elif recsys_type == RecsysType.TWHIN.value:
        twhin_tokenizer, twhin_model = load_model("Twitter/twhin-bert-base")
        models = (twhin_tokenizer, twhin_model)
        return models
    elif recsys_type == RecsysType.REDDIT.value or recsys_type == RecsysType.RANDOM.value:
        return None
    else:
        raise ValueError(f"Unknown recsys type: {recsys_type}")


def rec_sys_random(user_table: List[Dict[str,
                                         Any]], post_table: List[Dict[str,
                                                                      Any]],
                   trace_table: List[Dict[str, Any]], rec_matrix: List[List],
                   max_rec_post_len: int) -> List[List]:
    """
    Randomly recommend posts to users.

    Args:
        user_table (List[Dict[str, Any]]): List of users.
        post_table (List[Dict[str, Any]]): List of posts.
        trace_table (List[Dict[str, Any]]): List of user interactions.
        rec_matrix (List[List]): Existing recommendation matrix.
        max_rec_post_len (int): Maximum number of recommended posts.

    Returns:
        List[List]: Updated recommendation matrix.
    """
    # 获取所有推文的ID
    post_ids = [post['post_id'] for post in post_table]

    if len(post_ids) <= max_rec_post_len:
        # 如果推文数量小于等于最大推荐数，每个用户获得所有推文ID
        new_rec_matrix = [post_ids] * (len(rec_matrix) - 1)
        new_rec_matrix = [None] + new_rec_matrix
    else:
        new_rec_matrix = [None]
        # 如果推文数量大于最大推荐数，每个用户随机获得指定数量的推文ID
        for _ in range(1, len(rec_matrix)):
            new_rec_matrix.append(random.sample(post_ids, max_rec_post_len))

    return new_rec_matrix

def calculate_hot_score(num_likes: int, num_dislikes: int,
                        created_at: datetime) -> int:
    """
    Compute the hot score for a post.

    Args:
        num_likes (int): Number of likes.
        num_dislikes (int): Number of dislikes.
        created_at (datetime): Creation time of the post.

    Returns:
        int: Hot score of the post.

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

def get_recommendations(user_index:int, cosine_similarities:List[List], items:Dict, top_k = 100):
    similarities = np.array(cosine_similarities[user_index])
    top_item_indices = similarities.argsort()[::-1][:top_k]
    recommended_items = [(list(items.keys())[i], similarities[i]) for i in top_item_indices]
    return recommended_items

def rec_sys_reddit(post_table: List[Dict[str, Any]], rec_matrix: List[List],
                   max_rec_post_len: int) -> List[List]:
    """
    Recommend posts based on Reddit-like hot score.

    Args:
        post_table (List[Dict[str, Any]]): List of posts.
        rec_matrix (List[List]): Existing recommendation matrix.
        max_rec_post_len (int): Maximum number of recommended posts.

    Returns:
        List[List]: Updated recommendation matrix.
    """
    # 获取所有推文的ID
    post_ids = [post['post_id'] for post in post_table]

    if len(post_ids) <= max_rec_post_len:
        # 如果推文数量小于等于最大推荐数，每个用户获得所有推文ID
        new_rec_matrix = [post_ids] * (len(rec_matrix) - 1)
        new_rec_matrix = [None] + new_rec_matrix
    else:
        # 该推荐系统的时间复杂度是O(post_num * log max_rec_post_len)
        all_hot_score = []
        for post in post_table:
            created_at_dt = datetime.strptime(post['created_at'],
                                              "%Y-%m-%d %H:%M:%S.%f")
            hot_score = calculate_hot_score(post['num_likes'],
                                            post['num_dislikes'],
                                            created_at_dt)
            all_hot_score.append((hot_score, post['post_id']))
        # 排序
        # print(all_hot_score)
        top_posts = heapq.nlargest(max_rec_post_len,
                                   all_hot_score,
                                   key=lambda x: x[0])
        top_post_ids = [post_id for _, post_id in top_posts]

        new_rec_matrix = [None]
        # 如果推文数量大于最大推荐数，每个用户随机获得指定数量的推文ID
        new_rec_matrix = [top_post_ids] * (len(rec_matrix) - 1)
        new_rec_matrix = [None] + new_rec_matrix

    return new_rec_matrix

def rec_sys_personalized(user_table: List[Dict[str, Any]],
                         post_table: List[Dict[str, Any]],
                         trace_table: List[Dict[str,
                                                Any]], rec_matrix: List[List],
                         max_rec_post_len: int) -> List[List]:
    """
    Recommend posts based on personalized similarity scores.

    Args:
        user_table (List[Dict[str, Any]]): List of users.
        post_table (List[Dict[str, Any]]): List of posts.
        trace_table (List[Dict[str, Any]]): List of user interactions.
        rec_matrix (List[List]): Existing recommendation matrix.
        max_rec_post_len (int): Maximum number of recommended posts.

    Returns:
        List[List]: Updated recommendation matrix.
    """
    # 获取所有推文的ID
    post_ids = [post['post_id'] for post in post_table]

    if len(post_ids) <= max_rec_post_len:
        # 如果推文数量小于等于最大推荐数，每个用户获得所有推文ID
        new_rec_matrix = [post_ids] * (len(rec_matrix) - 1)
        new_rec_matrix = [None] + new_rec_matrix
    else:
        new_rec_matrix = [None]
        # 如果推文数量大于最大推荐数，每个用户随机获得personalized推文ID
        for idx in range(1, len(rec_matrix)):
            user_id = user_table[idx - 1]['user_id']
            user_bio = user_table[idx - 1]['bio']
            # filter out posts that belong to the user
            available_post_contents = [(post['post_id'], post['content'])
                                       for post in post_table
                                       if post['user_id'] != user_id]
            # calculate similarity between user bio and post text
            post_scores = []
            for post_id, post_content in available_post_contents:
                if model is not None:
                    user_embedding = model.encode(user_bio)
                    post_embedding = model.encode(post_content)
                    similarity = np.dot(user_embedding, post_embedding) / (
                        np.linalg.norm(user_embedding) *
                        np.linalg.norm(post_embedding))
                else:
                    similarity = random.random()

                post_scores.append((post_id, similarity))

            # sort posts by similarity
            post_scores.sort(key=lambda x: x[1], reverse=True)

            # extract post ids
            rec_post_ids = [
                post_id for post_id, _ in post_scores[:max_rec_post_len]
            ]
            new_rec_matrix.append(rec_post_ids)

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

    model = get_recsys_model(recsys_type="twhin-bert")
    (twhin_tokenizer, twhin_model)  = model

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
        # 获取每个用户最近发的一条推特，根据最近发的推特进行推荐（这里面可以）
        user_previous_tweet =  {tweet['user_id']-1: tweet['content'] for tweet in tweet_table} 
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
            rec_items = get_recommendations(user_index, cosine_similarities, items, top_k=max_rec_tweet_len)
            new_rec_ids = [item for item, _ in rec_items]
            new_rec_matrix.append(new_rec_ids)

    return new_rec_matrix

def normalize_similarity_adjustments(post_scores, base_similarity,
                                     like_similarity, dislike_similarity):
    """
    Normalize the adjustments to keep them in scale with overall similarities.

    Args:
        post_scores (list): List of post scores.
        base_similarity (float): Base similarity score.
        like_similarity (float): Similarity score for liked posts.
        dislike_similarity (float): Similarity score for disliked posts.

    Returns:
        float: Adjusted similarity score.
    """
    if len(post_scores) == 0:
        return base_similarity

    max_score = max(post_scores, key=lambda x: x[1])[1]
    min_score = min(post_scores, key=lambda x: x[1])[1]
    score_range = max_score - min_score
    adjustment = (like_similarity - dislike_similarity) * (score_range / 2)
    return base_similarity + adjustment


def swap_random_posts(rec_post_ids, post_ids, swap_percent=0.1):
    """
    Swap a percentage of recommended posts with random posts.

    Args:
        rec_post_ids (list): List of recommended post IDs.
        post_ids (list): List of all post IDs.
        swap_percent (float): Percentage of posts to swap.

    Returns:
        list: Updated list of recommended post IDs.
    """
    num_to_swap = int(len(rec_post_ids) * swap_percent)
    posts_to_swap = random.sample(post_ids, num_to_swap)
    indices_to_replace = random.sample(range(len(rec_post_ids)), num_to_swap)

    for idx, new_post in zip(indices_to_replace, posts_to_swap):
        rec_post_ids[idx] = new_post

    return rec_post_ids


def get_trace_contents(user_id, action, post_table, trace_table):
    """
    Get the contents of posts that a user has interacted with.

    Args:
        user_id (str): ID of the user.
        action (str): Type of action (like or unlike).
        post_table (list): List of posts.
        trace_table (list): List of user interactions.

    Returns:
        list: List of post contents.
    """
    # Get post IDs from trace table for the given user and action
    trace_post_ids = [
        trace['post_id'] for trace in trace_table
        if (trace['user_id'] == user_id and trace['action'] == action)
    ]
    # Fetch post contents from post table where post IDs match those in the
    # trace
    trace_contents = [
        post['content'] for post in post_table
        if post['post_id'] in trace_post_ids
    ]
    return trace_contents


def rec_sys_personalized_with_trace(
    user_table: List[Dict[str, Any]],
    post_table: List[Dict[str, Any]],
    trace_table: List[Dict[str, Any]],
    rec_matrix: List[List],
    max_rec_post_len: int,
    swap_rate: float = 0.1,
) -> List[List]:
    """
    This version:
    1. If the number of posts is less than or equal to the maximum
        recommended length, each user gets all post IDs

    2. Otherwise:
        - For each user, get a like-trace pool and dislike-trace pool from the
            trace table
        - For each user, calculate the similarity between the user's bio and
            the post text
        - Use the trace table to adjust the similarity score
        - Swap 10% of the recommended posts with the random posts

    Personalized recommendation system that uses user interaction traces.

    Args:
        user_table (List[Dict[str, Any]]): List of users.
        post_table (List[Dict[str, Any]]): List of posts.
        trace_table (List[Dict[str, Any]]): List of user interactions.
        rec_matrix (List[List]): Existing recommendation matrix.
        max_rec_post_len (int): Maximum number of recommended posts.
        swap_rate (float): Percentage of posts to swap for diversity.

    Returns:
        List[List]: Updated recommendation matrix.
    """
    # 获取所有推文的ID
    post_ids = [post['post_id'] for post in post_table]
    if len(post_ids) <= max_rec_post_len:
        # 如果推文数量小于等于最大推荐数，每个用户获得所有推文ID
        new_rec_matrix = [post_ids] * (len(rec_matrix) - 1)
        new_rec_matrix = [None] + new_rec_matrix
    else:
        new_rec_matrix = [None]
        # 如果推文数量大于最大推荐数，每个用户随机获得personalized推文ID
        for idx in range(1, len(rec_matrix)):
            user_id = user_table[idx - 1]['user_id']
            user_bio = user_table[idx - 1]['bio']
            # filter out posts that belong to the user
            available_post_contents = [(post['post_id'], post['content'])
                                       for post in post_table
                                       if post['user_id'] != user_id]

            # filter out like-trace and dislike-trace
            like_trace_contents = get_trace_contents(user_id,
                                                     ActionType.LIKE.value,
                                                     post_table, trace_table)
            dislike_trace_contents = get_trace_contents(
                user_id, ActionType.UNLIKE.value, post_table, trace_table)
            # calculate similarity between user bio and post text
            post_scores = []
            for post_id, post_content in available_post_contents:
                if model is not None:
                    user_embedding = model.encode(user_bio)
                    post_embedding = model.encode(post_content)
                    base_similarity = np.dot(
                        user_embedding,
                        post_embedding) / (np.linalg.norm(user_embedding) *
                                           np.linalg.norm(post_embedding))
                    post_scores.append((post_id, base_similarity))
                else:
                    post_scores.append((post_id, random.random()))

            new_post_scores = []
            # adjust similarity based on like and dislike traces
            for _post_id, _base_similarity in post_scores:
                _post_content = post_table[post_ids.index(_post_id)]['content']
                like_similarity = sum(
                    np.dot(model.encode(_post_content), model.encode(like)) /
                    (np.linalg.norm(model.encode(_post_content)) *
                     np.linalg.norm(model.encode(like)))
                    for like in like_trace_contents) / len(
                        like_trace_contents) if like_trace_contents else 0
                dislike_similarity = sum(
                    np.dot(model.encode(_post_content), model.encode(dislike))
                    / (np.linalg.norm(model.encode(_post_content)) *
                       np.linalg.norm(model.encode(dislike)))
                    for dislike in dislike_trace_contents) / len(
                        dislike_trace_contents
                    ) if dislike_trace_contents else 0

                # Normalize and apply adjustments
                adjusted_similarity = normalize_similarity_adjustments(
                    post_scores, _base_similarity, like_similarity,
                    dislike_similarity)
                new_post_scores.append((_post_id, adjusted_similarity))

            # sort posts by similarity
            new_post_scores.sort(key=lambda x: x[1], reverse=True)
            # extract post ids
            rec_post_ids = [
                post_id for post_id, _ in new_post_scores[:max_rec_post_len]
            ]

            if swap_rate > 0:
                # swap the recommended posts with random posts
                swap_free_ids = [
                    post_id for post_id in post_ids
                    if post_id not in rec_post_ids and post_id not in [
                        trace['post_id']
                        for trace in trace_table if trace['user_id']
                    ]
                ]
                rec_post_ids = swap_random_posts(rec_post_ids, swap_free_ids,
                                                 swap_rate)

            new_rec_matrix.append(rec_post_ids)

    return new_rec_matrix
