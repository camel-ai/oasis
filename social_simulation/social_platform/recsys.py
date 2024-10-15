'''注意需要在写入rec_matrix的时候判断是否超过max_rec_post_len'''
from ast import literal_eval
import heapq
import random
from datetime import datetime
from math import log
from typing import Any, Dict, List

from colorama import Back
from .typing import ActionType, RecsysType
import numpy as np
from yaml import safe_load
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import os
import time
import logging
from transformers import AutoTokenizer, AutoModel
from .process_recsys_posts import generate_post_vector
from sklearn.metrics.pairwise import cosine_similarity
rec_log = logging.getLogger(name='social.rec')
rec_log.setLevel('DEBUG')

# 先设置为None，后续再在recsys函数里赋一次值
model = None
twhin_tokenizer, twhin_model = None, None

# Create the TF-IDF model
tfidf_vectorizer = TfidfVectorizer()
# prepare the twhin model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# twhin_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="/mnt/hwfile/trustai/zhangzaibin/twhin-bert-base",model_max_length=512 ) # TODO change the pretrained_model_path
# twhin_model = AutoModel.from_pretrained(pretrained_model_name_or_path="/mnt/hwfile/trustai/zhangzaibin/twhin-bert-base").to(device)
# STmodel = SentenceTransformer('/mnt/petrelfs/zhengzirui/social-simulation/models/models--sentence-transformers--paraphrase-MiniLM-L6-v2/snapshots/3bf4ae7445aa77c8daaef06518dd78baffff53c9').to(device)
# twhin_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="/ibex/user/yangz0h/open_source_llm/twhin-bert-base",model_max_length=512 ) # TODO change the pretrained_model_path
# twhin_model = AutoModel.from_pretrained(pretrained_model_name_or_path="/ibex/user/yangz0h/open_source_llm/twhin-bert-base").to(device)
twhin_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="Twitter/twhin-bert-base",model_max_length=512 ) # TODO change the pretrained_model_path
twhin_model = AutoModel.from_pretrained(pretrained_model_name_or_path="Twitter/twhin-bert-base").to(device)


# 每个用户的所有历史推特和最近一条推特
user_previous_post_all = {}
user_previous_post = {}
user_profiles = []
# 获取 {post_id: content} dict
t_items = {}
# 获取 {uid: follower_count} dict
# 这里还是需要保证agent注册时按顺序来的，存在user_id=agent_id+1的关系 乱序注册这里就会有问题
u_items = {}
# 获取所有推文的创建时间，根据时间远近来赋分
date_score = []
# 获取所有推文的作者的粉丝数
fans_score = []

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

def get_recsys_model(recsys_type:str=None):
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
# Move model to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if model is not None:
    model.to(device)
else:
    print('Model not available, using random similarity.')
    pass

# 重置全局变量
def reset_globals():
    global user_previous_post_all, user_previous_post, user_profiles, t_items, u_items, date_score, fans_score
    user_previous_post_all = {}
    user_previous_post = {}
    user_profiles = []
    t_items = {}
    u_items = {}
    date_score = []
    fans_score = []


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
    new_rec_matrix = []
    if len(post_ids) <= max_rec_post_len:
        # 如果推文数量小于等于最大推荐数，每个用户获得所有推文ID
        new_rec_matrix = [post_ids] * len(rec_matrix)
    else:
        # 如果推文数量大于最大推荐数，每个用户随机获得指定数量的推文ID
        for _ in range(len(rec_matrix)):
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

def get_recommendations(user_index, cosine_similarities, items, score, top_n = 100, ):
    similarities = np.array(cosine_similarities[user_index])
    similarities = similarities * score
    top_item_indices = similarities.argsort()[::-1][:top_n]
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
        new_rec_matrix = [post_ids] * len(rec_matrix)
    else:
        # 该推荐系统的时间复杂度是O(post_num * log max_rec_post_len)
        all_hot_score = []
        for post in post_table:
            try:
                created_at_dt = datetime.strptime(post['created_at'],
                                                "%Y-%m-%d %H:%M:%S.%f")
            except Exception:
                created_at_dt = datetime.strptime(post['created_at'],
                                                "%Y-%m-%d %H:%M:%S")
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

        # 如果推文数量大于最大推荐数，每个用户随机获得指定数量的推文ID
        new_rec_matrix = [top_post_ids] * len(rec_matrix)

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
    global model
    if model == None or type(model) == tuple:
        model = get_recsys_model(recsys_type="twitter")

    # 获取所有推文的ID
    post_ids = [post['post_id'] for post in post_table]
    print(f'Running personalized recommendation for {len(user_table)} users......')
    start_time = time.time()
    new_rec_matrix = []
    if len(post_ids) <= max_rec_post_len:
        # If the number of posts is less than or equal to the maximum recommended length, each user gets all post IDs
        new_rec_matrix = [post_ids] * len(rec_matrix)
    else:
        # If the number of posts is greater than the maximum recommended length, each user gets personalized post IDs
        user_bios = [user['bio'] if 'bio' in user and user['bio'] is not None else '' for user in user_table]
        post_contents = [post['content'] for post in post_table]

        if model:
            user_embeddings = model.encode(user_bios,
                                           convert_to_tensor=True,
                                           device=device)
            post_embeddings = model.encode(post_contents,
                                            convert_to_tensor=True,
                                            device=device)

            # Compute dot product similarity
            dot_product = torch.matmul(user_embeddings, post_embeddings.T)

            # Compute norm
            user_norms = torch.norm(user_embeddings, dim=1)
            post_norms = torch.norm(post_embeddings, dim=1)

            # Compute cosine similarity
            similarities = dot_product / (user_norms[:, None] * post_norms[None, :])

        else:
            # Generate random similarities
            similarities = torch.rand(len(user_table), len(post_table))


        # Iterate through each user to generate personalized recommendations.
        for user_index, user in enumerate(user_table):
            # Filter out posts made by the current user.
            filtered_post_indices = [
                i for i, post in enumerate(post_table) if post['user_id'] != user['user_id']
            ]

            user_similarities = similarities[user_index, filtered_post_indices]

            # Get the corresponding post IDs for the filtered posts.
            filtered_post_ids = [post_table[i]['post_id'] for i in filtered_post_indices]

            # Determine the top posts based on the similarities, limited by max_rec_post_len.
            _, top_indices = torch.topk(user_similarities, k=min(max_rec_post_len, len(filtered_post_ids)))

            top_post_ids = [filtered_post_ids[i] for i in top_indices.tolist()]

            # Append the top post IDs to the new recommendation matrix.
            new_rec_matrix.append(top_post_ids)

    end_time = time.time()
    print(f'Personalized recommendation time: {end_time - start_time:.6f}s')
    return new_rec_matrix


def get_like_post_id(user_id, action, trace_table):
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
        literal_eval(trace['info'])["post_id"] for trace in trace_table
        if (trace['user_id'] == user_id and trace['action'] == action)
    ]

    # 只取最近点赞的5条post,如果不够则用最新点赞的post padding
    # 只取id，不取content是因为后面会算一遍所有post的embedding，把这个拿出来单独再算一遍非常耗时，尤其是agent的数量级比较大的时候
    if len(trace_post_ids) < 5 and len(trace_post_ids) > 0:
        trace_post_ids += [trace_post_ids[-1]] * (5-len(trace_post_ids))
    elif len(trace_post_ids) > 5:
        trace_post_ids = trace_post_ids[-5:]
    else:
        trace_post_ids = [0]

    return trace_post_ids

# 计算过去like的post与target post的余弦相似度并取平均值
def calculate_like_similarity(liked_vectors, target_vectors):
    # 计算向量的范数
    liked_norms = np.linalg.norm(liked_vectors, axis=1)
    target_norms = np.linalg.norm(target_vectors, axis=1)
    # 计算点积
    dot_products = np.dot(target_vectors, liked_vectors.T)
    # 计算余弦相似度
    cosine_similarities = dot_products / np.outer(target_norms, liked_norms)
    # 取平均值
    average_similarities = np.mean(cosine_similarities, axis=1)
    
    return average_similarities

def rec_sys_personalized_twh(
    user_table: List[Dict[str, Any]],
    post_table: List[Dict[str, Any]],
    latest_post_count: int,
    trace_table: List[Dict[str, Any]],
    rec_matrix: List[List],
    max_rec_post_len: int,
    # source_post_indexs: List[int],
    recall_only: bool = False,
    enable_like_score: bool = False
) -> List[List]:
    # 设置一些全局变量，减少时间消耗
    global date_score, fans_score, t_items, u_items, user_previous_post, user_previous_post_all, user_profiles
    # 获取 uid: follower_count dict
    # 只更新一次，除非要加入中途介入新用户的功能。
    if (not u_items) or len(u_items) != len(user_table):
        u_items = {user['user_id']: user["num_followers"] for user in user_table}
    if not user_previous_post_all or len(user_previous_post_all) != len(user_table):
        # 每个user都要有一个历史推特列表
        user_previous_post_all = {index: [] for index in range(len(user_table))}
        user_previous_post = {index: "" for index in range(len(user_table))}
    if not user_profiles or len(user_profiles) != len(user_table):
        for user in user_table:
            if user['bio'] == None:
                user_profiles.append('This user does not have profile')
            else:
                user_profiles.append(user['bio'])

    current_time = int(os.environ["SANDBOX_TIME"])
    if len(t_items) < len(post_table):
        for post in post_table[-latest_post_count:]:
            # 获取 {post_id: content} dict，只更新最新发出的推特
            t_items[post['post_id']] = post['content']
            # 更新用户发出的历史推特
            user_previous_post_all[post['user_id']].append(post['content'])
            user_previous_post[post['user_id']] = post['content']
            # 获取所有推文的创建时间，根据时间远近来赋分, 需要注意的是这种算法最多只能跑90个时间步
            date_score.append(np.log( (271.8 - (current_time - int(post['created_at'])))/100))
            # 获取post的受众群体数量, 根据粉丝数量来赋分
            try:
                fans_score.append(np.log(u_items[post['user_id']] + 1) / np.log(1000))
            except Exception as e:
                print(e)
                import pdb;
                # pdb.set_trace()

    date_score_np = np.array(date_score)
    # fan_score [1, 2.x]
    fans_score_np = np.array(fans_score)
    fans_score_np = np.where(fans_score_np < 1, 1, fans_score_np)

    if enable_like_score:
        # 与之前like的内容计算相似度，先收集trace中的like post id
        like_post_ids_all = []
        for user in user_table:
            user_id = user['agent_id']
            like_post_ids = get_like_post_id(user_id, ActionType.LIKE_POST.value, trace_table)
            like_post_ids_all.append(like_post_ids)
         
    scores = date_score_np * fans_score_np
    new_rec_matrix = []
    if len(post_table) <= max_rec_post_len:
        # 如果推文数量小于等于最大推荐数，每个用户获得所有推文ID
        tids = [t['post_id'] for t in post_table]
        new_rec_matrix = [tids] * (len(rec_matrix))

    else: 
        # 如果推文数量大于最大推荐数，每个用户随机获得personalized推文ID

        # 这里需要过一遍所有user，去更新他的profile，是一个比较耗时的操作
        for post_user_index in user_previous_post:
            try:
                # 直接使用最新推特替换profile的方法会导致推荐系统向已经转发了该推特的用户重复推送其他转发了该推特的repost
                # user_profiles[post_user_index] = user_previous_post[post_user_index]
                # 这里改为向user char的最后加上对于Recent post的内容介绍
                update_profile = f" # Recent post:{user_previous_post[post_user_index]}"
                if user_previous_post[post_user_index] != "":
                    # 如果没有更新 recent post，加上这一段
                    if "# Recent post:" not in user_profiles[post_user_index]:
                        user_profiles[post_user_index] += update_profile
                    # 如果profile中有recent post，但不是该用户最新发的推，将其置换掉
                    elif update_profile not in user_profiles[post_user_index]:
                        user_profiles[post_user_index] = user_profiles[post_user_index].split("# Recent post:")[0] + update_profile
            except:
                print("update previous post failed")
        
        corpus = user_profiles + list(t_items.values())
        tweet_vector_start_t = time.time()
        all_post_vector_list = generate_post_vector(twhin_model, twhin_tokenizer, corpus, batch_size=1000)
        tweet_vector_end_t = time.time()
        rec_log.info(f"twhin model cost time: {tweet_vector_end_t-tweet_vector_start_t}")
        user_vector = all_post_vector_list[:len(user_profiles)]
        posts_vector = all_post_vector_list[len(user_profiles):]

        if enable_like_score:
            # 遍历所有like post id，从posts_vector把like post vectors收集起来用矩阵加速运算
            like_posts_vectors = []
            for user_idx, like_post_ids in enumerate(like_post_ids_all):
                if len(like_post_ids) != 1:
                    for like_post_id in like_post_ids:
                        try:
                            like_posts_vectors.append(posts_vector[like_post_id-1])
                        except:
                            like_posts_vectors.append(user_vector[user_idx])
                else:
                    like_posts_vectors += [user_vector[user_idx] for _ in range(5)]
            try:
                like_posts_vectors = torch.stack(like_posts_vectors).view(len(user_table), 5, posts_vector.shape[1])
            except:
                import pdb
                pdb.set_trace()
        get_similar_start_t = time.time()
        cosine_similarities = cosine_similarity(user_vector, posts_vector)
        get_similar_end_t = time.time()
        rec_log.info(f"get cosine_similarity time: {get_similar_end_t-get_similar_start_t}")
        if enable_like_score:
            for user_index, profile in enumerate(user_profiles):
                user_like_posts_vector = like_posts_vectors[user_index]
                like_scores = calculate_like_similarity(user_like_posts_vector, posts_vector)
                try:
                    scores = scores + like_scores
                except:
                    import pdb;pdb.set_trace()

        cosine_similarities = cosine_similarities * scores
        cosine_similarities = torch.tensor(cosine_similarities)
        value, indices = torch.topk(cosine_similarities, max_rec_post_len, dim=1, largest=True, sorted=True)

        matrix_list = indices.cpu().numpy()
        post_list = list(t_items.keys())
        for rec_ids in matrix_list:
            rec_ids = [post_list[i] for i in rec_ids]
            new_rec_matrix.append(rec_ids)
       
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

    start_time = time.time()

    # 获取所有推文的ID
    new_rec_matrix = []
    post_ids = [post['post_id'] for post in post_table]
    if len(post_ids) <= max_rec_post_len:
        # 如果推文数量小于等于最大推荐数，每个用户获得所有推文ID
        new_rec_matrix = [post_ids] * (len(rec_matrix) - 1)
    else:
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
                                                     ActionType.LIKE_POST.value,
                                                     post_table, trace_table)
            dislike_trace_contents = get_trace_contents(
                user_id, ActionType.UNLIKE_POST.value, post_table, trace_table)
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
    end_time = time.time()
    print(f'Personalized recommendation time: {end_time - start_time:.6f}s')
    return new_rec_matrix
