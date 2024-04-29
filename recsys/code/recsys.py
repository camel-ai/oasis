# File: recsys/code/recsys.py
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer

class RecSys:
    """
    A recommendation system for generating personalized tweet recommendations for users.

    Attributes:
        db_file (str): The file path to the SQLite database.
        HOMEMAX (int): The maximum number of tweets to recommend to each user.
        model: An instance of SentenceTransformer model for text embedding.
    """

    def __init__(self, db_file):
        """
        Initializes the RecSys instance.

        Args:
            db_file (str): The file path to the SQLite database.
        """
        self.db_file = db_file
        self.HOMEMAX = 100  # 定义最大推文数
        try:
            self.model = SentenceTransformer('bert-base-nli-mean-tokens')
        except Exception as e:
            print("An error occurred while loading the model:", str(e))
            self.model = None

    def fetch_data(self):
        """
        Fetches user data and tweet data from the SQLite database.

        Returns:
            tuple: A tuple containing two lists - users and tweets.
                   Each list contains tuples representing the data of each user/tweet.
        """
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM user")
        users = cursor.fetchall()
        
        cursor.execute("SELECT * FROM tweets")
        tweets = cursor.fetchall()
        
        conn.close()
        return users, tweets

    def clean_data(self, tweets):
        """
        Cleans the tweet data by filtering out sensitive content or performing other preprocessing.

        Args:
            tweets (list): A list of tuples representing the tweet data.

        Returns:
            list: A list of cleaned tweet data.
        """
        # 过滤敏感内容或其他清洗过程
        return [tweet for tweet in tweets if 'chink' not in tweet[1]]

    def calculate_similarity(self, tweets):
        """
        Calculates the similarity between tweets using text embeddings.

        Args:
            tweets (list): A list of tuples representing the tweet data.

        Returns:
            array: A 2D numpy array representing the cosine similarity between tweets.
        """
        embeddings = self.model.encode([tweet[1] for tweet in tweets])
        cosine_similarity = np.dot(embeddings, embeddings.T)
        return cosine_similarity

    def store_recommendations(self, user_recommendations):
        """
        Stores the tweet recommendations for each user in the SQLite database.

        Args:
            user_recommendations (dict): A dictionary where keys are user IDs and values are lists of recommended tweet IDs.
        """
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        for user_id, recommendations in user_recommendations.items():
            rec_str = ','.join(str(r) for r in recommendations)
            cursor.execute("INSERT INTO rec (user_id, recommendations) VALUES (?, ?)", (user_id, rec_str))
        
        conn.commit()
        conn.close()

    async def run(self, channel):
        """
        Runs the recommendation system.

        Fetches data, cleans it, calculates recommendations, and stores them in the database.
        """
        while True:
            await channel.receive_from('Twitter')
            users, tweets = self.fetch_data()
            cleaned_tweets = self.clean_data(tweets)
        
            if len(cleaned_tweets) <= self.HOMEMAX:
                # 如果推文数量小于等于 HOMEMAX，直接为每个用户推荐所有推文
                recommendations = [tweet[0] for tweet in cleaned_tweets]  # 假设 tweet[0] 是推文ID
                user_recommendations = {user[0]: recommendations for user in users}
            else:
                # 使用模型计算相似度并选择最相关的推文
                similarities = self.calculate_similarity(cleaned_tweets)
                user_recommendations = {}
                for user in users:
                    user_id = user[0]
                    user_recommendations[user_id] = np.argsort(-similarities[user_id])[:self.HOMEMAX]

            self.store_recommendations(user_recommendations)

# 使用示例
db_file = 'testDB.db'
rec_sys = RecSys(db_file)
rec_sys.run()
