# File: SandboxTwitterModule/infra/twitter_api.py
import asyncio
from datetime import datetime

class TwitterAPI:
    def __init__(self, db):
        self.db = db  # This should be an instance of TwitterDB

    async def signup(self, username, password, name, description):
        existing_user = await self.db.get_user_by_username(username)
        if existing_user is None:
            hashed_password = await self.db.hash_password(password)
            user_id = await self.db.insert_user(username, name, description, datetime.now(), 'adult', hashed_password)
            return {'status': 'success', 'user_id': user_id}
        else:
            return {'status': 'error', 'message': 'Username already exists'}

    async def login(self, username, password):
        user = await self.db.get_user_by_username(username)
        if user and await self.db.check_password(user['password'], password):
            return {'status': 'success', 'user_id': user['id']}
        return {'status': 'error', 'message': 'Invalid login credentials'}

    async def tweet(self, user_id, content):
        tweet_id = await self.db.insert_tweet(user_id, content, datetime.now())
        return {'status': 'success', 'tweet_id': tweet_id}

    async def like_tweet(self, user_id, tweet_id):
        result = await self.db.add_tweet_like(tweet_id)
        if result:
            return {'status': 'success'}
        else:
            return {'status': 'error', 'message': 'Failed to like tweet'}

    async def retweet(self, user_id, original_tweet_id):
        original_tweet = await self.db.get_tweet(original_tweet_id)
        if original_tweet:
            retweet_content = f"RT @{original_tweet['username']}: {original_tweet['content']}"
            retweet_id = await self.db.insert_tweet(user_id, retweet_content, datetime.now())
            return {'status': 'success', 'retweet_id': retweet_id}
        else:
            return {'status': 'error', 'message': 'Original tweet not found'}

    async def home_get(self, user_id):
        recommended_tweets = await self.db.get_recommendations(user_id)
        return {'status': 'success', 'tweets': recommended_tweets}

    async def home_refresh(self, user_id):
        # Simulate updating recommendations by selecting new random tweets
        await self.db.refresh_recommendations(user_id)
        updated_tweets = await self.db.get_recommendations(user_id)
        return {'status': 'success', 'tweets': updated_tweets}
