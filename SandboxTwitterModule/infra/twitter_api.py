# File: SandboxTwitterModule/infra/twitter_api.py
import asyncio
import bcrypt
from datetime import datetime

class TwitterAPI:
    def __init__(self, db):
        self.db = db  # This should be an instance of TwitterDB


    async def signup(self, real_name, username, password, description):
        try:
            # 检查 real_name 是否已存在
            if self.db.get_user_by_real_name(real_name):
                return {'status': 'error', 'message': 'The same real_name already exists. Please login directly.'}

            # 尝试创建新用户，确保 username 是唯一的
            while self.db.account_exists_by_username(username):
                username = self._generate_new_username(username)

            user_id = self.db.insert_user(real_name, username, description, datetime.now(), 'adult', password)
            if user_id:
                trace_info = f'[signup] User (real_name={real_name}, username={username}) signed up at {datetime.now()}'
                trace_id = self.db.add_trace(user_id, 'signup', trace_info)
                return {'status': 'success', 'user_id': user_id, 'username': username, 'trace_id': trace_id}
            else:
                return {'status': 'error', 'message': 'Signup failed'}
        except Exception as e:
            return {'status': 'error', 'message': f'An error occurred: {str(e)}'}

    def _generate_new_username(self, username):
        """ Generate a new username by appending a random number or incrementing an existing number """
        import random
        return f"{username}_{random.randint(1, 9999)}"



    async def login_by_realname(self, real_name, password):
        """Login the user using their real name, and verify their password."""
        user = self.db.get_user_by_real_name(real_name)
        if user and bcrypt.checkpw(password.encode('utf-8'), user.password):
            # Log the login action
            self.db.update_user_login_state(user.id, True)  # Set user state to online
            trace_info = f'[login] User (real_name={real_name}) logged in at {datetime.now()} by its real_name'
            trace_id = self.db.add_trace(user.id, 'login', trace_info)
            return {'status': 'success', 'user_id': user.id, 'trace_id': trace_id}
        return {'status': 'error', 'message': 'Invalid login credentials'}

    async def login_by_username(self, username, password):
        """Login the user using their username, and verify their password."""
        user = self.db.get_user_by_username(username)
        if user and bcrypt.checkpw(password.encode('utf-8'), user.password):
            # Log the login action
            trace_info = f'[login] User (username={username}) logged in at {datetime.now()} by its username'
            trace_id = self.db.add_trace(user.id, 'login', trace_info)
            return {'status': 'success', 'user_id': user.id, 'trace_id': trace_id}
        return {'status': 'error', 'message': 'Invalid login credentials'}



    async def logout(self, user_id):
        """Log out the user and record the logout action."""
        user = self.db.get_user_by_id(user_id)  # 使用新方法通过 ID 获取用户
        if user and user.login_state:  # Ensure the user is online before attempting to logout
            self.db.update_user_login_state(user_id, False)  # Set user state to offline
            trace_info = f'[logout] User (real_name={user.name}) logged out at {datetime.now()}'
            trace_id = self.db.add_trace(user_id, 'logout', trace_info)
            return {'status': 'success', 'trace_id': trace_id}
        return {'status': 'error', 'message': 'User is not logged in or already logged out'}



    async def tweet(self, user_id, content):
        result = self.db.add_tweet(user_id, content)
        return result

    async def fetch_tweets_by_user(self, user_id):
        tweets = self.db.get_tweets_by_user_id(user_id)
        return tweets




    async def fetch_user_id_by_real_name(self, real_name):
        """Asynchronously fetch user_id by real name from the database."""
        try:
            return await asyncio.get_event_loop().run_in_executor(None, self.db.get_user_id_by_real_name, real_name)
        except Exception as e:
            print(f"Error fetching user ID for {real_name}: {e}")
            return None

    async def fetch_all_users(self):
        """Asynchronously fetch all users."""
        try:
            return await asyncio.get_event_loop().run_in_executor(None, self.db.get_all_users)
        except Exception as e:
            print("Error fetching all users:", e)
            return []

    async def follow_user(self, follower_id, followee_id):
        """Asynchronously follow a user."""
        print(f"[api.py follow_user] User(id={follower_id}) is preparing to follow User(id={followee_id})")
        try:
            return await asyncio.get_event_loop().run_in_executor(None, self.db.follow_user, follower_id, followee_id)
        except Exception as e:
            print(f"[api.py follow_user]Error following user {followee_id} by user {follower_id}: {e}")
            return {'status': 'error', 'message': str(e)}


    async def get_following_ids(self, user_id):
        """Asynchronously get the list of user IDs that the given user is following."""
        try:
            return await asyncio.get_event_loop().run_in_executor(None, self.db.get_following_ids, user_id)
        except Exception as e:
            print(f"[api.py get_following_ids] Error fetching following IDs for user {user_id}: {e}")
            return []


    async def unfollow_user(self, follower_id, followee_id):
        print(f"[api.py unfollow_user] User(id={follower_id}) is preparing to unfollow User(id={followee_id})")
        try:
            return await asyncio.get_event_loop().run_in_executor(None, self.db.unfollow_user, follower_id, followee_id)
        except Exception as e:
            print(f"[api.py unfollow_user] Error unfollowing user {followee_id} by user {follower_id}: {e}")
            return {'status': 'error', 'message': str(e)}






















'''

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


    async def fetch_timeline(self, user_id):
        """获取用户的时间线，即他们关注的人的最新推文。"""
        tweets = await self.db.get_timeline_tweets(user_id)
        return {'status': 'success', 'tweets': tweets}


'''