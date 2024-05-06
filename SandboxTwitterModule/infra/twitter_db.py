# File: SandboxTwitterModule/infra/twitter_db.py
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, Table, JSON, func, Boolean
from sqlalchemy import select, delete, insert, update
from sqlalchemy.orm import scoped_session, sessionmaker, relationship, declarative_base, joinedload
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import NullPool
import bcrypt
import json
import logging
from datetime import datetime

from ..core.decorators import function_call_logger

Base = declarative_base()

# 定义数据库模型
class User(Base):
    __tablename__ = 'user'
    id = Column(Integer, primary_key=True)
    name = Column(String)  # name 就是 twitterUserAgent 的 real_name
    description = Column(Text)
    created_at = Column(DateTime)
    child_or_adult = Column(String)
    password = Column(String)  # 存储加密后的密码
    login_state = Column(Boolean, default=False)  # 新增字段表示登录状态，默认为False
    tweets = relationship('Tweet', back_populates='user')
    traces = relationship('Trace', back_populates='user')
    twitter_account = relationship('TwitterAccount', back_populates='user', uselist=False)

class TwitterAccount(Base):
    __tablename__ = 'twitter_account'
    account_id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('user.id'))
    username = Column(String, unique=True)  # 用户名从User表移至此处
    following = Column(JSON)
    followers = Column(JSON)
    mute = Column(JSON)
    block = Column(JSON)
    home_timeline = Column(JSON)  # 新增字段，存储推文ID数组
    user = relationship('User', back_populates='twitter_account')

class Tweet(Base):
    __tablename__ = 'tweet'
    tweet_id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('user.id'))
    content = Column(Text)
    time = Column(DateTime)
    likes = Column(Integer, default=0)
    user = relationship('User', back_populates='tweets')

class Trace(Base):
    __tablename__ = 'trace'
    trace_id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('user.id'))
    time = Column(DateTime)
    action = Column(Text)
    info = Column(Text)
    user = relationship('User', back_populates='traces')

class Recommendation(Base):
    __tablename__ = 'rec'
    rec_id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('user.id'))
    tweet_ids = Column(JSON)
    user = relationship('User', backref='recommendation')



# TwitterDB 类定义
class TwitterDB:
    @function_call_logger
    def __init__(self, engine=None, db_uri='sqlite:///twitter_simulation.db'):
        if engine:
            self.engine = engine
        else:
            if db_uri:
                self.engine = create_engine(db_uri, echo=True, poolclass=NullPool)
            else:
                logging.error("No database URI or engine provided.")
                raise ValueError("A database URI or engine must be provided.")

        self.Session = scoped_session(sessionmaker(bind=self.engine))
        Base.metadata.create_all(self.engine)
        self.db_info = db_uri if db_uri else "Engine provided, no URI available"

    def get_db_info(self):
        """返回数据库的描述信息。"""
        return self.db_info

    def __del__(self):
        self.Session.remove()


    def get_all_users(self):
        session = self.Session()
        try:
            users = session.query(User).all()
            return users
        except SQLAlchemyError as e:
            logging.error(f"Database error while retrieving all users: {e}")
            raise
        finally:
            session.close()

    def user_exists(self, user_id):
        session = self.Session()
        try:
            user = session.query(User).filter(User.id == user_id).one_or_none()
            return user is not None
        finally:
            session.close()


    def update_user_login_state(self, user_id, is_online):
        """Update the user's login state in the database."""
        session = self.Session()
        try:
            user = session.query(User).filter(User.id == user_id).one()
            user.login_state = is_online
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()


    def get_user_by_id(self, user_id):
        with self.Session() as session:
            try:
                # 使用 joinedload (Eager Loading) 预加载关联的 TwitterAccount，以防止在对象脱离 Session (Session) 后遇到懒加载 (Lazy Loading) 的问题
                user = session.query(User).filter(User.id == user_id).options(joinedload(User.twitter_account)).one_or_none()
                return user
            except SQLAlchemyError as e:
                # 处理可能的SQLAlchemy错误
                logging.error(f"[db.py get_user_by_id] Database error while retrieving user by ID: {e}")
                session.rollback()
                raise


    def get_user_by_real_name(self, real_name):
        """根据 用户的真实姓名 查询用户信息。如果用户存在，返回用户对象；否则返回None。"""
        session = self.Session()
        try:
            user = session.query(User).filter(User.name == real_name).first()
            return user
        except SQLAlchemyError as e:
            logging.error(f"Database error while retrieving user by real name: {e}")
            raise
        finally:
            session.close()


    def get_user_by_username(self, username):
        """根据 用户名 查询用户信息。如果用户存在，返回用户对象；否则返回None。"""
        session = self.Session()
        try:
            account = session.query(TwitterAccount).filter(TwitterAccount.username == username).first()
            return account.user if account else None # 请注意这里返回的是 user 对象，而不是 account 对象
        except SQLAlchemyError as e:
            logging.error(f"Database error while retrieving user by username: {e}")
            raise
        finally:
            session.close()


    def account_exists_by_username(self, username):
        """检查用户名是否存在于TwitterAccount表中"""
        session = self.Session()
        try:
            account = session.query(TwitterAccount).filter(TwitterAccount.username == username).first()
            return account is not None # 请注意这里返回的是 account 对象，而不是 account_id
        except SQLAlchemyError as e:
            logging.error(f"Database error while checking username: {e}")
            raise
        finally:
            session.close()


    def insert_user(self, username, name, description, created_at, child_or_adult, password):
        session = self.Session()
        try:
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            user = User(name=name, description=description, created_at=created_at, child_or_adult=child_or_adult, password=hashed_password)
            session.add(user)
            session.commit()
            account = TwitterAccount(user_id=user.id, username=username)
            session.add(account)
            session.commit()
            return user.id
        except SQLAlchemyError as e:
            session.rollback()
            logging.error(f"Error during user insertion: {e}")
            return None
        finally:
            session.close()



    def add_trace(self, user_id, action, info):
        """记录用户的行为追踪信息"""
        session = self.Session()
        try:
            trace = Trace(user_id=user_id, action=action, info=info, time=datetime.now())
            session.add(trace)
            session.commit()
            return trace.trace_id
        except SQLAlchemyError as e:
            session.rollback()
            logging.error(f"Database error while adding trace: {e}")
            raise
        finally:
            session.close()





    def add_tweet(self, user_id, content):
        session = self.Session()
        try:
            tweet = Tweet(user_id=user_id, content=content, time=datetime.now())
            session.add(tweet)
            session.flush()  # Ensure tweet_id is created
            trace_info = f'[tweet] User (id={user_id}) tweeted: {content} at {datetime.now()}'
            self.add_trace(user_id, 'tweet', trace_info)
            session.commit()
            return {'status': 'success', 'message': 'Tweet added and traced successfully', 'tweet_id': tweet.tweet_id}
        except Exception as e:
            session.rollback()
            return {'status': 'error', 'message': str(e)}
        finally:
            session.close()


    def get_tweets_by_user_id(self, user_id):
        session = self.Session()
        try:
            # 查询特定用户的所有推文，并按时间降序排列
            tweets = session.query(Tweet).filter(Tweet.user_id == user_id).order_by(Tweet.time.desc()).all()
            return tweets
        except Exception as e:
            print(f"Error retrieving tweets for user {user_id}: {str(e)}")
            return []
        finally:
            session.close()


    def get_user_id_by_real_name(self, real_name):
        session = self.Session()
        try:
            user = session.query(User).filter(User.name == real_name).one()
            return user.id
        except Exception as e:
            session.rollback()
            print("Failed to fetch user ID by real name:", str(e))
            return None
        finally:
            session.close()


    def follow_user(self, follower_id, followee_id):
        session = self.Session()
        try:
            print(f"[db.py follow_user] Attempting to follow: follower_id={follower_id}, followee_id={followee_id}")
            follower_account = session.query(TwitterAccount).filter_by(user_id=follower_id).one_or_none()
            followee_account = session.query(TwitterAccount).filter_by(user_id=followee_id).one_or_none()

            if follower_account and followee_account:
                if followee_id not in json.loads(follower_account.following or '[]'):
                    following_list = json.loads(follower_account.following or '[]')
                    following_list.append(followee_id)
                    follower_account.following = json.dumps(following_list)

                    followers_list = json.loads(followee_account.followers or '[]')
                    followers_list.append(follower_id)
                    followee_account.followers = json.dumps(followers_list)
                    trace_info = f'[follow] User (id={follower_id}) followed user (id={followee_id}) at {datetime.now()}'
                    self.add_trace(follower_id, 'follow', trace_info)

                    session.commit()
                    print(f"[db.py follow_user] Follow relationship[User(id={follower_id})->User(id={followee_id})] successfully created")

                    return {'status': 'success', 'message': 'User followed successfully'}
                return {'status': 'error', 'message': 'Already following this user'}
            return {'status': 'error', 'message': 'Invalid follower_id or followee_id'}
        except Exception as e:
            session.rollback()
            print("[db.py follow_user] Error occurred:", str(e))
            return {'status': 'error', 'message': str(e)}
        finally:
            session.close()




    def unfollow_user(self, follower_id, followee_id):
        session = self.Session()
        try:
            follower_account = session.query(TwitterAccount).filter_by(user_id=follower_id).one_or_none()
            followee_account = session.query(TwitterAccount).filter_by(user_id=followee_id).one_or_none()

            if follower_account and followee_account:
                if followee_id in json.loads(follower_account.following or '[]'):
                    following_list = json.loads(follower_account.following or '[]')
                    following_list.remove(followee_id)
                    follower_account.following = json.dumps(following_list)

                    followers_list = json.loads(followee_account.followers or '[]')
                    followers_list.remove(follower_id)
                    followee_account.followers = json.dumps(followers_list)

                    trace_info = f'[unfollow] User (id={follower_id}) unfollowed user (id={followee_id}) at {datetime.now()}'
                    self.add_trace(follower_id, 'unfollow', trace_info)

                    session.commit()
                    print(f"[db.py unfollow_user] Follow relationship[User(id={follower_id})->User(id={followee_id})] successfully unfollowed")

                    return {'status': 'success', 'message': 'User unfollowed successfully'}
                return {'status': 'error', 'message': 'Not following this user'}
            return {'status': 'error', 'message': 'Invalid follower_id or followee_id'}
        except Exception as e:
            session.rollback()
            print("[db.py unfollow_user] Error occurred:", str(e))
            return {'status': 'error', 'message': str(e)}
        finally:
            session.close()


    def get_following_ids(self, user_id):
        session = self.Session()
        try:
            user_account = session.query(TwitterAccount).filter_by(user_id=user_id).one_or_none()
            if user_account:
                print(f"User account found for user_id {user_id}, following: {user_account.following}")
            else:
                print(f"No user account found for user_id {user_id}")

            if user_account and user_account.following:
                following_ids = json.loads(user_account.following)
                print(f"Following IDs for user_id {user_id}: {following_ids}")
                return following_ids
            else:
                print(f"No following data or invalid JSON for user_id {user_id}")
            return []
        except Exception as e:
            print(f"[db.py get_following_ids] Database error while fetching following IDs for user {user_id}: {e}")
            return []
        finally:
            session.close()


























'''


    # for tem !!!

    def delete_user(self, user_id): # by_id
        session = self.Session()
        try:
            user = session.query(User).filter(User.id == user_id).one()
            session.delete(user)
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            logging.error(f"Database error while deleting user: {e}")
            raise
        finally:
            session.close()

    def delete_user_by_username(self, username): # by_username
        session = self.Session()
        try:
            user = session.query(User).filter(User.username == username).one_or_none()
            if user:
                session.delete(user)
                session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            logging.error(f"Database error during user deletion: {e}")
            raise
        finally:
            session.close()


    def update_user(self, user_id, update_fields):
        session = self.Session()
        try:
            user = session.query(User).filter(User.id == user_id).one()
            for key, value in update_fields.items():
                setattr(user, key, value)
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            logging.error(f"Database error while updating user: {e}")
            raise
        finally:
            session.close()

    def search_users(self, filter_by):
        """ 根据指定条件搜索用户 """
        session = self.Session()
        try:
            query = session.query(User)
            for attr, value in filter_by.items():
                query = query.filter(getattr(User, attr) == value)
            users = query.all()
            return users
        except SQLAlchemyError as e:
            logging.error(f"Database error while searching users: {e}")
            raise
        finally:
            session.close()

    def bulk_delete_users(self, user_ids):
        """ 批量删除指定的用户列表 """
        session = self.Session()
        try:
            for user_id in user_ids:
                user = session.query(User).filter(User.id == user_id).one()
                session.delete(user)
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            logging.error(f"Database error during bulk deletion: {e}")
            raise
        finally:
            session.close()


    def update_tweet(self, tweet_id, new_content):
        """ 更新推文内容 """
        session = self.Session()
        try:
            tweet = session.query(Tweet).filter(Tweet.tweet_id == tweet_id).one()
            tweet.content = new_content
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            logging.error(f"Database error while updating tweet: {e}")
            raise
        finally:
            session.close()

    def search_tweets(self, filter_by):
        """ 根据指定条件搜索推文 """
        session = self.Session()
        try:
            query = session.query(Tweet)
            for attr, value in filter_by.items():
                query = query.filter(getattr(Tweet, attr) == value)
            tweets = query.all()
            return tweets
        except SQLAlchemyError as e:
            logging.error(f"Database error while searching tweets: {e}")
            raise
        finally:
            session.close()

    def delete_tweet(self, tweet_id):
        """ 删除指定的推文 """
        session = self.Session()
        try:
            tweet = session.query(Tweet).filter(Tweet.tweet_id == tweet_id).one()
            session.delete(tweet)
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            logging.error(f"Database error while deleting tweet: {e}")
            raise
        finally:
            session.close()


    def count_tweets(self):
        """ 统计系统中的总推文数 """
        session = self.Session()
        try:
            tweet_count = session.query(Tweet).count()
            return tweet_count
        except SQLAlchemyError as e:
            logging.error(f"Database error while counting tweets: {e}")
            raise
        finally:
            session.close()


    def get_traces_by_user(self, user_id):
        """ 获取特定用户的所有追踪信息 """
        session = self.Session()
        try:
            traces = session.query(Trace).filter(Trace.user_id == user_id).all()
            return traces
        except SQLAlchemyError as e:
            logging.error(f"Database error while retrieving traces: {e}")
            raise
        finally:
            session.close()



    def add_twitter_account(self, user_id, following=None, followers=None, mute=None, block=None):
        """ 添加用户的Twitter账户信息 """
        session = self.Session()
        try:
            twitter_account = TwitterAccount(
                user_id=user_id,
                following=json.dumps(following or []),
                followers=json.dumps(followers or []),
                mute=json.dumps(mute or []),
                block=json.dumps(block or [])
            )
            session.add(twitter_account)
            session.commit()
            return twitter_account.account_id
        except SQLAlchemyError as e:
            session.rollback()
            logging.error(f"Database error while adding Twitter account: {e}")
            raise
        finally:
            session.close()

    def get_twitter_account_by_user(self, user_id):
        """ 获取用户的Twitter账户信息 """
        session = self.Session()
        try:
            account = session.query(TwitterAccount).filter(TwitterAccount.user_id == user_id).one()
            return account
        except SQLAlchemyError as e:
            logging.error(f"Database error while retrieving Twitter account: {e}")
            raise
        finally:
            session.close()


    def refresh_home_timeline(self, user_id):
        """刷新指定用户的首页时间线"""
        session = self.Session()
        try:
            # 获取用户的当前推荐
            recommendation = session.query(Recommendation).filter(Recommendation.user_id == user_id).one_or_none()
            if recommendation:
                # 更新 twitterAccount table 中的 home_timeline 字段
                twitter_account = session.query(TwitterAccount).filter(TwitterAccount.user_id == user_id).one()
                twitter_account.home_timeline = json.dumps(recommendation.tweet_ids)
                session.commit()
                return {'status': 'success', 'message': 'Home timeline updated'}
            return {'status': 'error', 'message': 'No recommendations found'}
        except SQLAlchemyError as e:
            session.rollback()
            logging.error(f"Database error while refreshing home timeline: {e}")
            raise
        finally:
            session.close()


'''



