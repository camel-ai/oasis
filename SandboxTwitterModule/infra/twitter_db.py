# File: SandboxTwitterModule/infra/twitter_db.py
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, Table, JSON, func
from sqlalchemy.orm import scoped_session, sessionmaker, relationship, declarative_base
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
    username = Column(String, unique=True)
    name = Column(String)
    description = Column(Text)
    created_at = Column(DateTime)
    child_or_adult = Column(String)
    password = Column(String)
    tweets = relationship('Tweet', back_populates='user')
    traces = relationship('Trace', back_populates='user')
    twitter_account = relationship('TwitterAccount', back_populates='user', uselist=False)

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

class TwitterAccount(Base):
    __tablename__ = 'twitter_account'
    account_id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('user.id'))
    following = Column(JSON)
    followers = Column(JSON)
    mute = Column(JSON)
    block = Column(JSON)
    user = relationship('User', back_populates='twitter_account')

# 多对多关系表
user_connections = Table('user_connections', Base.metadata,
    Column('follower_id', Integer, ForeignKey('user.id'), primary_key=True),
    Column('followed_id', Integer, ForeignKey('user.id'), primary_key=True)
)

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



    # for tem !!!
    def get_user_by_username(self, username):
        """根据用户名查询用户信息"""
        session = self.Session()
        try:
            user = session.query(User).filter(User.username == username).first()
            return user
        except SQLAlchemyError as e:
            logging.error(f"Database error while retrieving user by username: {e}")
            raise
        finally:
            session.close()


    def insert_user(self, username, name, description, created_at, child_or_adult, password):
        session = self.Session()
        try:
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            user = User(username=username, name=name, description=description, created_at=created_at, child_or_adult=child_or_adult, password=hashed_password)
            session.add(user)
            session.commit()
            return user.id
        except SQLAlchemyError as e:
            session.rollback()
            logging.error(f"Database error: {e}")
            raise
        finally:
            session.close()

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

    def delete_user(self, user_id):
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



    def create_tweet(self, user_id, content):
        """ 创建新推文 """
        session = self.Session()
        try:
            tweet = Tweet(user_id=user_id, content=content, time=datetime.now())
            session.add(tweet)
            session.commit()
            return tweet.tweet_id
        except SQLAlchemyError as e:
            session.rollback()
            logging.error(f"Database error while creating tweet: {e}")
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

    def get_tweets_by_user(self, user_id):
        """ 获取特定用户的所有推文 """
        session = self.Session()
        try:
            tweets = session.query(Tweet).filter(Tweet.user_id == user_id).all()
            return tweets
        except SQLAlchemyError as e:
            logging.error(f"Database error while retrieving tweets for user {user_id}: {e}")
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



    def add_recommendation(self, user_id, tweet_ids):
        """ 添加推荐 """
        session = self.Session()
        try:
            recommendation = Recommendation(user_id=user_id, tweet_ids=json.dumps(tweet_ids))
            session.add(recommendation)
            session.commit()
            return recommendation.rec_id
        except SQLAlchemyError as e:
            session.rollback()
            logging.error(f"Database error while adding recommendation: {e}")
            raise
        finally:
            session.close()

    def update_recommendation(self, rec_id, new_tweet_ids):
        """ 更新推荐内容 """
        session = self.Session()
        try:
            recommendation = session.query(Recommendation).filter(Recommendation.rec_id == rec_id).one()
            recommendation.tweet_ids = json.dumps(new_tweet_ids)
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            logging.error(f"Database error while updating recommendation: {e}")
            raise
        finally:
            session.close()

    def get_recommendation_by_user(self, user_id):
        """ 获取特定用户的推荐 """
        session = self.Session()
        try:
            recommendation = session.query(Recommendation).filter(Recommendation.user_id == user_id).one()
            return json.loads(recommendation.tweet_ids)
        except SQLAlchemyError as e:
            logging.error(f"Database error while retrieving recommendation for user {user_id}: {e}")
            raise
        finally:
            session.close()

    def delete_recommendation(self, rec_id):
        """ 删除指定的推荐 """
        session = self.Session()
        try:
            recommendation = session.query(Recommendation).filter(Recommendation.rec_id == rec_id).one()
            session.delete(recommendation)
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            logging.error(f"Database error while deleting recommendation: {e}")
            raise
        finally:
            session.close()

    def get_all_recommendations(self):
        """ 获取所有推荐 """
        session = self.Session()
        try:
            recommendations = session.query(Recommendation).all()
            return [(rec.user_id, json.loads(rec.tweet_ids)) for rec in recommendations]
        except SQLAlchemyError as e:
            logging.error(f"Database error while retrieving all recommendations: {e}")
            raise
        finally:
            session.close()

    def filter_recommendations(self, conditions):
        """ 根据特定条件过滤推荐 """
        session = self.Session()
        try:
            query = session.query(Recommendation)
            for attr, value in conditions.items():
                if hasattr(Recommendation, attr):
                    query = query.filter(getattr(Recommendation, attr) == value)
            recommendations = query.all()
            return recommendations
        except SQLAlchemyError as e:
            logging.error(f"Database error while filtering recommendations: {e}")
            raise
        finally:
            session.close()

    def add_bulk_recommendations(self, recommendations):
        """ 批量添加推荐 """
        session = self.Session()
        try:
            for recommendation in recommendations:
                rec = Recommendation(user_id=recommendation['user_id'], tweet_ids=json.dumps(recommendation['tweet_ids']))
                session.add(rec)
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            logging.error(f"Database error while adding bulk recommendations: {e}")
            raise
        finally:
            session.close()

    def get_recommendation_details(self, rec_id):
        """ 获取推荐详细信息，包括关联的推文内容 """
        session = self.Session()
        try:
            recommendation = session.query(Recommendation).filter(Recommendation.rec_id == rec_id).one()
            tweet_ids = json.loads(recommendation.tweet_ids)
            tweets = session.query(Tweet).filter(Tweet.tweet_id.in_(tweet_ids)).all()
            return {'recommendation_id': rec_id, 'tweets': tweets}
        except SQLAlchemyError as e:
            logging.error(f"Database error while retrieving recommendation details: {e}")
            raise
        finally:
            session.close()

    def update_real_time_recommendation(self, user_id, new_tweet_ids):
        """ 实时更新用户的推荐列表 """
        session = self.Session()
        try:
            recommendation = session.query(Recommendation).filter(Recommendation.user_id == user_id).one()
            current_tweet_ids = json.loads(recommendation.tweet_ids)
            updated_tweet_ids = list(set(current_tweet_ids + new_tweet_ids))  # 合并列表并去重
            recommendation.tweet_ids = json.dumps(updated_tweet_ids)
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            logging.error(f"Database error while updating real-time recommendation: {e}")
            raise
        finally:
            session.close()

    def evaluate_recommendation_effectiveness(self, rec_id):
        """评估指定推荐的效果，基于用户互动（如点赞、评论等）"""
        session = self.Session()
        try:
            recommendation = session.query(Recommendation).filter(Recommendation.rec_id == rec_id).one()
            tweet_ids = json.loads(recommendation.tweet_ids)  # 这是一个列表
            tweets = session.query(Tweet).filter(Tweet.tweet_id.in_(tweet_ids)).all()  # 使用 .in_() 来处理列表
            total_interactions = sum(tweet.likes for tweet in tweets)  # 以点赞数作为互动指标
            return total_interactions
        except SQLAlchemyError as e:
            logging.error(f"Database error while evaluating recommendation effectiveness: {e}")
            raise
        finally:
            session.close()

    def record_recommendation_history(self, rec_id, user_id, interaction_details):
        """ 记录用户对推荐的反应，例如点击、喜欢或评论 """
        session = self.Session()
        try:
            history_entry = Trace(user_id=user_id, action='recommendation_interaction', info=json.dumps({
                'rec_id': rec_id,
                'interactions': interaction_details
            }), time=datetime.now())
            session.add(history_entry)
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            logging.error(f"Database error while recording recommendation history: {e}")
            raise
        finally:
            session.close()

    def evaluate_complex_recommendation_effectiveness(self, user_id):
        """ 基于多种互动数据评估推荐效果，包括点击、喜欢和评论 """
        session = self.Session()
        try:
            interactions = session.query(Trace).filter(Trace.user_id == user_id, Trace.action == 'recommendation_interaction').all()
            total_interactions = 0
            for interaction in interactions:
                details = json.loads(interaction.info)
                for key, value in details['interactions'].items():
                    total_interactions += value  # 假设每种互动都等价地增加一个点
            return total_interactions
        except SQLAlchemyError as e:
            logging.error(f"Database error while evaluating complex recommendation effectiveness: {e}")
            raise
        finally:
            session.close()



    def add_trace(self, user_id, action, info):
        """ 添加用户的行为追踪信息 """
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


