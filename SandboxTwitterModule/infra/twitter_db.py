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
