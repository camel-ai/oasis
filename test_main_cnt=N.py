# File: test_main_cnt=N.py
import pytest
from datetime import datetime
from SandboxTwitterModule.infra.twitter_db import TwitterDB, Base, User, Tweet, Trace, Recommendation, TwitterAccount
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, Table, JSON, func
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import NullPool
import bcrypt
import json
import logging


import sys
from SandboxTwitterModule.sandbox_twitter import SandboxTwitter as ST

N_agent_cnt = 10

@pytest.fixture(scope="module")
def db():
    # 创建内存数据库引擎
    engine = create_engine('sqlite:///:memory:', echo=True)
    # 使用从TwitterDB模块导入的Base创建所有表
    Base.metadata.create_all(engine)
    _Session = scoped_session(sessionmaker(bind=engine))

    # 创建数据库实例，确保构造器可以接受engine
    db_instance = TwitterDB(engine=engine)
    yield db_instance

    _Session.remove()
    engine.dispose()


def test_main(monkeypatch):
    # 模拟输入
    monkeypatch.setattr('sys.stdin', sys.stdin)

    # 模拟stdin的输入
    monkeypatch.setattr('builtins.input', lambda _: N_agent_cnt)  # 替换输入函数，返回 N_agent_cnt

    # 测试主函数
    my_twitter = ST()
    my_twitter.run()
    
    assert my_twitter.AgentsGraph.count == N_agent_cnt  # 假设这个方法返回用户数量
