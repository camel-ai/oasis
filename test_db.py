# File: test_db.py
import pytest
from datetime import datetime
from SandboxTwitterModule.infra.twitter_db import TwitterDB, User, Base
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker

import sys
from SandboxTwitterModule.sandbox_twitter import SandboxTwitter as ST


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

def test_insert_user(db):
    """ 测试插入用户功能 """
    user_id = db.insert_user(
        username="testuser",
        name="Test User",
        description="A test user",
        created_at=datetime.now(),
        child_or_adult="adult",
        password="testpass"
    )
    assert user_id is not None, "Failed to insert user"

    # 检查用户是否正确插入数据库
    session = db.Session()
    user = session.query(User).filter_by(id=user_id).first()
    session.close()
    assert user is not None and user.username == "testuser", "User was not inserted correctly"

def test_get_all_users(db):
    """ 测试获取所有用户功能 """
    # 首先插入一些测试用户
    db.insert_user("user1", "User One", "Description One", datetime.now(), "adult", "pass1")
    db.insert_user("user2", "User Two", "Description Two", datetime.now(), "adult", "pass2")

    users = db.get_all_users()
    assert len(users) >= 2, "Failed to retrieve all users"

def test_delete_user(db):
    """ 测试删除用户功能 """
    user_id = db.insert_user("userdel", "User Delete", "Description Delete", datetime.now(), "adult", "passdel")
    db.delete_user(user_id)

    # 检查用户是否已被删除
    session = db.Session()
    user = session.query(User).filter_by(id=user_id).first()
    session.close()
    assert user is None, "User was not deleted correctly"

def test_update_user(db):
    """ 测试更新用户信息功能 """
    user_id = db.insert_user("userup", "User Update", "Description Update", datetime.now(), "adult", "passup")
    db.update_user(user_id, {"name": "Updated Name", "description": "Updated Description"})

    # 检查用户信息是否已更新
    session = db.Session()
    user = session.query(User).filter_by(id=user_id).first()
    session.close()
    assert user.name == "Updated Name" and user.description == "Updated Description", "User was not updated correctly"




def test_main(monkeypatch):
    # 模拟输入
    monkeypatch.setattr('sys.stdin', sys.stdin)

    # 模拟stdin的输入
    monkeypatch.setattr('builtins.input', lambda _: "10")  # 替换输入函数，返回 "10"

    # 测试主函数
    my_twitter = ST()

    assert my_twitter.AgentsGraph.count == 10  # 假设这个方法返回用户数量