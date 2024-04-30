# File: test_db.py
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

def test_search_users(db):
    """ 测试根据条件搜索用户功能 """
    db.insert_user("searchuser1", "Search User One", "Description", datetime.now(), "adult", "pass1")
    db.insert_user("searchuser2", "Search User Two", "Description", datetime.now(), "adult", "pass2")

    # 搜索名字包含 'One' 的用户
    users = db.search_users({"name": "Search User One"})
    assert len(users) == 1 and users[0].username == "searchuser1", "Search function failed to find the right users"

def test_bulk_delete_users(db):
    """ 测试批量删除用户功能 """
    user1_id = db.insert_user("bulkuser1", "Bulk User One", "Description", datetime.now(), "adult", "pass1")
    user2_id = db.insert_user("bulkuser2", "Bulk User Two", "Description", datetime.now(), "adult", "pass2")

    db.bulk_delete_users([user1_id, user2_id])

    # 检查用户是否被删除
    session = db.Session()
    users = session.query(User).filter(User.id.in_([user1_id, user2_id])).all()
    session.close()
    assert len(users) == 0, "Bulk delete failed, some users were not deleted"



def test_create_tweet(db):
    """ 测试创建推文功能 """
    user_id = db.insert_user("tweetuser", "Tweet User", "User for tweets", datetime.now(), "adult", "tweetpass")
    tweet_id = db.create_tweet(user_id, "Hello, Twitter!")
    assert tweet_id is not None, "Failed to create tweet"

    # 检查推文是否正确创建
    session = db.Session()
    tweet = session.query(Tweet).filter_by(tweet_id=tweet_id).first()
    session.close()
    assert tweet is not None and tweet.content == "Hello, Twitter!", "Tweet was not inserted correctly"

def test_update_tweet(db):
    """ 测试更新推文内容功能 """
    user_id = db.insert_user("update_tweet_user", "User for updating tweets", "Updating tweets", datetime.now(), "adult", "updatetweetpass")
    tweet_id = db.create_tweet(user_id, "Initial content")
    db.update_tweet(tweet_id, "Updated content")

    # 检查推文内容是否更新
    session = db.Session()
    tweet = session.query(Tweet).filter_by(tweet_id=tweet_id).first()
    session.close()
    assert tweet.content == "Updated content", "Tweet content was not updated correctly"

def test_search_tweets(db):
    """ 测试搜索推文功能 """
    user_id = db.insert_user("searchtweetuser", "Search Tweet User", "Search tweets", datetime.now(), "adult", "searchtweet")
    db.create_tweet(user_id, "Search this tweet")
    tweets = db.search_tweets({"content": "Search this tweet"})
    
    assert len(tweets) == 1 and tweets[0].content == "Search this tweet", "Failed to search the tweet correctly"

def test_delete_tweet(db):
    """ 测试删除推文功能 """
    user_id = db.insert_user("delete_tweet_user", "Delete Tweet User", "User for delete", datetime.now(), "adult", "deletetweetpass")
    tweet_id = db.create_tweet(user_id, "Delete this tweet")
    db.delete_tweet(tweet_id)

    # 检查推文是否被删除
    session = db.Session()
    tweet = session.query(Tweet).filter_by(tweet_id=tweet_id).first()
    session.close()
    assert tweet is None, "Tweet was not deleted correctly"

def test_get_tweets_by_user(db):
    """ 测试获取特定用户的所有推文功能 """
    user_id = db.insert_user("tweet_list_user", "Tweet List User", "List of tweets", datetime.now(), "adult", "listtweets")
    db.create_tweet(user_id, "First tweet")
    db.create_tweet(user_id, "Second tweet")
    tweets = db.get_tweets_by_user(user_id)

    assert len(tweets) == 2, "Did not retrieve correct number of tweets for user"

def test_count_tweets(db):
    """ 测试统计推文数量功能 """
    user_id = db.insert_user("count_tweet_user", "Count Tweet User", "Counting tweets", datetime.now(), "adult", "counttweet")
    db.create_tweet(user_id, "Count this tweet")
    db.create_tweet(user_id, "Count another tweet")
    total_count = db.count_tweets()

    assert total_count >= 2, "Tweet count did not match expected number"




def test_add_recommendation(db):
    """ 测试添加推荐功能 """
    user_id = db.insert_user("rec_user", "Recommendation User", "User for recommendation", datetime.now(), "adult", "recpass")
    tweet_ids = [db.create_tweet(user_id, f"Tweet {i}") for i in range(5)]
    rec_id = db.add_recommendation(user_id, tweet_ids)

    # 检查推荐是否正确添加
    session = db.Session()
    recommendation = session.query(Recommendation).filter_by(rec_id=rec_id).first()
    session.close()
    assert recommendation is not None and json.loads(recommendation.tweet_ids) == tweet_ids, "Recommendation was not added correctly"

def test_update_recommendation(db):
    """ 测试更新推荐内容功能 """
    user_id = db.insert_user("update_rec_user", "Update Rec User", "Updating recommendations", datetime.now(), "adult", "updaterecpass")
    tweet_ids = [db.create_tweet(user_id, f"Tweet {i}") for i in range(2)]
    rec_id = db.add_recommendation(user_id, tweet_ids)
    new_tweet_ids = [db.create_tweet(user_id, f"New Tweet {i}") for i in range(3)]
    db.update_recommendation(rec_id, new_tweet_ids)

    # 检查推荐是否更新
    session = db.Session()
    recommendation = session.query(Recommendation).filter_by(rec_id=rec_id).first()
    session.close()
    assert json.loads(recommendation.tweet_ids) == new_tweet_ids, "Recommendation was not updated correctly"

def test_get_recommendation_by_user(db):
    """ 测试获取特定用户的推荐功能 """
    user_id = db.insert_user("get_rec_user", "Get Rec User", "Getting recommendations", datetime.now(), "adult", "getrecpass")
    tweet_ids = [db.create_tweet(user_id, f"Tweet {i}") for i in range(3)]
    db.add_recommendation(user_id, tweet_ids)
    fetched_tweet_ids = db.get_recommendation_by_user(user_id)

    assert fetched_tweet_ids == tweet_ids, "Failed to fetch the correct recommendations"

def test_delete_recommendation(db):
    """ 测试删除推荐功能 """
    user_id = db.insert_user("user_del_rec", "User Del Rec", "User for deleting recommendation", datetime.now(), "adult", "delrecpass")
    tweet_ids = [db.create_tweet(user_id, f"Delete Rec Tweet {i}") for i in range(1)]
    rec_id = db.add_recommendation(user_id, tweet_ids)
    db.delete_recommendation(rec_id)

    # 检查推荐是否被删除
    session = db.Session()
    recommendation = session.query(Recommendation).filter_by(rec_id=rec_id).first()
    session.close()
    assert recommendation is None, "Recommendation was not deleted correctly"

def test_get_all_recommendations(db):
    """ 测试获取所有推荐功能 """
    user_id = db.insert_user("user_all_recs", "User All Recs", "User for all recommendations", datetime.now(), "adult", "allrecspass")
    tweet_ids = [db.create_tweet(user_id, "All Recs Tweet") for i in range(2)]
    db.add_recommendation(user_id, tweet_ids)

    recommendations = db.get_all_recommendations()
    assert len(recommendations) > 0, "Failed to retrieve all recommendations"

def test_filter_recommendations(db):
    """ 测试按条件过滤推荐功能 """
    user_id = db.insert_user("user_filter_recs", "User Filter Recs", "User for filtering recommendations", datetime.now(), "adult", "filterrecspass")
    tweet_ids = [db.create_tweet(user_id, "Filter Recs Tweet") for i in range(3)]
    rec_id = db.add_recommendation(user_id, tweet_ids)
    
    # 以用户 ID 作为筛选条件
    filtered_recs = db.filter_recommendations({'user_id': user_id})
    assert len(filtered_recs) == 1 and filtered_recs[0].rec_id == rec_id, "Failed to filter recommendations correctly"

def test_add_bulk_recommendations(db):
    """ 测试批量添加推荐功能 """
    # 先清理所有现有推荐
    db.Session.query(Recommendation).delete()
    db.Session.commit()

    user_id = db.insert_user("bulk_rec_user", "Bulk Rec User", "User for bulk recommendations", datetime.now(), "adult", "bulkrecpass")
    bulk_recommendations = [
        {'user_id': user_id, 'tweet_ids': [db.create_tweet(user_id, f"Bulk Tweet {i}") for i in range(2)]},
        {'user_id': user_id, 'tweet_ids': [db.create_tweet(user_id, f"Bulk Tweet {i}") for i in range(2, 4)]}
    ]
    db.add_bulk_recommendations(bulk_recommendations)

    # 验证推荐是否正确添加
    recommendations = db.get_all_recommendations()
    assert len(recommendations) == 2, "Bulk recommendations were not added correctly"

def test_get_recommendation_details(db):
    """ 测试获取推荐详细信息功能 """
    user_id = db.insert_user("rec_detail_user", "Rec Detail User", "User for recommendation details", datetime.now(), "adult", "recdetailpass")
    tweet_ids = [db.create_tweet(user_id, "Detailed Rec Tweet") for _ in range(3)]
    rec_id = db.add_recommendation(user_id, tweet_ids)
    rec_details = db.get_recommendation_details(rec_id)

    assert rec_details['recommendation_id'] == rec_id and len(rec_details['tweets']) == 3, "Recommendation details were not retrieved correctly"

def test_update_real_time_recommendation(db):
    """ 测试实时更新推荐列表功能 """
    user_id = db.insert_user("real_time_rec_user", "Real Time Rec User", "User for real-time updates", datetime.now(), "adult", "realtimeupdate")
    initial_tweet_ids = [db.create_tweet(user_id, "Initial Tweet") for _ in range(2)]
    new_tweet_ids = [db.create_tweet(user_id, "New Tweet") for _ in range(2)]
    db.add_recommendation(user_id, initial_tweet_ids)
    db.update_real_time_recommendation(user_id, new_tweet_ids)
    
    recommendation_details = db.get_recommendation_by_user(user_id)
    assert len(recommendation_details) == 4, "Real-time recommendation update failed"

def test_evaluate_recommendation_effectiveness(db):
    """测试评估推荐效果功能"""
    user_id = db.insert_user("eval_rec_user", "Eval Rec User", "User for evaluating recommendations", datetime.now(), "adult", "evalrec")
    tweet_ids = [db.create_tweet(user_id, "Evaluable Tweet") for _ in range(3)]
    rec_id = db.add_recommendation(user_id, tweet_ids)
    effectiveness = db.evaluate_recommendation_effectiveness(rec_id)
    assert effectiveness >= 0, "Failed to evaluate recommendation effectiveness"

def test_record_recommendation_history(db):
    """ 测试记录推荐历史功能 """
    user_id = db.insert_user("history_user", "History User", "User for history", datetime.now(), "adult", "historypass")
    rec_id = db.add_recommendation(user_id, [db.create_tweet(user_id, "Historical Tweet")])
    db.record_recommendation_history(rec_id, user_id, {'clicks': 5, 'likes': 3})

    # 检查历史记录是否正确添加
    session = db.Session()
    history_entries = session.query(Trace).filter(Trace.user_id == user_id).all()
    session.close()
    assert len(history_entries) == 1, "Recommendation history was not recorded correctly"

def test_evaluate_complex_recommendation_effectiveness(db):
    """ 测试复杂推荐效果评估功能 """
    user_id = db.insert_user("complex_eval_user", "Complex Eval User", "User for complex evaluation", datetime.now(), "adult", "complexpass")
    rec_id = db.add_recommendation(user_id, [db.create_tweet(user_id, "Complex Eval Tweet")])
    db.record_recommendation_history(rec_id, user_id, {'clicks': 10, 'likes': 5, 'comments': 2})

    effectiveness = db.evaluate_complex_recommendation_effectiveness(user_id)
    assert effectiveness == 17, "Complex recommendation effectiveness was not evaluated correctly"




def test_add_trace(db):
    """测试添加追踪信息功能"""
    user_id = db.insert_user("trace_user", "Trace User", "A user to test trace functionality", datetime.now(), "adult", "tracepass")
    trace_id = db.add_trace(user_id, "Login", "User logged in")
    assert trace_id is not None, "Failed to add trace"

def test_get_traces_by_user(db):
    """测试获取用户追踪信息功能"""
    unique_username = f"trace_user_{datetime.now().timestamp()}"
    user_id = db.insert_user(
        username=unique_username,
        name="Trace User",
        description="A user to test trace functionality",
        created_at=datetime.now(),
        child_or_adult="adult",
        password="tracepass"
    )
    db.add_trace(user_id, "Login", "User logged in")
    traces = db.get_traces_by_user(user_id)
    assert traces is not None and len(traces) > 0, "Failed to retrieve traces or no traces found"




def test_add_twitter_account(db):
    """测试添加Twitter账户功能"""
    user_id = db.insert_user("account_user", "Account User", "A user to test account functionality", datetime.now(), "adult", "accountpass")
    account_id = db.add_twitter_account(user_id)
    assert account_id is not None, "Failed to add Twitter account"

def test_get_twitter_account_by_user(db):
    """测试获取Twitter账户信息功能"""
    unique_username = f"account_user_{datetime.now().timestamp()}"
    user_id = db.insert_user(
        username=unique_username,
        name="Account User",
        description="A user to test account functionality",
        created_at=datetime.now(),
        child_or_adult="adult",
        password="accountpass"
    )
    db.add_twitter_account(user_id)
    account = db.get_twitter_account_by_user(user_id)
    assert account is not None, "Failed to retrieve Twitter account"




def test_main(monkeypatch):
    # 模拟输入
    monkeypatch.setattr('sys.stdin', sys.stdin)

    # 模拟stdin的输入
    monkeypatch.setattr('builtins.input', lambda _: "10")  # 替换输入函数，返回 "10"

    # 测试主函数
    my_twitter = ST()

    assert my_twitter.AgentsGraph.count == 10  # 假设这个方法返回用户数量