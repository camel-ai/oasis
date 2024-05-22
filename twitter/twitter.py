from __future__ import annotations

import random
import sqlite3
from datetime import datetime, timedelta
from typing import Any

from .database import create_db, fetch_rec_table_as_matrix, fetch_table_from_db
from .recsys import rec_sys_random
from .typing import ActionType


class Twitter:

    def __init__(self, db_path: str, channel: Any):
        create_db(db_path)
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db_cursor = self.db.cursor()
        self.channel = channel

        # channel传进的操作数量
        self.ope_cnt = -1
        # 推荐系统缓存更新的时间间隔（以传进来的操作数为单位）
        self.rec_update_time = 20

        # twitter内部推荐系统refresh一次返回的推文数量
        self.refresh_tweet_count = 5
        # rec table(buffer)中每个用户的最大tweet数量
        self.max_rec_tweet_len = 50

        # twitter内部定义的热搜规则参数
        self.trend_num_days = 7
        self.trend_top_k = 10

    @staticmethod
    def _not_signup_error_message(agent_id):
        return {
            "success":
            False,
            "error": (f"Agent {agent_id} have not signed up and have no "
                      f"user id.")
        }

    def _execute_db_command(self, command, args=(), commit=False):
        self.db_cursor.execute(command, args)
        if commit:
            self.db.commit()
        return self.db_cursor

    async def running(self):
        while True:
            message_id, data = await self.channel.receive_from()
            if message_id:
                self.ope_cnt += 1
            agent_id, message, action = data
            action = ActionType(action)

            if (self.ope_cnt % self.rec_update_time == 0
                    and action != ActionType.REFRESH):
                print('Successfully update rec table.')
                self.ope_cnt += 1
                await self.update_rec_table()

            if action == ActionType.EXIT:
                self.db_cursor.close()
                self.db.close()
                break

            elif action == ActionType.UPDATE_REC:
                await self.update_rec_table()

            elif action == ActionType.SIGNUP:
                result = await self.signup(agent_id=agent_id,
                                           user_message=message)
                await self.channel.send_to((message_id, agent_id, result))

            elif action == ActionType.REFRESH:
                result = await self.refresh(agent_id=agent_id)
                await self.channel.send_to((message_id, agent_id, result))

            elif action == ActionType.CREATE_TWEET:
                result = await self.create_tweet(agent_id=agent_id,
                                                 content=message)
                await self.channel.send_to((message_id, agent_id, result))

            elif action == ActionType.LIKE:
                result = await self.like(agent_id=agent_id, tweet_id=message)
                await self.channel.send_to((message_id, agent_id, result))

            elif action == ActionType.UNLIKE:
                result = await self.unlike(agent_id=agent_id, tweet_id=message)
                await self.channel.send_to((message_id, agent_id, result))

            elif action == ActionType.SEARCH_TWEET:
                result = await self.search_tweets(agent_id=agent_id,
                                                  query=message)
                await self.channel.send_to((message_id, agent_id, result))

            elif action == ActionType.SEARCH_USER:
                result = await self.search_user(agent_id=agent_id,
                                                query=message)
                await self.channel.send_to((message_id, agent_id, result))

            elif action == ActionType.FOLLOW:
                result = await self.follow(agent_id=agent_id,
                                           followee_id=message)
                await self.channel.send_to((message_id, agent_id, result))

            elif action == ActionType.UNFOLLOW:
                result = await self.unfollow(agent_id=agent_id,
                                             followee_id=message)
                await self.channel.send_to((message_id, agent_id, result))

            elif action == ActionType.MUTE:
                result = await self.mute(agent_id=agent_id, mutee_id=message)
                await self.channel.send_to((message_id, agent_id, result))

            elif action == ActionType.UNMUTE:
                result = await self.unmute(agent_id=agent_id, mutee_id=message)
                await self.channel.send_to((message_id, agent_id, result))

            elif action == ActionType.TREND:
                result = await self.trend(agent_id=agent_id)
                await self.channel.send_to((message_id, agent_id, result))

            elif action == ActionType.RETWEET:
                result = await self.retweet(agent_id=agent_id, tweet_id=message)
                await self.channel.send_to((message_id, agent_id, result))

            else:
                raise ValueError(f"Action {action} is not supported")

    def _check_agent_userid(self, agent_id):
        try:
            user_query = ("SELECT user_id FROM user WHERE agent_id = ?")
            # Assuming execute_db_query_async returns a list of query results
            results = self._execute_db_command(user_query, (agent_id, ))
            # Fetch the first row of the query result
            first_row = results.fetchone()
            if first_row:
                user_id = first_row[0]
                return user_id
            else:
                return None
        except Exception as e:
            # Log or handle the error as appropriate
            print(f"Error querying user_id for agent_id {agent_id}: {e}")
            return None

    # 注册
    async def signup(self, agent_id, user_message):
        # 允许重名，user_id是主键
        user_name, name, bio = user_message
        current_time = datetime.now()
        try:
            if self._check_agent_userid(agent_id):
                user_id = self._check_agent_userid(agent_id)
                return {
                    "success":
                    False,
                    "error":
                    (f"Agent {agent_id} have already signed up with user "
                     f"id: {user_id}")
                }
            # 插入用户记录
            user_insert_query = (
                "INSERT INTO user (agent_id, user_name, name, bio, created_at,"
                " num_followings, num_followers) VALUES (?, ?, ?, ?, ?, ?, ?)")
            self._execute_db_command(
                user_insert_query,
                (agent_id, user_name, name, bio, current_time, 0, 0),
                commit=True)
            user_id = self.db_cursor.lastrowid
            # 准备trace记录的信息
            action_info = {"name": name, "user_name": user_name, "bio": bio}
            trace_insert_query = (
                "INSERT INTO trace (user_id, created_at, action, info) "
                "VALUES (?, ?, ?, ?)")
            self._execute_db_command(
                trace_insert_query,
                (user_id, current_time, ActionType.SIGNUP.value,
                 str(action_info)),
                commit=True)

            return {"success": True, "user_id": user_id}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def refresh(self, agent_id: int):
        # output不变，执行内容是从rec table取特定id的tweet
        current_time = datetime.now()
        try:
            user_id = self._check_agent_userid(agent_id)
            if not user_id:
                return self._not_signup_error_message(agent_id)

            # 从rec表中获取指定user_id的所有tweet_id
            rec_query = "SELECT tweet_id FROM rec WHERE user_id = ?"
            self._execute_db_command(rec_query, (user_id, ))
            rec_results = self.db_cursor.fetchall()

            tweet_ids = [row[0] for row in rec_results]
            selected_tweet_ids = tweet_ids

            # 如果tweet_id数量 >= self.refresh_tweet_count，则随机选择指定数量的tweet_id
            if len(tweet_ids) >= self.refresh_tweet_count:
                selected_tweet_ids = random.sample(tweet_ids,
                                                   self.refresh_tweet_count)

            # 根据选定的tweet_id从tweet表中获取tweet详情
            placeholders = ', '.join('?' for _ in selected_tweet_ids)
            # 构造SQL查询字符串
            tweet_query = (
                f"SELECT tweet_id, user_id, content, created_at, num_likes "
                f"FROM tweet WHERE tweet_id IN ({placeholders})")
            self._execute_db_command(tweet_query, selected_tweet_ids)
            results = self.db_cursor.fetchall()
            if not results:
                return {"success": False, "message": "No tweets found."}

            # 记录操作到trace表
            action_info = {}
            trace_insert_query = (
                "INSERT INTO trace (user_id, created_at, action, info) "
                "VALUES (?, ?, ?, ?)")
            self._execute_db_command(
                trace_insert_query,
                (user_id, current_time, ActionType.REFRESH.value,
                 str(action_info)),
                commit=True)

            # 将结果的每个元组转换为字典
            tweets = [{
                "tweet_id": tweet_id,
                "user_id": user_id,
                "content": content,
                "created_at": created_at,
                "num_likes": num_likes
            } for tweet_id, user_id, content, created_at, num_likes in results]
            return {"success": True, "tweets": tweets}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def update_rec_table(self):
        # Recsys(trace/user/tweet table), 结果是刷新了rec table
        user_table = fetch_table_from_db(self.db_cursor, 'user')
        tweet_table = fetch_table_from_db(self.db_cursor, 'tweet')
        trace_table = fetch_table_from_db(self.db_cursor, 'trace')
        rec_matrix = fetch_rec_table_as_matrix(self.db_cursor)

        new_rec_matrix = rec_sys_random(user_table, tweet_table, trace_table,
                                        rec_matrix, self.max_rec_tweet_len)
        # 构建SQL语句以删除rec表中的所有记录
        sql_query = "DELETE FROM rec"
        # 使用封装好的_execute_db_command函数执行SQL语句
        self._execute_db_command(sql_query, commit=True)
        for user_id in range(1, len(new_rec_matrix)):
            for tweet_id in new_rec_matrix[user_id]:
                sql_query = (
                    "INSERT INTO rec (user_id, tweet_id) VALUES (?, ?)")
                self._execute_db_command(sql_query, (user_id, tweet_id),
                                         commit=True)

    async def create_tweet(self, agent_id: int, content: str):
        current_time = datetime.now()
        try:
            user_id = self._check_agent_userid(agent_id)
            if not user_id:
                return self._not_signup_error_message(agent_id)

            # 插入推文记录
            tweet_insert_query = (
                "INSERT INTO tweet (user_id, content, created_at, num_likes) "
                "VALUES (?, ?, ?, ?)")
            self._execute_db_command(tweet_insert_query,
                                     (user_id, content, current_time, 0),
                                     commit=True)
            tweet_id = self.db_cursor.lastrowid
            # 准备trace记录的信息
            action_info = {"content": content, "tweet_id": tweet_id}
            trace_insert_query = (
                "INSERT INTO trace (user_id, created_at, action, info) "
                "VALUES (?, ?, ?, ?)")
            self._execute_db_command(
                trace_insert_query,
                (user_id, current_time, ActionType.CREATE_TWEET.value,
                 str(action_info)), commit=True)
            return {"success": True, "tweet_id": tweet_id}

        except Exception as e:
            return {"success": False, "error": str(e)}
        
    async def retweet(self, agent_id: int, tweet_id:int):
        current_time = datetime.now()
        try:
            user_id = self._check_agent_userid(agent_id)
            if not user_id:
                return self._not_signup_error_message(agent_id)
            
            # 查询要转发的推特内容
            sql_query = (
                "SELECT tweet_id, user_id, content, created_at, num_likes "
                "FROM tweet "
                "WHERE tweet_id = ? ")
            # 执行数据库查询
            self._execute_db_command(
                sql_query,
                (tweet_id,),
                commit=True)
            results = self.db_cursor.fetchall()
            if not results:
                return {"success": False, "error": "Tweet not found."}
            
            orig_content = results[0][2]
            orig_like = results[0][-1]
            orig_user_id = results[0][1]

            retweet_content = f"user{agent_id+1} retweet from user{str(orig_user_id)}. origianl_tweet: {orig_content}" # 转发的推特标识一下是从哪个user转的，方便判断
            
            # 确保此前未转发过
            retweet_check_query = (
                "SELECT * FROM 'tweet' WHERE content LIKE ? ")
            self._execute_db_command(retweet_check_query, (retweet_content,))
            if self.db_cursor.fetchone():
                # 已存在转发记录
                return {
                    "success": False,
                    "error": "Retweet record already exists."
                }
            
            # 插入转推推文记录
            tweet_insert_query = (
                "INSERT INTO tweet (user_id, content, created_at, num_likes) "
                "VALUES (?, ?, ?, ?)")

            self._execute_db_command(tweet_insert_query,
                                     (user_id, retweet_content, current_time, orig_like),
                                     commit=True)
            
            tweet_id = self.db_cursor.lastrowid
            # 准备trace记录的信息
            action_info = {"content": retweet_content, "tweet_id": tweet_id}
            trace_insert_query = (
                "INSERT INTO trace (user_id, created_at, action, info) "
                "VALUES (?, ?, ?, ?)")
            self._execute_db_command(
                trace_insert_query,
                (user_id, current_time, ActionType.RETWEET.value,
                 str(action_info)), commit=True)
            
            return {"success": True, "tweet_id": tweet_id}

        except Exception as e:
            return {"success": False, "error": str(e)}
        
    async def retweet(self, agent_id: int, tweet_id:int):
        current_time = datetime.now()
        try:
            user_id = self._check_agent_userid(agent_id)
            if not user_id:
                return self._not_signup_error_message(agent_id)
            
            # 查询要转发的推特内容
            sql_query = (
                "SELECT tweet_id, user_id, content, created_at, num_likes "
                "FROM tweet "
                "WHERE tweet_id = ? ")
            # 执行数据库查询
            self._execute_db_command(
                sql_query,
                (tweet_id,),
                commit=True)
            results = self.db_cursor.fetchall()
            if not results:
                return {"success": False, "error": "Tweet not found."}
            
            orig_content = results[0][2]
            orig_like = results[0][-1]
            orig_user_id = results[0][1]

            retweet_content = f"user{agent_id+1} retweet from user{str(orig_user_id)}. origianl_tweet: {orig_content}" # 转发的推特标识一下是从哪个user转的，方便判断
            
            # 确保此前未转发过
            retweet_check_query = (
                "SELECT * FROM 'tweet' WHERE content LIKE ? ")
            self._execute_db_command(retweet_check_query, (retweet_content,))
            if self.db_cursor.fetchone():
                # 已存在转发记录
                return {
                    "success": False,
                    "error": "Retweet record already exists."
                }
            
            # 插入转推推文记录
            tweet_insert_query = (
                "INSERT INTO tweet (user_id, content, created_at, num_likes) "
                "VALUES (?, ?, ?, ?)")

            self._execute_db_command(tweet_insert_query,
                                     (user_id, retweet_content, current_time, orig_like),
                                     commit=True)
            
            tweet_id = self.db_cursor.lastrowid
            # 准备trace记录的信息
            action_info = {"content": retweet_content, "tweet_id": tweet_id}
            trace_insert_query = (
                "INSERT INTO trace (user_id, created_at, action, info) "
                "VALUES (?, ?, ?, ?)")
            self._execute_db_command(
                trace_insert_query,
                (user_id, current_time, ActionType.RETWEET.value,
                 str(action_info)), commit=True)
            
            return {"success": True, "tweet_id": tweet_id}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def like(self, agent_id: int, tweet_id: int):
        current_time = datetime.now()
        try:
            user_id = self._check_agent_userid(agent_id)
            if not user_id:
                return self._not_signup_error_message(agent_id)
            # 检查是否已经存在点赞记录
            like_check_query = (
                "SELECT * FROM 'like' WHERE tweet_id = ? AND user_id = ?")
            self._execute_db_command(like_check_query, (tweet_id, user_id))
            if self.db_cursor.fetchone():
                # 已存在点赞记录
                return {
                    "success": False,
                    "error": "Like record already exists."
                }

            # 更新tweet表中的点赞数
            tweet_update_query = (
                "UPDATE tweet SET num_likes = num_likes + 1 WHERE tweet_id = ?"
            )
            self._execute_db_command(tweet_update_query, (tweet_id, ),
                                     commit=True)

            # 在like表中添加记录
            like_insert_query = (
                "INSERT INTO 'like' (tweet_id, user_id, created_at) "
                "VALUES (?, ?, ?)")
            self._execute_db_command(like_insert_query,
                                     (tweet_id, user_id, current_time),
                                     commit=True)
            like_id = self.db_cursor.lastrowid  # 获取刚刚插入的点赞记录的ID

            # 记录操作到trace表
            action_info = {"tweet_id": tweet_id, "like_id": like_id}
            trace_insert_query = (
                "INSERT INTO trace (user_id, created_at, action, info) "
                "VALUES (?, ?, ?, ?)")
            self._execute_db_command(trace_insert_query,
                                     (user_id, current_time,
                                      ActionType.LIKE.value, str(action_info)),
                                     commit=True)
            return {"success": True, "like_id": like_id}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def unlike(self, agent_id: int, tweet_id: int):
        current_time = datetime.now()
        try:
            user_id = self._check_agent_userid(agent_id)
            if not user_id:
                return self._not_signup_error_message(agent_id)

            # 检查是否已经存在点赞记录
            like_check_query = (
                "SELECT * FROM 'like' WHERE tweet_id = ? AND user_id = ?")
            self._execute_db_command(like_check_query, (tweet_id, user_id))
            result = self.db_cursor.fetchone()

            if not result:
                # 没有存在点赞记录
                return {
                    "success": False,
                    "error": "Like record does not exist."
                }

            # Get the `like_id`
            like_id, _, _, _ = result

            # 更新tweet表中的点赞数
            tweet_update_query = (
                "UPDATE tweet SET num_likes = num_likes - 1 WHERE tweet_id = ?"
            )
            self._execute_db_command(
                tweet_update_query,
                (tweet_id, ),
                commit=True,
            )

            # 在like表中删除记录
            like_delete_query = ("DELETE FROM 'like' WHERE like_id = ?")
            self._execute_db_command(
                like_delete_query,
                (like_id, ),
                commit=True,
            )

            # 记录操作到trace表
            action_info = {"tweet_id": tweet_id, "like_id": like_id}
            trace_insert_query = (
                "INSERT INTO trace (user_id, created_at, action, info) "
                "VALUES (?, ?, ?, ?)")
            self._execute_db_command(
                trace_insert_query,
                (user_id, current_time, ActionType.UNLIKE.value,
                 str(action_info)),
                commit=True)
            return {"success": True, "like_id": like_id}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def search_tweets(self, agent_id: int, query: str):
        current_time = datetime.now()
        try:
            user_id = self._check_agent_userid(agent_id)
            if not user_id:
                return self._not_signup_error_message(agent_id)
            # 更新SQL查询，以便同时根据content、tweet_id和user_id进行搜索
            # 注意：CAST是必要的，因为tweet_id和user_id是整数类型，而搜索的query是字符串类型
            sql_query = (
                "SELECT tweet_id, user_id, content, created_at, num_likes "
                "FROM tweet "
                "WHERE content LIKE ? OR CAST(tweet_id AS TEXT) LIKE ? OR "
                "CAST(user_id AS TEXT) LIKE ?")
            # 执行数据库查询
            self._execute_db_command(
                sql_query,
                ('%' + query + '%', '%' + query + '%', '%' + query + '%'),
                commit=True)
            results = self.db_cursor.fetchall()

            # 记录操作到trace表
            action_info = {"query": query}
            trace_insert_query = (
                "INSERT INTO trace (user_id, created_at, action, info) "
                "VALUES (?, ?, ?, ?)")
            self._execute_db_command(
                trace_insert_query,
                (0, current_time, "search_tweets",
                 str(action_info)),  # 假设user_id为0表示系统操作或未指定用户
                commit=True)
            # 记录操作到trace表
            action_info = {"query": query}
            trace_insert_query = (
                "INSERT INTO trace (user_id, created_at, action, info) "
                "VALUES (?, ?, ?, ?)")
            self._execute_db_command(
                trace_insert_query,
                (user_id, current_time, ActionType.SEARCH_TWEET.value,
                 str(action_info)),
                commit=True)

            # 如果没有找到结果，返回一个指示失败的字典
            if not results:
                return {
                    "success": False,
                    "message": "No tweets found matching the query."
                }

            # 将结果的每个元组转换为字典
            tweets = [{
                "tweet_id": tweet_id,
                "user_id": user_id,
                "content": content,
                "created_at": created_at,
                "num_likes": num_likes
            } for tweet_id, user_id, content, created_at, num_likes in results]

            return {"success": True, "tweets": tweets}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def search_user(self, agent_id: int, query: str):
        try:
            user_id = self._check_agent_userid(agent_id)
            if not user_id:
                return self._not_signup_error_message(agent_id)
            current_time = datetime.now()
            sql_query = (
                "SELECT user_id, user_name, name, bio, created_at, "
                "num_followings, num_followers "
                "FROM user "
                "WHERE user_name LIKE ? OR name LIKE ? OR bio LIKE ? OR "
                "CAST(user_id AS TEXT) LIKE ?")
            # 改写为使用 execute_db_command 方法
            self._execute_db_command(sql_query,
                                     ('%' + query + '%', '%' + query + '%',
                                      '%' + query + '%', '%' + query + '%'),
                                     commit=True)
            results = self.db_cursor.fetchall()

            # 记录操作到trace表
            action_info = {"query": query}
            trace_insert_query = (
                "INSERT INTO trace (user_id, created_at, action, info) "
                "VALUES (?, ?, ?, ?)")
            self._execute_db_command(
                trace_insert_query,
                (user_id, current_time, ActionType.SEARCH_USER.value,
                 str(action_info)),
                commit=True)

            # If no results found, return a dict with 'success' key as False:
            if not results:
                return {
                    "success": False,
                    "message": "No users found matching the query."
                }

            # Convert each tuple in results to a dictionary:
            users = [{
                "user_id": user_id,
                "user_name": user_name,
                "name": name,
                "bio": bio,
                "created_at": created_at,
                "num_followings": num_followings,
                "num_followers": num_followers
            } for user_id, user_name, name, bio, created_at, num_followings,
                     num_followers in results]
            return {"success": True, "users": users}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def follow(self, agent_id: int, followee_id: int):
        current_time = datetime.now()
        try:
            user_id = self._check_agent_userid(agent_id)
            if not user_id:
                return self._not_signup_error_message(agent_id)
            # 检查是否已经存在关注记录
            follow_check_query = ("SELECT * FROM follow WHERE follower_id = ? "
                                  "AND followee_id = ?")
            self._execute_db_command(follow_check_query,
                                     (user_id, followee_id))
            if self.db_cursor.fetchone():
                # 已存在关注记录
                return {
                    "success": False,
                    "error": "Follow record already exists."
                }

            # 在follow表中添加记录
            follow_insert_query = (
                "INSERT INTO follow (follower_id, followee_id, created_at) "
                "VALUES (?, ?, ?)")
            self._execute_db_command(follow_insert_query,
                                     (user_id, followee_id, current_time),
                                     commit=True)
            follow_id = self.db_cursor.lastrowid  # 获取刚刚插入的关注记录的ID

            # 更新user表中的following字段
            user_update_query1 = (
                "UPDATE user SET num_followings = num_followings + 1 "
                "WHERE user_id = ?")
            self._execute_db_command(user_update_query1, (user_id, ),
                                     commit=True)

            # 更新user表中的follower字段
            user_update_query2 = (
                "UPDATE user SET num_followers = num_followers + 1 "
                "WHERE user_id = ?")
            self._execute_db_command(user_update_query2, (followee_id, ),
                                     commit=True)

            # 记录操作到trace表
            action_info = {"follow_id": follow_id}
            trace_insert_query = (
                "INSERT INTO trace (user_id, created_at, action, info) "
                "VALUES (?, ?, ?, ?)")
            self._execute_db_command(
                trace_insert_query,
                (user_id, current_time, ActionType.FOLLOW.value,
                 str(action_info)),
                commit=True)
            return {"success": True, "follow_id": follow_id}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def unfollow(self, agent_id: int, followee_id: int):
        current_time = datetime.now()
        try:
            user_id = self._check_agent_userid(agent_id)
            if not user_id:
                return self._not_signup_error_message(agent_id)
            # 检查是否存在关注记录，并获取其ID
            follow_check_query = (
                "SELECT follow_id FROM follow WHERE follower_id = ? AND "
                "followee_id = ?")
            self._execute_db_command(follow_check_query,
                                     (user_id, followee_id))
            follow_record = self.db_cursor.fetchone()
            if not follow_record:
                return {
                    "success": False,
                    "error": "Follow record does not exist."
                }
            follow_id = follow_record[0]  # 假设ID位于查询结果的第一列

            # 在follow表中删除记录
            follow_delete_query = "DELETE FROM follow WHERE follow_id = ?"
            self._execute_db_command(follow_delete_query, (follow_id, ),
                                     commit=True)

            # 更新user表中的following字段
            user_update_query1 = (
                "UPDATE user SET num_followings = num_followings - 1 "
                "WHERE user_id = ?")
            self._execute_db_command(user_update_query1, (user_id, ),
                                     commit=True)

            # 更新user表中的follower字段
            user_update_query2 = (
                "UPDATE user SET num_followers = num_followers - 1 "
                "WHERE user_id = ?")
            self._execute_db_command(user_update_query2, (followee_id, ),
                                     commit=True)

            # 记录操作到trace表
            action_info = {"followee_id": followee_id}
            trace_insert_query = (
                "INSERT INTO trace (user_id, created_at, action, info) "
                "VALUES (?, ?, ?, ?)")
            self._execute_db_command(
                trace_insert_query,
                (user_id, current_time, ActionType.UNFOLLOW.value,
                 str(action_info)),
                commit=True)
            return {
                "success": True,
                "follow_id": follow_id  # 返回被删除的关注记录ID
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def mute(self, agent_id: int, mutee_id: int):
        current_time = datetime.now()
        try:
            user_id = self._check_agent_userid(agent_id)
            if not user_id:
                return self._not_signup_error_message(agent_id)
            # 检查是否已经存在禁言记录
            mute_check_query = ("SELECT * FROM mute WHERE muter_id = ? AND "
                                "mutee_id = ?")
            self._execute_db_command(mute_check_query, (user_id, mutee_id))
            if self.db_cursor.fetchone():
                # 已存在禁言记录
                return {
                    "success": False,
                    "error": "Mute record already exists."
                }
            # 在mute表中添加记录
            mute_insert_query = (
                "INSERT INTO mute (muter_id, mutee_id, created_at) "
                "VALUES (?, ?, ?)")
            self._execute_db_command(mute_insert_query,
                                     (user_id, mutee_id, current_time),
                                     commit=True)
            mute_id = self.db_cursor.lastrowid  # 获取刚刚插入的禁言记录的ID

            # 记录操作到trace表
            action_info = {"mutee_id": mutee_id}
            trace_insert_query = (
                "INSERT INTO trace (user_id, created_at, action, info) "
                "VALUES (?, ?, ?, ?)")
            self._execute_db_command(trace_insert_query,
                                     (user_id, current_time,
                                      ActionType.MUTE.value, str(action_info)),
                                     commit=True)
            return {"success": True, "mute_id": mute_id}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def unmute(self, agent_id: int, mutee_id: int):
        current_time = datetime.now()
        try:
            user_id = self._check_agent_userid(agent_id)
            if not user_id:
                return self._not_signup_error_message(agent_id)
            # 检查是否存在指定的禁言记录，并获取mute_id
            mute_check_query = (
                "SELECT mute_id FROM mute WHERE muter_id = ? AND mutee_id = ?")
            self._execute_db_command(mute_check_query, (user_id, mutee_id))
            mute_record = self.db_cursor.fetchone()
            if not mute_record:
                # 如果不存在禁言记录
                return {"success": False, "error": "No mute record exists."}
            mute_id = mute_record[0]

            # 从mute表中删除指定的禁言记录
            mute_delete_query = ("DELETE FROM mute WHERE mute_id = ?")
            self._execute_db_command(mute_delete_query, (mute_id, ),
                                     commit=True)

            # 记录解除禁言操作到trace表
            action_info = {"mutee_id": mutee_id}
            trace_insert_query = (
                "INSERT INTO trace (user_id, created_at, action, info) "
                "VALUES (?, ?, ?, ?)")
            self._execute_db_command(
                trace_insert_query,
                (user_id, current_time, ActionType.UNMUTE.value,
                 str(action_info)),
                commit=True)
            return {"success": True, "mute_id": mute_id}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def trend(self, agent_id: int):
        """
        Get the top K trending tweets in the last num_days days.
        """
        try:
            user_id = self._check_agent_userid(agent_id)
            if not user_id:
                return self._not_signup_error_message(agent_id)
            current_time = datetime.now()
            # 计算搜索的起始时间
            start_time = current_time - timedelta(days=self.trend_num_days)

            # 构建SQL查询语句
            sql_query = """
                SELECT user_id, tweet_id, content, created_at, num_likes
                FROM tweet
                WHERE created_at >= ?
                ORDER BY num_likes DESC
                LIMIT ?
            """
            # 执行数据库查询
            self._execute_db_command(sql_query, (start_time, self.trend_top_k),
                                     commit=True)
            results = self.db_cursor.fetchall()

            # 如果没有找到结果，返回一个指示失败的字典
            if not results:
                return {
                    "success": False,
                    "message": "No trending tweets in the specified period."
                }
            # 将结果的每个元组转换为字典
            tweets = [{
                "tweet_id": tweet_id,
                "user_id": user_id,
                "content": content,
                "created_at": created_at,
                "num_likes": num_likes
            } for tweet_id, user_id, content, created_at, num_likes in results]

            trace_insert_query = """
                INSERT INTO trace (user_id, created_at, action, info)
                VALUES (?, ?, ?, ?)
            """
            self._execute_db_command(trace_insert_query,
                                     (user_id, current_time, "trend", None),
                                     commit=True)
            return {"success": True, "tweets": tweets}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # 同步函数，不在running部分使用，直接被generator模块调用
    def generate_agents(self, agents_info_list: list):
        r"""Generate the agents using :obj:`agents_info`.

        Args:
            agents_info_list (List[Dict]): list of agent info
        """
        # TODO assert the database is running
        for index, item in enumerate(agents_info_list):
            # Insert agents info to USER database
            self.db_cursor.execute(
                ("INSERT INTO user (user_name, name, bio, created_at, "
                 "num_followings, num_followers) VALUES (?, ?, ?, ?, ?, ?)"),
                (item["user_name"], item["name"], item["bio"],
                 item["created_at"], item["num_followings"],
                 item["num_followers"]))
            self.db.commit()

    # 同步函数，不在running部分使用，直接被generator模块调用
    def update_tweets(self, tweet_info_dict: dict):
        r"""Update tweets into the database.

        Args:
            tweet_info_dict (dict): Dictionary of tweet information.
        """
        self.db_cursor.execute(
            ("INSERT INTO tweet (tweet_id, user_id, content, created_at, "
             "num_likes) VALUES (?, ?, ?, ?, ?)"),
            (tweet_info_dict["tweet_id"], tweet_info_dict["user_id"],
             tweet_info_dict["content"], tweet_info_dict["created_at"],
             tweet_info_dict["num_likes"]))
        self.db.commit()
