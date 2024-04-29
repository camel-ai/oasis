from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from typing import Any

from generator.create_database import get_db_path
from info_server.core.type import ActionType


class Twitter:

    def __init__(self):
        db_path = get_db_path()
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db_cursor = self.db.cursor()

        # twitter内部定义的热搜规则参数
        self.trend_num_days = 7
        self.trend_top_k = 10

    def execute_db_command(self, command, args=(), commit=False):
        self.db_cursor.execute(command, args)
        if commit:
            self.db.commit()
        return self.db_cursor

    async def running(self, channel: Any):
        while True:
            data = await channel.receive_from("Agent Group")
            user_id, message, action = data

            action = ActionType(action)

            if action == ActionType.EXIT:
                self.db_cursor.close()
                self.db.close()
                break

            elif action == 'refresh':
                result = await self.refresh(user_id=id, channel=channel)
                await channel.send_to("Agent Group", (id, result))

            elif action == ActionType.SEARCH_USER:
                result = await self.search_user(user_id=user_id, query=message)
                await channel.send_to("Agent Group", (user_id, result))

            elif action == ActionType.SEARCH_TWEET:
                result = await self.search_tweets(user_id=user_id,
                                                  query=message)
                await channel.send_to("Agent Group", (user_id, result))

            elif action == ActionType.CREATE_TWEET:
                result = await self.create_tweet(user_id=user_id,
                                                 content=message)
                await channel.send_to("Agent Group", (user_id, result))

            elif action == ActionType.LIKE:
                result = await self.like(user_id=user_id, tweet_id=message)
                await channel.send_to("Agent Group", (user_id, result))

            elif action == ActionType.UNLIKE:
                result = await self.unlike(user_id=user_id, tweet_id=message)
                await channel.send_to("Agent Group", (user_id, result))

            elif action == ActionType.FOLLOW:
                result = await self.follow(user_id=user_id,
                                           followee_id=message)
                await channel.send_to("Agent Group", (user_id, result))

            elif action == ActionType.UNFOLLOW:
                result = await self.unfollow(user_id=user_id,
                                             followee_id=message)
                await channel.send_to("Agent Group", (user_id, result))

            elif action == ActionType.MUTE:
                result = await self.mute(user_id=user_id, mutee_id=message)
                await channel.send_to("Agent Group", (user_id, result))

            elif action == ActionType.UNMUTE:
                result = await self.unmute(user_id=user_id, mutee_id=message)
                await channel.send_to("Agent Group", (user_id, result))

            elif action == ActionType.TREND:
                result = await self.trend(user_id=user_id)
                await channel.send_to("Agent Group", (user_id, result))

            else:
                raise ValueError(f"Action {action} is not supported")

    # 注：暂未测试refresh
    async def refresh(self, channel, user_id):
        try:
            await channel.send_to('RecSys', user_id)
            content = await channel.receive_from('RecSys', user_id=user_id)
            # 期望content格式：
            '''
            tweets = [{
                "tweet_id": tweet_id,
                "user_id": user_id,
                "content": content,
                "created_at": created_at,
                "num_likes": num_likes
            } for tweet_id, user_id, content, created_at, num_likes in results]
            '''
            current_time = datetime.now()
            trace_insert_query = """
                INSERT INTO trace (user_id, created_at, action, info)
                VALUES (?, ?, ?, ?)
            """

            # 调用execute_db_command函数执行数据库插入操作
            self.execute_db_command(
                trace_insert_query,
                args=(user_id, current_time, "refresh", None),
                commit=True
            )
            return {"success": True, "tweets": content}
        except Exception as e:
            print(f"Error refreshing: {e}")
            return {"success": False, "error": str(e)}

    async def create_tweet(self, user_id: int, content: str):
        current_time = datetime.now()
        print(current_time)
        try:
            # 插入推文记录
            tweet_insert_query = (
                "INSERT INTO tweet (user_id, content, created_at, num_likes) "
                "VALUES (?, ?, ?, ?)")
            self.execute_db_command(tweet_insert_query,
                                    (user_id, content, current_time, 0),
                                    commit=True)
            tweet_id = self.db_cursor.lastrowid
            # 准备trace记录的信息
            action_info = {"content": content, "tweet_id": tweet_id}
            trace_insert_query = (
                "INSERT INTO trace (user_id, created_at, action, info) "
                "VALUES (?, ?, ?, ?)")
            self.execute_db_command(
                trace_insert_query,
                (user_id, current_time, "create_tweet", str(action_info)),
                commit=True)
            return {"success": True, "tweet_id": tweet_id}

        except Exception as e:
            print(f"Error creating tweet: {e}")
            return {"success": False, "error": str(e)}

    async def search_user(self, user_id: int, query: str):
        try:
            current_time = datetime.now()
            sql_query = (
                "SELECT user_id, user_name, name, bio, created_at, "
                "num_followings, num_followers "
                "FROM user "
                "WHERE user_name LIKE ? OR name LIKE ? OR bio LIKE ? OR "
                "CAST(user_id AS TEXT) LIKE ?")
            # 改写为使用 execute_db_command 方法
            self.execute_db_command(sql_query,
                                    ('%' + query + '%', '%' + query + '%',
                                     '%' + query + '%', '%' + query + '%'),
                                    commit=True)
            results = self.db_cursor.fetchall()

            # 记录操作到trace表
            action_info = {"query": query}
            trace_insert_query = (
                "INSERT INTO trace (user_id, created_at, action, info) "
                "VALUES (?, ?, ?, ?)")
            self.execute_db_command(
                trace_insert_query,
                (user_id, current_time, "search_user", str(action_info)),
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

    async def search_tweets(self, user_id: int, query: str):
        try:
            current_time = datetime.now()
            # 更新SQL查询，以便同时根据content、tweet_id和user_id进行搜索
            # 注意：CAST是必要的，因为tweet_id和user_id是整数类型，而搜索的query是字符串类型
            sql_query = (
                "SELECT tweet_id, user_id, content, created_at, num_likes "
                "FROM tweet "
                "WHERE content LIKE ? OR CAST(tweet_id AS TEXT) LIKE ? OR "
                "CAST(user_id AS TEXT) LIKE ?")
            # 执行数据库查询
            self.execute_db_command(
                sql_query,
                ('%' + query + '%', '%' + query + '%', '%' + query + '%'),
                commit=True)
            results = self.db_cursor.fetchall()

            # 记录操作到trace表
            action_info = {"query": query}
            trace_insert_query = (
                "INSERT INTO trace (user_id, created_at, action, info) "
                "VALUES (?, ?, ?, ?)")
            self.execute_db_command(
                trace_insert_query,
                (0, current_time, "search_tweets",
                 str(action_info)),  # 假设user_id为0表示系统操作或未指定用户
                commit=True)
            # 记录操作到trace表
            action_info = {"query": query}
            trace_insert_query = (
                "INSERT INTO trace (user_id, created_at, action, info) "
                "VALUES (?, ?, ?, ?)")
            self.execute_db_command(
                trace_insert_query,
                (user_id, current_time, "search_tweets", str(action_info)),
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

    async def like(self, user_id: int, tweet_id: int):
        current_time = datetime.now()
        try:
            # 检查是否已经存在点赞记录
            like_check_query = (
                "SELECT * FROM 'like' WHERE tweet_id = ? AND user_id = ?")
            self.execute_db_command(like_check_query, (tweet_id, user_id))
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
            self.execute_db_command(tweet_update_query, (tweet_id, ),
                                    commit=True)

            # 在like表中添加记录
            like_insert_query = (
                "INSERT INTO 'like' (tweet_id, user_id, created_at) "
                "VALUES (?, ?, ?)")
            self.execute_db_command(like_insert_query,
                                    (tweet_id, user_id, current_time),
                                    commit=True)
            like_id = self.db_cursor.lastrowid  # 获取刚刚插入的点赞记录的ID

            # 记录操作到trace表
            action_info = {"tweet_id": tweet_id, "like_id": like_id}
            trace_insert_query = (
                "INSERT INTO trace (user_id, created_at, action, info) "
                "VALUES (?, ?, ?, ?)")
            self.execute_db_command(
                trace_insert_query,
                (user_id, current_time, "like", str(action_info)),
                commit=True)
            return {"success": True, "like_id": like_id}
        except Exception as e:
            print(f"Error liking tweet: {e}")
            return {"success": False, "error": str(e)}

    async def unlike(self, user_id: int, tweet_id: int):
        current_time = datetime.now()
        try:
            # 检查是否已经存在点赞记录
            like_check_query = (
                "SELECT * FROM 'like' WHERE tweet_id = ? AND user_id = ?")
            self.execute_db_command(like_check_query, (tweet_id, user_id))
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
            self.execute_db_command(
                tweet_update_query,
                (tweet_id, ),
                commit=True,
            )

            # 在like表中删除记录
            like_delete_query = ("DELETE FROM 'like' WHERE like_id = ?")
            self.execute_db_command(
                like_delete_query,
                (like_id, ),
                commit=True,
            )

            # 记录操作到trace表
            action_info = {"tweet_id": tweet_id, "like_id": like_id}
            trace_insert_query = (
                "INSERT INTO trace (user_id, created_at, action, info) "
                "VALUES (?, ?, ?, ?)")
            self.execute_db_command(
                trace_insert_query,
                (user_id, current_time, "unlike", str(action_info)),
                commit=True)
            return {"success": True, "like_id": like_id}
        except Exception as e:
            print(f"Error liking tweet: {e}")
            return {"success": False, "error": str(e)}

    async def follow(self, user_id: int, followee_id: int):
        current_time = datetime.now()
        try:
            # 检查是否已经存在关注记录
            follow_check_query = ("SELECT * FROM follow WHERE follower_id = ? "
                                  "AND followee_id = ?")
            self.execute_db_command(follow_check_query, (user_id, followee_id))
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
            self.execute_db_command(follow_insert_query,
                                    (user_id, followee_id, current_time),
                                    commit=True)
            follow_id = self.db_cursor.lastrowid  # 获取刚刚插入的关注记录的ID

            # 更新user表中的following字段
            user_update_query1 = (
                "UPDATE user SET num_followings = num_followings + 1 "
                "WHERE user_id = ?")
            self.execute_db_command(user_update_query1, (user_id, ),
                                    commit=True)

            # 更新user表中的follower字段
            user_update_query2 = (
                "UPDATE user SET num_followers = num_followers + 1 "
                "WHERE user_id = ?")
            self.execute_db_command(user_update_query2, (followee_id, ),
                                    commit=True)

            # 记录操作到trace表
            action_info = {"follow_id": follow_id}
            trace_insert_query = (
                "INSERT INTO trace (user_id, created_at, action, info) "
                "VALUES (?, ?, ?, ?)")
            self.execute_db_command(
                trace_insert_query,
                (user_id, current_time, "follow", str(action_info)),
                commit=True)
            return {"success": True, "follow_id": follow_id}
        except Exception as e:
            print(f"Error following user: {e}")
            return {"success": False, "error": str(e)}

    async def unfollow(self, user_id: int, followee_id: int):
        current_time = datetime.now()
        try:
            # 检查是否存在关注记录，并获取其ID
            follow_check_query = (
                "SELECT follow_id FROM follow WHERE follower_id = ? AND "
                "followee_id = ?"
            )
            self.execute_db_command(
                follow_check_query, (user_id, followee_id)
            )
            follow_record = self.db_cursor.fetchone()
            if not follow_record:
                return {
                    "success": False,
                    "error": "Follow record does not exist."
                }
            follow_id = follow_record[0]  # 假设ID位于查询结果的第一列

            # 在follow表中删除记录
            follow_delete_query = "DELETE FROM follow WHERE follow_id = ?"
            self.execute_db_command(follow_delete_query, (follow_id,),
                                    commit=True)

            # 更新user表中的following字段
            user_update_query1 = (
                "UPDATE user SET num_followings = num_followings - 1 "
                "WHERE user_id = ?"
            )
            self.execute_db_command(user_update_query1, (user_id,),
                                    commit=True)

            # 更新user表中的follower字段
            user_update_query2 = (
                "UPDATE user SET num_followers = num_followers - 1 "
                "WHERE user_id = ?"
            )
            self.execute_db_command(user_update_query2, (followee_id,),
                                    commit=True)

            # 记录操作到trace表
            action_info = {"followee_id": followee_id}
            trace_insert_query = (
                "INSERT INTO trace (user_id, created_at, action, info) "
                "VALUES (?, ?, ?, ?)"
            )
            self.execute_db_command(
                trace_insert_query,
                (user_id, current_time, "unfollow", str(action_info)),
                commit=True)
            return {
                "success": True,
                "follow_id": follow_id  # 返回被删除的关注记录ID
            }
        except Exception as e:
            print(f"Error unfollowing user: {e}")
            return {"success": False, "error": str(e)}

    async def mute(self, user_id: int, mutee_id: int):
        current_time = datetime.now()
        try:
            # 检查是否已经存在禁言记录
            mute_check_query = ("SELECT * FROM mute WHERE muter_id = ? AND "
                                "mutee_id = ?")
            self.execute_db_command(mute_check_query, (user_id, mutee_id))
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
            self.execute_db_command(mute_insert_query,
                                    (user_id, mutee_id, current_time),
                                    commit=True)
            mute_id = self.db_cursor.lastrowid  # 获取刚刚插入的禁言记录的ID

            # 记录操作到trace表
            action_info = {"mutee_id": mutee_id}
            trace_insert_query = (
                "INSERT INTO trace (user_id, created_at, action, info) "
                "VALUES (?, ?, ?, ?)")
            self.execute_db_command(
                trace_insert_query,
                (user_id, current_time, "mute", str(action_info)),
                commit=True)
            return {"success": True, "mute_id": mute_id}
        except Exception as e:
            print(f"Error muting user: {e}")
            return {"success": False, "error": str(e)}

    async def unmute(self, user_id: int, mutee_id: int):
        current_time = datetime.now()
        try:
            # 检查是否存在指定的禁言记录，并获取mute_id
            mute_check_query = (
                "SELECT mute_id FROM mute WHERE muter_id = ? AND mutee_id = ?"
            )
            self.execute_db_command(mute_check_query, (user_id, mutee_id))
            mute_record = self.db_cursor.fetchone()
            if not mute_record:
                # 如果不存在禁言记录
                return {
                    "success": False,
                    "error": "No mute record exists."
                }
            mute_id = mute_record[0]

            # 从mute表中删除指定的禁言记录
            mute_delete_query = (
                "DELETE FROM mute WHERE mute_id = ?"
            )
            self.execute_db_command(mute_delete_query, (mute_id,), commit=True)

            # 记录解除禁言操作到trace表
            action_info = {"mutee_id": mutee_id}
            trace_insert_query = (
                "INSERT INTO trace (user_id, created_at, action, info) "
                "VALUES (?, ?, ?, ?)"
            )
            self.execute_db_command(
                trace_insert_query,
                (user_id, current_time, "unmute", str(action_info)),
                commit=True)
            return {
                "success": True,
                "mute_id": mute_id
            }
        except Exception as e:
            print(f"Error unmuting user: {e}")
            return {"success": False, "error": str(e)}

    async def trend(self, user_id: int):
        """
        Get the top K trending tweets in the last num_days days.
        """
        try:
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
            self.execute_db_command(
                sql_query,
                (start_time, self.trend_top_k),
                commit=True
            )
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
            self.execute_db_command(
                trace_insert_query,
                (user_id, current_time, "trend", None),
                commit=True
            )
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
