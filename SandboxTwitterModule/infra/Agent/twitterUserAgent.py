# File: SandboxTwitterModule/infra/Agent/twitterUserAgent.py
import networkx as nx
import random
import asyncio
import logging


from ..twitter_api import TwitterAPI


class TwitterUserAgent:
    def __init__(self, agent_id, real_name):
        self.db = None
        self.api = None
        self.user_id = None
        self.agent_id = agent_id
        self.real_name = real_name
        self.profile = {
            'nodes': [],  # Relationships with other agents
            'edges': [],  # Relationship details
            'other_info': {'mbti': None, 'activity_level': 'Medium'},
        }
        self.memory = {'system_one': [], 'system_two': []}
        self.home_content = []

    async def perform_random_action(self):
        """根据 agent 的个性和情境随机选择行为"""
        activities = ['tweet_random', 'follow_random_user', 'unfollow_random_user', 'choose_tweet_to_like']
        probabilities = [0.25, 0.25, 0.25, 0.25]  # 添加点赞行为的概率

        # 调整概率以反映代理的活跃度
        if self.profile['other_info']['activity_level'] == 'High':
            probabilities = [0.25, 0.25, 0.25, 0.25]  # 对于活跃用户，均等分配概率
        elif self.profile['other_info']['activity_level'] == 'Low':
            probabilities = [0.15, 0.25, 0.25, 0.35]  # 对于不太活跃的用户，增加点赞的概率，减少发推的概率

        chosen_activity = random.choices(activities, weights=probabilities, k=1)[0]
        await getattr(self, chosen_activity)()  # 调用选择的方法
