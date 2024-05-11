# File: SandboxTwitterModule/infra/Agent/twitterUserAgent.py
import networkx as nx
import random
import asyncio
import logging


from SandboxTwitterModule.infra import Twitter


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

    async def perform_action(self, action_type, message):
        """根据传入的action_type和message执行action, 并得到返回值"""
        pass
        
