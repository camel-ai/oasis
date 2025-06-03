# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
import asyncio
import json
import os
import os.path as osp
import sqlite3
import tempfile

import pytest
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

import oasis
from oasis import ActionType, ManualAction, generate_twitter_agent_graph
from oasis.social_agent.agent import SocialAgent
from oasis.social_platform.channel import Channel
from oasis.social_platform.config import UserInfo
from oasis.social_platform.platform import Platform

parent_folder = osp.dirname(osp.abspath(__file__))
test_db_filepath = osp.join(parent_folder, "test_interview.db")


@pytest.fixture
def setup_interview_test():
    """Setup fixture for interview tests."""
    if os.path.exists(test_db_filepath):
        os.remove(test_db_filepath)
    yield
    # Cleanup after test
    if os.path.exists(test_db_filepath):
        os.remove(test_db_filepath)


@pytest.mark.asyncio
async def test_single_interview_action(setup_interview_test):
    """Test conducting a single interview with an agent."""
    agents = []
    channel = Channel()
    infra = Platform(test_db_filepath, channel)
    task = asyncio.create_task(infra.running())

    try:
        # Create and sign up a test agent
        real_name = "TestAgent"
        description = "A test agent for interview testing."
        profile = {
            "nodes": [],
            "edges": [],
            "other_info": {
                "user_profile":
                "I am a test agent with strong opinions on technology.",
                "mbti": "INTJ",
                "activity_level": ["online"] * 24,
                "activity_level_frequency": [5] * 24,
                "active_threshold": [0.5] * 24,
            },
        }
        user_info = UserInfo(name=real_name,
                             description=description,
                             profile=profile)
        agent = SocialAgent(agent_id=0, user_info=user_info, channel=channel)

        # Sign up the agent
        return_message = await agent.env.action.sign_up(
            "testuser", "TestUser", "A test bio.")
        assert return_message["success"] is True
        agents.append(agent)

        # Conduct an interview
        interview_prompt = "What are your thoughts on artificial intelligence?"
        return_message = await agent.env.action.interview(interview_prompt)
        assert return_message["success"] is True
        assert "interview_id" in return_message

        # Verify the interview was recorded in the database
        conn = sqlite3.connect(test_db_filepath)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT user_id, info, action
            FROM trace
            WHERE action = ? AND user_id = ?
        """, (ActionType.INTERVIEW.value, 0))

        interview_records = cursor.fetchall()
        assert len(interview_records) == 1

        user_id, info_json, action = interview_records[0]
        assert user_id == 0
        assert action == ActionType.INTERVIEW.value

        info = json.loads(info_json)
        assert info["prompt"] == interview_prompt
        assert "interview_id" in info

        conn.close()

    finally:
        await channel.write_to_receive_queue((None, None, ActionType.EXIT))
        await task


@pytest.mark.asyncio
async def test_multiple_interviews_action(setup_interview_test):
    """Test conducting multiple interviews with different agents."""
    agents = []
    channel = Channel()
    infra = Platform(test_db_filepath, channel)
    task = asyncio.create_task(infra.running())

    try:
        # Create and sign up multiple test agents
        for i in range(3):
            real_name = f"TestAgent{i}"
            description = f"Test agent {i} for interview testing."
            profile = {
                "nodes": [],
                "edges": [],
                "other_info": {
                    "user_profile":
                    f"I am test agent {i} with unique perspectives.",
                    "mbti": "INTJ",
                    "activity_level": ["online"] * 24,
                    "activity_level_frequency": [5] * 24,
                    "active_threshold": [0.5] * 24,
                },
            }
            user_info = UserInfo(name=real_name,
                                 description=description,
                                 profile=profile)
            agent = SocialAgent(agent_id=i,
                                user_info=user_info,
                                channel=channel)

            # Sign up the agent
            return_message = await agent.env.action.sign_up(
                f"testuser{i}", f"TestUser{i}", f"Test bio {i}.")
            assert return_message["success"] is True
            agents.append(agent)

        # Conduct interviews with different prompts
        interview_prompts = [
            "What is your opinion on climate change?",
            "How do you feel about social media?",
            "What are your thoughts on remote work?"
        ]

        for i, (agent, prompt) in enumerate(zip(agents, interview_prompts)):
            return_message = await agent.env.action.interview(prompt)
            assert return_message["success"] is True
            assert "interview_id" in return_message

        # Verify all interviews were recorded
        conn = sqlite3.connect(test_db_filepath)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT user_id, info, action
            FROM trace
            WHERE action = ?
            ORDER BY user_id
        """, (ActionType.INTERVIEW.value, ))

        interview_records = cursor.fetchall()
        assert len(interview_records) == 3

        for i, (user_id, info_json, action) in enumerate(interview_records):
            assert user_id == i
            assert action == ActionType.INTERVIEW.value

            info = json.loads(info_json)
            assert info["prompt"] == interview_prompts[i]
            assert "interview_id" in info

        conn.close()

    finally:
        await channel.write_to_receive_queue((None, None, ActionType.EXIT))
        await task


@pytest.mark.asyncio
async def test_interview_with_environment():
    """Test interview functionality using the full OASIS environment."""
    # Create a temporary database file
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
        db_path = tmp_file.name

    try:
        openai_model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O_MINI,
        )

        available_actions = [
            ActionType.CREATE_POST,
            ActionType.LIKE_POST,
            ActionType.REPOST,
            ActionType.FOLLOW,
            ActionType.DO_NOTHING,
        ]

        # Create a minimal agent graph for testing
        agent_graph = await generate_twitter_agent_graph(
            profile_path=("data/twitter_dataset/anonymous_topic_200_1h/"
                          "False_Business_0.csv"),
            model=openai_model,
            available_actions=available_actions,
        )

        # Create environment
        env = oasis.make(
            agent_graph=agent_graph,
            platform=oasis.DefaultPlatformType.TWITTER,
            database_path=db_path,
        )

        await env.reset()

        # Test single interview action
        actions = {}
        actions[env.agent_graph.get_agent(0)] = ManualAction(
            action_type=ActionType.INTERVIEW,
            action_args={"prompt": "What is your favorite color and why?"})
        await env.step(actions)

        # Test multiple interviews in one step
        actions = {}
        actions[env.agent_graph.get_agent(1)] = ManualAction(
            action_type=ActionType.INTERVIEW,
            action_args={"prompt": "What do you think about technology?"})
        actions[env.agent_graph.get_agent(2)] = ManualAction(
            action_type=ActionType.INTERVIEW,
            action_args={"prompt": "How do you spend your free time?"})
        await env.step(actions)

        # Test mixing interview with other actions
        actions = {}
        actions[env.agent_graph.get_agent(0)] = ManualAction(
            action_type=ActionType.CREATE_POST,
            action_args={"content": "Hello world!"})
        actions[env.agent_graph.get_agent(3)] = ManualAction(
            action_type=ActionType.INTERVIEW,
            action_args={"prompt": "What motivates you in life?"})
        await env.step(actions)

        await env.close()

        # Verify interview data in database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT user_id, info, created_at
            FROM trace
            WHERE action = ?
            ORDER BY created_at
        """, (ActionType.INTERVIEW.value, ))

        interview_records = cursor.fetchall()
        assert len(interview_records) == 4  # 4 interviews conducted

        # Check interview content
        expected_prompts = {
            "What is your favorite color and why?",
            "What do you think about technology?",
            "How do you spend your free time?", "What motivates you in life?"
        }

        # Get all records and verify content exists
        actual_prompts = set()
        for user_id, info_json, timestamp in interview_records:
            info = json.loads(info_json)
            actual_prompts.add(info["prompt"])
            assert "interview_id" in info
            assert "response" in info

        # Use set comparison to verify all expected prompts exist
        assert actual_prompts == expected_prompts, (
            f"Missing prompts: {expected_prompts - actual_prompts}, "
            f"Unexpected prompts: {actual_prompts - expected_prompts}")

        conn.close()

    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.remove(db_path)


@pytest.mark.asyncio
async def test_interview_data_retrieval(setup_interview_test):
    """Test retrieving and analyzing interview data from the database."""
    agents = []
    channel = Channel()
    infra = Platform(test_db_filepath, channel)
    task = asyncio.create_task(infra.running())

    try:
        # Create and sign up test agents
        for i in range(2):
            real_name = f"TestAgent{i}"
            description = f"Test agent {i}."
            profile = {
                "nodes": [],
                "edges": [],
                "other_info": {
                    "user_profile": f"Agent {i} profile",
                    "mbti": "INTJ",
                    "activity_level": ["online"] * 24,
                    "activity_level_frequency": [5] * 24,
                    "active_threshold": [0.5] * 24,
                },
            }
            user_info = UserInfo(name=real_name,
                                 description=description,
                                 profile=profile)
            agent = SocialAgent(agent_id=i,
                                user_info=user_info,
                                channel=channel)

            return_message = await agent.env.action.sign_up(
                f"testuser{i}", f"TestUser{i}", f"Bio {i}.")
            assert return_message["success"] is True
            agents.append(agent)

        # Conduct interviews
        interview_data = [
            (0, "What is your favorite programming language?"),
            (1, "How do you approach problem-solving?"),
            (0,
             "What are your career goals?"),  # Second interview with agent 0
        ]

        for agent_id, prompt in interview_data:
            return_message = await agents[agent_id].env.action.interview(prompt
                                                                         )
            assert return_message["success"] is True

        # Test data retrieval functions
        conn = sqlite3.connect(test_db_filepath)
        cursor = conn.cursor()

        # Test 1: Get all interviews
        cursor.execute(
            """
            SELECT user_id, info, created_at
            FROM trace
            WHERE action = ?
            ORDER BY created_at
        """, (ActionType.INTERVIEW.value, ))

        all_interviews = cursor.fetchall()
        assert len(all_interviews) == 3

        # Test 2: Get interviews for specific agent
        cursor.execute(
            """
            SELECT user_id, info, created_at
            FROM trace
            WHERE action = ? AND user_id = ?
            ORDER BY created_at
        """, (ActionType.INTERVIEW.value, 0))

        agent_0_interviews = cursor.fetchall()
        assert len(agent_0_interviews) == 2

        # Test 3: Verify interview content
        for i, (user_id, info_json, timestamp) in enumerate(all_interviews):
            info = json.loads(info_json)
            expected_agent_id, expected_prompt = interview_data[i]
            assert user_id == expected_agent_id
            assert info["prompt"] == expected_prompt
            assert "interview_id" in info

        conn.close()

    finally:
        await channel.write_to_receive_queue((None, None, ActionType.EXIT))
        await task


@pytest.mark.asyncio
async def test_interview_error_handling(setup_interview_test):
    """Test error handling in interview functionality."""
    agents = []
    channel = Channel()
    infra = Platform(test_db_filepath, channel)
    task = asyncio.create_task(infra.running())

    try:
        # Create and sign up a test agent
        real_name = "TestAgent"
        description = "Test agent."
        profile = {
            "nodes": [],
            "edges": [],
            "other_info": {
                "user_profile": "Test profile",
                "mbti": "INTJ",
                "activity_level": ["online"] * 24,
                "activity_level_frequency": [5] * 24,
                "active_threshold": [0.5] * 24,
            },
        }
        user_info = UserInfo(name=real_name,
                             description=description,
                             profile=profile)
        agent = SocialAgent(agent_id=0, user_info=user_info, channel=channel)

        return_message = await agent.env.action.sign_up(
            "testuser", "TestUser", "Test bio.")
        assert return_message["success"] is True
        agents.append(agent)

        # Test with empty prompt
        return_message = await agent.env.action.interview("")
        assert return_message[
            "success"] is True  # Empty prompt should still work

        # Test with very long prompt
        long_prompt = "What do you think about your state."
        return_message = await agent.env.action.interview(long_prompt)
        assert return_message["success"] is True

        # Verify both interviews were recorded
        conn = sqlite3.connect(test_db_filepath)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT COUNT(*)
            FROM trace
            WHERE action = ? AND user_id = ?
        """, (ActionType.INTERVIEW.value, 0))

        count = cursor.fetchone()[0]
        assert count == 2

        conn.close()

    finally:
        await channel.write_to_receive_queue((None, None, ActionType.EXIT))
        await task


if __name__ == "__main__":
    # Run tests individually for debugging
    async def run_tests():
        print("Running interview action tests...")

        # You can run individual tests here for debugging
        # await test_single_interview_action(None)
        # await test_multiple_interviews_action(None)
        # await test_interview_data_retrieval(None)
        # await test_interview_error_handling(None)

        print("All tests completed!")

    # asyncio.run(run_tests())
