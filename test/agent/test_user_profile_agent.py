# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========

import pytest
import json
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
from typing import Dict, Any
from camel.models import ModelFactory
from camel.types import ModelPlatformType

# Import the modules to test
from oasis.user_profile_agent.agent import PreferenceAgent
from oasis.user_profile_agent.types import (UserPreferenceProfile,
                                            ProfileSummary, Preferences,
                                            CommunityProfile)
from oasis.user_profile_agent.enums import (InterestEnum, ContentStyleEnum,
                                            InteractionStyleEnum,
                                            ActivePeriodEnum,
                                            BehavioralArchetypeEnum,
                                            CommunityRoleEnum)


class TestPreferenceAgent:
    """Test cases for PreferenceAgent class"""

    @pytest.fixture
    def preference_agent(self):
        """Create a mock PreferenceAgent instance for testing"""
        # Create a mock agent to avoid model initialization issues
        agent = Mock(spec=PreferenceAgent)
        agent._postprocess_response = PreferenceAgent._postprocess_response.__get__(agent)
        return agent

    @pytest.fixture
    def sample_user_history(self):
        """Sample user history data for testing"""
        return {
            "actions": {
                "create_post": [{
                    "timestamp": "2024-01-01T10:00:00",
                    "content":
                    "Discussing the latest trends in AI and machine learning",
                    "engagement": {
                        "likes": 15,
                        "comments": 3
                    }
                }, {
                    "timestamp": "2024-01-02T14:30:00",
                    "content": "Sharing insights on blockchain technology",
                    "engagement": {
                        "likes": 8,
                        "comments": 2
                    }
                }],
                "create_comment": [{
                    "timestamp": "2024-01-01T11:00:00",
                    "content":
                    "Great analysis! I agree with your points on decentralization.",
                    "post_id": "123"
                }],
                "like": [{
                    "timestamp": "2024-01-01T09:00:00",
                    "post_id": "456"
                }, {
                    "timestamp": "2024-01-01T15:00:00",
                    "post_id": "789"
                }]
            }
        }

    @pytest.fixture
    def sample_previous_profile(self):
        """Sample previous profile for testing"""
        return {
            "profile_summary": {
                "interests": ["Artificial Intelligence", "Software Development"],
                "preferences": {
                    "content_style": "Technical and analytical",
                    "content_style_tags": ["Long-form Posts"],
                    "interaction_style": "Professional and engaging",
                    "interaction_style_tags": ["Formal & Professional"],
                    "active_periods": ["Morning"]
                },
                "behavioral_summary":
                "Active content creator with focus on technology",
                "behavioral_archetype_tags": ["Content Creator"],
                "community_profile": {
                    "affinity": "Technology communities",
                    "potential_role": "Expert contributor",
                    "potential_role_tags": ["Expert Contributor"]
                }
            }
        }

    def test_postprocess_response_string_to_list_conversion(
            self, preference_agent):
        """Test _postprocess_response method converts strings to lists"""
        response_content = {
            "profile_summary": {
                "preferences": {
                    "content_style_tags": "Long-form Posts",
                    "interaction_style_tags": "Formal & Professional",
                    "active_periods": "Morning"
                },
                "behavioral_archetype_tags": "Content Creator",
                "community_profile": {
                    "potential_role_tags": "Expert Contributor"
                },
                "interests": "Artificial Intelligence, Software Development"
            }
        }

        result = preference_agent._postprocess_response(response_content)

        # Check that strings were converted to lists
        prefs = result["profile_summary"]["preferences"]
        assert isinstance(prefs["content_style_tags"], list)
        assert prefs["content_style_tags"] == ["Long-form Posts"]
        assert isinstance(prefs["interaction_style_tags"], list)
        assert isinstance(prefs["active_periods"], list)

        assert isinstance(
            result["profile_summary"]["behavioral_archetype_tags"], list)
        assert isinstance(
            result["profile_summary"]["community_profile"]
            ["potential_role_tags"], list)
        assert isinstance(result["profile_summary"]["interests"], list)
        assert result["profile_summary"]["interests"] == [
            "Artificial Intelligence", "Software Development"
        ]

    def test_postprocess_response_handles_active_periods_tags(
            self, preference_agent):
        """Test _postprocess_response handles incorrect field name active_periods_tags"""
        response_content = {
            "profile_summary": {
                "preferences": {
                    "active_periods_tags": ["Morning", "Evening"]
                }
            }
        }

        result = preference_agent._postprocess_response(response_content)

        prefs = result["profile_summary"]["preferences"]
        assert "active_periods" in prefs
        assert prefs["active_periods"] == ["Morning", "Evening"]
        assert "active_periods_tags" not in prefs

    @pytest.mark.asyncio
    async def test_analyse_with_valid_data(self, preference_agent,
                                           sample_user_history,
                                           sample_previous_profile):
        """Test analyse method with valid input data"""
        # Mock the LLM response
        mock_response = {
            "profile_summary": {
                "interests":
                ["Artificial Intelligence", "Software Development", "Cryptocurrency & Blockchain"],
                "preferences": {
                    "content_style":
                    "Technical and analytical with practical insights",
                    "content_style_tags":
                    ["Long-form Posts", "Opinion & Editorials"],
                    "interaction_style":
                    "Professional and engaging",
                    "interaction_style_tags":
                    ["Formal & Professional", "Analytical & Inquisitive"],
                    "active_periods": ["Morning", "Afternoon"]
                },
                "behavioral_summary":
                "Active content creator with expertise in emerging technologies",
                "behavioral_archetype_tags":
                ["Content Creator", "Thought Leader"],
                "community_profile": {
                    "affinity": "Technology and blockchain communities",
                    "potential_role": "Expert contributor and thought leader",
                    "potential_role_tags":
                    ["Expert Contributor", "Thought Leader"]
                }
            }
        }

        with patch.object(preference_agent, 'astep',
                          new_callable=AsyncMock) as mock_astep:
            mock_astep.return_value = Mock(msg=Mock(
                content=json.dumps(mock_response)))

            result = await preference_agent.analyse(
                (sample_user_history, sample_previous_profile, {}),
                previous_profile=sample_previous_profile)

            assert result is not None
            assert "profile_summary" in result
            assert result["profile_summary"]["interests"] == [
                "Artificial Intelligence", "Software Development", "Cryptocurrency & Blockchain"
            ]

    @pytest.mark.asyncio
    async def test_analyse_with_invalid_json_response(self, preference_agent,
                                                      sample_user_history):
        """Test analyse method handles invalid JSON response"""
        with patch.object(preference_agent, 'astep',
                          new_callable=AsyncMock) as mock_astep:
            mock_astep.return_value = Mock(msg=Mock(
                content="Invalid JSON response"))

            result = await preference_agent.analyse(
                (sample_user_history, {}, {}), previous_profile=None)

            assert result is None

    @pytest.mark.asyncio
    async def test_analyse_with_exception(self, preference_agent,
                                          sample_user_history):
        """Test analyse method handles exceptions gracefully"""
        with patch.object(preference_agent, 'astep',
                          new_callable=AsyncMock) as mock_astep:
            mock_astep.side_effect = Exception("LLM service error")

            result = await preference_agent.analyse(
                (sample_user_history, {}, {}), previous_profile=None)

            assert result is None


class TestUserProfileTypes:
    """Test cases for user profile data types"""

    def test_preferences_model_validation(self):
        """Test Preferences model validation"""
        preferences_data = {
            "content_style": "Technical and detailed",
            "content_style_tags": ["Long-form Posts"],
            "interaction_style": "Professional",
            "interaction_style_tags": ["Formal & Professional"],
            "active_periods": ["Morning"]
        }

        preferences = Preferences(**preferences_data)
        assert preferences.content_style == "Technical and detailed"
        assert preferences.content_style_tags == [
            ContentStyleEnum.LONG_FORM_POSTS
        ]

    def test_community_profile_model_validation(self):
        """Test CommunityProfile model validation"""
        community_data = {
            "affinity": "Technology communities",
            "potential_role": "Expert contributor",
            "potential_role_tags": ["Expert Contributor"]
        }

        community_profile = CommunityProfile(**community_data)
        assert community_profile.affinity == "Technology communities"
        assert community_profile.potential_role_tags == [
            CommunityRoleEnum.EXPERT_CONTRIBUTOR
        ]

    def test_profile_summary_model_validation(self):
        """Test ProfileSummary model validation"""
        profile_data = {
            "interests": ["Artificial Intelligence", "Software Development"],
            "preferences": {
                "content_style": "Technical",
                "content_style_tags": ["Long-form Posts"],
                "interaction_style": "Professional",
                "interaction_style_tags": ["Formal & Professional"],
                "active_periods": ["Morning"]
            },
            "behavioral_summary": "Active technology enthusiast",
            "behavioral_archetype_tags": ["Content Creator"],
            "community_profile": {
                "affinity": "Tech communities",
                "potential_role": "Contributor",
                "potential_role_tags": ["Expert Contributor"]
            }
        }

        profile_summary = ProfileSummary(**profile_data)
        assert len(profile_summary.interests) == 2
        assert profile_summary.behavioral_archetype_tags == [
            BehavioralArchetypeEnum.CONTENT_CREATOR
        ]

    def test_user_preference_profile_model_validation(self):
        """Test UserPreferenceProfile model validation"""
        profile_data = {
            "profile_summary": {
                "interests": ["Artificial Intelligence"],
                "preferences": {
                    "content_style": "Technical",
                    "content_style_tags": ["Long-form Posts"],
                    "interaction_style": "Professional",
                    "interaction_style_tags": ["Formal & Professional"],
                    "active_periods": ["Morning"]
                },
                "behavioral_summary": "Tech enthusiast",
                "behavioral_archetype_tags": ["Content Creator"],
                "community_profile": {
                    "affinity": "Tech communities",
                    "potential_role": "Contributor",
                    "potential_role_tags": ["Expert Contributor"]
                }
            }
        }

        user_profile = UserPreferenceProfile(**profile_data)
        assert user_profile.profile_summary.interests == [
            InterestEnum.ARTIFICIAL_INTELLIGENCE
        ]

    def test_string_to_list_conversion(self):
        """Test automatic string to list conversion in validators"""
        # Test with string input that should be converted to list
        preferences_data = {
            "content_style_tags": "Long-form Posts",  # String instead of list
            "interaction_style_tags": "Formal & Professional",
            "active_periods": "Morning"
        }

        preferences = Preferences(**preferences_data)
        assert isinstance(preferences.content_style_tags, list)
        assert preferences.content_style_tags == [
            ContentStyleEnum.LONG_FORM_POSTS
        ]
        assert isinstance(preferences.interaction_style_tags, list)
        assert isinstance(preferences.active_periods, list)


class TestEnums:
    """Test cases for enum classes"""

    def test_interest_enum_values(self):
        """Test InterestEnum has expected values"""
        assert InterestEnum.ARTIFICIAL_INTELLIGENCE.value == "Artificial Intelligence"
        assert InterestEnum.SOFTWARE_DEVELOPMENT.value == "Software Development"
        assert InterestEnum.ECONOMICS_MARKETS.value == "Economics & Markets"

    def test_content_style_enum_values(self):
        """Test ContentStyleEnum has expected values"""
        assert ContentStyleEnum.LONG_FORM_POSTS.value == "Long-form Posts"
        assert ContentStyleEnum.SHORT_QUICK_UPDATES.value == "Short & Quick Updates"

    def test_interaction_style_enum_values(self):
        """Test InteractionStyleEnum has expected values"""
        assert InteractionStyleEnum.FORMAL_PROFESSIONAL.value == "Formal & Professional"
        assert InteractionStyleEnum.CASUAL_FRIENDLY.value == "Casual & Friendly"

    def test_active_period_enum_values(self):
        """Test ActivePeriodEnum has expected English values"""
        assert ActivePeriodEnum.MORNING.value == "Morning"
        assert ActivePeriodEnum.AFTERNOON.value == "Afternoon"
        assert ActivePeriodEnum.EVENING.value == "Evening"
        assert ActivePeriodEnum.NIGHT.value == "Night"

    def test_behavioral_archetype_enum_values(self):
        """Test BehavioralArchetypeEnum has expected values"""
        assert BehavioralArchetypeEnum.CONTENT_CREATOR.value == "Content Creator"
        assert BehavioralArchetypeEnum.THOUGHT_LEADER.value == "Thought Leader"
        assert BehavioralArchetypeEnum.ACTIVE_PARTICIPANT.value == "Active Participant"

    def test_community_role_enum_values(self):
        """Test CommunityRoleEnum has expected values"""
        assert CommunityRoleEnum.EXPERT_CONTRIBUTOR.value == "Expert Contributor"
        assert CommunityRoleEnum.ACTIVE_PARTICIPANT.value == "Active Participant"
