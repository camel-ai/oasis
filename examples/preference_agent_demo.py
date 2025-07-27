#!/usr/bin/env python3
"""
PreferenceAgent Demo - Enhanced User Profile Analysis

This example demonstrates the new PreferenceAgent class that provides AI-powered
user behavior analysis and structured profile generation for social media simulation.

The PreferenceAgent analyzes user historical data and generates comprehensive profiles
including interests, behavioral preferences, interaction styles, and community roles.
"""

import asyncio
import json
from typing import Dict, Any

from oasis.user_profile_agent.agent import PreferenceAgent
from oasis.user_profile_agent.types import UserPreferenceProfile
from camel.models import ModelFactory
from camel.types import ModelType, ModelPlatformType


def create_sample_user_data():
    """Create sample user data to demonstrate PreferenceAgent analysis."""

    # Sample user historical behavior data
    user_history = {
        "user_id": 1,
        "actions": {
            "create_post": [
                "Just discovered a fascinating new ML algorithm for natural language processing! The attention mechanism improvements are incredible. #MachineLearning #AI",
                "Sharing my latest tutorial on Python decorators. Hope this helps fellow developers! 🐍 #Programming #Tutorial",
                "The future of AI in software development is here. Excited to see how GitHub Copilot evolves! #AI #SoftwareDevelopment",
                "Working on a new deep learning project. The transformer architecture never ceases to amaze me. #DeepLearning #Research"
            ],
            "create_comment": [
                "Great question! Here's how you can optimize that algorithm for better performance...",
                "I've had similar experiences with this framework. The key is to understand the underlying architecture.",
                "Excellent point about model efficiency. Have you considered using quantization techniques?",
                "This is exactly what the community needs - more practical tutorials like this!"
            ],
            "like_post": [],
            "sign_up": [],
            "refresh": [],
            "do_nothing": []
        },
        "time_distribution": {
            "morning": 2,
            "afternoon": 5,
            "evening": 8,
            "night": 3
        }
    }

    # Sample existing user profile data (could be empty for new users)
    existing_profile = {
        "basic_info": {
            "name":
            "TechExpert_Alice",
            "bio":
            "Senior software engineer passionate about AI and machine learning"
        }
    }

    # Sample agent profile information
    agent_profile = {
        "agent_id": 1,
        "name": "TechExpert_Alice",
        "bio":
        "Senior software engineer passionate about AI and machine learning. Loves sharing technical insights and helping others learn.",
        "personality_traits": ["analytical", "helpful", "detail-oriented"],
        "background": "Computer Science PhD, 8 years industry experience"
    }

    return user_history, existing_profile, agent_profile


async def demonstrate_preference_agent():
    """Demonstrate the PreferenceAgent's profile analysis capabilities."""

    print("🤖 PreferenceAgent Demo - AI-Powered User Profile Analysis")
    print("=" * 60)

    # Initialize the model for PreferenceAgent
    print("🔧 Initializing AI model for profile analysis...")
    try:
        # Use Qwen model with Aliyun endpoint for better regional compatibility
        model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.QWEN_MAX,
            url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        print("✅ Model initialized successfully (using Qwen via Aliyun)")
    except Exception as e:
        print(f"❌ Failed to initialize model: {e}")
        print("💡 Make sure you have set your API keys and model configuration")
        return

    # Create PreferenceAgent instance
    print("\n🧠 Creating PreferenceAgent...")
    preference_agent = PreferenceAgent(model=model)
    print("✅ PreferenceAgent created successfully")

    # Prepare sample user data
    print("\n📊 Preparing sample user data...")
    user_history, existing_profile, agent_profile = create_sample_user_data()

    print("📝 Sample user data prepared:")
    print(f"   • Posts created: {len(user_history['actions']['create_post'])}")
    print(
        f"   • Comments made: {len(user_history['actions']['create_comment'])}"
    )
    print(
        f"   • Most active period: evening ({user_history['time_distribution']['evening']} actions)"
    )
    print(f"   • User bio: {agent_profile['bio']}")

    # Demonstrate profile analysis
    await demonstrate_profile_analysis(preference_agent, user_history,
                                       existing_profile, agent_profile)

    # Show different scenarios
    await demonstrate_new_user_scenario(preference_agent)

    print("\n🎯 PreferenceAgent Demo Completed!")


async def demonstrate_profile_analysis(preference_agent, user_history,
                                       existing_profile, agent_profile):
    """Demonstrate comprehensive profile analysis."""

    print("\n🔍 Analyzing User Profile with AI...")
    print("-" * 40)

    try:
        # Call the PreferenceAgent to analyze user data
        print("🤖 Running AI analysis on user behavior patterns...")

        # The analyse method expects a tuple of (history, profile, agent_profiles)
        user_profiles_tuple = (user_history, existing_profile, agent_profile)

        # Perform the analysis
        analyzed_profile = await preference_agent.analyse(user_profiles_tuple)

        print("✅ Profile analysis completed!")

        # Check if we got valid results
        if analyzed_profile and 'profile_summary' in analyzed_profile:
            # Display the results
            display_analysis_results(analyzed_profile)
        else:
            print("⚠️  API returned no data, showing mock results instead...")
            analyzed_profile = get_mock_profile_analysis()
            display_analysis_results(analyzed_profile)

    except Exception as e:
        print(f"❌ Profile analysis failed: {e}")
        print(
            "💡 Showing mock results to demonstrate expected functionality...")

        # Show realistic mock results
        analyzed_profile = get_mock_profile_analysis()
        display_analysis_results(analyzed_profile)


def display_analysis_results(profile: Dict[str, Any]):
    """Display the analyzed profile results in a readable format."""

    print("\n📊 AI Analysis Results:")
    print("=" * 30)

    if not profile or 'profile_summary' not in profile:
        print("❌ No valid profile data received")
        return

    summary = profile['profile_summary']

    # Display interests
    interests = summary.get('interests', [])
    print(f"🎯 Identified Interests ({len(interests)}):")
    for interest in interests:
        print(f"   • {interest}")

    # Display preferences
    if 'preferences' in summary:
        prefs = summary['preferences']
        print(f"\n✍️  Content Style:")
        print(f"   Description: {prefs.get('content_style', 'N/A')}")
        print(f"   Tags: {prefs.get('content_style_tags', [])}")

        print(f"\n💬 Interaction Style:")
        print(f"   Description: {prefs.get('interaction_style', 'N/A')}")
        print(f"   Tags: {prefs.get('interaction_style_tags', [])}")

        print(f"\n🕒 Active Periods: {prefs.get('active_periods', [])}")

    # Display behavioral summary
    behavioral_summary = summary.get('behavioral_summary', '')
    if behavioral_summary:
        print(f"\n🧠 Behavioral Analysis:")
        print(f"   {behavioral_summary}")

    behavioral_tags = summary.get('behavioral_archetype_tags', [])
    if behavioral_tags:
        print(f"   Archetype Tags: {behavioral_tags}")

    # Display community profile
    if 'community_profile' in summary:
        cp = summary['community_profile']
        print(f"\n👥 Community Profile:")
        print(f"   Affinity: {cp.get('affinity', 'N/A')}")
        print(f"   Potential Role: {cp.get('potential_role', 'N/A')}")
        print(f"   Role Tags: {cp.get('potential_role_tags', [])}")

    print(f"\n💾 Complete Profile Structure:")
    print(json.dumps(profile, indent=2, ensure_ascii=False))


async def demonstrate_new_user_scenario(preference_agent):
    """Demonstrate profile generation for a new user with minimal data."""

    print("\n🆕 New User Scenario - Profile Generation from Bio Only")
    print("-" * 50)

    # Create data for a new user with no history
    new_user_history = {
        "user_id": 2,
        "actions": {
            "create_post": [],
            "create_comment": [],
            "like_post": [],
            "sign_up": [],
            "refresh": [],
            "do_nothing": []
        },
        "time_distribution": {
            "morning": 0,
            "afternoon": 0,
            "evening": 0,
            "night": 0
        }
    }

    new_user_profile = {}

    new_agent_profile = {
        "agent_id": 2,
        "name": "BusinessGuru_Bob",
        "bio":
        "Entrepreneur and business strategist. Focuses on market trends, startup advice, and investment opportunities.",
        "personality_traits": ["ambitious", "strategic", "networking-focused"]
    }

    print("📝 New user data:")
    print(f"   • Name: {new_agent_profile['name']}")
    print(f"   • Bio: {new_agent_profile['bio']}")
    print(f"   • Historical actions: None")

    try:
        print("\n🤖 Generating profile from bio and personality traits...")

        new_user_tuple = (new_user_history, new_user_profile,
                          new_agent_profile)
        new_analyzed_profile = await preference_agent.analyse(new_user_tuple)

        print("✅ Profile generated successfully!")

        # Check if we got valid results
        if not new_analyzed_profile or 'profile_summary' not in new_analyzed_profile:
            print("⚠️  API returned no data, showing mock results instead...")
            new_analyzed_profile = get_mock_new_user_profile()

        if new_analyzed_profile and 'profile_summary' in new_analyzed_profile:
            summary = new_analyzed_profile['profile_summary']
            interests = summary.get('interests', [])
            print(f"\n🎯 Predicted Interests: {interests}")

            if 'preferences' in summary:
                content_tags = summary['preferences'].get(
                    'content_style_tags', [])
                interaction_tags = summary['preferences'].get(
                    'interaction_style_tags', [])
                print(f"✍️  Predicted Content Style: {content_tags}")
                print(f"💬 Predicted Interaction Style: {interaction_tags}")

    except Exception as e:
        print(f"❌ New user profile generation failed: {e}")
        print(
            "💡 Showing mock results to demonstrate expected functionality...")

        # Show realistic mock results for new user
        new_analyzed_profile = get_mock_new_user_profile()
        if new_analyzed_profile and 'profile_summary' in new_analyzed_profile:
            summary = new_analyzed_profile['profile_summary']
            interests = summary.get('interests', [])
            print(f"\n🎯 Predicted Interests: {interests}")

            if 'preferences' in summary:
                content_tags = summary['preferences'].get(
                    'content_style_tags', [])
                interaction_tags = summary['preferences'].get(
                    'interaction_style_tags', [])
                print(f"✍️  Predicted Content Style: {content_tags}")
                print(f"💬 Predicted Interaction Style: {interaction_tags}")


def get_mock_profile_analysis():
    """Generate mock profile analysis data for demonstration."""
    return {
        "profile_summary": {
            "interests":
            ["Artificial Intelligence", "Software Development", "Technology"],
            "preferences": {
                "content_style":
                "Technical and educational content with detailed explanations and practical examples",
                "content_style_tags":
                ["Educational & Tutorials", "Professional Insights"],
                "interaction_style":
                "Helpful and supportive, professional tone with focus on knowledge sharing",
                "interaction_style_tags":
                ["Formal & Professional", "Supportive & Encouraging"],
                "active_periods": ["下午", "晚上"]
            },
            "behavioral_summary":
            "User demonstrates expert technical knowledge and actively helps others learn. Shows consistent pattern of sharing detailed tutorials and engaging in helpful discussions about AI and programming topics.",
            "behavioral_archetype_tags":
            ["Thought Leader", "Knowledge Seeker"],
            "community_profile": {
                "affinity":
                "Technical communities focused on AI and software development, particularly those emphasizing education and knowledge sharing",
                "potential_role":
                "Technical mentor and knowledge contributor who helps others understand complex concepts",
                "potential_role_tags":
                ["Expert Contributor", "Mentor & Guide"]
            }
        }
    }


def get_mock_new_user_profile():
    """Generate mock profile for new user scenario."""
    return {
        "profile_summary": {
            "interests": [
                "Business Strategy", "Economics & Markets",
                "Investing & Personal Finance"
            ],
            "preferences": {
                "content_style":
                "Strategic business insights with market analysis and practical advice for entrepreneurs",
                "content_style_tags":
                ["Professional Insights", "Opinion & Editorials"],
                "interaction_style":
                "Confident and strategic, focused on networking and business opportunities",
                "interaction_style_tags":
                ["Formal & Professional", "Direct & Straightforward"],
                "active_periods": ["上午", "下午"]
            },
            "behavioral_summary":
            "User exhibits strong business acumen and entrepreneurial mindset. Likely to share market insights, startup advice, and investment strategies with a focus on practical business applications.",
            "behavioral_archetype_tags":
            ["Thought Leader", "Active Participant"],
            "community_profile": {
                "affinity":
                "Business and entrepreneurship communities, particularly those focused on market trends and investment opportunities",
                "potential_role":
                "Business advisor and strategic thinker who provides market insights and startup guidance",
                "potential_role_tags":
                ["Expert Contributor", "Leader & Organizer"]
            }
        }
    }


def show_expected_output_structure():
    """Show the expected output structure of PreferenceAgent analysis."""

    print("\n📋 Expected PreferenceAgent Output Structure:")
    print("-" * 45)

    expected_structure = {
        "profile_summary": {
            "interests":
            ["Artificial Intelligence", "Software Development", "Technology"],
            "preferences": {
                "content_style":
                "Technical and educational content with detailed explanations",
                "content_style_tags":
                ["Educational & Tutorials", "Professional Insights"],
                "interaction_style":
                "Helpful and supportive, professional tone",
                "interaction_style_tags":
                ["Formal & Professional", "Supportive & Encouraging"],
                "active_periods": ["下午", "晚上"]
            },
            "behavioral_summary":
            "User demonstrates expert technical knowledge and actively helps others learn",
            "behavioral_archetype_tags":
            ["Thought Leader", "Knowledge Seeker"],
            "community_profile": {
                "affinity":
                "Technical communities focused on AI and software development",
                "potential_role": "Technical mentor and knowledge contributor",
                "potential_role_tags":
                ["Expert Contributor", "Mentor & Guide"]
            }
        }
    }

    print(json.dumps(expected_structure, indent=2, ensure_ascii=False))


def print_feature_overview():
    """Print an overview of PreferenceAgent features."""

    print("\n" + "=" * 60)
    print("🌟 PREFERENCEAGENT - AI-POWERED PROFILE ANALYSIS")
    print("=" * 60)
    print("""
🔧 KEY FEATURES:
   • AI-powered behavioral analysis using LLM
   • Structured profile generation with Pydantic validation
   • Support for both data-rich and new user scenarios
   • Comprehensive categorization using standardized enums
   • Automatic data type conversion and error handling

📊 ANALYSIS COMPONENTS:
   • User Interests: Categorized from predefined enum values
   • Content Style: How users create and share content
   • Interaction Style: Communication patterns and tone
   • Active Periods: Temporal behavior patterns
   • Behavioral Archetypes: Personality and engagement types
   • Community Roles: Social roles and group affiliations

🤖 TECHNICAL CAPABILITIES:
   • Handles incomplete or missing data gracefully
   • Converts string responses to proper array formats
   • Provides fallback mechanisms for API failures
   • Supports both historical data analysis and bio-based prediction
   • Generates structured JSON output for easy integration

💡 USE CASES:
   • Dynamic user profiling in social media simulations
   • Personalized content recommendation systems
   • Behavioral pattern analysis for research
   • Community formation and role assignment
   • User segmentation and targeting
""")
    print("=" * 60)


if __name__ == "__main__":
    print_feature_overview()

    try:
        asyncio.run(demonstrate_preference_agent())
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
