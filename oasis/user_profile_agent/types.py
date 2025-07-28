from typing import List, Optional, Union
from pydantic import BaseModel, Field, field_validator
from oasis.user_profile_agent.enums import ContentStyleEnum, InteractionStyleEnum, ActivePeriodEnum, BehavioralArchetypeEnum, CommunityRoleEnum, InterestEnum
import json


class Preferences(BaseModel):
    """
    Defines the user's behavioral preferences.
    Descriptive text is retained, and parallel enumeration tag lists are added for normalization.
    """
    content_style: str = Field(
        default=None,
        description=
        "String. Detailed descriptive text about the user's content creation style, e.g., 'The user mainly publishes long, detailed posts to share professional insights.'"
    )
    content_style_tags: List[ContentStyleEnum] = Field(
        default=[],
        description=
        "Array of strings. Standardized list of content style tags. Must be an array of strings, each chosen from the following enum values: "
        + ", ".join([e.value for e in ContentStyleEnum]) +
        ". Example: ['Long-form Posts', 'Opinion & Editorials']")
    interaction_style: str = Field(
        default=None,
        description=
        "String. Detailed descriptive text about the user's interaction style, e.g., 'The user communicates in a formal and professional tone, often engaging in inquisitive discussions.'"
    )
    interaction_style_tags: List[InteractionStyleEnum] = Field(
        default=[],
        description=
        "Array of strings. Standardized list of interaction style tags. Must be an array of strings, each chosen from the following enum values: "
        + ", ".join([e.value for e in InteractionStyleEnum]) +
        ". Example: ['Formal & Professional', 'Analytical & Inquisitive']")
    active_periods: List[ActivePeriodEnum] = Field(
        default=[],
        description=
        "Array of strings. List of user's core active period enums. Must be an array of strings, each chosen from the following enum values: "
        + ", ".join([e.value for e in ActivePeriodEnum]) +
        ". Example: ['Evening', 'Late Night']")

    @field_validator('content_style_tags',
                     'interaction_style_tags',
                     'active_periods',
                     mode='before')
    @classmethod
    def convert_string_to_list(cls, v):
        """Automatically convert a string to list format"""
        if isinstance(v, str):
            return [v]
        return v


class CommunityProfile(BaseModel):
    """Describes the user's potential community affiliations and roles."""
    affinity: str = Field(
        default=None,
        description=
        "String. Detailed description of the specific types of communities the user might join, e.g., 'The user may join academic circles focused on technology policy and ethics.'"
    )
    potential_role: str = Field(
        default=None,
        description=
        "String. Detailed description of the social roles the user might play in these communities, e.g., 'They may act as an expert contributor, sharing in-depth insights.'"
    )
    potential_role_tags: List[CommunityRoleEnum] = Field(
        default=[],
        description=
        "Array of strings. Standardized list of potential community role tags. Must be an array of strings, each chosen from the following enum values: "
        + ", ".join([e.value for e in CommunityRoleEnum]) +
        ". Example: ['Expert Contributor', 'Active Participant']")

    @field_validator('potential_role_tags', mode='before')
    @classmethod
    def convert_string_to_list(cls, v):
        """Automatically convert a string to list format"""
        if isinstance(v, str):
            return [v]
        return v


class ProfileSummary(BaseModel):
    """
    Summary of the user's profile.
    """
    interests: List[InterestEnum] = Field(
        ...,
        description=
        "Array of strings. List of user's core interest tags. Must be an array of strings, each chosen from the following enum values: "
        + ", ".join([e.value for e in InterestEnum]) +
        ". Example: ['Economics & Markets', 'Investing & Personal Finance', 'Business Strategy']"
    )
    preferences: Preferences = Field(
        default=None,
        description=
        "Object. Detailed information about the user's behavioral preferences, including content style, interaction style, and active periods."
    )
    behavioral_summary: str = Field(
        default=None,
        description=
        "String. Qualitative description of the user's participation patterns and behavioral archetypes, e.g., 'The user follows a pattern of observing first and then deeply engaging as a content creator.'"
    )
    behavioral_archetype_tags: List[BehavioralArchetypeEnum] = Field(
        default=[],
        description=
        "Array of strings. Standardized list of user's behavioral archetype tags. Must be an array of strings, each chosen from the following enum values: "
        + ", ".join([e.value for e in BehavioralArchetypeEnum]) +
        ". Example: ['Content Creator', 'Thought Leader']")
    community_profile: CommunityProfile = Field(
        default=None,
        description=
        "Object. Detailed information about the user's community affiliation and roles, including community affinity and potential roles."
    )

    @field_validator('interests', 'behavioral_archetype_tags', mode='before')
    @classmethod
    def convert_string_to_list(cls, v):
        """Automatically convert a string to list format"""
        if isinstance(v, str):
            return [v]
        return v


class UserPreferenceProfile(BaseModel):
    """
    A structured user profile generated from behavioral analysis.
    This model strictly follows the required JSON output format.
    """
    profile_summary: ProfileSummary = Field(
        ...,
        description=
        "Object. Detailed summary of the user's profile, including interests, preferences, behavioral summary, and community profile."
    )

    @classmethod
    def description(cls) -> str:
        """Generate a JSON Schema description for use as the LLM response format."""
        schema = cls.model_json_schema()
        return json.dumps(schema, indent=2)
