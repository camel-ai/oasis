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
from __future__ import annotations

import json
from typing import List, Tuple
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from .types import *


class PreferenceAgent(ChatAgent):

    def _format_system_message(self) -> BaseMessage:
        return BaseMessage.make_user_message(
            role_name="User Behavior Analyst",
            content=
            "You are an expert social media behavior analyst. Your task is to analyze user behavior patterns and generate structured profiles."
        )

    def _format_user_prompt(self, history: dict, profile: dict,
                            agent_profiles: dict) -> str:
        user_history = json.dumps(history, indent=2) if history else ""
        user_profile = json.dumps(profile, indent=2) if profile else ""
        agent_profile = json.dumps(agent_profiles,
                                   indent=2) if agent_profiles else ""

        # Define the prompt as a regular string to avoid f-string parsing issues
        # with the JSON block.
        static_prompt = static_prompt = """You are a highly precise data processing agent. Your ONLY function is to analyze user data and map it to a rigid JSON structure using ONLY the predefined tags provided. Any deviation from the exact tag values will cause a system failure.

**CRITICAL DIRECTIVE**: For every field that requires tags, you MUST use the exact, case-sensitive, character-for-character string values from the "Valid Tags" lists provided in each section.
-   DO NOT invent, create, or add any new tags.
-   DO NOT paraphrase, simplify, or shorten the tags. For example, if a valid tag is "Economics & Markets", you MUST use "Economics & Markets", NOT "Economics".
-   DO NOT extract partial keywords from the valid tags.
-   IMPORTANT: Your output MUST be a single, valid JSON object and nothing else. Do not include any text before or after the JSON.

---

### Analysis Workflow
Your analysis process depends on the input data provided below. READ THIS CAREFULLY.

* **Case A: If you see a section titled "The following is the historical data of the user":**
    You MUST base your entire analysis and all generated fields directly on the JSON data provided in that section.

* **Case B: If the "historical data of the user" section is ABSENT or `null`:**
    This means you must generate a profile based on imagination. Use the "profile data of the user" (e.g., their bio and name) to first mentally create a short, plausible history of actions (e.g., 1-2 posts or comments) that are consistent with that profile. Then, base your entire final JSON output on this **imagined history**. Do NOT output the imagined history itself; it is only an internal tool for you to generate the final, structured JSON profile.

---

Your Instructions:
For each section below, you will provide a descriptive text summary AND select the most fitting tags from the predefined lists where applicable.

1.  **Identify User ID**: Extract the user identifier.

2.  **Analyze User Interests**:
    * Analyze the user's content and map their topics to the EXACT string values from the `Valid InterestEnum Tags` list below. No other values are permitted.
    * You MUST select 3 to 5 tags from this list.
    * **Valid `InterestEnum` Tags**: "Artificial Intelligence", "Software Development", "Cryptocurrency & Blockchain", "Economics & Markets", "Investing & Personal Finance", "Travel & Adventure", "Health & Wellness", "Gaming", "Movies & TV Shows", "Music", "Books & Literature", "Politics & Governance", "Social Issues & Advocacy", "Education", "Philosophy", "Hospitality Industry", "Sustainability", "Eco-tourism", "Business Strategy", "Technology in Hospitality", "Logistics and Distribution Networks", "Cultural Intersections"

3.  **Analyze Behavioral Preferences**:
    * **Content Style**: Provide a descriptive text. Then, map the findings to 1-2 tags chosen from the EXACT `Valid ContentStyleEnum Tags` list below.
        -   **Valid `ContentStyleEnum` Tags**: "Long-form Posts", "Short & Quick Updates", "Visual Content", "Opinion & Editorials", "Educational & Tutorials", "News & Updates", "Personal Stories", "Professional Insights", "Questions & Inquiries"
    * **Interaction Style**: Provide a descriptive text. Then, map the findings to 1-2 tags chosen from the EXACT `Valid InteractionStyleEnum Tags` list below.
        -   **Valid `InteractionStyleEnum` Tags**: "Friendly & Supportive", "Formal & Professional", "Casual & Friendly", "Analytical & Inquisitive", "Supportive & Encouraging", "Humorous & Witty", "Direct & Straightforward", "Thoughtful & Reflective"
    * **Active Periods**: Select one or more tags from the EXACT `Valid ActivePeriodEnum Tags` list below.
        -   **Valid `ActivePeriodEnum` Tags**: "上午", "下午", "晚上", "深夜"

4.  **Summarize Behavioral Patterns**:
    * Provide a qualitative summary text for `behavioral_summary`.
    * Then, map the user's archetype to 1-2 tags chosen from the EXACT `Valid BehavioralArchetypeEnum Tags` list below.
        -   **Valid `BehavioralArchetypeEnum` Tags**: "Content Creator", "Active Participant", "Observer & Lurker", "Thought Leader", "Community Builder", "Knowledge Seeker", "Casual User"

5.  **Analyze Community & Network Profile**:
    * **Affinity**: Provide a descriptive text for the `affinity` field.
    * **Potential Role**: Provide a descriptive text for `potential_role`. Then, map the role to 1-2 tags chosen from the EXACT `Valid CommunityRoleEnum Tags` list below.
        -   **Valid `CommunityRoleEnum` Tags**: "Leader & Organizer", "Mentor & Guide", "Expert Contributor", "Active Participant", "Newcomer & Learner"

**CRITICAL REQUIREMENTS - MANDATORY COMPLIANCE:**

🚨 **ARRAY FORMAT REQUIREMENT**: 
- `interests` → MUST BE ARRAY: ["value1", "value2"]
- `content_style_tags` → MUST BE ARRAY: ["value1", "value2"]
- `interaction_style_tags` → MUST BE ARRAY: ["value1", "value2"]
- `active_periods` → MUST BE ARRAY: ["value1", "value2"]
- `behavioral_archetype_tags` → MUST BE ARRAY: ["value1", "value2"]
- `potential_role_tags` → MUST BE ARRAY: ["value1", "value2"]

🚨 **NEVER USE SINGLE STRINGS FOR THESE FIELDS**
❌ WRONG: "content_style_tags": "Long-form Posts"
✅ CORRECT: "content_style_tags": ["Long-form Posts"]

🚨 **ALL FIELDS MUST BE PRESENT**
- interests, preferences, behavioral_summary, behavioral_archetype_tags, community_profile
- content_style, content_style_tags, interaction_style, interaction_style_tags, active_periods
- affinity, potential_role, potential_role_tags

**FINAL CHECK**: Before outputting JSON, verify EVERY field ending with "_tags", "interests", or "active_periods" is an ARRAY with square brackets []

**Must strictly follow the following JSON structure format (example):**

```json
{
  "profile_summary": {
    "interests": ["Business Strategy", "Economics & Markets", "Hospitality Industry"],
    "preferences": {
      "content_style": "专业深度分析，结合行业经验分享实用见解",
      "content_style_tags": ["Professional Insights", "Educational & Tutorials"],
      "interaction_style": "正式专业但友好支持，乐于分享经验和建议",
      "interaction_style_tags": ["Formal & Professional", "Supportive & Encouraging"],
      "active_periods": ["下午", "晚上"]
    },
    "behavioral_summary": "用户表现为经验丰富的行业专家，积极参与专业讨论，乐于分享见解和建议，具有明显的思想领袖特质",
    "behavioral_archetype_tags": ["Thought Leader", "Knowledge Seeker"],
    "community_profile": {
      "affinity": "倾向于加入商业、经济和酒店旅游行业的专业社区，重视知识分享和行业交流",
      "potential_role": "在专业社区中担任经验分享者和指导者角色，为新人提供行业见解",
      "potential_role_tags": ["Expert Contributor", "Mentor & Guide"]
    }
  }
}
```

**Note: The above is a complete example, you must generate a complete JSON structure containing all fields.**

**Important Notes:**
- All fields marked as "string arrays" must be in array format, not single strings
- All enum values must be strictly selected from the above lists, cannot be created arbitrarily
- JSON structure must be complete, cannot miss any required fields

**Example Output:**
```json
{
  "profile_summary": {
    "interests": ["Economics & Markets", "Investing & Personal Finance", "Business Strategy"],
    "preferences": {
      "content_style": "The user primarily writes long, detailed posts to share professional opinions.",
      "content_style_tags": ["Long-form Posts", "Professional Insights"],
      "interaction_style": "Their tone is formal and professional, often engaging in inquisitive discussions.",
      "interaction_style_tags": ["Formal & Professional", "Analytical & Inquisitive"],
      "active_periods": ["晚上", "下午"]
    },
    "behavioral_summary": "The user follows a pattern of observing first, then engaging deeply as a content creator.",
    "behavioral_archetype_tags": ["Content Creator", "Thought Leader"],
    "community_profile": {
      "affinity": "The user would likely join academic circles focused on technology policy and ethics.",
      "potential_role": "They would act as an expert contributor, sharing deep insights.",
      "potential_role_tags": ["Expert Contributor"]
    }
  }
}
```

The following is the historical data of the user:  
{user_history}

The following is the profile data of the user:  
{user_profile}

The following is the agent profile of the user:  
{agent_profile}
"""

        prompt = f"""
The following is the historical data of the user:  
{user_history}

The following is the profile data of the user:  
{user_profile}

The following is the agent profile of the user:  
{agent_profile}
"""

        return static_prompt + prompt

    def _postprocess_response(self, response_content: dict) -> dict:
        """
        Post-process model response, converting string tag fields to list format.
        This handles cases where the model returns single strings instead of arrays.
        """
        if isinstance(response_content,
                      dict) and 'profile_summary' in response_content:
            profile = response_content['profile_summary']

            # Handle tag fields in preferences
            if 'preferences' in profile:
                prefs = profile['preferences']
                # Convert strings to lists
                for field in [
                        'content_style_tags', 'interaction_style_tags',
                        'active_periods'
                ]:
                    if field in prefs and isinstance(prefs[field], str):
                        prefs[field] = [prefs[field]]

                # Handle possible incorrect field names
                if 'active_periods_tags' in prefs:
                    if 'active_periods' not in prefs:
                        prefs['active_periods'] = prefs[
                            'active_periods_tags'] if isinstance(
                                prefs['active_periods_tags'],
                                list) else [prefs['active_periods_tags']]
                    del prefs['active_periods_tags']

            # Handle behavioral_archetype_tags
            if 'behavioral_archetype_tags' in profile and isinstance(
                    profile['behavioral_archetype_tags'], str):
                profile['behavioral_archetype_tags'] = [
                    profile['behavioral_archetype_tags']
                ]

            # Handle tag fields in community_profile
            if 'community_profile' in profile:
                cp = profile['community_profile']
                if 'potential_role_tags' in cp and isinstance(
                        cp['potential_role_tags'], str):
                    cp['potential_role_tags'] = [cp['potential_role_tags']]

            # Handle interests field, split if it's a comma-separated string
            if 'interests' in profile and isinstance(profile['interests'],
                                                     str):
                profile['interests'] = [
                    i.strip() for i in profile['interests'].split(',')
                ]

        return response_content

    async def analyse(self, user_profiles: Tuple[dict, dict, dict]) -> dict:
        r"""
        Analyze user profile.
        
        Args:
            user_profiles: Tuple containing user history, personal data, and agent information
        
        Returns:
            Updated user profile data
        """
        input_content = self._format_user_prompt(user_profiles[0],
                                                 user_profiles[1],
                                                 user_profiles[2])
        user_msg = BaseMessage.make_user_message(role_name="User",
                                                 content=input_content)

        try:
            # First try using strict response_format
            response = await self.astep(user_msg,
                                        response_format=UserPreferenceProfile)
            print(f'Strict response successful: {response}')

            # Ensure the return is in dictionary format, not string
            content = response.msg.content
            if isinstance(content, str):
                # If it's a string, try to parse as JSON
                try:
                    import json
                    content = json.loads(content)
                except json.JSONDecodeError:
                    print(
                        f"Warning: Failed to parse response as JSON: {content}"
                    )
            elif hasattr(content, 'model_dump'):
                # If it's a Pydantic model, convert to dictionary
                content = content.model_dump()
            elif hasattr(content, 'dict'):
                # Compatible with older Pydantic versions
                content = content.dict()

            return content
        except Exception as e:
            print(f"Strict response failed: {e}")
            print("Falling back to flexible parsing...")

    def __init__(self, model):
        system_message = self._format_system_message()
        super().__init__(system_message=system_message,
                         model=model,
                         tools=[],
                         single_iteration=True,
                         scheduling_strategy='random_model')
