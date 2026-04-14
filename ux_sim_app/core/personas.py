"""
Persona generator: uses LLM to create a focus group of realistic customer personas
from a scraped website + optional user-supplied business context.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional

from ux_sim_app.core.llm import chat, text_content
from ux_sim_app.core.config import TEXT_MODEL


@dataclass
class Persona:
    username: str
    name: str
    age: int
    gender: str
    country: str
    mbti: str
    bio: str
    persona_type: str
    goals: List[str] = field(default_factory=list)
    frustrations: List[str] = field(default_factory=list)

    def system_prompt(self, mode: str = "content") -> str:
        base = (
            f"You are {self.name} (@{self.username}), a {self.age}-year-old "
            f"{self.gender} from {self.country}. MBTI: {self.mbti}. "
            f"Type: {self.persona_type}. Bio: {self.bio}. "
            f"Goals: {', '.join(self.goals)}. "
            f"Frustrations: {', '.join(self.frustrations)}."
        )
        if mode == "content":
            return base + (
                "\n\nYou are a social media user. React authentically to content "
                "using the social_action tool. Be in-character and opinionated."
            )
        if mode == "browser":
            return base + (
                "\n\nYou are browsing a website as this persona. Describe your "
                "experience finding information, navigating menus, and completing tasks. "
                "Be specific about what confused or delighted you."
            )
        if mode == "visual":
            return base + (
                "\n\nYou are a consumer reacting to brand imagery. Give honest, "
                "in-character emotional and aesthetic feedback. Be specific."
            )
        return base

    def to_dict(self) -> dict:
        return asdict(self)


PERSONA_SCHEMA = {
    "type": "function",
    "function": {
        "name": "create_focus_group",
        "description": "Create a focus group of customer personas.",
        "parameters": {
            "type": "object",
            "properties": {
                "personas": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "username": {"type": "string"},
                            "name": {"type": "string"},
                            "age": {"type": "integer"},
                            "gender": {"type": "string"},
                            "country": {"type": "string"},
                            "mbti": {"type": "string"},
                            "bio": {"type": "string"},
                            "persona_type": {"type": "string"},
                            "goals": {"type": "array", "items": {"type": "string"}},
                            "frustrations": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["username", "name", "age", "gender", "country",
                                     "mbti", "bio", "persona_type", "goals", "frustrations"],
                    },
                }
            },
            "required": ["personas"],
        },
    },
}


async def generate_personas(
    website_text: str,
    website_url: str,
    business_context: str = "",
    customer_profile: str = "",
    num_personas: int = 5,
    real_world_context: str = "",
) -> List[Persona]:
    """Generate a focus group of personas from website content + optional user context.

    Parameters
    ----------
    real_world_context : str, optional
        A synthesized briefing of live community signals (Reddit, HN, X, etc.)
        gathered by the Real World Data tab. When provided, personas will reflect
        current public sentiment about the brand or topic.
    """

    user_context_block = ""
    if business_context.strip():
        user_context_block += f"\n\nAdditional business context from the owner:\n{business_context}"
    if customer_profile.strip():
        user_context_block += f"\n\nOwner's description of their ideal customer:\n{customer_profile}"
    if real_world_context.strip():
        user_context_block += (
            f"\n\nReal-world community signals (gathered live from Reddit, Hacker News, "
            f"social media and other sources):\n{real_world_context}\n"
            f"Use these signals to shape the opinions, frustrations, and attitudes of the "
            f"personas so they reflect what real people are currently saying about this brand."
        )

    prompt = f"""You are a UX research specialist. Based on the following website content,
generate {num_personas} distinct, realistic customer personas who would visit this website.

Website URL: {website_url}

Website content (auto-scraped):
{website_text[:3000]}
{user_context_block}

Create personas that represent the realistic diversity of this business's customer base.
Each persona should have specific goals when visiting the website and realistic frustrations
they might encounter. Make them feel like real people, not archetypes.

Use the create_focus_group tool to return the personas."""

    resp = await chat(
        messages=[
            {"role": "system", "content": "You are a UX research expert. Always use the provided tool."},
            {"role": "user", "content": prompt},
        ],
        tools=[PERSONA_SCHEMA],
        tool_choice={"type": "function", "function": {"name": "create_focus_group"}},
        max_tokens=3000,
    )

    tc = resp["choices"][0]["message"].get("tool_calls", [])
    if not tc:
        raise ValueError("LLM did not return personas via tool call.")

    data = json.loads(tc[0]["function"]["arguments"])
    personas = []
    for p in data.get("personas", []):
        personas.append(Persona(
            username=p.get("username", "user"),
            name=p.get("name", "Unknown"),
            age=int(p.get("age", 35)),
            gender=p.get("gender", "unknown"),
            country=p.get("country", "Australia"),
            mbti=p.get("mbti", "ENFP"),
            bio=p.get("bio", ""),
            persona_type=p.get("persona_type", "General User"),
            goals=p.get("goals", []),
            frustrations=p.get("frustrations", []),
        ))
    return personas
