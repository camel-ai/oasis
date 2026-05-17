"""
Pydantic models shared across the API and simulation engine.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import IntEnum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Simulation Mode ────────────────────────────────────────────────────────────

class SimulationMode(IntEnum):
    CONTENT = 1    # Social-media content feedback (OASIS core)
    BROWSER = 2    # Website interaction via Playwright / Browserbase MCP
    VISUAL  = 3    # Visual-input analysis via VLM


# ── Persona / Focus Group ──────────────────────────────────────────────────────

class PersonaProfile(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    username: str
    bio: str
    age: Optional[int] = None
    gender: Optional[str] = None
    country: Optional[str] = None
    mbti: Optional[str] = None
    user_profile: str = ""   # rich narrative description


class FocusGroup(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str = ""
    personas: List[PersonaProfile] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def agent_count(self) -> int:
        return len(self.personas)


# ── API Request / Response ─────────────────────────────────────────────────────

class GeneratePersonasRequest(BaseModel):
    business_description: str
    customer_profile: str
    num_personas: int = Field(default=5, ge=1, le=50)
    blablador_api_key: Optional[str] = None


class GeneratePersonasResponse(BaseModel):
    job_id: str
    status: str
    focus_group_id: Optional[str] = None
    personas: Optional[List[PersonaProfile]] = None


class FocusGroupSummary(BaseModel):
    id: str
    name: str
    agent_count: int


class ListFocusGroupsResponse(BaseModel):
    focus_groups: List[FocusGroupSummary]


class SimulationParameters(BaseModel):
    mode: SimulationMode = SimulationMode.CONTENT
    # Mode 2 specific
    target_url: Optional[str] = None
    use_browserbase_fallback: bool = True
    # Mode 3 specific
    image_urls: Optional[List[str]] = None   # pre-uploaded image URLs
    image_base64: Optional[List[str]] = None  # base64-encoded images
    # General
    max_rounds: int = 1
    extra: Dict[str, Any] = Field(default_factory=dict)


class StartSimulationRequest(BaseModel):
    focus_group_id: str
    content_type: str = "text"          # text | url | image | mixed
    content_payload: str = ""           # the main text/URL payload
    parameters: SimulationParameters = Field(
        default_factory=SimulationParameters
    )


class SimulationStatus(BaseModel):
    job_id: str
    status: str                          # queued | running | completed | failed
    mode: SimulationMode = SimulationMode.CONTENT
    progress_percentage: float = 0.0
    message: str = ""
    results: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── Persona-generation via LLM ─────────────────────────────────────────────────

PERSONA_GENERATION_PROMPT = """\
You are a market research expert. Generate {num_personas} distinct, realistic \
user personas for the following context.

Business Description:
{business_description}

Target Customer Profile:
{customer_profile}

Return ONLY a valid JSON array of objects. Each object must have these keys:
  "username"     – a plausible social-media handle (no spaces)
  "name"         – full display name
  "bio"          – 1-2 sentence Twitter/Reddit-style bio
  "age"          – integer
  "gender"       – string
  "country"      – string
  "mbti"         – 4-letter MBTI type
  "persona"      – 3-5 sentence rich personality + behaviour description

Do not include any markdown fences or extra text. Output only the JSON array.
"""
