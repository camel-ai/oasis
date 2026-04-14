"""
PersonaManager – generates, stores, and retrieves focus groups of personas.

Persona generation is driven by an LLM (OpenAI-compatible) so that any
business description + customer profile can produce a realistic cohort.
The resulting profiles are persisted as JSON files under
  simulation_backend/data/focus_groups/<id>.json
so they survive server restarts.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional

import httpx

from simulation_backend.core.models import (
    FocusGroup,
    FocusGroupSummary,
    PersonaProfile,
    PERSONA_GENERATION_PROMPT,
)
from simulation_backend.core.settings import get_settings

log = logging.getLogger("simulation_backend.persona_manager")


class PersonaManager:
    """Thread-safe in-memory + disk-backed store for focus groups."""

    def __init__(self) -> None:
        self._groups: Dict[str, FocusGroup] = {}
        self._settings = get_settings()
        os.makedirs(self._settings.focus_groups_dir, exist_ok=True)
        self._load_from_disk()

    # ── Persistence ────────────────────────────────────────────────────────

    def _group_path(self, group_id: str) -> str:
        return os.path.join(
            self._settings.focus_groups_dir, f"{group_id}.json"
        )

    def _save_to_disk(self, group: FocusGroup) -> None:
        path = self._group_path(group.id)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(group.model_dump_json(indent=2))
        log.debug("Saved focus group %s to %s", group.id, path)

    def _load_from_disk(self) -> None:
        dir_path = self._settings.focus_groups_dir
        for fname in os.listdir(dir_path):
            if not fname.endswith(".json"):
                continue
            try:
                with open(os.path.join(dir_path, fname), encoding="utf-8") as fh:
                    data = json.load(fh)
                group = FocusGroup(**data)
                self._groups[group.id] = group
                log.info("Loaded focus group '%s' (%s)", group.name, group.id)
            except Exception as exc:
                log.warning("Could not load %s: %s", fname, exc)

    # ── Public API ─────────────────────────────────────────────────────────

    def list_groups(self) -> List[FocusGroupSummary]:
        return [
            FocusGroupSummary(
                id=g.id, name=g.name, agent_count=g.agent_count
            )
            for g in self._groups.values()
        ]

    def get_group(self, group_id: str) -> Optional[FocusGroup]:
        return self._groups.get(group_id)

    def save_group(self, group: FocusGroup) -> FocusGroup:
        self._groups[group.id] = group
        self._save_to_disk(group)
        return group

    def delete_group(self, group_id: str) -> bool:
        if group_id not in self._groups:
            return False
        del self._groups[group_id]
        path = self._group_path(group_id)
        if os.path.exists(path):
            os.remove(path)
        return True

    # ── LLM-based generation ───────────────────────────────────────────────

    async def generate_personas(
        self,
        business_description: str,
        customer_profile: str,
        num_personas: int = 5,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> FocusGroup:
        """
        Call an OpenAI-compatible endpoint to generate personas, then
        persist them as a new FocusGroup.
        """
        settings = self._settings
        effective_key = api_key or settings.openai_api_key
        effective_url = (base_url or settings.openai_base_url).rstrip("/")

        prompt = PERSONA_GENERATION_PROMPT.format(
            num_personas=num_personas,
            business_description=business_description,
            customer_profile=customer_profile,
        )

        payload = {
            "model": settings.default_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that outputs only valid JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.8,
        }

        headers = {
            "Authorization": f"Bearer {effective_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{effective_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()

        raw = resp.json()["choices"][0]["message"]["content"].strip()

        # Strip markdown fences if the model added them
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        agent_data: List[dict] = json.loads(raw)

        personas: List[PersonaProfile] = []
        for item in agent_data[:num_personas]:
            personas.append(
                PersonaProfile(
                    id=str(uuid.uuid4()),
                    name=item.get("name", "Unknown"),
                    username=item.get("username", "user"),
                    bio=item.get("bio", ""),
                    age=item.get("age"),
                    gender=item.get("gender"),
                    country=item.get("country"),
                    mbti=item.get("mbti"),
                    user_profile=item.get("persona", ""),
                )
            )

        group_name = (
            f"Focus Group – {business_description[:40].strip()}…"
            if len(business_description) > 40
            else f"Focus Group – {business_description.strip()}"
        )
        group = FocusGroup(
            id=str(uuid.uuid4()),
            name=group_name,
            description=f"Generated for: {customer_profile[:100]}",
            personas=personas,
            created_at=datetime.utcnow(),
        )
        return self.save_group(group)

    # ── OASIS AgentGraph builder ───────────────────────────────────────────

    def build_agent_graph(
        self,
        group: FocusGroup,
        available_actions=None,
        model=None,
    ):
        """
        Convert a FocusGroup into an OASIS AgentGraph.
        Imported lazily to avoid hard dependency when OASIS is not installed.
        """
        try:
            from oasis.social_agent.agent import SocialAgent
            from oasis.social_agent.agent_graph import AgentGraph
            from oasis.social_platform.config.user import UserInfo
        except ImportError as exc:
            raise RuntimeError(
                "OASIS package not found. Install it with: pip install oasis-sim"
            ) from exc

        agent_graph = AgentGraph()
        for idx, persona in enumerate(group.personas):
            profile = {
                "nodes": [],
                "edges": [],
                "other_info": {
                    "user_profile": persona.user_profile,
                    "mbti": persona.mbti or "",
                    "gender": persona.gender or "",
                    "age": persona.age or 0,
                    "country": persona.country or "",
                },
            }
            user_info = UserInfo(
                name=persona.name,
                description=persona.bio,
                profile=profile,
                recsys_type="twitter",
            )
            agent = SocialAgent(
                agent_id=idx,
                user_info=user_info,
                model=model,
                agent_graph=agent_graph,
                available_actions=available_actions,
            )
            agent_graph.add_agent(agent)
        return agent_graph


# ── Singleton ──────────────────────────────────────────────────────────────────
_persona_manager: Optional[PersonaManager] = None


def get_persona_manager() -> PersonaManager:
    global _persona_manager
    if _persona_manager is None:
        _persona_manager = PersonaManager()
    return _persona_manager
