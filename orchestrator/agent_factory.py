from __future__ import annotations

import asyncio
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from camel.models import BaseModelBackend

from generation.emission_policy import EmissionPolicy, PersonaConfig
from generation.extended_agent import ExtendedSocialAgent
from oasis.social_agent.agent_graph import AgentGraph
from oasis.social_platform.channel import Channel
from oasis.social_platform.config import UserInfo
from oasis.social_platform.typing import ActionType
from orchestrator.sidecar_logger import SidecarLogger


def _infer_primary_label(username: str) -> str:
    lowered = username.lower()
    if lowered.startswith("incel_"):
        return "incel"
    if lowered.startswith("misinfo_"):
        return "misinfo"
    if lowered.startswith("benign_"):
        return "benign"
    # default benign
    return "benign"


async def build_agent_graph_from_csv(
    personas_csv: Path,
    model: BaseModelBackend,
    channel: Channel,
    available_actions: Optional[List[ActionType]] = None,
    emission_policy: Optional[EmissionPolicy] = None,
    sidecar_logger: Optional[SidecarLogger] = None,
    run_seed: int = 314159,
    **extended_kwargs: Any,
) -> AgentGraph:
    r"""Create an AgentGraph using ExtendedSocialAgent from a simple personas CSV.

    CSV columns expected: username, description, user_char, primary_label,
    secondary_label (optional), allowed_labels (JSON list), label_mode_cap
    """
    graph = AgentGraph()
    with personas_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for idx, row in enumerate(rows):
        username: str = row.get("username") or f"user_{idx}"
        description: str = row.get("description", "")
        user_char: str = row.get("user_char", "")

        # Require explicit persona columns; do not infer
        primary = row.get("primary_label")
        label_mode_cap = row.get("label_mode_cap")
        allowed_raw = row.get("allowed_labels")
        if primary is None or label_mode_cap is None or allowed_raw is None:
            raise ValueError(
                "Persona CSV must include `primary_label`, `label_mode_cap`, and `allowed_labels`."
            )
        try:
            allowed = json.loads(allowed_raw)
        except Exception as exc:
            raise ValueError(
                f"Invalid JSON in `allowed_labels` for username={username}"
            ) from exc

        # Optional per-persona emission configuration
        emission_params = None
        emission_params_raw = row.get("emission_params_json")
        if emission_params_raw:
            try:
                parsed = json.loads(emission_params_raw)
                if isinstance(parsed, dict):
                    emission_params = {str(k): float(v) for k, v in parsed.items()}
            except Exception:
                emission_params = None

        # Optional per-persona label pair preferences
        pair_probs = None
        pair_probs_raw = row.get("pair_probs_json")
        if pair_probs_raw:
            try:
                parsed = json.loads(pair_probs_raw)
                if isinstance(parsed, dict):
                    pair_probs = {str(k): float(v) for k, v in parsed.items()}
            except Exception:
                pair_probs = None

        persona_cfg = PersonaConfig(
            persona_id=f"{primary}_{idx:04d}",
            primary_label=str(primary),
            allowed_labels=allowed,
            label_mode_cap=str(label_mode_cap),
            benign_on_none_prob=0.6,
            max_labels_per_post=2,
            emission_probs=emission_params,
            pair_probs=pair_probs,
        )

        user_info = UserInfo(
            name=username,
            description=description,
            profile={"other_info": {"user_profile": user_char}},
            recsys_type="twitter",
        )
        agent = ExtendedSocialAgent(
            agent_id=idx,
            user_info=user_info,
            channel=channel,
            model=model,
            agent_graph=graph,
            available_actions=available_actions,
            persona_cfg=persona_cfg,
            emission_policy=emission_policy,
            sidecar_logger=sidecar_logger,
            run_seed=run_seed,
            **extended_kwargs,
        )
        graph.add_agent(agent)
    return graph


