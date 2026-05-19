"""
/api/v1/personas  –  Focus group & persona management endpoints.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query

from simulation_backend.core.models import (
    FocusGroup,
    GeneratePersonasRequest,
    GeneratePersonasResponse,
    ListFocusGroupsResponse,
)
from simulation_backend.core.persona_manager import (
    PersonaManager,
    get_persona_manager,
)

log = logging.getLogger("simulation_backend.api.personas")
router = APIRouter(prefix="/api/v1/personas", tags=["Personas"])


@router.post(
    "/generate",
    response_model=GeneratePersonasResponse,
    summary="Generate a new focus group of personas via LLM",
)
async def generate_personas(
    body: GeneratePersonasRequest,
    pm: PersonaManager = Depends(get_persona_manager),
) -> GeneratePersonasResponse:
    """
    Calls the configured LLM to generate *num_personas* realistic personas
    based on the supplied business description and customer profile.
    Returns the full focus group immediately (synchronous generation).
    """
    try:
        group = await pm.generate_personas(
            business_description=body.business_description,
            customer_profile=body.customer_profile,
            num_personas=body.num_personas,
            api_key=body.blablador_api_key,
        )
    except Exception as exc:
        log.exception("Persona generation failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return GeneratePersonasResponse(
        job_id=group.id,
        status="completed",
        focus_group_id=group.id,
        personas=group.personas,
    )


@router.get(
    "",
    response_model=ListFocusGroupsResponse,
    summary="List all available focus groups",
)
def list_focus_groups(
    pm: PersonaManager = Depends(get_persona_manager),
) -> ListFocusGroupsResponse:
    return ListFocusGroupsResponse(focus_groups=pm.list_groups())


@router.get(
    "/{group_id}",
    response_model=FocusGroup,
    summary="Get a specific focus group with all persona details",
)
def get_focus_group(
    group_id: str,
    pm: PersonaManager = Depends(get_persona_manager),
) -> FocusGroup:
    group = pm.get_group(group_id)
    if group is None:
        raise HTTPException(status_code=404, detail="Focus group not found")
    return group


@router.delete(
    "/{group_id}",
    summary="Delete a focus group",
)
def delete_focus_group(
    group_id: str,
    pm: PersonaManager = Depends(get_persona_manager),
) -> dict:
    deleted = pm.delete_group(group_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Focus group not found")
    return {"success": True, "message": f"Focus group {group_id} deleted."}
