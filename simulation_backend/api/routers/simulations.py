"""
/api/v1/simulations  –  Simulation job management endpoints.
"""
from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException

from simulation_backend.core.job_store import JobStore, get_job_store
from simulation_backend.core.models import (
    SimulationStatus,
    StartSimulationRequest,
)
from simulation_backend.core.orchestrator import (
    SimulationOrchestrator,
    get_orchestrator,
)

log = logging.getLogger("simulation_backend.api.simulations")
router = APIRouter(prefix="/api/v1/simulations", tags=["Simulations"])


@router.post(
    "",
    response_model=SimulationStatus,
    status_code=202,
    summary="Start a new simulation (async)",
)
def start_simulation(
    body: StartSimulationRequest,
    orchestrator: SimulationOrchestrator = Depends(get_orchestrator),
    job_store: JobStore = Depends(get_job_store),
) -> SimulationStatus:
    """
    Enqueues a simulation job. Returns immediately with a *job_id* that can
    be polled via GET /api/v1/simulations/{job_id}.
    """
    job_id = orchestrator.enqueue(body)
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=500, detail="Job creation failed")
    return job


@router.get(
    "",
    response_model=List[SimulationStatus],
    summary="List all simulation jobs",
)
def list_simulations(
    job_store: JobStore = Depends(get_job_store),
) -> List[SimulationStatus]:
    return job_store.list_all()


@router.get(
    "/{job_id}",
    response_model=SimulationStatus,
    summary="Poll the status of a simulation job",
)
def get_simulation(
    job_id: str,
    job_store: JobStore = Depends(get_job_store),
) -> SimulationStatus:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.delete(
    "/{job_id}",
    summary="Cancel a running simulation job",
)
def cancel_simulation(
    job_id: str,
    job_store: JobStore = Depends(get_job_store),
) -> dict:
    cancelled = job_store.cancel(job_id)
    return {"success": cancelled, "job_id": job_id}
