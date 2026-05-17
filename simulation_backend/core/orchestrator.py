"""
SimulationOrchestrator – routes simulation jobs to the correct mode handler
and updates the JobStore with progress and results.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Coroutine, Dict, Optional

from simulation_backend.core.job_store import JobStore
from simulation_backend.core.models import (
    FocusGroup,
    SimulationMode,
    SimulationParameters,
    StartSimulationRequest,
)
from simulation_backend.core.persona_manager import PersonaManager

log = logging.getLogger("simulation_backend.orchestrator")


class SimulationOrchestrator:
    def __init__(self, persona_manager: PersonaManager, job_store: JobStore) -> None:
        self._pm = persona_manager
        self._js = job_store

    # ── Public ─────────────────────────────────────────────────────────────

    def enqueue(self, request: StartSimulationRequest) -> str:
        """Create a job record and schedule the simulation coroutine."""
        job = self._js.create(mode=request.parameters.mode)
        coro = self._run(job.job_id, request)
        self._js.submit(job.job_id, coro)
        return job.job_id

    # ── Internal ───────────────────────────────────────────────────────────

    async def _run(self, job_id: str, request: StartSimulationRequest) -> None:
        self._js.update(job_id, status="running", progress=0.0,
                        message="Starting simulation…")

        group = self._pm.get_group(request.focus_group_id)
        if group is None:
            self._js.update(
                job_id,
                status="failed",
                message=f"Focus group '{request.focus_group_id}' not found.",
            )
            return

        async def _progress(pct: float, name: str) -> None:
            self._js.update(
                job_id,
                progress=pct,
                message=f"Processing persona: {name} ({pct:.0f}%)",
            )

        try:
            results = await self._dispatch(
                group=group,
                request=request,
                progress_callback=_progress,
            )
            self._js.update(
                job_id,
                status="completed",
                progress=100.0,
                message="Simulation complete.",
                results=results,
            )
        except Exception as exc:
            log.exception("Simulation %s failed: %s", job_id, exc)
            self._js.update(
                job_id,
                status="failed",
                message=str(exc),
            )

    async def _dispatch(
        self,
        group: FocusGroup,
        request: StartSimulationRequest,
        progress_callback: Callable,
    ) -> Dict[str, Any]:
        mode = request.parameters.mode

        if mode == SimulationMode.CONTENT:
            from simulation_backend.modes.mode1_content import run_content_simulation
            return await run_content_simulation(
                group=group,
                content_payload=request.content_payload,
                content_type=request.content_type,
                parameters=request.parameters,
                progress_callback=progress_callback,
            )

        elif mode == SimulationMode.BROWSER:
            from simulation_backend.modes.mode2_browser import run_browser_simulation
            target_url = (
                request.parameters.target_url
                or request.content_payload
            )
            return await run_browser_simulation(
                group=group,
                target_url=target_url,
                task_description=request.content_payload,
                parameters=request.parameters,
                progress_callback=progress_callback,
            )

        elif mode == SimulationMode.VISUAL:
            from simulation_backend.modes.mode3_visual import run_visual_simulation
            return await run_visual_simulation(
                group=group,
                content_payload=request.content_payload,
                parameters=request.parameters,
                progress_callback=progress_callback,
            )

        else:
            raise ValueError(f"Unknown simulation mode: {mode}")


# ── Singleton ──────────────────────────────────────────────────────────────────
_orchestrator: Optional[SimulationOrchestrator] = None


def get_orchestrator() -> SimulationOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        from simulation_backend.core.persona_manager import get_persona_manager
        from simulation_backend.core.job_store import get_job_store
        _orchestrator = SimulationOrchestrator(
            persona_manager=get_persona_manager(),
            job_store=get_job_store(),
        )
    return _orchestrator
