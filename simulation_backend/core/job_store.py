"""
In-memory job store for simulation tasks.

Each simulation run is tracked as a SimulationStatus object keyed by job_id.
The store is intentionally lightweight – no external broker required.
For production use, replace with Redis or a proper task queue.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from typing import Callable, Coroutine, Dict, Optional

from simulation_backend.core.models import SimulationMode, SimulationStatus


class JobStore:
    def __init__(self) -> None:
        self._jobs: Dict[str, SimulationStatus] = {}
        self._tasks: Dict[str, asyncio.Task] = {}

    # ── CRUD ───────────────────────────────────────────────────────────────

    def create(self, mode: SimulationMode) -> SimulationStatus:
        job = SimulationStatus(
            job_id=str(uuid.uuid4()),
            status="queued",
            mode=mode,
        )
        self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Optional[SimulationStatus]:
        return self._jobs.get(job_id)

    def list_all(self) -> list[SimulationStatus]:
        return list(self._jobs.values())

    def update(
        self,
        job_id: str,
        *,
        status: Optional[str] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        results: Optional[dict] = None,
    ) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        if status is not None:
            job.status = status
        if progress is not None:
            job.progress_percentage = progress
        if message is not None:
            job.message = message
        if results is not None:
            job.results = results
        job.updated_at = datetime.utcnow()

    # ── Task management ────────────────────────────────────────────────────

    def submit(
        self,
        job_id: str,
        coro: Coroutine,
    ) -> asyncio.Task:
        """Schedule a coroutine as a background asyncio task."""
        loop = asyncio.get_event_loop()
        task = loop.create_task(coro)
        self._tasks[job_id] = task

        def _on_done(t: asyncio.Task) -> None:
            exc = t.exception()
            if exc:
                self.update(
                    job_id,
                    status="failed",
                    message=str(exc),
                )
            self._tasks.pop(job_id, None)

        task.add_done_callback(_on_done)
        return task

    def cancel(self, job_id: str) -> bool:
        task = self._tasks.get(job_id)
        if task and not task.done():
            task.cancel()
            return True
        return False


# ── Singleton ──────────────────────────────────────────────────────────────────
_job_store: Optional[JobStore] = None


def get_job_store() -> JobStore:
    global _job_store
    if _job_store is None:
        _job_store = JobStore()
    return _job_store
