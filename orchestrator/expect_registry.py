from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class ExpectedRecord:
    step_idx: int
    tokens: List[str]
    insertion_fallback: bool = False


class ExpectRegistry:
    r"""Async-safe registry for expected tokens per agent and step."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._store: Dict[int, ExpectedRecord] = {}

    async def set_expected(self, agent_id: int, step_idx: int, tokens: List[str]) -> None:
        async with self._lock:
            self._store[agent_id] = ExpectedRecord(step_idx=step_idx, tokens=list(tokens or []))

    async def get_expected(self, agent_id: int) -> Optional[ExpectedRecord]:
        async with self._lock:
            return self._store.get(agent_id)

    async def note_insertion(self, agent_id: int) -> None:
        async with self._lock:
            rec = self._store.get(agent_id)
            if rec:
                rec.insertion_fallback = True

    async def consume(self, agent_id: int) -> Optional[ExpectedRecord]:
        async with self._lock:
            return self._store.pop(agent_id, None)


