"""
Mode 2 – Browser-Action Simulation (Website Interaction)

Each persona gets its own browser session and navigates the target URL.
The persona's LLM decides which browser actions to take based on its
system prompt and the current page DOM.

Execution strategy:
  1. Primary: local Playwright sessions, up to max_browser_sessions in parallel.
  2. Fallback: if Playwright is unavailable OR the page is unreachable after
     retries, delegate to Browserbase MCP (requires BROWSERBASE_API_KEY).
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

import httpx

from simulation_backend.core.models import (
    FocusGroup,
    PersonaProfile,
    SimulationMode,
    SimulationParameters,
)
from simulation_backend.core.prompt_adapter import build_system_prompt
from simulation_backend.core.settings import get_settings
from simulation_backend.modes.browser_session import (
    BrowserSession,
    BrowserSessionPool,
    _PLAYWRIGHT_AVAILABLE,
)
from simulation_backend.modes.browserbase_client import BrowserbaseMCPClient

log = logging.getLogger("simulation_backend.mode2")

# Maximum browser tool-call rounds per persona
_MAX_STEPS = 8

# ── Public entry point ─────────────────────────────────────────────────────────

async def run_browser_simulation(
    group: FocusGroup,
    target_url: str,
    task_description: str,
    parameters: SimulationParameters,
    progress_callback=None,
) -> Dict[str, Any]:
    settings = get_settings()
    total = len(group.personas)
    pool = BrowserSessionPool(max_sessions=settings.max_browser_sessions)

    async def _run_one(idx: int, persona: PersonaProfile) -> Dict[str, Any]:
        async with pool.semaphore:
            use_fallback = (
                not _PLAYWRIGHT_AVAILABLE
                or not parameters.use_browserbase_fallback is False
            )
            try:
                result = await _playwright_persona_session(
                    persona=persona,
                    target_url=target_url,
                    task_description=task_description,
                    pool=pool,
                    settings=settings,
                )
            except Exception as exc:
                log.warning(
                    "Playwright failed for %s (%s), trying Browserbase fallback",
                    persona.name, exc,
                )
                if parameters.use_browserbase_fallback and settings.browserbase_api_key:
                    result = await _browserbase_persona_session(
                        persona=persona,
                        target_url=target_url,
                        task_description=task_description,
                        api_key=settings.browserbase_api_key,
                        settings=settings,
                    )
                else:
                    result = {
                        "persona_id": persona.id,
                        "persona_name": persona.name,
                        "status": "failed",
                        "error": str(exc),
                        "steps": [],
                        "summary": "",
                    }
            if progress_callback:
                pct = round((idx + 1) / total * 100, 1)
                coro = progress_callback(pct, persona.name)
                if asyncio.iscoroutine(coro):
                    await coro
            return result

    tasks = [_run_one(i, p) for i, p in enumerate(group.personas)]
    responses = await asyncio.gather(*tasks, return_exceptions=False)
    await pool.close_all()

    return {
        "mode": SimulationMode.BROWSER,
        "target_url": target_url,
        "persona_responses": responses,
        "aggregate": _aggregate_browser_responses(responses),
    }


# ── Playwright path ────────────────────────────────────────────────────────────

async def _playwright_persona_session(
    persona: PersonaProfile,
    target_url: str,
    task_description: str,
    pool: BrowserSessionPool,
    settings,
) -> Dict[str, Any]:
    session = pool.get_session(persona.id.__hash__() % 10_000)
    steps: List[Dict[str, Any]] = []

    try:
        await session.start()
        nav = await session.navigate(target_url)
        steps.append({"action": "navigate", "result": nav})

        dom_result = await session.extract_dom()
        dom_snippet = dom_result.get("dom", "")[:4000]

        # Ask the LLM what to do next
        for _round in range(_MAX_STEPS):
            next_action = await _decide_browser_action(
                persona=persona,
                task_description=task_description,
                dom_snippet=dom_snippet,
                steps=steps,
                settings=settings,
            )
            action_type = next_action.get("action", "done")
            if action_type == "done":
                break

            result = await _execute_browser_action(session, next_action)
            steps.append({"action": action_type, "args": next_action, "result": result})

            # Refresh DOM after action
            dom_result = await session.extract_dom()
            dom_snippet = dom_result.get("dom", "")[:4000]

        summary = await _summarise_session(
            persona=persona,
            task_description=task_description,
            steps=steps,
            settings=settings,
        )
    finally:
        await session.close()
        await pool.remove_session(session.agent_id)

    return {
        "persona_id": persona.id,
        "persona_name": persona.name,
        "status": "completed",
        "driver": "playwright",
        "steps": steps,
        "summary": summary,
    }


async def _decide_browser_action(
    persona: PersonaProfile,
    task_description: str,
    dom_snippet: str,
    steps: List[Dict],
    settings,
) -> Dict[str, Any]:
    system_prompt = build_system_prompt(persona, SimulationMode.BROWSER)
    history = json.dumps(
        [{"action": s["action"], "result": str(s.get("result", ""))[:200]}
         for s in steps],
        indent=2,
    )
    user_msg = (
        f"Task: {task_description}\n\n"
        f"Current page DOM (truncated):\n{dom_snippet}\n\n"
        f"Steps taken so far:\n{history}\n\n"
        "What is your next action? Use the browser_action tool."
    )
    schema = _browser_action_schema()
    payload = {
        "model": settings.default_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        "tools": [{"type": "function", "function": schema}],
        "tool_choice": {"type": "function",
                        "function": {"name": "browser_action"}},
        "temperature": 0.5,
    }
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{settings.openai_base_url.rstrip('/')}/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {settings.openai_api_key}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
        data = resp.json()
        tool_calls = (
            data["choices"][0].get("message", {}).get("tool_calls", [])
        )
        if tool_calls:
            return json.loads(tool_calls[0]["function"]["arguments"])
    except Exception as exc:
        log.warning("LLM browser-action decision failed: %s", exc)
    return {"action": "done", "reasoning": "LLM unavailable"}


def _browser_action_schema() -> dict:
    return {
        "name": "browser_action",
        "description": "Choose the next browser action for this persona.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["navigate", "click", "type_text", "scroll", "done"],
                    "description": "The browser action to perform.",
                },
                "url": {"type": "string", "description": "URL for navigate."},
                "selector": {
                    "type": "string",
                    "description": "CSS selector for click/type_text.",
                },
                "text": {
                    "type": "string",
                    "description": "Text to type for type_text action.",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Why this action was chosen.",
                },
            },
            "required": ["action", "reasoning"],
        },
    }


async def _execute_browser_action(
    session: BrowserSession, action_args: Dict[str, Any]
) -> Dict[str, Any]:
    action = action_args.get("action", "done")
    if action == "navigate":
        return await session.navigate(action_args.get("url", ""))
    elif action == "click":
        return await session.click(action_args.get("selector", "body"))
    elif action == "type_text":
        return await session.type_text(
            action_args.get("selector", "input"),
            action_args.get("text", ""),
        )
    elif action == "scroll":
        # Simple JS scroll
        try:
            await session._page.evaluate("window.scrollBy(0, 600)")
            return {"status": "success", "action": "scroll"}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}
    return {"status": "done"}


async def _summarise_session(
    persona: PersonaProfile,
    task_description: str,
    steps: List[Dict],
    settings,
) -> str:
    system_prompt = build_system_prompt(persona, SimulationMode.BROWSER)
    history = json.dumps(
        [{"action": s["action"], "result": str(s.get("result", ""))[:300]}
         for s in steps],
        indent=2,
    )
    user_msg = (
        f"You have just completed browsing the website for the task: "
        f"'{task_description}'.\n\n"
        f"Here is what you did:\n{history}\n\n"
        "Summarise your experience as this persona: what you found, "
        "how you felt, and whether you completed the task."
    )
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{settings.openai_base_url.rstrip('/')}/chat/completions",
                json={
                    "model": settings.default_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_msg},
                    ],
                    "temperature": 0.7,
                },
                headers={
                    "Authorization": f"Bearer {settings.openai_api_key}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as exc:
        return f"Summary unavailable: {exc}"


# ── Browserbase fallback path ──────────────────────────────────────────────────

async def _browserbase_persona_session(
    persona: PersonaProfile,
    target_url: str,
    task_description: str,
    api_key: str,
    settings,
) -> Dict[str, Any]:
    steps: List[Dict[str, Any]] = []
    try:
        async with BrowserbaseMCPClient(api_key=api_key) as client:
            start_result = await client.start()
            steps.append({"action": "start", "result": start_result})

            nav_result = await client.navigate(target_url)
            steps.append({"action": "navigate", "result": nav_result})

            await asyncio.sleep(2)

            obs_result = await client.observe()
            steps.append({"action": "observe", "result": obs_result})

            act_result = await client.act("Scroll down to explore the page")
            steps.append({"action": "act", "result": act_result})

        summary = await _summarise_session(
            persona=persona,
            task_description=task_description,
            steps=steps,
            settings=settings,
        )
        return {
            "persona_id": persona.id,
            "persona_name": persona.name,
            "status": "completed",
            "driver": "browserbase_mcp",
            "steps": steps,
            "summary": summary,
        }
    except Exception as exc:
        return {
            "persona_id": persona.id,
            "persona_name": persona.name,
            "status": "failed",
            "driver": "browserbase_mcp",
            "error": str(exc),
            "steps": steps,
            "summary": "",
        }


# ── Aggregate ──────────────────────────────────────────────────────────────────

def _aggregate_browser_responses(responses: List[Dict[str, Any]]) -> Dict[str, Any]:
    completed = sum(1 for r in responses if r.get("status") == "completed")
    failed = len(responses) - completed
    drivers: Dict[str, int] = {}
    for r in responses:
        d = r.get("driver", "unknown")
        drivers[d] = drivers.get(d, 0) + 1
    return {
        "total_personas": len(responses),
        "completed": completed,
        "failed": failed,
        "driver_distribution": drivers,
    }
