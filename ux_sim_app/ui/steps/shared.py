"""
shared.py
=========
Shared imports, async helpers, and module-level state used by all step modules.
Importing this module is the single source of truth for the background event loop
and the CAPTCHA session registry.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path

# Ensure the app package is importable when run as a module
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import ux_sim_app.core.config as cfg
from ux_sim_app.core.config import (
    OPENAI_API_KEY, BROWSERBASE_API_KEY, REPORTS_DIR, TEXT_MODEL, VISION_MODEL
)
from ux_sim_app.core.scraper import scrape
from ux_sim_app.core.personas import generate_personas, Persona
from ux_sim_app.modes.runner import run_mode1, run_mode2, run_mode3, SimulationResult
from ux_sim_app.ux.scanner import scan_website
from ux_sim_app.report.slide_generator import build_report_data, render_html, html_to_pdf, IssueSlide
from ux_sim_app.report.redesign_client import generate_redesign, sanitise_for_embed
from ux_sim_app.integrations import notebooklm as nlm
from ux_sim_app.integrations.notebooklm import UX_CATEGORIES
from ux_sim_app.integrations.real_world_data import gather_and_synthesize
from ux_sim_app.core.captcha_guard import preflight_check, CaptchaGuardResult

logger = logging.getLogger(__name__)

# ── CAPTCHA session registry ──────────────────────────────────────────────────
# Maps URL → storage_state_path (Playwright JSON) from the pre-flight check.
# Downstream Mode 2 and UX scan look up this dict to inherit cleared sessions.
_captcha_sessions: dict = {}

# ── Persistent background event loop ─────────────────────────────────────────
# A single background thread hosts a persistent event loop that is reused for
# every _run() call. This avoids spawning a new thread + loop per LLM call.

_bg_loop: asyncio.AbstractEventLoop | None = None
_bg_thread: threading.Thread | None = None
_bg_lock = threading.Lock()


def _get_bg_loop() -> asyncio.AbstractEventLoop:
    """Return the shared background event loop, creating it on first call."""
    global _bg_loop, _bg_thread
    with _bg_lock:
        if _bg_loop is None or not _bg_loop.is_running():
            _bg_loop = asyncio.new_event_loop()

            def _run_forever():
                asyncio.set_event_loop(_bg_loop)
                _bg_loop.run_forever()

            _bg_thread = threading.Thread(
                target=_run_forever, daemon=True, name="oasis-bg-loop"
            )
            _bg_thread.start()
    return _bg_loop


def _run(coro):
    """Submit a coroutine to the shared background event loop and block until done."""
    import concurrent.futures
    loop = _get_bg_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()
