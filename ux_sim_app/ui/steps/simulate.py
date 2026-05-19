"""steps/simulate.py — Step 2: Run simulation modes in parallel."""
from __future__ import annotations

import asyncio
import json

from .shared import (
    _run, _captcha_sessions,
    Persona, SimulationResult,
    run_mode1, run_mode2, run_mode3,
    logger,
)


def step_run_simulations(
    url: str,
    personas_json: str,
    image_urls_json: str,
    run_mode1_flag: bool,
    run_mode2_flag: bool,
    run_mode3_flag: bool,
    content_items_text: str,
):
    """Run selected simulation modes in parallel.
    MUST yield exactly 3 values on every yield:
      (status_text, sim_results_json, video_recordings_json)
    """
    _EMPTY_RESULTS = "{}"
    _EMPTY_VIDS = "[]"

    url = url or ""
    personas_json = personas_json or ""
    image_urls_json = image_urls_json or ""
    content_items_text = content_items_text or ""

    if not personas_json or personas_json in ("{}", ""):
        yield "❌ Please generate personas first (Tab 1).", _EMPTY_RESULTS, _EMPTY_VIDS
        return

    try:
        persona_dicts = json.loads(personas_json)
        personas = [Persona(**p) for p in persona_dicts]
    except Exception as exc:
        yield f"❌ Failed to parse personas: {exc}", _EMPTY_RESULTS, _EMPTY_VIDS
        return

    try:
        image_urls = json.loads(image_urls_json) if image_urls_json else []
    except Exception:
        image_urls = []

    results: list[SimulationResult] = []
    all_video_recordings: list = []

    # ── Build coroutines to run in parallel ───────────────────────────────────
    coros = {}
    _ss_path = _captcha_sessions.get(url.strip())

    if run_mode2_flag:
        coros["mode2"] = run_mode2(personas, url.strip(), storage_state_path=_ss_path)

    if run_mode3_flag:
        if not image_urls:
            yield "⚠️ Mode 3: No images found on the website. Skipping.", _EMPTY_RESULTS, _EMPTY_VIDS
        else:
            coros["mode3"] = run_mode3(personas, image_urls)

    if run_mode1_flag:
        items = [c.strip() for c in content_items_text.strip().split("\n---\n") if c.strip()]
        if not items:
            items = [content_items_text.strip()] if content_items_text.strip() else []
        if not items:
            yield "⚠️ Mode 1: No content items provided. Skipping.", _EMPTY_RESULTS, _EMPTY_VIDS
        else:
            coros["mode1"] = run_mode1(personas, items)

    if not coros:
        yield "⚠️ No simulation modes were run. Please select at least one mode.", _EMPTY_RESULTS, _EMPTY_VIDS
        return

    mode_labels = {
        "mode1": "Mode 1 – Content Simulation",
        "mode2": "Mode 2 – Browser Usability Simulation",
        "mode3": "Mode 3 – Visual Simulation",
    }
    running = ", ".join(mode_labels[k] for k in coros)
    yield f"⏳ Running in parallel: {running}...", _EMPTY_RESULTS, _EMPTY_VIDS

    async def _run_all():
        keys = list(coros.keys())
        raw = await asyncio.gather(*[coros[k] for k in keys], return_exceptions=True)
        return dict(zip(keys, raw))

    try:
        mode_results = _run(_run_all())
    except Exception as exc:
        yield f"❌ Simulation error: {exc}", _EMPTY_RESULTS, _EMPTY_VIDS
        return

    summaries = []
    for key, r in mode_results.items():
        if isinstance(r, Exception):
            summaries.append(f"⚠️ {mode_labels[key]} error: {r}")
            continue
        if key == "mode2":
            results.append(r)
            all_video_recordings.extend(r.video_recordings or [])
            conv = int(r.aggregate.get("conversion_intent_rate", 0) * 100)
            n_vids = len(r.video_recordings or [])
            summaries.append(f"✅ Mode 2: Conversion intent {conv}%, {n_vids} recording(s)")
        elif key == "mode3":
            results.append(r)
            avg = r.aggregate.get("average_resonance", 0)
            summaries.append(f"✅ Mode 3: Avg resonance {avg}/10")
        elif key == "mode1":
            if isinstance(r, list):
                results.extend(r)
                avg_eng = sum(x.aggregate.get("engagement_rate", 0) for x in r) / max(len(r), 1)
            else:
                results.append(r)
                avg_eng = r.aggregate.get("engagement_rate", 0)
            summaries.append(f"✅ Mode 1: Avg engagement {int(avg_eng * 100)}%")

    if not results:
        yield "⚠️ All simulation modes failed. Check logs for details.", _EMPTY_RESULTS, _EMPTY_VIDS
        return

    results_json = json.dumps([r.to_dict() for r in results], indent=2)
    vids_json = json.dumps(all_video_recordings, indent=2)
    yield "\n".join(summaries), results_json, vids_json
