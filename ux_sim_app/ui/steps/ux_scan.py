"""steps/ux_scan.py — Step 3: UX scan + NotebookLM configuration helpers."""
from __future__ import annotations

import json
import traceback
import uuid
from datetime import datetime

from .shared import (
    _run, _captcha_sessions,
    scan_website, nlm, UX_CATEGORIES,
    logger,
)


# ── NotebookLM helpers ────────────────────────────────────────────────────────

def nlm_authenticate():
    """Trigger NotebookLM OAuth. Returns (status_message, notebook_label)."""
    try:
        ok, msg = _run(nlm.authenticate_notebooklm())
        state = nlm.get_state()
        nb_label = state.get("notebook_title") or ("✅ Authenticated" if ok else "⏳ Awaiting login")
        return msg, nb_label
    except Exception as e:
        return f"❌ Auth error: {e}", "—"


def nlm_connect(notebook_id_or_title: str):
    """Connect to or create a NotebookLM notebook."""
    try:
        ok, msg = _run(nlm.connect_notebook(notebook_id_or_title or "UX Best Practices"))
        state = nlm.get_state()
        nb_label = state.get("notebook_title") or "—"
        return msg, nb_label
    except Exception as e:
        return f"❌ Connect error: {e}", "—"


def nlm_disconnect():
    """Disconnect from NotebookLM."""
    try:
        _run(nlm.disconnect())
        return "Disconnected from NotebookLM.", "—"
    except Exception as e:
        return f"Error: {e}", "—"


def nlm_run_config_loop(business_context: str, personas_json: str):
    """Run the UX configuration loop across all 15 UX categories. Yields progress."""
    persona_summary = ""
    try:
        pds = json.loads(personas_json or "[]")
        if isinstance(pds, list) and pds:
            names = [p.get("name", "") for p in pds[:3]]
            ages = [str(p.get("age", "")) for p in pds[:3]]
            persona_summary = f"Personas include: {', '.join(names)} (ages {', '.join(ages)})"
    except Exception:
        pass

    state = nlm.get_state()
    source = "NotebookLM" if (state["authenticated"] and state["notebook_id"]) else "built-in knowledge base"
    total = len(UX_CATEGORIES)

    yield f"⏳ Starting UX configuration loop ({total} categories) using {source}...", "{}"

    results: dict = {}
    for i, cat in enumerate(UX_CATEGORIES):
        yield f"⏳ [{i+1}/{total}] Querying: {cat['name']}...", json.dumps(results)
        try:
            bp = _run(nlm.query_category_best_practices(
                cat,
                business_context=business_context or "",
                persona_summary=persona_summary,
            ))
            results[cat["id"]] = bp
        except Exception as exc:
            results[cat["id"]] = f"Error: {exc}"

    yield f"✅ Configuration loop complete. {total} categories loaded from {source}.", json.dumps(results)


# ── Step 3: UX Scan ───────────────────────────────────────────────────────────

def step_ux_scan(url: str):
    """Run the full UX scan.
    MUST yield exactly 2 values on every yield.
    """
    _EMPTY_UX = "{}"

    if not url or not url.strip():
        yield "❌ Please enter a URL first (Tab 1).", _EMPTY_UX
        return

    # Show which knowledge source will be used for the AI critique
    _nlm_state = nlm.get_state()
    _nlm_connected = bool(_nlm_state.get("notebook_id"))
    _nb_title = _nlm_state.get("notebook_title", "")
    _kb_source = (
        f"\U0001f4d3 NotebookLM: {_nb_title}"
        if _nlm_connected
        else "\U0001f4da Built-in knowledge base"
    )
    yield f"⏳ Running UX scan — knowledge source: {_kb_source}...", _EMPTY_UX

    try:
        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        _ss_path = _captcha_sessions.get(url.strip())
        ux_report = _run(scan_website(url.strip(), run_id, storage_state_path=_ss_path))

        ux_json = json.dumps({
            "url": ux_report.url,
            "run_id": run_id,
            "overall_score": ux_report.overall_score,
            "overall_summary": ux_report.overall_summary,
            "strengths": ux_report.strengths,
            "weaknesses": ux_report.weaknesses,
            "screenshots": ux_report.screenshots,
            "heuristic_checks": ux_report.heuristic_checks,
            "dimensions": [
                {
                    "name": d.name,
                    "score": d.score,
                    "feedback": d.feedback,
                    "issues": [
                        {
                            "category": i.category,
                            "severity": i.severity,
                            "description": i.description,
                            "recommendation": i.recommendation,
                        }
                        for i in d.issues
                    ],
                }
                for d in ux_report.dimensions
            ],
            "recommendations": ux_report.recommendations,
            "error": ux_report.error,
        }, indent=2)

        score = ux_report.overall_score
        heur = ux_report.heuristic_checks.get("heuristic_score", 0)
        dims = len(ux_report.dimensions)
        issues = sum(len(d.issues) for d in ux_report.dimensions)
        yield (
            f"✅ UX scan complete. Score: {score}/100 · Heuristic: {heur}/100 · "
            f"{dims} dimensions · {issues} issues found.",
            ux_json,
        )

    except Exception as exc:
        yield f"❌ UX scan error: {exc}\n{traceback.format_exc()}", _EMPTY_UX
