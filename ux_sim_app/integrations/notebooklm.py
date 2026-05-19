"""NotebookLM MCP Integration for the OASIS UX Simulation App.

Uses the notebooklm-py library (teng-lin/notebooklm-py) to:
  1. Authenticate with Google NotebookLM via `notebooklm login` CLI
     (opens a browser for Google OAuth, saves Playwright storage state to
     ~/.notebooklm/storage_state.json)
  2. Create or connect to a UX Best Practices notebook
  3. Run an agent configuration loop that queries the notebook for
     UX best practices per heuristic category, personalising the
     UX scan based on the business context and persona profiles

The correct auth flow is:
  - `notebooklm login`  →  browser OAuth  →  storage_state.json saved
  - `NotebookLMClient.from_storage()`  →  loads auth from file
  - `async with client:` context manager for each API call

UX Heuristic Categories (based on Nielsen's 10 + extended audit):
  1.  Visibility of System Status
  2.  Match Between System and Real World
  3.  User Control and Freedom
  4.  Consistency and Standards
  5.  Error Prevention
  6.  Recognition Rather Than Recall
  7.  Flexibility and Efficiency of Use
  8.  Aesthetic and Minimalist Design
  9.  Help Users Recognise, Diagnose, and Recover from Errors
  10. Help and Documentation
  11. Accessibility and Inclusive Design
  12. Mobile Responsiveness
  13. Visual Hierarchy and Cognitive Load
  14. Trust Signals and Social Proof
  15. Conversion Optimisation and CTAs
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── UX heuristic categories ────────────────────────────────────────────────────
UX_CATEGORIES: List[Dict[str, str]] = [
    {
        "id": "visibility",
        "name": "Visibility of System Status",
        "description": "Does the site keep users informed about what is happening through feedback?",
        "query": "What are the best practices for visibility of system status in web UX? What are common mistakes and how to fix them?",
    },
    {
        "id": "real_world",
        "name": "Match Between System and Real World",
        "description": "Does the site use language and concepts familiar to the user?",
        "query": "What are best practices for matching system language to real-world user expectations in web design? Common mistakes and fixes?",
    },
    {
        "id": "user_control",
        "name": "User Control and Freedom",
        "description": "Can users easily undo actions and navigate freely?",
        "query": "What are best practices for user control and freedom in UX? How to design clear exit points and undo mechanisms?",
    },
    {
        "id": "consistency",
        "name": "Consistency and Standards",
        "description": "Does the site follow platform conventions and internal consistency?",
        "query": "What are best practices for consistency and standards in web UI design? Common inconsistency mistakes and solutions?",
    },
    {
        "id": "error_prevention",
        "name": "Error Prevention",
        "description": "Does the design prevent problems before they occur?",
        "query": "What are UX best practices for error prevention in web forms and navigation? Common mistakes and design patterns?",
    },
    {
        "id": "recognition",
        "name": "Recognition Rather Than Recall",
        "description": "Are options, actions, and information visible rather than memorised?",
        "query": "What are best practices for recognition over recall in UX? How to design menus, navigation, and CTAs to reduce cognitive load?",
    },
    {
        "id": "flexibility",
        "name": "Flexibility and Efficiency of Use",
        "description": "Does the site serve both novice and expert users?",
        "query": "What are best practices for flexibility and efficiency of use in web UX? How to design for both new and returning users?",
    },
    {
        "id": "aesthetic",
        "name": "Aesthetic and Minimalist Design",
        "description": "Does the design avoid irrelevant or rarely needed information?",
        "query": "What are best practices for aesthetic minimalist web design? How to reduce visual noise and improve focus?",
    },
    {
        "id": "error_recovery",
        "name": "Error Recognition and Recovery",
        "description": "Are error messages clear and do they help users recover?",
        "query": "What are best practices for error messages and recovery flows in web UX? Common mistakes and effective patterns?",
    },
    {
        "id": "help",
        "name": "Help and Documentation",
        "description": "Is help easy to find and task-focused?",
        "query": "What are best practices for help documentation and onboarding in web UX? When and how to surface help content?",
    },
    {
        "id": "accessibility",
        "name": "Accessibility and Inclusive Design",
        "description": "Is the site usable by people with disabilities? WCAG compliance?",
        "query": "What are WCAG 2.1 best practices for web accessibility? Common accessibility mistakes and how to fix them?",
    },
    {
        "id": "mobile",
        "name": "Mobile Responsiveness",
        "description": "Does the site work well on mobile devices?",
        "query": "What are best practices for mobile-responsive web design? Common mobile UX mistakes and solutions?",
    },
    {
        "id": "visual_hierarchy",
        "name": "Visual Hierarchy and Cognitive Load",
        "description": "Is information structured to guide the eye and reduce mental effort?",
        "query": "What are best practices for visual hierarchy and reducing cognitive load in web design? Common mistakes and design patterns?",
    },
    {
        "id": "trust",
        "name": "Trust Signals and Social Proof",
        "description": "Does the site build credibility and trust with users?",
        "query": "What are best practices for trust signals, social proof, and credibility in web design? Common trust mistakes?",
    },
    {
        "id": "conversion",
        "name": "Conversion Optimisation and CTAs",
        "description": "Are calls-to-action clear, compelling, and well-placed?",
        "query": "What are best practices for CTA design and conversion optimisation in web UX? Common CTA mistakes and improvements?",
    },
]

# ── State ──────────────────────────────────────────────────────────────────────
_state: Dict[str, Any] = {
    "authenticated": False,
    "notebook_id": None,
    "notebook_title": None,
    "client": None,  # kept for API compatibility; not used (we use context managers)
    "best_practices_cache": {},  # category_id → best practices text
    "status": "Not connected",
}

_lock = threading.Lock()


def get_state() -> Dict[str, Any]:
    with _lock:
        return dict(_state)


def _update_state(**kwargs: Any) -> None:
    with _lock:
        _state.update(kwargs)


# ── Auth helpers ───────────────────────────────────────────────────────────────

def _get_storage_path(storage_path: Optional[str] = None) -> Path:
    """Return the Playwright storage state path used by `notebooklm login`."""
    if storage_path:
        return Path(storage_path)
    try:
        from notebooklm.paths import get_storage_path as _gsp  # type: ignore
        return _gsp()
    except Exception:
        return Path.home() / ".notebooklm" / "storage_state.json"


def _is_authenticated(storage_path: Optional[str] = None) -> bool:
    """Check if a valid storage_state.json exists from a previous `notebooklm login`."""
    sp = _get_storage_path(storage_path)
    if not sp.exists():
        return False
    try:
        data = json.loads(sp.read_text())
        cookies = data.get("cookies", [])
        # Must have at least one Google auth cookie
        return any(
            c.get("name") in ("SID", "HSID", "SSID", "APISID", "SAPISID", "__Secure-1PSID")
            and "google" in c.get("domain", "")
            for c in cookies
        )
    except Exception:
        return False


# ── Package check ──────────────────────────────────────────────────────────────

def check_notebooklm_installed() -> bool:
    """Check if notebooklm-py is installed."""
    try:
        import notebooklm  # noqa: F401
        return True
    except ImportError:
        return False


def install_notebooklm() -> tuple[bool, str]:
    """Install notebooklm-py if not present."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "notebooklm-py", "--quiet"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            return True, "notebooklm-py installed successfully."
        return False, f"Install failed: {result.stderr[:300]}"
    except Exception as e:
        return False, f"Install error: {e}"


# ── Auth and connection ────────────────────────────────────────────────────────

async def authenticate_notebooklm(storage_path: Optional[str] = None) -> tuple[bool, str]:
    """Authenticate with NotebookLM via `notebooklm login` CLI.

    The correct auth flow is:
      1. Run `notebooklm login` — opens a browser for Google OAuth
      2. User completes login and presses ENTER in the terminal
      3. Playwright storage state is saved to ~/.notebooklm/storage_state.json
      4. Subsequent calls use `NotebookLMClient.from_storage()` to load auth

    This function:
    - Checks for an existing valid storage state first (quick path)
    - If not found, launches `notebooklm login` in a background subprocess
    - Returns instructions to the user to complete the browser login

    Returns (already_authenticated, message).
    """
    if not check_notebooklm_installed():
        ok, msg = install_notebooklm()
        if not ok:
            return False, msg

    sp = _get_storage_path(storage_path)

    # Already authenticated — verify by loading the storage state
    if _is_authenticated(storage_path):
        try:
            from notebooklm import NotebookLMClient  # type: ignore
            async with await NotebookLMClient.from_storage(str(sp)) as client:
                # Quick connectivity check
                _ = await client.notebooks.list()
            _update_state(
                authenticated=True,
                status="✅ Authenticated with NotebookLM (existing session)",
            )
            return True, (
                "✅ Already authenticated — existing session found.\n\n"
                "Click **Connect Notebook** to link a notebook, then "
                "**Run Config Loop** to load best practices."
            )
        except Exception as e:
            # Storage state exists but may be expired — fall through to re-login
            logger.warning("Existing storage state invalid: %s", e)

    # No valid session — launch `notebooklm login` in a background thread
    # (it opens a browser window; we can't await it in Gradio's event loop)
    _update_state(status="⏳ Launching browser login...")

    def _run_login():
        try:
            result = subprocess.run(
                [sys.executable, "-m", "notebooklm", "login"],
                timeout=300,
            )
            if result.returncode == 0 and _is_authenticated(storage_path):
                _update_state(
                    authenticated=True,
                    status="✅ Login complete — session saved",
                )
            else:
                _update_state(
                    status="⚠️ Login window closed — authentication not confirmed. Try again."
                )
        except subprocess.TimeoutExpired:
            _update_state(status="⚠️ Login timed out (5 min). Please try again.")
        except Exception as exc:
            _update_state(status=f"❌ Login error: {exc}")

    t = threading.Thread(target=_run_login, daemon=True)
    t.start()

    return False, (
        "🌐 **A browser window has opened for Google login.**\n\n"
        "1. Complete the Google sign-in in the browser window\n"
        "2. Wait until you see the NotebookLM homepage\n"
        "3. Return here and click **Authenticate** again to confirm\n\n"
        f"Session will be saved to: `{sp}`"
    )


async def connect_notebook(notebook_id_or_title: str) -> tuple[bool, str]:
    """Connect to an existing notebook by ID or title, or create a new one.

    Uses `NotebookLMClient.from_storage()` as a context manager for each call.
    Stores only the notebook_id/title in state (not the client object).
    """
    if not _is_authenticated():
        return False, (
            "❌ Not authenticated. Please click **Authenticate** first "
            "and complete the Google login."
        )

    try:
        from notebooklm import NotebookLMClient  # type: ignore

        async with await NotebookLMClient.from_storage() as client:
            notebooks = await client.notebooks.list()
            matched = None
            search = (notebook_id_or_title or "").strip().lower()
            for nb in notebooks:
                if search and (
                    search in nb.id.lower() or search in nb.title.lower()
                ):
                    matched = nb
                    break

            if matched:
                _update_state(
                    authenticated=True,
                    notebook_id=matched.id,
                    notebook_title=matched.title,
                    status=f"✅ Connected to notebook: {matched.title}",
                )
                return True, f"✅ Connected to existing notebook: **{matched.title}**"

            # Create new notebook
            title = notebook_id_or_title.strip() if notebook_id_or_title.strip() else "UX Best Practices"
            nb = await client.notebooks.create(title)
            _update_state(
                authenticated=True,
                notebook_id=nb.id,
                notebook_title=nb.title,
                status=f"✅ Created new notebook: {nb.title}",
            )
            return True, f"✅ Created new notebook: **{nb.title}**"

    except Exception as e:
        msg = f"❌ Failed to connect to notebook: {e}"
        _update_state(status=msg)
        return False, msg


# ── Agent configuration loop ───────────────────────────────────────────────────

async def query_category_best_practices(
    category: Dict[str, str],
    business_context: str = "",
    persona_summary: str = "",
) -> str:
    """Query the connected notebook for best practices for a single UX category.

    Falls back to a built-in knowledge base if NotebookLM is not connected.
    """
    state = get_state()

    # Check cache first
    cache_key = category["id"]
    if cache_key in state["best_practices_cache"]:
        return state["best_practices_cache"][cache_key]

    # Build a personalised query
    context_clause = ""
    if business_context:
        context_clause += f"\n\nBusiness context: {business_context[:300]}"
    if persona_summary:
        context_clause += f"\n\nTarget users: {persona_summary[:300]}"

    query = category["query"] + context_clause

    # Try NotebookLM first
    if state["authenticated"] and state["notebook_id"] and _is_authenticated():
        try:
            from notebooklm import NotebookLMClient  # type: ignore
            async with await NotebookLMClient.from_storage() as client:
                result = await client.chat.ask(state["notebook_id"], query)
                answer = result.answer if hasattr(result, "answer") else str(result)
                # Cache and return
                with _lock:
                    _state["best_practices_cache"][cache_key] = answer
                return answer
        except Exception as e:
            logger.warning("NotebookLM query failed for %s: %s", category["id"], e)

    # Fallback: use built-in knowledge base
    return _builtin_best_practices(category["id"])


async def run_ux_configuration_loop(
    categories: Optional[List[str]] = None,
    business_context: str = "",
    persona_summary: str = "",
    on_progress: Optional[Any] = None,
) -> Dict[str, str]:
    """Run the full agent configuration loop across all (or selected) UX categories.

    Returns a dict mapping category_id → best_practices_text.
    Calls on_progress(category_name, index, total) after each category completes.
    """
    selected = [c for c in UX_CATEGORIES if not categories or c["id"] in categories]
    results: Dict[str, str] = {}

    for i, cat in enumerate(selected):
        if on_progress:
            try:
                on_progress(cat["name"], i, len(selected))
            except Exception:
                pass

        bp = await query_category_best_practices(cat, business_context, persona_summary)
        results[cat["id"]] = bp
        # Small delay to avoid rate limiting
        await asyncio.sleep(0.3)

    return results


def get_best_practices_for_scan(category_ids: List[str]) -> Dict[str, str]:
    """Synchronous helper to get cached best practices for the UX scanner."""
    state = get_state()
    result = {}
    for cid in category_ids:
        if cid in state["best_practices_cache"]:
            result[cid] = state["best_practices_cache"][cid]
        else:
            result[cid] = _builtin_best_practices(cid)
    return result


# ── Built-in fallback knowledge base ──────────────────────────────────────────

_BUILTIN_KB: Dict[str, str] = {
    "visibility": (
        "Best practices: Provide real-time feedback for all user actions (loading spinners, "
        "progress bars, success/error messages). Use clear status indicators. Avoid silent failures. "
        "Common mistakes: No loading state, no confirmation after form submission, broken links with no 404 page."
    ),
    "real_world": (
        "Best practices: Use plain language, avoid jargon. Match terminology to user mental models. "
        "Use familiar icons with text labels. Common mistakes: Technical error codes shown to users, "
        "industry jargon in navigation, dates in non-localised formats."
    ),
    "user_control": (
        "Best practices: Provide clear back/undo options. Allow users to cancel processes. "
        "Avoid irreversible actions without confirmation. Common mistakes: No breadcrumbs, "
        "forms that lose data on back navigation, no way to undo destructive actions."
    ),
    "consistency": (
        "Best practices: Use consistent colours, typography, button styles, and terminology throughout. "
        "Follow platform conventions (e.g. blue underlined links). Common mistakes: Different button "
        "styles for the same action, inconsistent heading hierarchy, mixed icon styles."
    ),
    "error_prevention": (
        "Best practices: Validate inputs inline before submission. Provide format hints. "
        "Use confirmation dialogs for destructive actions. Common mistakes: No inline validation, "
        "unclear required field indicators, no confirmation before delete."
    ),
    "recognition": (
        "Best practices: Keep navigation visible. Use descriptive labels. Show recently visited items. "
        "Avoid hiding key actions in menus. Common mistakes: Icon-only navigation without labels, "
        "hidden menus, no search functionality on large sites."
    ),
    "flexibility": (
        "Best practices: Offer keyboard shortcuts for power users. Provide filters and sorting. "
        "Remember user preferences. Common mistakes: No search, no filtering, no way to customise "
        "the experience, forcing all users through the same flow."
    ),
    "aesthetic": (
        "Best practices: Remove decorative elements that add no value. Use whitespace generously. "
        "Limit colour palette. Prioritise content over chrome. Common mistakes: Cluttered homepages, "
        "too many competing CTAs, excessive animations, low contrast text."
    ),
    "error_recovery": (
        "Best practices: Write error messages in plain language. Explain what went wrong and how to fix it. "
        "Offer a clear path forward. Common mistakes: Generic 'Something went wrong' messages, "
        "no guidance on how to recover, error messages that disappear too quickly."
    ),
    "help": (
        "Best practices: Provide contextual help close to where users need it. Use tooltips for complex fields. "
        "Make FAQ/support easy to find. Common mistakes: Help buried in footer, no tooltips on complex forms, "
        "support contact hidden or hard to find."
    ),
    "accessibility": (
        "Best practices: Ensure WCAG 2.1 AA compliance. Provide alt text for images. Use sufficient colour contrast "
        "(4.5:1 for text). Ensure keyboard navigability. Support screen readers. Common mistakes: Missing alt text, "
        "low contrast, non-keyboard-accessible menus, missing ARIA labels."
    ),
    "mobile": (
        "Best practices: Use responsive layouts. Ensure tap targets are at least 44×44px. Avoid hover-only interactions. "
        "Test on real devices. Common mistakes: Text too small on mobile, tap targets too close together, "
        "horizontal scrolling, desktop-only features."
    ),
    "visual_hierarchy": (
        "Best practices: Use size, weight, and colour to guide the eye. Place the most important content above the fold. "
        "Use F-pattern or Z-pattern layouts. Limit font sizes to 3-4 levels. Common mistakes: Everything the same size, "
        "no clear primary CTA, too many competing focal points."
    ),
    "trust": (
        "Best practices: Display social proof (reviews, testimonials, logos). Show security badges on checkout. "
        "Use professional photography. Display contact information prominently. Common mistakes: No reviews, "
        "stock photos, hidden contact info, no SSL indicator, no privacy policy link."
    ),
    "conversion": (
        "Best practices: Use action-oriented CTA text ('Book a Table', not 'Submit'). Place CTAs above the fold. "
        "Use contrasting colours for primary CTAs. Reduce friction in conversion flows. Common mistakes: "
        "Weak CTA text, CTAs buried below the fold, too many competing CTAs, long checkout forms."
    ),
}


def _builtin_best_practices(category_id: str) -> str:
    return _BUILTIN_KB.get(
        category_id,
        "No specific best practices available for this category. Apply general UX principles."
    )


# ── Notebook listing helper ────────────────────────────────────────────────────

async def list_notebooks() -> tuple[bool, List[Dict[str, str]]]:
    """List all notebooks in the authenticated account."""
    if not _is_authenticated():
        return False, []
    try:
        from notebooklm import NotebookLMClient  # type: ignore
        async with await NotebookLMClient.from_storage() as client:
            notebooks = await client.notebooks.list()
            return True, [{"id": nb.id, "title": nb.title} for nb in notebooks]
    except Exception as e:
        logger.warning("Failed to list notebooks: %s", e)
        return False, []


async def disconnect() -> None:
    """Disconnect from NotebookLM and clear state."""
    _update_state(
        authenticated=False,
        notebook_id=None,
        notebook_title=None,
        client=None,
        best_practices_cache={},
        status="Disconnected",
    )
