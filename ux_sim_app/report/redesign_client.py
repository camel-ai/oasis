"""
redesign_client.py
==================
Client for the HuggingFace Space:
  https://agents-mcp-hackathon-website-generator.hf.space

Exposes two main entry points:

  screenshot_to_html(screenshot_url_or_path, nebius_api_key, ux_context)
    → (analysis: str, html_code: str)

  improve_html_with_ux(html_code, ux_issues, nebius_api_key)
    → improved_html_code: str

The HF Space uses a Gradio 2-step POST/GET pattern:
  1. POST  → returns {"event_id": "..."}
  2. GET   → SSE stream; last "data:" line contains the result JSON array

We implement this pattern in Python using httpx with a short polling loop.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Optional, Tuple

import httpx

HF_BASE = "https://agents-mcp-hackathon-website-generator.hf.space/gradio_api/call"
DEFAULT_TIMEOUT = 90  # seconds to wait for the HF space to respond


# ---------------------------------------------------------------------------
# Low-level Gradio SSE client
# ---------------------------------------------------------------------------

async def _gradio_call(
    endpoint: str,
    data: list,
    timeout: int = DEFAULT_TIMEOUT,
) -> list:
    """
    Call a Gradio Space endpoint using the 2-step POST/GET SSE pattern.

    Returns the parsed result list from the final SSE "data:" event.
    """
    url_post = f"{HF_BASE}/{endpoint}"

    async with httpx.AsyncClient(timeout=timeout) as client:
        # Step 1: POST to queue the job
        resp = await client.post(
            url_post,
            json={"data": data},
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        event_id = resp.json().get("event_id")
        if not event_id:
            raise ValueError(f"No event_id returned from {endpoint}: {resp.text}")

        # Step 2: GET the SSE stream
        url_get = f"{HF_BASE}/{endpoint}/{event_id}"
        last_data: Optional[list] = None

        async with client.stream("GET", url_get, timeout=timeout) as stream:
            async for line in stream.aiter_lines():
                line = line.strip()
                if line.startswith("data:"):
                    payload = line[5:].strip()
                    if payload:
                        try:
                            last_data = json.loads(payload)
                        except json.JSONDecodeError:
                            pass  # partial line, keep waiting

        if last_data is None:
            raise RuntimeError(f"No data received from SSE stream for {endpoint}/{event_id}")

        return last_data


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def screenshot_to_html_async(
    screenshot_url: str,
    nebius_api_key: str,
    ux_context: str = "",
) -> Tuple[str, str]:
    """
    Convert a screenshot URL to HTML code using the HF Space.

    Parameters
    ----------
    screenshot_url : str
        Public URL of the screenshot image (or local path — will be uploaded).
    nebius_api_key : str
        Nebius API key required by the HF Space.
    ux_context : str
        Optional UX context / improvement prompt appended to the default prompt.

    Returns
    -------
    (analysis, html_code) : Tuple[str, str]
    """
    prompt = (
        "Analyse this website screenshot and generate clean, modern, responsive HTML/CSS "
        "that improves the UX based on best practices. "
        "Focus on: clear navigation, readable typography, accessible colour contrast, "
        "obvious calls-to-action, and mobile-friendly layout."
    )
    if ux_context:
        prompt += f" Additional context: {ux_context}"

    payload_image = {
        "path": screenshot_url,
        "meta": {"_type": "gradio.FileData"},
    }

    result = await _gradio_call(
        endpoint="screenshot_to_code",
        data=[payload_image, nebius_api_key],
        timeout=DEFAULT_TIMEOUT,
    )

    # result is [analysis_str, html_code_str]
    analysis = result[0] if len(result) > 0 else ""
    html_code = result[1] if len(result) > 1 else ""
    return analysis, html_code


async def improve_html_async(
    html_code: str,
    ux_issues: list[str],
    nebius_api_key: str,
) -> str:
    """
    Take existing HTML and a list of UX issues, then ask the HF Space to
    generate an improved version that addresses those issues.

    Uses the /generate_html_code endpoint with a rich description.
    """
    issues_text = "\n".join(f"- {issue}" for issue in ux_issues)
    description = (
        f"Improve the following HTML to fix these UX issues:\n{issues_text}\n\n"
        f"Requirements: maintain the same content and structure, but fix the UX problems. "
        f"Use modern CSS (flexbox/grid), good colour contrast, clear CTAs, and readable fonts.\n\n"
        f"Original HTML:\n```html\n{html_code[:3000]}\n```"  # truncate to avoid token limits
    )

    result = await _gradio_call(
        endpoint="generate_html_code",
        data=[description, nebius_api_key],
        timeout=DEFAULT_TIMEOUT,
    )

    return result[0] if result else html_code


# ---------------------------------------------------------------------------
# Synchronous wrappers (for use in Gradio sync callbacks)
# ---------------------------------------------------------------------------

def screenshot_to_html(
    screenshot_url: str,
    nebius_api_key: str,
    ux_context: str = "",
) -> Tuple[str, str]:
    """Synchronous wrapper for screenshot_to_html_async."""
    import concurrent.futures

    def _worker():
        return asyncio.run(screenshot_to_html_async(screenshot_url, nebius_api_key, ux_context))

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_worker).result(timeout=DEFAULT_TIMEOUT + 10)


def improve_html(
    html_code: str,
    ux_issues: list[str],
    nebius_api_key: str,
) -> str:
    """Synchronous wrapper for improve_html_async."""
    import concurrent.futures

    def _worker():
        return asyncio.run(improve_html_async(html_code, ux_issues, nebius_api_key))

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_worker).result(timeout=DEFAULT_TIMEOUT + 10)


# ---------------------------------------------------------------------------
# Redesign pipeline: screenshot → HTML → UX-improved HTML
# ---------------------------------------------------------------------------

def generate_redesign(
    screenshot_url: str,
    ux_issues: list[str],
    nebius_api_key: str,
) -> dict:
    """
    Full pipeline:
      1. Convert screenshot to initial HTML (screenshot_to_code endpoint)
      2. Apply UX improvements to that HTML (generate_html_code endpoint)

    Returns a dict with:
      {
        "analysis": str,          # AI analysis of the original screenshot
        "initial_html": str,      # First-pass HTML from screenshot
        "improved_html": str,     # UX-improved HTML
        "error": str | None,
      }
    """
    result = {
        "analysis": "",
        "initial_html": "",
        "improved_html": "",
        "error": None,
    }

    if not nebius_api_key or not nebius_api_key.strip():
        result["error"] = "Nebius API key is required for the screenshot-to-HTML redesign feature."
        return result

    if not screenshot_url:
        result["error"] = "No screenshot URL provided."
        return result

    try:
        # Step 1: Screenshot → HTML
        analysis, initial_html = screenshot_to_html(
            screenshot_url=screenshot_url,
            nebius_api_key=nebius_api_key,
            ux_context="; ".join(ux_issues[:3]) if ux_issues else "",
        )
        result["analysis"] = analysis
        result["initial_html"] = initial_html

        # Step 2: HTML → UX-improved HTML (only if we have issues to fix)
        if ux_issues and initial_html:
            improved = improve_html(
                html_code=initial_html,
                ux_issues=ux_issues,
                nebius_api_key=nebius_api_key,
            )
            result["improved_html"] = improved
        else:
            result["improved_html"] = initial_html

    except Exception as exc:
        result["error"] = str(exc)

    return result


# ---------------------------------------------------------------------------
# HTML sanitiser — strips <script> and <body>/<html> wrappers for embedding
# ---------------------------------------------------------------------------

def sanitise_for_embed(html_code: str, max_height: int = 420) -> str:
    """
    Prepare generated HTML for safe embedding in an <iframe srcdoc>.

    - Removes <script> tags (security)
    - Wraps in a minimal shell if no <html> tag present
    - Adds overflow:auto and max-height to the body
    """
    if not html_code:
        return "<p style='color:#888;font-size:12px;padding:12px;'>No redesign generated.</p>"

    # Strip script tags
    html_code = re.sub(r'<script[\s\S]*?</script>', '', html_code, flags=re.IGNORECASE)

    # If it's a snippet (no <html>), wrap it
    if "<html" not in html_code.lower():
        html_code = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  body {{ margin: 0; padding: 8px; font-family: 'Inter', sans-serif;
         overflow: auto; max-height: {max_height}px; }}
</style>
</head>
<body>
{html_code}
</body>
</html>"""
    else:
        # Inject body style
        html_code = re.sub(
            r'(<body[^>]*>)',
            rf'\1<style>body{{overflow:auto;max-height:{max_height}px;}}</style>',
            html_code,
            count=1,
            flags=re.IGNORECASE,
        )

    return html_code
