"""
redesign_client.py
==================
Generates an AI HTML redesign from a website screenshot.

Primary path  : OpenAI GPT-4o Vision  (requires OPENAI_API_KEY — already used by the app)
Optional path : HuggingFace Space      (requires NEBIUS_API_KEY — only if explicitly set)

IMPORTANT: The HuggingFace Space (agents-mcp-hackathon/website-generator) ALWAYS requires
a valid Nebius JWT token.  Empty string, OpenAI keys, and the demo key pre-filled in the
Space UI all return HTTP 401.  Therefore the OpenAI Vision path is the reliable default and
the HF Space is an opt-in upgrade when the user provides their own Nebius key.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
from typing import Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HF_BASE = "https://agents-mcp-hackathon-website-generator.hf.space/gradio_api/call"
DEFAULT_TIMEOUT = 90

_STRIP_FENCE = re.compile(r"^```(?:html)?\s*|```\s*$", re.MULTILINE)

_REDESIGN_PROMPT = (
    "You are a senior UX/UI redesign expert. "
    "You are given a screenshot of a website page. "
    "Output ONLY a complete, self-contained HTML page (no markdown fences, no explanation) "
    "that shows a meaningfully improved redesign of the navigation and hero/main section. "
    "Requirements:\n"
    "- Modern CSS (flexbox/grid), good contrast, clear visual hierarchy\n"
    "- A prominent CTA button\n"
    "- Preserve the brand colours and name\n"
    "- Mobile-friendly viewport meta tag\n"
    "- No external dependencies (inline all CSS)\n"
    "The output must start with <!DOCTYPE html> and end with </html>."
)

_ANALYSIS_PROMPT = (
    "In 1-2 sentences, describe the main UX problem visible in this screenshot "
    "and what the redesign should fix."
)


# ---------------------------------------------------------------------------
# HTML sanitiser
# ---------------------------------------------------------------------------

def sanitise_for_embed(html_code: str, max_height: int = 420) -> str:
    """
    Prepare generated HTML for safe embedding in an <iframe srcdoc>.
    - Strips markdown fences
    - Removes <script> tags
    - Injects overflow:auto so the iframe scrolls rather than overflows
    """
    if not html_code:
        return "<p style='color:#888;font-size:12px;padding:12px;'>No redesign generated.</p>"

    # Strip markdown fences
    html_code = _STRIP_FENCE.sub("", html_code).strip()

    # Strip script tags
    html_code = re.sub(r"<script[\s\S]*?</script>", "", html_code, flags=re.IGNORECASE)

    if "<html" not in html_code.lower():
        html_code = (
            "<!DOCTYPE html><html><head>"
            "<meta charset='UTF-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<style>body{{margin:0;padding:8px;font-family:sans-serif;"
            f"overflow:auto;max-height:{max_height}px;}}</style>"
            "</head><body>"
            + html_code
            + "</body></html>"
        )
    else:
        html_code = re.sub(
            r"(<body[^>]*>)",
            rf"\1<style>body{{overflow:auto;max-height:{max_height}px;}}</style>",
            html_code,
            count=1,
            flags=re.IGNORECASE,
        )

    return html_code


# ---------------------------------------------------------------------------
# Screenshot helper (Playwright)
# ---------------------------------------------------------------------------

async def _take_screenshot(url: str) -> Optional[bytes]:
    """Capture a JPEG screenshot of *url* using Playwright. Returns None on failure."""
    try:
        from playwright.async_api import async_playwright  # type: ignore
        async with async_playwright() as p:
            browser = await p.chromium.launch(args=["--no-sandbox"])
            page = await browser.new_page(viewport={"width": 1280, "height": 800})
            await page.goto(url, timeout=20_000, wait_until="domcontentloaded")
            await page.wait_for_timeout(2500)
            data = await page.screenshot(type="jpeg", quality=65)
            await browser.close()
            return data
    except Exception as exc:
        logger.warning("Screenshot failed for %s: %s", url, exc)
        return None


# ---------------------------------------------------------------------------
# Primary: OpenAI GPT-4o Vision
# ---------------------------------------------------------------------------

async def _redesign_via_openai(
    screenshot_bytes: bytes,
    openai_key: str,
    model: str = "gpt-4o",
) -> Tuple[str, str]:
    """Returns (analysis_text, html_code). Raises on failure."""
    img_b64 = base64.b64encode(screenshot_bytes).decode()
    image_payload = {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}", "detail": "low"},
    }

    async with httpx.AsyncClient(timeout=90) as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "max_tokens": 3000,
                "messages": [
                    {"role": "user", "content": [image_payload, {"type": "text", "text": _REDESIGN_PROMPT}]}
                ],
            },
        )
    resp.raise_for_status()
    html_code = _STRIP_FENCE.sub("", resp.json()["choices"][0]["message"]["content"]).strip()

    # Brief analysis (cheap mini call)
    analysis = ""
    try:
        async with httpx.AsyncClient(timeout=30) as client2:
            r2 = await client2.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"},
                json={
                    "model": "gpt-4o-mini",
                    "max_tokens": 120,
                    "messages": [
                        {"role": "user", "content": [image_payload, {"type": "text", "text": _ANALYSIS_PROMPT}]}
                    ],
                },
            )
        r2.raise_for_status()
        analysis = r2.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        logger.debug("Analysis mini-call failed: %s", exc)

    return analysis, html_code


# ---------------------------------------------------------------------------
# Optional: HuggingFace Space (requires Nebius key)
# ---------------------------------------------------------------------------

async def _gradio_call(endpoint: str, data: list, timeout: int = DEFAULT_TIMEOUT) -> list:
    """2-step Gradio POST/GET SSE pattern."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            f"{HF_BASE}/{endpoint}",
            json={"data": data},
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        event_id = resp.json().get("event_id")
        if not event_id:
            raise ValueError(f"No event_id from {endpoint}: {resp.text}")

        last_data: Optional[list] = None
        async with client.stream("GET", f"{HF_BASE}/{endpoint}/{event_id}", timeout=timeout) as stream:
            async for line in stream.aiter_lines():
                line = line.strip()
                if line.startswith("data:"):
                    payload = line[5:].strip()
                    if payload:
                        try:
                            last_data = json.loads(payload)
                        except json.JSONDecodeError:
                            pass

    if last_data is None:
        raise RuntimeError(f"No data from SSE stream for {endpoint}/{event_id}")
    return last_data


async def _redesign_via_hf_space(screenshot_url: str, nebius_key: str) -> Tuple[str, str]:
    """Returns (analysis_text, html_code). Raises if Nebius key is missing or call fails."""
    if not nebius_key:
        raise ValueError("Nebius API key is required for the HuggingFace Space path.")

    result = await _gradio_call(
        "screenshot_to_code",
        [{"path": screenshot_url, "meta": {"_type": "gradio.FileData"}}, nebius_key],
    )
    analysis = result[0] if len(result) > 0 else ""
    html_code = result[1] if len(result) > 1 else ""

    if "Error" in (analysis or "") or not html_code:
        raise RuntimeError(f"HF Space returned error: {analysis}")

    return analysis, html_code


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def generate_redesign(
    screenshot_url: str,
    openai_key: str = "",
    nebius_key: str = "",
    vision_model: str = "gpt-4o",
) -> Tuple[str, str, str]:
    """
    Generate an AI HTML redesign for the given screenshot URL.

    Strategy:
      1. If nebius_key is set → try HuggingFace Space (Qwen2.5-VL + DeepSeek-V3)
      2. Fall back to (or use directly) OpenAI GPT-4o Vision

    Returns:
        (analysis_text, html_code, html_sanitised_for_srcdoc)
    """
    openai_key = openai_key or os.environ.get("OPENAI_API_KEY", "")
    nebius_key = nebius_key or os.environ.get("NEBIUS_API_KEY", "")

    # --- Try HuggingFace Space if Nebius key is available ---
    if nebius_key:
        try:
            logger.info("Trying HuggingFace Space redesign for %s", screenshot_url)
            analysis, html_code = await _redesign_via_hf_space(screenshot_url, nebius_key)
            logger.info("HF Space redesign succeeded.")
            return analysis, html_code, sanitise_for_embed(html_code)
        except Exception as exc:
            logger.warning("HF Space failed (%s), falling back to OpenAI Vision.", exc)

    # --- OpenAI GPT-4o Vision path ---
    if not openai_key:
        raise RuntimeError(
            "No API key available for redesign generation. "
            "Set OPENAI_API_KEY (always works) or NEBIUS_API_KEY (for HF Space)."
        )

    # Take a fresh Playwright screenshot
    screenshot_bytes = await _take_screenshot(screenshot_url)
    if screenshot_bytes is None:
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                r = await client.get(screenshot_url)
                r.raise_for_status()
                screenshot_bytes = r.content
        except Exception as exc:
            raise RuntimeError(f"Could not obtain screenshot for {screenshot_url}: {exc}") from exc

    logger.info("Generating redesign via OpenAI %s for %s", vision_model, screenshot_url)
    analysis, html_code = await _redesign_via_openai(screenshot_bytes, openai_key, vision_model)
    return analysis, html_code, sanitise_for_embed(html_code)


def generate_redesign_sync(
    screenshot_url: str,
    openai_key: str = "",
    nebius_key: str = "",
    vision_model: str = "gpt-4o",
) -> Tuple[str, str, str]:
    """Synchronous wrapper around generate_redesign."""
    return asyncio.run(generate_redesign(screenshot_url, openai_key, nebius_key, vision_model))


# ---------------------------------------------------------------------------
# Legacy sync wrappers (kept for backward compatibility with app.py)
# ---------------------------------------------------------------------------

def generate_redesign(  # type: ignore[no-redef]  # noqa: F811
    screenshot_url: str,
    ux_issues: list = None,
    nebius_api_key: str = "",
    openai_key: str = "",
    vision_model: str = "gpt-4o",
) -> dict:
    """
    Legacy dict-returning wrapper used by app.py step_generate_report.

    Returns:
        {
            "analysis": str,
            "initial_html": str,
            "improved_html": str,
            "html_sanitised": str,
            "error": str | None,
        }
    """
    result = {
        "analysis": "",
        "initial_html": "",
        "improved_html": "",
        "html_sanitised": "",
        "error": None,
    }
    try:
        analysis, html_code, html_sanitised = generate_redesign_sync(
            screenshot_url=screenshot_url,
            openai_key=openai_key,
            nebius_key=nebius_api_key,
            vision_model=vision_model,
        )
        result["analysis"] = analysis
        result["initial_html"] = html_code
        result["improved_html"] = html_code
        result["html_sanitised"] = html_sanitised
    except Exception as exc:
        result["error"] = str(exc)
    return result
