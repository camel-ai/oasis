"""
redesign_client.py
==================
Generates an AI HTML redesign from a website screenshot using OpenAI Vision.

Pipeline
--------
1. Capture a Playwright screenshot of the target URL (JPEG, 1280×800)
2. Send the screenshot to GPT-4o Vision with a UX redesign prompt
3. Receive a complete, self-contained HTML page as the redesign
4. Optionally run a second call to GPT-4o-mini for a brief analysis sentence
5. Sanitise the HTML for safe embedding in an <iframe srcdoc>

No external services other than the OpenAI API are used.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import re
from typing import Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_REDESIGN_SYSTEM = (
    "You are a senior UX/UI engineer. "
    "Output ONLY a complete, self-contained HTML page — no markdown fences, "
    "no explanation, no comments outside the HTML. "
    "The page must start with <!DOCTYPE html> and end with </html>."
)

_REDESIGN_USER = (
    "You are given a screenshot of a website. "
    "Generate a meaningfully improved HTML redesign of the visible section. "
    "Requirements:\n"
    "  • Modern CSS (flexbox/grid), good colour contrast, clear visual hierarchy\n"
    "  • Prominent CTA button\n"
    "  • Preserve the brand name and colour palette\n"
    "  • Mobile-friendly viewport meta tag\n"
    "  • All CSS inlined — no external stylesheets or scripts\n"
    "  • Focus on fixing navigation clarity, readability, and trust signals\n"
    "{ux_context}"
)

_ANALYSIS_USER = (
    "In one or two sentences, describe the main UX problem visible in this "
    "screenshot and what the redesign should prioritise fixing."
)

_STRIP_FENCE = re.compile(r"^```(?:html)?\s*|```\s*$", re.MULTILINE)


# ---------------------------------------------------------------------------
# Screenshot capture (Playwright)
# ---------------------------------------------------------------------------

async def _capture_screenshot(url: str) -> Optional[bytes]:
    """Return a JPEG screenshot of *url* at 1280×800. Returns None on failure."""
    try:
        from playwright.async_api import async_playwright  # type: ignore
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(args=["--no-sandbox", "--disable-dev-shm-usage"])
            page = await browser.new_page(viewport={"width": 1280, "height": 800})
            await page.goto(url, timeout=25_000, wait_until="domcontentloaded")
            await page.wait_for_timeout(2500)
            data = await page.screenshot(type="jpeg", quality=70)
            await browser.close()
            return data
    except Exception as exc:
        logger.warning("Playwright screenshot failed for %s: %s", url, exc)
        return None


# ---------------------------------------------------------------------------
# OpenAI Vision calls
# ---------------------------------------------------------------------------

async def _call_vision(
    image_bytes: bytes,
    user_prompt: str,
    system_prompt: str,
    api_key: str,
    model: str,
    max_tokens: int,
) -> str:
    """Send an image + text prompt to an OpenAI vision model and return the text response."""
    img_b64 = base64.b64encode(image_bytes).decode()
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}",
                            "detail": "low",
                        },
                    },
                    {"type": "text", "text": user_prompt},
                ],
            },
        ],
    }
    async with httpx.AsyncClient(timeout=90) as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


async def _screenshot_html(html_code: str) -> Optional[bytes]:
    """
    Render a self-contained HTML string in a headless Playwright browser
    and return a JPEG screenshot (960×540, matching the slide canvas).
    Returns None on failure.
    """
    try:
        from playwright.async_api import async_playwright  # type: ignore
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(args=["--no-sandbox", "--disable-dev-shm-usage"])
            page = await browser.new_page(viewport={"width": 960, "height": 540})
            await page.set_content(html_code, wait_until="domcontentloaded")
            await page.wait_for_timeout(800)
            data = await page.screenshot(type="jpeg", quality=80, clip={"x": 0, "y": 0, "width": 960, "height": 540})
            await browser.close()
            return data
    except Exception as exc:
        logger.warning("Redesign screenshot failed: %s", exc)
        return None


async def _generate_redesign_async(
    screenshot_url: str,
    openai_key: str,
    vision_model: str = "gpt-4o",
    ux_context: str = "",
) -> Tuple[str, str, Optional[str]]:
    """
    Core async pipeline.

    Returns
    -------
    (analysis_text, html_code, redesign_screenshot_data_uri | None)
    """
    # 1. Obtain screenshot bytes
    screenshot_bytes: Optional[bytes] = None

    if screenshot_url.startswith("http"):
        screenshot_bytes = await _capture_screenshot(screenshot_url)
        if screenshot_bytes is None:
            # Fallback: download the image directly (e.g. already a PNG/JPEG URL)
            try:
                async with httpx.AsyncClient(timeout=20) as client:
                    r = await client.get(screenshot_url)
                    r.raise_for_status()
                    screenshot_bytes = r.content
            except Exception as exc:
                raise RuntimeError(
                    f"Could not obtain screenshot for {screenshot_url}: {exc}"
                ) from exc
    else:
        # Local file path
        with open(screenshot_url, "rb") as fh:
            screenshot_bytes = fh.read()

    if not screenshot_bytes:
        raise RuntimeError(f"Empty screenshot for {screenshot_url}")

    # 2. Build the redesign prompt
    context_line = (
        f"\nAdditional UX context to address: {ux_context}" if ux_context else ""
    )
    user_prompt = _REDESIGN_USER.format(ux_context=context_line)

    # 3. Generate redesign HTML
    html_code = await _call_vision(
        image_bytes=screenshot_bytes,
        user_prompt=user_prompt,
        system_prompt=_REDESIGN_SYSTEM,
        api_key=openai_key,
        model=vision_model,
        max_tokens=3500,
    )
    # Strip any accidental markdown fences
    html_code = _STRIP_FENCE.sub("", html_code).strip()

    # 4. Generate brief analysis (cheap mini call)
    analysis = ""
    try:
        analysis = await _call_vision(
            image_bytes=screenshot_bytes,
            user_prompt=_ANALYSIS_USER,
            system_prompt="You are a concise UX analyst.",
            api_key=openai_key,
            model="gpt-4o-mini",
            max_tokens=120,
        )
    except Exception as exc:
        logger.debug("Analysis call failed (non-fatal): %s", exc)

    # 5. Render redesign HTML to a screenshot for the slide panel
    redesign_screenshot_uri: Optional[str] = None
    try:
        shot_bytes = await _screenshot_html(html_code)
        if shot_bytes:
            b64 = base64.b64encode(shot_bytes).decode()
            redesign_screenshot_uri = f"data:image/jpeg;base64,{b64}"
    except Exception as exc:
        logger.debug("Redesign screenshot step failed (non-fatal): %s", exc)

    return analysis, html_code, redesign_screenshot_uri


# ---------------------------------------------------------------------------
# HTML sanitiser — safe for <iframe srcdoc>
# ---------------------------------------------------------------------------

def sanitise_for_embed(html_code: str, max_height: int = 420) -> str:
    """
    Prepare generated HTML for safe embedding in an <iframe srcdoc>.

    - Strips markdown fences
    - Removes <script> tags
    - Wraps bare snippets in a minimal HTML shell
    - Injects overflow:auto so the iframe scrolls
    """
    if not html_code:
        return "<p style='color:#888;font-size:12px;padding:12px;'>No redesign generated.</p>"

    html_code = _STRIP_FENCE.sub("", html_code).strip()
    html_code = re.sub(r"<script[\s\S]*?</script>", "", html_code, flags=re.IGNORECASE)

    if "<html" not in html_code.lower():
        html_code = (
            "<!DOCTYPE html><html>"
            "<head><meta charset='UTF-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<style>body{{margin:0;padding:8px;font-family:sans-serif;"
            f"overflow:auto;max-height:{max_height}px;}}</style></head>"
            f"<body>{html_code}</body></html>"
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
# Public synchronous API (used by app.py)
# ---------------------------------------------------------------------------

def generate_redesign(
    screenshot_url: str,
    ux_issues: Optional[list] = None,
    openai_key: str = "",
    vision_model: str = "gpt-4o",
) -> dict:
    """
    Generate an AI HTML redesign for the given screenshot URL.

    Parameters
    ----------
    screenshot_url : str
        Public URL or local file path of the page screenshot.
    ux_issues : list[str], optional
        List of UX issues to address in the redesign prompt.
    openai_key : str
        OpenAI API key. Falls back to the OPENAI_API_KEY environment variable.
    vision_model : str
        OpenAI vision model to use. Defaults to "gpt-4o".

    Returns
    -------
    dict with keys:
        analysis       – one-sentence UX analysis of the original screenshot
        html_code      – raw HTML redesign
        html_sanitised – HTML safe for <iframe srcdoc> embedding
        error          – error message string, or None on success
    """
    result: dict = {
        "analysis": "",
        "html_code": "",
        "html_sanitised": "",
        "redesign_screenshot": "",   # base64 data URI of the rendered redesign
        "error": None,
    }

    key = (openai_key or "").strip() or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        result["error"] = (
            "OPENAI_API_KEY is not set. Add it in the ⚙️ Settings tab."
        )
        return result

    if not screenshot_url:
        result["error"] = "No screenshot URL provided."
        return result

    ux_context = "; ".join(ux_issues[:3]) if ux_issues else ""

    try:
        analysis, html_code, redesign_screenshot = asyncio.run(
            _generate_redesign_async(
                screenshot_url=screenshot_url,
                openai_key=key,
                vision_model=vision_model,
                ux_context=ux_context,
            )
        )
        result["analysis"] = analysis
        result["html_code"] = html_code
        result["html_sanitised"] = sanitise_for_embed(html_code)
        result["redesign_screenshot"] = redesign_screenshot or ""
    except Exception as exc:
        result["error"] = str(exc)
        logger.error("Redesign generation failed: %s", exc)

    return result
