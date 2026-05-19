"""
captcha_guard.py
================
Pre-flight CAPTCHA guard for OASIS — powered by CloakBrowser.

Strategy
--------
1. Launch a CloakBrowser context (stealth Chromium with source-level C++
   fingerprint patches). CloakBrowser passes Cloudflare Turnstile, reCAPTCHA
   v3 (scores 0.9), FingerprintJS, and BrowserScan out of the box.
2. Navigate to the target URL and wait for network idle.
3. Detect known CAPTCHA / bot-challenge signatures in the DOM.
4. If detected:
   a. Attempt a humanized DOM click (Turnstile / "I am human" checkbox).
      CloakBrowser's humanize=True makes the click indistinguishable from a
      real user (Bézier curve, realistic aim point, hold duration).
   b. If still blocked and a 2captcha API key is configured, extract the
      site-key and submit to 2Captcha; inject the returned token.
   c. If still blocked after CAPTCHA_TIMEOUT seconds → raise CaptchaAbortError.
5. On success: extract verified page text and save Playwright storage state
   (cookies + localStorage) to a temp file so downstream sessions inherit it.
6. Return a CaptchaGuardResult containing:
   - status: "clear" | "solved" | "aborted" | "disabled"
   - page_text: verified body text (empty on abort)
   - storage_state_path: path to JSON session file (None on abort)
   - message: human-readable status for the UI

Why CloakBrowser instead of standard Playwright?
-------------------------------------------------
Standard Playwright is immediately detected by Cloudflare Turnstile and
FingerprintJS because navigator.webdriver=true and CDP automation signals
are present. Patching these at the JavaScript level (playwright-stealth) is
fragile and breaks with every Chrome update.

CloakBrowser patches Chromium at the C++ source level — 49 patches covering
canvas, WebGL, audio, fonts, GPU, screen, WebRTC, network timing, automation
signals, and CDP input behavior. Detection systems see a real browser because
it IS a real browser.

Environment variables (all optional)
--------------------------------------
TWOCAPTCHA_API_KEY      — 2Captcha API key for last-resort solving
CAPTCHA_TIMEOUT         — seconds to wait for a solve before aborting (default 45)
CAPTCHA_ENABLED         — set to "false" to skip the guard entirely (default "true")
CLOAKBROWSER_PROXY      — HTTP or SOCKS5 proxy for the browser session
CLOAKBROWSER_GEOIP      — auto-detect timezone/locale from proxy exit IP
CLOAKBROWSER_HUMANIZE   — enable human-like mouse/keyboard/scroll (default true)
CLOAKBROWSER_HUMAN_PRESET — "default" or "careful"
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Optional

# ── Configuration ──────────────────────────────────────────────────────────────
TWOCAPTCHA_API_KEY: str = os.getenv("TWOCAPTCHA_API_KEY", "")
CAPTCHA_TIMEOUT: int = int(os.getenv("CAPTCHA_TIMEOUT", "45"))
CAPTCHA_ENABLED: bool = os.getenv("CAPTCHA_ENABLED", "true").lower() != "false"

# CloakBrowser settings (also read from config.py if available)
try:
    from ux_sim_app.core.config import (
        CLOAKBROWSER_PROXY,
        CLOAKBROWSER_GEOIP,
        CLOAKBROWSER_HUMANIZE,
        CLOAKBROWSER_HUMAN_PRESET,
    )
except ImportError:
    CLOAKBROWSER_PROXY = os.getenv("CLOAKBROWSER_PROXY", "")
    CLOAKBROWSER_GEOIP = os.getenv("CLOAKBROWSER_GEOIP", "false").lower() == "true"
    CLOAKBROWSER_HUMANIZE = os.getenv("CLOAKBROWSER_HUMANIZE", "true").lower() != "false"
    CLOAKBROWSER_HUMAN_PRESET = os.getenv("CLOAKBROWSER_HUMAN_PRESET", "default")

# ── Known CAPTCHA DOM signatures ───────────────────────────────────────────────
_CAPTCHA_SELECTORS = [
    # Google reCAPTCHA v2 / v3
    "iframe[src*='recaptcha']",
    "div.g-recaptcha",
    "div[data-sitekey]",
    # hCaptcha
    "iframe[src*='hcaptcha']",
    "div.h-captcha",
    # Cloudflare Turnstile
    "div.cf-turnstile",
    "#cf-turnstile",
    "iframe[src*='challenges.cloudflare.com']",
    # Cloudflare browser challenge page
    "#cf-challenge-running",
    "#cf-please-wait",
    ".cf-browser-verification",
    "#challenge-form",
    # Generic bot-wall patterns
    "form#challenge-form",
    "[id*='captcha']",
    "[class*='captcha']",
]

# Selectors for the "I am human" checkbox (Turnstile / reCAPTCHA)
_CHECKBOX_SELECTORS = [
    ".recaptcha-checkbox",
    "span.recaptcha-checkbox-border",
    "div.cf-turnstile",
    "#cf-turnstile",
    "iframe[title*='challenge']",
]


# ── Result dataclass ───────────────────────────────────────────────────────────
@dataclass
class CaptchaGuardResult:
    status: str                               # "clear" | "solved" | "aborted" | "disabled"
    page_text: str = ""                       # verified body text (empty on abort)
    storage_state_path: Optional[str] = None  # path to Playwright storage state JSON
    message: str = ""                         # human-readable status for the UI
    captcha_type: Optional[str] = None        # detected CAPTCHA type (for logging)


class CaptchaAbortError(RuntimeError):
    """Raised when a CAPTCHA cannot be solved within the timeout."""


# ── Main entry point ───────────────────────────────────────────────────────────
async def preflight_check(url: str) -> CaptchaGuardResult:
    """
    Run the pre-flight CAPTCHA guard for *url* using CloakBrowser.

    Returns a CaptchaGuardResult. If status == "aborted", callers MUST stop
    processing and surface the .message to the user.
    """
    if not CAPTCHA_ENABLED:
        return CaptchaGuardResult(
            status="disabled",
            message="CAPTCHA guard disabled (CAPTCHA_ENABLED=false). Proceeding without check.",
        )

    # ── Import CloakBrowser (with Playwright fallback if not installed) ─────────
    try:
        from cloakbrowser import launch_context_async as _launch_ctx
        _using_cloak = True
    except ImportError:
        # Graceful fallback to standard Playwright if cloakbrowser is not installed
        _using_cloak = False
        from playwright.async_api import async_playwright as _apw

    storage_fd, storage_path = tempfile.mkstemp(suffix=".json", prefix="oasis_session_")
    os.close(storage_fd)

    try:
        if _using_cloak:
            result = await _run_with_cloakbrowser(url, storage_path, _launch_ctx)
        else:
            result = await _run_with_playwright(url, storage_path, _apw)
    except Exception as exc:
        try:
            os.unlink(storage_path)
        except OSError:
            pass
        return CaptchaGuardResult(
            status="aborted",
            message=f"❌ Analysis aborted: unexpected error during pre-flight check — {exc}",
        )

    return result


# ── CloakBrowser implementation ────────────────────────────────────────────────
async def _run_with_cloakbrowser(url: str, storage_path: str, launch_ctx) -> CaptchaGuardResult:
    """Run the pre-flight check using CloakBrowser (stealth Chromium)."""
    from playwright.async_api import TimeoutError as PWTimeout

    # Build launch kwargs
    launch_kwargs: dict = {
        "viewport": {"width": 1280, "height": 800},
        "locale": "en-US",
    }
    if CLOAKBROWSER_HUMANIZE:
        launch_kwargs["humanize"] = True
        launch_kwargs["human_preset"] = CLOAKBROWSER_HUMAN_PRESET
    if CLOAKBROWSER_PROXY:
        launch_kwargs["proxy"] = CLOAKBROWSER_PROXY
        if CLOAKBROWSER_GEOIP:
            launch_kwargs["geoip"] = True
    else:
        # No proxy — set a sensible default timezone
        launch_kwargs["timezone"] = "America/New_York"

    ctx = await launch_ctx(**launch_kwargs)

    try:
        page = await ctx.new_page()

        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            # Give JS challenges time to render (Cloudflare Turnstile takes ~1.5s)
            await page.wait_for_timeout(2500)
        except PWTimeout:
            await ctx.close()
            os.unlink(storage_path)
            return CaptchaGuardResult(
                status="aborted",
                message="❌ Analysis aborted: the website did not respond within 30 seconds.",
            )
        except Exception as exc:
            await ctx.close()
            os.unlink(storage_path)
            return CaptchaGuardResult(
                status="aborted",
                message=f"❌ Analysis aborted: navigation error — {exc}",
            )

        # ── Step 1: Detect CAPTCHA ─────────────────────────────────────────────
        captcha_type = await _detect_captcha(page)

        if captcha_type is None:
            # No CAPTCHA — CloakBrowser successfully passed bot detection
            page_text = await _extract_text(page)
            await ctx.storage_state(path=storage_path)
            await ctx.close()
            return CaptchaGuardResult(
                status="clear",
                page_text=page_text,
                storage_state_path=storage_path,
                message="✅ Website accessible — CloakBrowser passed bot detection, no CAPTCHA.",
            )

        # ── Step 2: Humanized checkbox click ──────────────────────────────────
        # CloakBrowser's humanize=True makes this click indistinguishable from
        # a real user — Bézier curve, realistic aim point, hold duration.
        clicked = await _try_checkbox_click(page)
        if clicked:
            await page.wait_for_timeout(3500)
            captcha_type_after = await _detect_captcha(page)
            if captcha_type_after is None:
                page_text = await _extract_text(page)
                await ctx.storage_state(path=storage_path)
                await ctx.close()
                return CaptchaGuardResult(
                    status="solved",
                    page_text=page_text,
                    storage_state_path=storage_path,
                    captcha_type=captcha_type,
                    message=f"✅ CAPTCHA ({captcha_type}) solved via humanized click.",
                )

        # ── Step 3: 2Captcha token injection (last resort) ────────────────────
        if TWOCAPTCHA_API_KEY:
            try:
                solved = await _solve_with_2captcha(page, url, captcha_type)
                if solved:
                    await page.wait_for_timeout(3000)
                    captcha_type_after = await _detect_captcha(page)
                    if captcha_type_after is None:
                        page_text = await _extract_text(page)
                        await ctx.storage_state(path=storage_path)
                        await ctx.close()
                        return CaptchaGuardResult(
                            status="solved",
                            page_text=page_text,
                            storage_state_path=storage_path,
                            captcha_type=captcha_type,
                            message=f"✅ CAPTCHA ({captcha_type}) solved via 2Captcha.",
                        )
            except Exception:
                pass  # fall through to abort

        # ── Step 4: Abort ──────────────────────────────────────────────────────
        await ctx.close()
        try:
            os.unlink(storage_path)
        except OSError:
            pass

        return CaptchaGuardResult(
            status="aborted",
            captcha_type=captcha_type,
            message=(
                f"❌ Analysis aborted: the website is protected by a {captcha_type} "
                f"challenge that could not be solved automatically.\n\n"
                f"**Options to proceed:**\n"
                f"1. Add `TWOCAPTCHA_API_KEY` to your `.env` file for automated solving.\n"
                f"2. Add `CLOAKBROWSER_PROXY` to route traffic through a residential IP "
                f"(reduces challenge frequency significantly).\n"
                f"3. Manually solve the CAPTCHA in a real browser and paste the cookies "
                f"into the session state field in ⚙️ Settings."
            ),
        )

    except Exception as exc:
        try:
            await ctx.close()
        except Exception:
            pass
        try:
            os.unlink(storage_path)
        except OSError:
            pass
        raise


# ── Standard Playwright fallback (if cloakbrowser not installed) ───────────────
async def _run_with_playwright(url: str, storage_path: str, async_playwright) -> CaptchaGuardResult:
    """Fallback: run the pre-flight check with standard Playwright."""
    from playwright.async_api import TimeoutError as PWTimeout

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled",
            ],
        )
        ctx = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
            locale="en-US",
            timezone_id="America/New_York",
        )
        # Minimal webdriver flag removal (less effective than CloakBrowser)
        await ctx.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', { get: () => undefined });"
        )
        page = await ctx.new_page()

        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            await page.wait_for_timeout(2000)
        except PWTimeout:
            await browser.close()
            os.unlink(storage_path)
            return CaptchaGuardResult(
                status="aborted",
                message="❌ Analysis aborted: the website did not respond within 30 seconds.",
            )
        except Exception as exc:
            await browser.close()
            os.unlink(storage_path)
            return CaptchaGuardResult(
                status="aborted",
                message=f"❌ Analysis aborted: navigation error — {exc}",
            )

        captcha_type = await _detect_captcha(page)
        if captcha_type is None:
            page_text = await _extract_text(page)
            await ctx.storage_state(path=storage_path)
            await browser.close()
            return CaptchaGuardResult(
                status="clear",
                page_text=page_text,
                storage_state_path=storage_path,
                message="✅ Website accessible — no CAPTCHA detected. (Install cloakbrowser for stealth mode.)",
            )

        clicked = await _try_checkbox_click(page)
        if clicked:
            await page.wait_for_timeout(3000)
            if await _detect_captcha(page) is None:
                page_text = await _extract_text(page)
                await ctx.storage_state(path=storage_path)
                await browser.close()
                return CaptchaGuardResult(
                    status="solved",
                    page_text=page_text,
                    storage_state_path=storage_path,
                    captcha_type=captcha_type,
                    message=f"✅ CAPTCHA ({captcha_type}) solved via checkbox click.",
                )

        if TWOCAPTCHA_API_KEY:
            try:
                if await _solve_with_2captcha(page, url, captcha_type):
                    await page.wait_for_timeout(3000)
                    if await _detect_captcha(page) is None:
                        page_text = await _extract_text(page)
                        await ctx.storage_state(path=storage_path)
                        await browser.close()
                        return CaptchaGuardResult(
                            status="solved",
                            page_text=page_text,
                            storage_state_path=storage_path,
                            captcha_type=captcha_type,
                            message=f"✅ CAPTCHA ({captcha_type}) solved via 2Captcha.",
                        )
            except Exception:
                pass

        await browser.close()
        try:
            os.unlink(storage_path)
        except OSError:
            pass
        return CaptchaGuardResult(
            status="aborted",
            captcha_type=captcha_type,
            message=(
                f"❌ Analysis aborted: {captcha_type} challenge could not be solved.\n\n"
                f"Install `cloakbrowser` for stealth mode (passes most challenges automatically), "
                f"or add `TWOCAPTCHA_API_KEY` to your `.env` for automated solving."
            ),
        )


# ── Shared helpers ─────────────────────────────────────────────────────────────

async def _detect_captcha(page) -> Optional[str]:
    """Return a short CAPTCHA type string if any challenge is detected, else None."""
    for sel in _CAPTCHA_SELECTORS:
        try:
            count = await page.locator(sel).count()
            if count > 0:
                if "recaptcha" in sel:
                    return "reCAPTCHA"
                if "hcaptcha" in sel:
                    return "hCaptcha"
                if "cloudflare" in sel or "cf-" in sel or "challenge" in sel:
                    return "Cloudflare Turnstile"
                return "CAPTCHA"
        except Exception:
            pass

    # Also check page title for common challenge phrases
    try:
        title = await page.title()
        if any(kw in title.lower() for kw in [
            "just a moment", "attention required", "security check",
            "ddos-guard", "are you human", "bot check",
        ]):
            return "Cloudflare / Bot-wall"
    except Exception:
        pass

    return None


async def _try_checkbox_click(page) -> bool:
    """Try to click the 'I am human' checkbox. Returns True if a click was made."""
    for sel in _CHECKBOX_SELECTORS:
        try:
            loc = page.locator(sel)
            if await loc.count() > 0:
                await loc.first.click(timeout=4000)
                return True
        except Exception:
            pass
    return False


async def _extract_text(page) -> str:
    """Extract visible body text from the page."""
    try:
        return await page.evaluate("() => document.body.innerText")
    except Exception:
        return ""


async def _solve_with_2captcha(page, url: str, captcha_type: str) -> bool:
    """
    Submit the CAPTCHA to 2Captcha and inject the returned token.
    Returns True if a token was successfully injected.

    Supports reCAPTCHA v2 and hCaptcha. Turnstile support is best-effort.
    Note: With CloakBrowser, this should rarely be needed.
    """
    import httpx

    # Extract site key from the page
    site_key = None
    try:
        site_key = await page.evaluate("""() => {
            const el = document.querySelector('[data-sitekey]');
            return el ? el.getAttribute('data-sitekey') : null;
        }""")
    except Exception:
        pass

    if not site_key:
        return False

    # Determine task type
    if captcha_type == "hCaptcha":
        task_type = "HCaptchaTaskProxyless"
    else:
        task_type = "RecaptchaV2TaskProxyless"

    async with httpx.AsyncClient(timeout=120) as client:
        # Submit task
        try:
            r = await client.post(
                "https://api.2captcha.com/createTask",
                json={
                    "clientKey": TWOCAPTCHA_API_KEY,
                    "task": {
                        "type": task_type,
                        "websiteURL": url,
                        "websiteKey": site_key,
                    },
                },
            )
            r.raise_for_status()
            task_id = r.json().get("taskId")
            if not task_id:
                return False
        except Exception:
            return False

        # Poll for result (up to CAPTCHA_TIMEOUT seconds)
        deadline = time.time() + CAPTCHA_TIMEOUT
        token = None
        while time.time() < deadline:
            await asyncio.sleep(5)
            try:
                r = await client.post(
                    "https://api.2captcha.com/getTaskResult",
                    json={"clientKey": TWOCAPTCHA_API_KEY, "taskId": task_id},
                )
                r.raise_for_status()
                data = r.json()
                if data.get("status") == "ready":
                    token = data.get("solution", {}).get("gRecaptchaResponse")
                    break
            except Exception:
                pass

        if not token:
            return False

    # Inject the token into the page
    try:
        await page.evaluate(f"""(token) => {{
            const ta = document.getElementById('g-recaptcha-response');
            if (ta) {{ ta.value = token; ta.style.display = 'block'; }}
            const ta2 = document.querySelector('textarea[name="h-captcha-response"]');
            if (ta2) {{ ta2.value = token; }}
            if (typeof ___grecaptcha_cfg !== 'undefined') {{
                Object.entries(___grecaptcha_cfg.clients || {{}}).forEach(([k, v]) => {{
                    const cb = v?.['']?.['']?.callback;
                    if (typeof cb === 'function') cb(token);
                }});
            }}
        }}""", token)
        return True
    except Exception:
        return False
