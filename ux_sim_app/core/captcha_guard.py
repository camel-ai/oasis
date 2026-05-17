"""
captcha_guard.py
================
Pre-flight CAPTCHA guard for OASIS.

Strategy
--------
1. Launch a real Playwright Chromium browser (non-headless capable, stealth flags).
2. Navigate to the target URL and wait for network idle.
3. Detect known CAPTCHA / bot-challenge signatures in the DOM.
4. If detected:
   a. Attempt a simple DOM click (Turnstile / "I am human" checkbox).
   b. If still blocked and a 2captcha API key is configured, extract the
      site-key and submit to 2Captcha; inject the returned token.
   c. If still blocked after CAPTCHA_TIMEOUT seconds → raise CaptchaAbortError.
5. On success: extract verified page text and save Playwright storage state
   (cookies + localStorage) to a temp file so downstream sessions inherit it.
6. Return a CaptchaGuardResult containing:
   - status: "clear" | "solved" | "aborted"
   - page_text: verified body text (empty on abort)
   - storage_state_path: path to JSON session file (None on abort)
   - message: human-readable status for the UI

Environment variables (all optional)
--------------------------------------
TWOCAPTCHA_API_KEY   — 2Captcha API key for automated solving
CAPTCHA_TIMEOUT      — seconds to wait for a solve before aborting (default 45)
CAPTCHA_ENABLED      — set to "false" to skip the guard entirely (default "true")
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Optional

# ── Configuration ──────────────────────────────────────────────────────────────
TWOCAPTCHA_API_KEY: str = os.getenv("TWOCAPTCHA_API_KEY", "")
CAPTCHA_TIMEOUT: int = int(os.getenv("CAPTCHA_TIMEOUT", "45"))
CAPTCHA_ENABLED: bool = os.getenv("CAPTCHA_ENABLED", "true").lower() != "false"

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
    status: str                          # "clear" | "solved" | "aborted" | "disabled"
    page_text: str = ""                  # verified body text (empty on abort)
    storage_state_path: Optional[str] = None  # path to Playwright storage state JSON
    message: str = ""                    # human-readable status for the UI
    captcha_type: Optional[str] = None   # detected CAPTCHA type (for logging)


class CaptchaAbortError(RuntimeError):
    """Raised when a CAPTCHA cannot be solved within the timeout."""


# ── Main entry point ───────────────────────────────────────────────────────────
async def preflight_check(url: str) -> CaptchaGuardResult:
    """
    Run the pre-flight CAPTCHA guard for *url*.

    Returns a CaptchaGuardResult. If status == "aborted", callers MUST stop
    processing and surface the .message to the user.
    """
    if not CAPTCHA_ENABLED:
        return CaptchaGuardResult(
            status="disabled",
            message="CAPTCHA guard disabled (CAPTCHA_ENABLED=false). Proceeding without check.",
        )

    from playwright.async_api import async_playwright, TimeoutError as PWTimeout

    storage_fd, storage_path = tempfile.mkstemp(suffix=".json", prefix="oasis_session_")
    os.close(storage_fd)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled",
                "--disable-features=IsolateOrigins,site-per-process",
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

        # Stealth: remove webdriver flag before any page load
        await ctx.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3] });
            Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
            window.chrome = { runtime: {} };
        """)

        page = await ctx.new_page()

        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            await page.wait_for_timeout(2000)  # let JS challenges render
        except PWTimeout:
            await browser.close()
            os.unlink(storage_path)
            return CaptchaGuardResult(
                status="aborted",
                message=f"❌ Analysis aborted: the website did not respond within 30 seconds.",
            )
        except Exception as exc:
            await browser.close()
            os.unlink(storage_path)
            return CaptchaGuardResult(
                status="aborted",
                message=f"❌ Analysis aborted: navigation error — {exc}",
            )

        # ── Step 1: Detect CAPTCHA ─────────────────────────────────────────────
        captcha_type = await _detect_captcha(page)

        if captcha_type is None:
            # No CAPTCHA — extract text and save session
            page_text = await _extract_text(page)
            await ctx.storage_state(path=storage_path)
            await browser.close()
            return CaptchaGuardResult(
                status="clear",
                page_text=page_text,
                storage_state_path=storage_path,
                message="✅ Website accessible — no CAPTCHA detected.",
            )

        # ── Step 2: Attempt simple checkbox click ──────────────────────────────
        clicked = await _try_checkbox_click(page)
        if clicked:
            await page.wait_for_timeout(3000)
            captcha_type_after = await _detect_captcha(page)
            if captcha_type_after is None:
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

        # ── Step 3: Attempt 2Captcha token injection ───────────────────────────
        if TWOCAPTCHA_API_KEY:
            try:
                solved = await _solve_with_2captcha(page, url, captcha_type)
                if solved:
                    await page.wait_for_timeout(3000)
                    captcha_type_after = await _detect_captcha(page)
                    if captcha_type_after is None:
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
            except Exception as exc:
                pass  # fall through to abort

        # ── Step 4: Abort ──────────────────────────────────────────────────────
        await browser.close()
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
                f"**To proceed:** add a TWOCAPTCHA_API_KEY to your .env file, or "
                f"manually solve the CAPTCHA in a real browser and paste the cookies "
                f"into the session state field in ⚙️ Settings."
            ),
        )


# ── Helpers ────────────────────────────────────────────────────────────────────

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

    # Also check page title / body text for common challenge phrases
    try:
        title = await page.title()
        if any(kw in title.lower() for kw in ["just a moment", "attention required",
                                               "security check", "ddos-guard",
                                               "are you human", "bot check"]):
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


async def _solve_with_2captcha(page, url: str, captcha_type: str) -> bool:
    """
    Submit the CAPTCHA to 2Captcha and inject the returned token.
    Returns True if a token was successfully injected.

    Supports reCAPTCHA v2 and hCaptcha. Turnstile support is best-effort.
    """
    import httpx

    # Extract site key from DOM
    site_key = await _extract_site_key(page, captcha_type)
    if not site_key:
        return False

    # Determine 2Captcha method
    if captcha_type == "reCAPTCHA":
        method = "userrecaptcha"
        token_field = "g-recaptcha-response"
        callback_fn = "___grecaptcha_cfg.clients[0].aa.aa.callback"
    elif captcha_type == "hCaptcha":
        method = "hcaptcha"
        token_field = "h-captcha-response"
        callback_fn = None
    else:
        return False  # Turnstile requires a different flow

    # Submit to 2Captcha
    async with httpx.AsyncClient(timeout=90.0) as client:
        # Step 1: Submit task
        resp = await client.post(
            "https://2captcha.com/in.php",
            data={
                "key": TWOCAPTCHA_API_KEY,
                "method": method,
                "googlekey": site_key,
                "pageurl": url,
                "json": 1,
            },
        )
        data = resp.json()
        if data.get("status") != 1:
            return False
        task_id = data["request"]

        # Step 2: Poll for result
        deadline = time.time() + CAPTCHA_TIMEOUT
        while time.time() < deadline:
            await asyncio.sleep(5)
            poll = await client.get(
                f"https://2captcha.com/res.php?key={TWOCAPTCHA_API_KEY}"
                f"&action=get&id={task_id}&json=1"
            )
            poll_data = poll.json()
            if poll_data.get("status") == 1:
                token = poll_data["request"]
                break
            if poll_data.get("request") not in ("CAPCHA_NOT_READY", "CAPTCHA_NOT_READY"):
                return False
        else:
            return False

    # Step 3: Inject token into the page
    await page.evaluate(f"""
        (token) => {{
            // Set the hidden textarea value
            const el = document.querySelector('#{token_field}') ||
                        document.querySelector('[name="{token_field}"]');
            if (el) el.value = token;

            // Try to fire the callback
            try {{
                const cfg = window.___grecaptcha_cfg;
                if (cfg) {{
                    const clients = cfg.clients;
                    if (clients) {{
                        Object.values(clients).forEach(c => {{
                            try {{
                                const cb = c && c.aa && c.aa.aa && c.aa.aa.callback;
                                if (typeof cb === 'function') cb(token);
                            }} catch(e) {{}}
                        }});
                    }}
                }}
            }} catch(e) {{}}

            // hCaptcha callback
            try {{
                if (window.hcaptcha) window.hcaptcha.execute();
            }} catch(e) {{}}
        }}
    """, token)

    return True


async def _extract_site_key(page, captcha_type: str) -> Optional[str]:
    """Extract the CAPTCHA site key from the page DOM."""
    selectors = {
        "reCAPTCHA": ["div[data-sitekey]", ".g-recaptcha[data-sitekey]",
                      "iframe[src*='recaptcha'][src*='k=']"],
        "hCaptcha":  ["div[data-sitekey]", ".h-captcha[data-sitekey]"],
    }
    for sel in selectors.get(captcha_type, []):
        try:
            loc = page.locator(sel).first
            if await loc.count() > 0:
                key = await loc.get_attribute("data-sitekey")
                if key:
                    return key
                # Try extracting from iframe src
                src = await loc.get_attribute("src") or ""
                import re
                m = re.search(r"[?&]k=([^&]+)", src)
                if m:
                    return m.group(1)
        except Exception:
            pass
    return None


async def _extract_text(page) -> str:
    """Extract clean body text from the current page."""
    try:
        from bs4 import BeautifulSoup
        html = await page.content()
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "head", "noscript"]):
            tag.decompose()
        return " ".join(soup.get_text(separator=" ").split())[:5000]
    except Exception:
        return ""
