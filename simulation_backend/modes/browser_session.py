"""
BrowserSession – a single Playwright browser session for one persona.

Adapted from feat/browser-actions-17194438393437691205 with:
  - RSS auto-discovery fallback for Cloudflare-blocked pages
  - Stealth mode via playwright-stealth when available
  - Clean async context-manager interface
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional
from urllib.parse import urljoin, urlparse

import httpx

log = logging.getLogger("simulation_backend.browser_session")

try:
    from bs4 import BeautifulSoup
    import feedparser
    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False

try:
    from playwright.async_api import (
        Browser,
        BrowserContext,
        Page,
        async_playwright,
    )
    _PLAYWRIGHT_AVAILABLE = True
except ImportError:
    _PLAYWRIGHT_AVAILABLE = False
    log.warning("Playwright not installed – browser sessions will be unavailable.")


class BrowserSession:
    """Manages a single headless Chromium session for one agent."""

    def __init__(self, agent_id: int) -> None:
        self.agent_id = agent_id
        self._playwright: Any = None
        self._browser: Optional[Any] = None
        self._context: Optional[Any] = None
        self._page: Optional[Any] = None
        self.started = False

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def start(self) -> Dict[str, Any]:
        if self.started:
            return {"status": "already_started", "agent_id": self.agent_id}
        if not _PLAYWRIGHT_AVAILABLE:
            raise RuntimeError(
                "Playwright is not installed. "
                "Run: pip install playwright && playwright install chromium"
            )
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        self._context = await self._browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
        )
        self._page = await self._context.new_page()

        # Apply stealth patch if available
        try:
            from playwright_stealth import stealth_async
            await stealth_async(self._page)
        except ImportError:
            pass

        self.started = True
        log.debug("Browser session started for agent %s", self.agent_id)
        return {"status": "started", "agent_id": self.agent_id}

    async def close(self) -> Dict[str, Any]:
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        self.started = False
        log.debug("Browser session closed for agent %s", self.agent_id)
        return {"status": "closed", "agent_id": self.agent_id}

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *args):
        await self.close()

    # ── Navigation ─────────────────────────────────────────────────────────

    async def navigate(self, url: str) -> Dict[str, Any]:
        if not self.started:
            raise RuntimeError("Browser not started – call start() first.")
        try:
            await self._page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        except Exception as exc:
            log.warning("Navigation error for %s: %s", url, exc)
            return {"status": "error", "url": url, "error": str(exc)}

        if await self._is_blocked():
            log.info("Page blocked (Cloudflare?), attempting RSS fallback for %s", url)
            rss_result = await self._rss_fallback(url)
            if rss_result:
                return rss_result

        return {"status": "success", "url": self._page.url}

    async def extract_dom(self) -> Dict[str, Any]:
        if not self.started:
            raise RuntimeError("Browser not started.")
        content = await self._page.content()
        return {"status": "success", "dom": content, "url": self._page.url}

    async def click(self, selector: str) -> Dict[str, Any]:
        if not self.started:
            raise RuntimeError("Browser not started.")
        try:
            await self._page.click(selector, timeout=10_000)
            return {"status": "success", "selector": selector}
        except Exception as exc:
            return {"status": "error", "selector": selector, "error": str(exc)}

    async def type_text(self, selector: str, text: str) -> Dict[str, Any]:
        if not self.started:
            raise RuntimeError("Browser not started.")
        try:
            await self._page.fill(selector, text)
            return {"status": "success", "selector": selector}
        except Exception as exc:
            return {"status": "error", "selector": selector, "error": str(exc)}

    async def fetch_rss_feed(self, url: str) -> Dict[str, Any]:
        """Directly fetch and parse an RSS/Atom feed."""
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                headers = {
                    "User-Agent": (
                        "Mozilla/5.0 (compatible; Googlebot/2.1; "
                        "+http://www.google.com/bot.html)"
                    )
                }
                resp = await client.get(url, headers=headers)
                resp.raise_for_status()
            if not _BS4_AVAILABLE:
                return {"status": "success", "raw": resp.text, "articles": []}
            import feedparser as fp
            feed = fp.parse(resp.text)
            articles = [
                {
                    "title": e.get("title", ""),
                    "link": e.get("link", ""),
                    "summary": e.get("summary", ""),
                    "published": e.get("published", ""),
                }
                for e in feed.entries[:20]
            ]
            return {"status": "success", "articles": articles}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    # ── Anti-bot helpers ───────────────────────────────────────────────────

    async def _is_blocked(self) -> bool:
        try:
            content = await self._page.content()
            lower = content.lower()
            if "cloudflare" in lower and (
                "<title>just a moment" in lower
                or "please wait while your request" in lower
            ):
                return True
        except Exception:
            pass
        return False

    async def _discover_rss(self, base_url: str) -> Optional[str]:
        parsed = urlparse(base_url)
        domain = f"{parsed.scheme}://{parsed.netloc}"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; Googlebot/2.1; "
                "+http://www.google.com/bot.html)"
            )
        }
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.get(domain, headers=headers)
                if resp.status_code == 200 and _BS4_AVAILABLE:
                    soup = BeautifulSoup(resp.text, "html.parser")
                    for tag in soup.find_all("link", type="application/rss+xml"):
                        if tag.get("href"):
                            return urljoin(domain, tag["href"])
        except Exception:
            pass
        for path in ["/feed", "/rss", "/feed.xml", "/rss.xml", "/news/rss"]:
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    test_url = urljoin(domain, path)
                    resp = await client.get(test_url, headers=headers)
                    ct = resp.headers.get("Content-Type", "").lower()
                    if resp.status_code == 200 and (
                        "xml" in ct or "rss" in resp.text.lower()[:200]
                    ):
                        return test_url
            except Exception:
                pass
        return None

    async def _rss_fallback(self, original_url: str) -> Optional[Dict[str, Any]]:
        rss_url = await self._discover_rss(original_url)
        if not rss_url:
            return None
        feed_data = await self.fetch_rss_feed(rss_url)
        if feed_data.get("status") != "success":
            return None
        for article in feed_data.get("articles", []):
            link = article.get("link", "")
            if link == original_url or original_url in link:
                html = (
                    f"<html><body>"
                    f"<h1>{article.get('title', '')}</h1>"
                    f"<p>{article.get('summary', '')}</p>"
                    f"</body></html>"
                )
                await self._page.set_content(html)
                return {
                    "status": "rss_fallback",
                    "url": original_url,
                    "title": article.get("title"),
                    "summary": article.get("summary"),
                }
        # Return feed homepage summary
        return {
            "status": "rss_feed",
            "url": rss_url,
            "articles": feed_data.get("articles", []),
        }


# ── Session pool ───────────────────────────────────────────────────────────────

class BrowserSessionPool:
    """
    Manages a pool of BrowserSession objects with a concurrency cap.
    """

    def __init__(self, max_sessions: int = 8) -> None:
        self._max = max_sessions
        self._semaphore = asyncio.Semaphore(max_sessions)
        self._sessions: dict[int, BrowserSession] = {}

    def get_session(self, agent_id: int) -> BrowserSession:
        if agent_id not in self._sessions:
            self._sessions[agent_id] = BrowserSession(agent_id)
        return self._sessions[agent_id]

    async def remove_session(self, agent_id: int) -> None:
        session = self._sessions.pop(agent_id, None)
        if session and session.started:
            await session.close()

    @property
    def semaphore(self) -> asyncio.Semaphore:
        return self._semaphore

    async def close_all(self) -> None:
        for session in list(self._sessions.values()):
            if session.started:
                await session.close()
        self._sessions.clear()
