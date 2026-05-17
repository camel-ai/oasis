"""
Website scraper: fetches text content, links, and image URLs from a target URL.
Uses httpx + BeautifulSoup for standard sites. When a CAPTCHA is detected the
scraper delegates to captcha_guard.py which uses a real Playwright browser to
solve the challenge before extracting content.
"""
from __future__ import annotations
import asyncio
import re
from dataclasses import dataclass, field
from typing import List, Optional
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

# Optional: captcha_guard is imported lazily to avoid circular imports
_captcha_guard = None

def _get_captcha_guard():
    global _captcha_guard
    if _captcha_guard is None:
        from ux_sim_app.core import captcha_guard as cg
        _captcha_guard = cg
    return _captcha_guard

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


@dataclass
class ScrapeResult:
    url: str
    title: str = ""
    description: str = ""
    body_text: str = ""
    links: List[str] = field(default_factory=list)
    image_urls: List[str] = field(default_factory=list)
    nav_links: List[str] = field(default_factory=list)
    error: Optional[str] = None


async def scrape(
    url: str,
    follow_links: int = 2,
    storage_state_path: Optional[str] = None,
) -> ScrapeResult:
    """
    Scrape a URL and optionally follow up to `follow_links` internal links
    (e.g. menu pages) to gather more content.

    If *storage_state_path* is provided (from a prior captcha_guard run), the
    scraper skips the httpx fetch and uses the already-verified page text that
    was extracted during the pre-flight check.
    """
    result = await _fetch_page(url, storage_state_path=storage_state_path)
    if result.error or follow_links <= 0:
        return result

    # Identify interesting sub-pages (menu, about, contact, etc.)
    interesting = _pick_interesting_links(result.links, url, follow_links)
    extra_texts = []
    for link in interesting:
        sub = await _fetch_page(link)
        if not sub.error and sub.body_text:
            extra_texts.append(f"\n\n--- {link} ---\n{sub.body_text[:1500]}")
            result.image_urls.extend(sub.image_urls)

    result.body_text = result.body_text + "".join(extra_texts)
    return result


async def _fetch_page(url: str, storage_state_path: Optional[str] = None) -> ScrapeResult:
    """Fetch a single page. Uses Playwright with session state when available,
    otherwise falls back to httpx."""
    html: str = ""

    if storage_state_path:
        # Use Playwright with the pre-cleared session state
        try:
            from playwright.async_api import async_playwright
            async with async_playwright() as pw:
                browser = await pw.chromium.launch(
                    headless=True,
                    args=["--no-sandbox", "--disable-dev-shm-usage"],
                )
                ctx = await browser.new_context(
                    user_agent=HEADERS["User-Agent"],
                    storage_state=storage_state_path,
                )
                page = await ctx.new_page()
                await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                await page.wait_for_timeout(1500)
                html = await page.content()
                await browser.close()
        except Exception as exc:
            return ScrapeResult(url=url, error=f"Playwright fetch failed: {exc}")
    else:
        try:
            async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
                r = await client.get(url, headers=HEADERS)
                r.raise_for_status()
                html = r.text
        except Exception as exc:
            return ScrapeResult(url=url, error=str(exc))

    soup = BeautifulSoup(html, "html.parser")

    # Title
    title = soup.title.string.strip() if soup.title else ""

    # Meta description
    meta = soup.find("meta", attrs={"name": re.compile("description", re.I)})
    description = meta.get("content", "").strip() if meta else ""

    # Body text (strip scripts/styles/nav/footer)
    for tag in soup(["script", "style", "nav", "footer", "head", "noscript"]):
        tag.decompose()
    body_text = " ".join(soup.get_text(separator=" ").split())[:5000]

    # Links
    base = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
    links = []
    nav_links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith("/"):
            href = base + href
        elif not href.startswith("http"):
            continue
        links.append(href)
        # Detect nav links (short anchor text)
        if len(a.get_text(strip=True)) < 30:
            nav_links.append(href)

    # Images
    image_urls = []
    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src", "")
        if src and not src.startswith("data:"):
            if src.startswith("//"):
                src = "https:" + src
            elif src.startswith("/"):
                src = base + src
            if src.startswith("http"):
                image_urls.append(src)

    return ScrapeResult(
        url=url,
        title=title,
        description=description,
        body_text=body_text,
        links=list(dict.fromkeys(links))[:40],
        image_urls=list(dict.fromkeys(image_urls))[:20],
        nav_links=list(dict.fromkeys(nav_links))[:15],
    )


def _pick_interesting_links(links: List[str], base_url: str, n: int) -> List[str]:
    """Pick up to n internal links that look like content pages (menu, about, etc.)."""
    base_domain = urlparse(base_url).netloc
    keywords = ["menu", "about", "food", "drink", "product", "service",
                 "lunch", "dinner", "breakfast", "contact", "event", "offer"]
    scored = []
    for link in links:
        if urlparse(link).netloc != base_domain:
            continue
        path = urlparse(link).path.lower()
        score = sum(1 for kw in keywords if kw in path)
        if score > 0:
            scored.append((score, link))
    scored.sort(reverse=True)
    return [l for _, l in scored[:n]]
