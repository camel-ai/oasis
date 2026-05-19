"""
Website scraper: fetches text content, links, and image URLs from a target URL.

Improvements (v3):
- storage_state_path is now propagated to ALL subpage fetches, not just the root.
- Playwright path uses CloakBrowser (stealth Chromium) when available, falling back
  to standard Playwright so the cleared CAPTCHA session is honoured on every page.
- Subpage fetches run concurrently (asyncio.gather) instead of sequentially.
- Scraper extracts structured sections (hero, features, pricing, CTA, footer) rather
  than a raw text blob, giving personas richer context.
- Body text limit raised to 8,000 chars; per-subpage limit raised to 2,500 chars.
"""
from __future__ import annotations
import asyncio
import re
from dataclasses import dataclass, field
from typing import List, Optional
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

_SECTION_SELECTORS = [
    # Hero / above-the-fold
    ("hero",    ["[class*='hero']", "[class*='banner']", "[class*='jumbotron']", "header main", "section:first-of-type"]),
    # Features / benefits
    ("features",["[class*='feature']", "[class*='benefit']", "[class*='why']", "[class*='how-it-works']"]),
    # Pricing
    ("pricing", ["[class*='pric']", "[class*='plan']", "[id*='pric']"]),
    # CTA / conversion
    ("cta",     ["[class*='cta']", "[class*='call-to-action']", "[class*='signup']", "[class*='get-started']"]),
    # Testimonials / social proof
    ("social",  ["[class*='testimonial']", "[class*='review']", "[class*='trust']", "[class*='social-proof']"]),
]


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
    Scrape a URL and concurrently follow up to `follow_links` internal links.

    *storage_state_path* (from captcha_guard) is passed to EVERY page fetch so
    the cleared session is honoured throughout the entire scrape, not just the root.
    """
    result = await _fetch_page(url, storage_state_path=storage_state_path)
    if result.error or follow_links <= 0:
        return result

    interesting = _pick_interesting_links(result.links, url, follow_links)

    # Fetch all subpages concurrently, passing the cleared session to each
    sub_results = await asyncio.gather(
        *[_fetch_page(link, storage_state_path=storage_state_path) for link in interesting],
        return_exceptions=True,
    )

    extra_texts = []
    for sub in sub_results:
        if isinstance(sub, Exception) or sub.error:
            continue
        if sub.body_text:
            extra_texts.append(f"\n\n--- {sub.url} ---\n{sub.body_text[:2500]}")
        result.image_urls.extend(sub.image_urls)

    result.body_text = result.body_text + "".join(extra_texts)
    # Deduplicate images
    result.image_urls = list(dict.fromkeys(result.image_urls))[:30]
    return result


async def _fetch_page(url: str, storage_state_path: Optional[str] = None) -> ScrapeResult:
    """Fetch a single page.

    When *storage_state_path* is provided, uses CloakBrowser (stealth Chromium)
    with the pre-cleared session so bot-detection walls are not re-triggered on
    subsequent pages. Falls back to standard Playwright if CloakBrowser is not
    installed, and to httpx if neither is available.
    """
    html: str = ""

    if storage_state_path:
        html = await _playwright_fetch(url, storage_state_path)
        if html.startswith("ERROR:"):
            return ScrapeResult(url=url, error=html[6:])
    else:
        try:
            async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
                r = await client.get(url, headers=HEADERS)
                r.raise_for_status()
                html = r.text
        except Exception as exc:
            return ScrapeResult(url=url, error=str(exc))

    return _parse_html(url, html)


async def _playwright_fetch(url: str, storage_state_path: str) -> str:
    """Fetch page HTML using CloakBrowser (preferred) or standard Playwright."""
    # Try CloakBrowser first
    try:
        from cloakbrowser import launch_context_async  # type: ignore
        ctx = await launch_context_async(
            humanize=False,  # speed over humanisation for scraping
            storage_state=storage_state_path,
        )
        page = await ctx.new_page()
        await page.goto(url, wait_until="domcontentloaded", timeout=60_000)
        await page.wait_for_timeout(1500)
        html = await page.content()
        await ctx.close()
        return html
    except ImportError:
        pass
    except Exception as exc:
        return f"ERROR:CloakBrowser fetch failed: {exc}"

    # Fallback: standard Playwright
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
            await page.goto(url, wait_until="domcontentloaded", timeout=60_000)
            await page.wait_for_timeout(1500)
            html = await page.content()
            await browser.close()
            return html
    except Exception as exc:
        return f"ERROR:Playwright fetch failed: {exc}"


def _parse_html(url: str, html: str) -> ScrapeResult:
    """Parse raw HTML into a ScrapeResult with structured section extraction."""
    soup = BeautifulSoup(html, "html.parser")
    base = f"{urlparse(url).scheme}://{urlparse(url).netloc}"

    # Title
    title = soup.title.string.strip() if soup.title and soup.title.string else ""

    # Meta description
    meta = soup.find("meta", attrs={"name": re.compile("description", re.I)})
    description = meta.get("content", "").strip() if meta else ""

    # ── Structured section extraction ─────────────────────────────────────────
    section_texts: List[str] = []
    for section_name, selectors in _SECTION_SELECTORS:
        for sel in selectors:
            try:
                el = soup.select_one(sel)
                if el:
                    text = " ".join(el.get_text(separator=" ").split())
                    if len(text) > 40:
                        section_texts.append(f"[{section_name.upper()}] {text[:600]}")
                        break
            except Exception:
                continue

    # ── Fallback: full body text ───────────────────────────────────────────────
    for tag in soup(["script", "style", "nav", "footer", "head", "noscript"]):
        tag.decompose()
    full_body = " ".join(soup.get_text(separator=" ").split())

    if section_texts:
        # Structured sections first, then remaining body text for context
        structured = "\n".join(section_texts)
        remaining_budget = 8000 - len(structured)
        body_text = structured + ("\n\n" + full_body[:max(remaining_budget, 0)] if remaining_budget > 200 else "")
    else:
        body_text = full_body[:8000]

    # ── Links ─────────────────────────────────────────────────────────────────
    links: List[str] = []
    nav_links: List[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith("/"):
            href = base + href
        elif not href.startswith("http"):
            continue
        links.append(href)
        if len(a.get_text(strip=True)) < 30:
            nav_links.append(href)

    # ── Images ────────────────────────────────────────────────────────────────
    image_urls: List[str] = []
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
        image_urls=list(dict.fromkeys(image_urls))[:30],
        nav_links=list(dict.fromkeys(nav_links))[:15],
    )


def _pick_interesting_links(links: List[str], base_url: str, n: int) -> List[str]:
    """Pick up to n internal links that look like content pages."""
    base_domain = urlparse(base_url).netloc
    keywords = [
        "menu", "about", "food", "drink", "product", "service", "feature",
        "pricing", "plan", "lunch", "dinner", "breakfast", "contact",
        "event", "offer", "how", "why", "solution", "use-case",
    ]
    scored = []
    for link in links:
        if urlparse(link).netloc != base_domain:
            continue
        path = urlparse(link).path.lower()
        score = sum(1 for kw in keywords if kw in path)
        if score > 0:
            scored.append((score, link))
    scored.sort(reverse=True)
    return [lnk for _, lnk in scored[:n]]
