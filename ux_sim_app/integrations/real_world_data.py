"""
real_world_data.py
==================
Optional real-world signal gathering for OASIS persona enrichment.

Gathers community signals from multiple sources in parallel, then uses an LLM
to synthesize a compact "Real-World Briefing" that is injected into persona
generation so that personas reflect what real people are actually saying about
the brand / product / topic right now.

All sources are OPTIONAL. Any source that is not configured (no key, no token)
is silently skipped. The synthesizer works with whatever data is available.

Sources
-------
Free (no key required):
  reddit      — top posts + top comments from relevant subreddits via Reddit
                public JSON API (no auth needed for read-only)
  hackernews  — Algolia HN Search API (public, no key)
  github      — GitHub REST API public endpoints (no auth, rate-limited to 60/hr)

Bring-your-own-key:
  x_twitter   — Requires a browser cookie token (log into x.com in any browser,
                paste the `auth_token` cookie value)
  bluesky     — Requires an App Password from bsky.app
  youtube     — Requires `yt-dlp` installed (`brew install yt-dlp` or
                `pip install yt-dlp`). No API key needed.
  scrapecreators — ScrapeCreators API key (10,000 free calls) for TikTok,
                Instagram, Threads, Pinterest, YouTube comments
  brave_search — Brave Search API key (2,000 free queries/month)
  perplexity  — OpenRouter key (pay-as-you-go via Perplexity Sonar model)

Usage
-----
    from ux_sim_app.integrations.real_world_data import gather_and_synthesize

    briefing = gather_and_synthesize(
        topic="Tootgarook Bistro parma night",
        keys={
            "x_auth_token": "...",
            "bluesky_handle": "user.bsky.social",
            "bluesky_app_password": "xxxx-xxxx-xxxx-xxxx",
            "scrapecreators_key": "...",
            "brave_key": "...",
            "openrouter_key": "...",   # for Perplexity Sonar
        },
        max_items_per_source=5,
        openai_key="sk-...",
        text_model="gpt-4.1-mini",
    )
    # briefing is a plain-text paragraph ready to inject into persona generation
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

class RealWorldItem:
    """A single piece of gathered content from any source."""
    __slots__ = ("source", "title", "body", "score", "url")

    def __init__(
        self,
        source: str,
        title: str,
        body: str = "",
        score: int = 0,
        url: str = "",
    ):
        self.source = source
        self.title = title
        self.body = body
        self.score = score
        self.url = url

    def to_text(self) -> str:
        parts = [f"[{self.source}]"]
        if self.title:
            parts.append(self.title)
        if self.body:
            parts.append(f"— {self.body[:300]}")
        if self.score:
            parts.append(f"(score: {self.score})")
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Source: Reddit (public JSON API — no auth required)
# ---------------------------------------------------------------------------

async def _gather_reddit(topic: str, max_items: int, client: httpx.AsyncClient) -> list[RealWorldItem]:
    items: list[RealWorldItem] = []
    try:
        # Search Reddit for the topic
        url = "https://www.reddit.com/search.json"
        params = {"q": topic, "sort": "relevance", "t": "month", "limit": max_items}
        headers = {"User-Agent": "OASIS-UX-Research/1.0"}
        r = await client.get(url, params=params, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
        for post in data.get("data", {}).get("children", [])[:max_items]:
            p = post.get("data", {})
            title = p.get("title", "")
            selftext = (p.get("selftext") or "")[:200]
            score = p.get("score", 0)
            permalink = "https://reddit.com" + p.get("permalink", "")
            items.append(RealWorldItem("Reddit", title, selftext, score, permalink))
    except Exception as exc:
        logger.debug("Reddit gather failed: %s", exc)
    return items


# ---------------------------------------------------------------------------
# Source: Hacker News (Algolia API — no auth required)
# ---------------------------------------------------------------------------

async def _gather_hackernews(topic: str, max_items: int, client: httpx.AsyncClient) -> list[RealWorldItem]:
    items: list[RealWorldItem] = []
    try:
        url = "https://hn.algolia.com/api/v1/search"
        params = {"query": topic, "tags": "story", "hitsPerPage": max_items}
        r = await client.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        for hit in data.get("hits", [])[:max_items]:
            title = hit.get("title", "")
            body = (hit.get("story_text") or "")[:200]
            score = hit.get("points", 0)
            hn_url = f"https://news.ycombinator.com/item?id={hit.get('objectID', '')}"
            items.append(RealWorldItem("Hacker News", title, body, score, hn_url))
    except Exception as exc:
        logger.debug("HN gather failed: %s", exc)
    return items


# ---------------------------------------------------------------------------
# Source: GitHub (public REST API — no auth, 60 req/hr)
# ---------------------------------------------------------------------------

async def _gather_github(topic: str, max_items: int, client: httpx.AsyncClient) -> list[RealWorldItem]:
    items: list[RealWorldItem] = []
    try:
        url = "https://api.github.com/search/repositories"
        params = {"q": topic, "sort": "stars", "order": "desc", "per_page": max_items}
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        r = await client.get(url, params=params, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
        for repo in data.get("items", [])[:max_items]:
            name = repo.get("full_name", "")
            desc = (repo.get("description") or "")[:200]
            stars = repo.get("stargazers_count", 0)
            html_url = repo.get("html_url", "")
            items.append(RealWorldItem("GitHub", name, desc, stars, html_url))
    except Exception as exc:
        logger.debug("GitHub gather failed: %s", exc)
    return items


# ---------------------------------------------------------------------------
# Source: Bluesky (App Password required)
# ---------------------------------------------------------------------------

async def _gather_bluesky(
    topic: str, max_items: int, client: httpx.AsyncClient,
    handle: str, app_password: str,
) -> list[RealWorldItem]:
    items: list[RealWorldItem] = []
    if not handle or not app_password:
        return items
    try:
        # Authenticate
        auth_r = await client.post(
            "https://bsky.social/xrpc/com.atproto.server.createSession",
            json={"identifier": handle, "password": app_password},
            timeout=15,
        )
        auth_r.raise_for_status()
        token = auth_r.json().get("accessJwt", "")
        if not token:
            return items

        # Search posts
        search_r = await client.get(
            "https://bsky.social/xrpc/app.bsky.feed.searchPosts",
            params={"q": topic, "limit": max_items},
            headers={"Authorization": f"Bearer {token}"},
            timeout=15,
        )
        search_r.raise_for_status()
        for post in search_r.json().get("posts", [])[:max_items]:
            record = post.get("record", {})
            text = record.get("text", "")
            like_count = post.get("likeCount", 0)
            items.append(RealWorldItem("Bluesky", text[:120], "", like_count))
    except Exception as exc:
        logger.debug("Bluesky gather failed: %s", exc)
    return items


# ---------------------------------------------------------------------------
# Source: ScrapeCreators (TikTok, Instagram, Threads, Pinterest, YT comments)
# ---------------------------------------------------------------------------

async def _gather_scrapecreators(
    topic: str, max_items: int, client: httpx.AsyncClient,
    api_key: str,
) -> list[RealWorldItem]:
    items: list[RealWorldItem] = []
    if not api_key:
        return items
    try:
        # TikTok hashtag search
        r = await client.get(
            "https://api.scrapecreators.com/v1/tiktok/hashtag",
            params={"hashtag": topic.replace(" ", ""), "limit": max_items},
            headers={"x-api-key": api_key},
            timeout=20,
        )
        if r.status_code == 200:
            for post in r.json().get("data", {}).get("posts", [])[:max_items]:
                desc = post.get("desc", "")[:200]
                plays = post.get("stats", {}).get("playCount", 0)
                items.append(RealWorldItem("TikTok", desc, "", plays))
    except Exception as exc:
        logger.debug("ScrapeCreators TikTok failed: %s", exc)
    return items


# ---------------------------------------------------------------------------
# Source: Brave Search (2,000 free queries/month)
# ---------------------------------------------------------------------------

async def _gather_brave(
    topic: str, max_items: int, client: httpx.AsyncClient,
    api_key: str,
) -> list[RealWorldItem]:
    items: list[RealWorldItem] = []
    if not api_key:
        return items
    try:
        r = await client.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": topic, "count": max_items, "result_filter": "web"},
            headers={"Accept": "application/json", "X-Subscription-Token": api_key},
            timeout=15,
        )
        r.raise_for_status()
        for result in r.json().get("web", {}).get("results", [])[:max_items]:
            title = result.get("title", "")
            desc = result.get("description", "")[:200]
            url = result.get("url", "")
            items.append(RealWorldItem("Brave Search", title, desc, 0, url))
    except Exception as exc:
        logger.debug("Brave Search gather failed: %s", exc)
    return items


# ---------------------------------------------------------------------------
# Source: Perplexity Sonar via OpenRouter
# ---------------------------------------------------------------------------

async def _gather_perplexity(
    topic: str, client: httpx.AsyncClient,
    openrouter_key: str,
) -> list[RealWorldItem]:
    items: list[RealWorldItem] = []
    if not openrouter_key:
        return items
    try:
        payload = {
            "model": "perplexity/sonar",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"What are people saying about '{topic}' right now? "
                        "Summarise the top 5 opinions, sentiments, or discussions "
                        "from social media and forums in 2-3 sentences each."
                    ),
                }
            ],
            "max_tokens": 600,
        }
        r = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json=payload,
            headers={
                "Authorization": f"Bearer {openrouter_key}",
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"].strip()
        items.append(RealWorldItem("Perplexity Sonar", f"Community sentiment on '{topic}'", text))
    except Exception as exc:
        logger.debug("Perplexity gather failed: %s", exc)
    return items


# ---------------------------------------------------------------------------
# Source: X / Twitter (browser auth_token cookie)
# ---------------------------------------------------------------------------

async def _gather_x_twitter(
    topic: str, max_items: int, client: httpx.AsyncClient,
    auth_token: str,
) -> list[RealWorldItem]:
    """
    Uses the internal Twitter search API with a browser auth_token cookie.
    This is a best-effort approach — Twitter may block it at any time.
    """
    items: list[RealWorldItem] = []
    if not auth_token:
        return items
    try:
        # Use the public search endpoint (requires auth_token cookie)
        headers = {
            "Cookie": f"auth_token={auth_token}",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Authorization": "Bearer AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs%3D1Zv7ttfk8LF81IUq16cHjhLTvJu4FA33AGWWjCpTnA",
        }
        # Note: Full Twitter API v2 requires elevated access; this uses the
        # guest token approach which is publicly documented
        guest_r = await client.post(
            "https://api.twitter.com/1.1/guest/activate.json",
            headers=headers,
            timeout=10,
        )
        if guest_r.status_code == 200:
            guest_token = guest_r.json().get("guest_token", "")
            headers["x-guest-token"] = guest_token
            search_r = await client.get(
                "https://api.twitter.com/2/search/adaptive.json",
                params={"q": topic, "count": max_items, "tweet_mode": "extended"},
                headers=headers,
                timeout=15,
            )
            if search_r.status_code == 200:
                tweets = search_r.json().get("globalObjects", {}).get("tweets", {})
                for tweet in list(tweets.values())[:max_items]:
                    text = tweet.get("full_text", tweet.get("text", ""))
                    likes = tweet.get("favorite_count", 0)
                    items.append(RealWorldItem("X/Twitter", text[:200], "", likes))
    except Exception as exc:
        logger.debug("X/Twitter gather failed: %s", exc)
    return items


# ---------------------------------------------------------------------------
# Core async orchestrator
# ---------------------------------------------------------------------------

async def _gather_all_async(
    topic: str,
    keys: dict,
    max_items_per_source: int = 5,
) -> list[RealWorldItem]:
    """
    Run all configured source gatherers in parallel.
    Returns a flat list of RealWorldItem objects sorted by score descending.
    """
    async with httpx.AsyncClient(follow_redirects=True) as client:
        tasks = [
            _gather_reddit(topic, max_items_per_source, client),
            _gather_hackernews(topic, max_items_per_source, client),
            _gather_github(topic, max_items_per_source, client),
        ]

        # Optional keyed sources
        if keys.get("bluesky_handle") and keys.get("bluesky_app_password"):
            tasks.append(_gather_bluesky(
                topic, max_items_per_source, client,
                keys["bluesky_handle"], keys["bluesky_app_password"],
            ))

        if keys.get("scrapecreators_key"):
            tasks.append(_gather_scrapecreators(
                topic, max_items_per_source, client, keys["scrapecreators_key"],
            ))

        if keys.get("brave_key"):
            tasks.append(_gather_brave(
                topic, max_items_per_source, client, keys["brave_key"],
            ))

        if keys.get("openrouter_key"):
            tasks.append(_gather_perplexity(topic, client, keys["openrouter_key"]))

        if keys.get("x_auth_token"):
            tasks.append(_gather_x_twitter(
                topic, max_items_per_source, client, keys["x_auth_token"],
            ))

        results = await asyncio.gather(*tasks, return_exceptions=True)

    all_items: list[RealWorldItem] = []
    for result in results:
        if isinstance(result, list):
            all_items.extend(result)
        elif isinstance(result, Exception):
            logger.debug("Source task exception: %s", result)

    # Sort by score descending, keep top 30
    all_items.sort(key=lambda x: x.score, reverse=True)
    return all_items[:30]


# ---------------------------------------------------------------------------
# LLM synthesizer
# ---------------------------------------------------------------------------

async def _synthesize_async(
    topic: str,
    items: list[RealWorldItem],
    openai_key: str,
    text_model: str,
) -> str:
    """
    Synthesize gathered items into a compact real-world briefing paragraph
    using the configured LLM.
    """
    if not items:
        return ""

    raw_data = "\n".join(item.to_text() for item in items)

    prompt = f"""You are a UX research analyst preparing a real-world community briefing.

Topic: {topic}

Below are raw signals gathered from Reddit, Hacker News, GitHub, social media, and search engines:

{raw_data}

Write a concise 3-5 sentence briefing that summarises:
1. What real users are saying about this topic right now
2. The dominant sentiments, frustrations, and desires
3. Any emerging trends or strong opinions

The briefing will be used to shape the opinions and attitudes of AI-generated UX research personas.
Write it as a factual summary, not as a list. Cite specific sources (e.g. "Reddit users note that...").
Do not invent information not present in the data above."""

    try:
        payload = {
            "model": text_model,
            "messages": [
                {"role": "system", "content": "You are a concise UX research analyst. Synthesise data into clear briefings."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 400,
        }
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                "https://api.openai.com/v1/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {openai_key}",
                    "Content-Type": "application/json",
                },
            )
            r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        logger.warning("Synthesis LLM call failed: %s", exc)
        # Fallback: return a plain concatenation of top items
        return "Real-world signals: " + " | ".join(
            item.title[:80] for item in items[:8]
        )


# ---------------------------------------------------------------------------
# Public synchronous API
# ---------------------------------------------------------------------------

def gather_and_synthesize(
    topic: str,
    keys: Optional[dict] = None,
    max_items_per_source: int = 5,
    openai_key: str = "",
    text_model: str = "gpt-4.1-mini",
) -> dict:
    """
    Gather real-world signals for *topic* from all configured sources,
    then synthesize them into a compact briefing paragraph.

    Parameters
    ----------
    topic : str
        The brand name, product, or topic to research (e.g. "Tootgarook Bistro").
    keys : dict, optional
        Optional API keys / tokens for additional sources:
          x_auth_token, bluesky_handle, bluesky_app_password,
          scrapecreators_key, brave_key, openrouter_key
    max_items_per_source : int
        Maximum items to gather per source (default 5).
    openai_key : str
        OpenAI API key for the synthesis step. Falls back to OPENAI_API_KEY env var.
    text_model : str
        Text model to use for synthesis.

    Returns
    -------
    dict with keys:
        briefing       – synthesized paragraph (empty string if nothing gathered)
        items          – list of raw item dicts [{source, title, body, score, url}]
        sources_used   – list of source names that returned data
        error          – error message or None
    """
    result = {
        "briefing": "",
        "items": [],
        "sources_used": [],
        "error": None,
    }

    if not topic or not topic.strip():
        result["error"] = "No topic provided."
        return result

    effective_key = (openai_key or "").strip() or os.environ.get("OPENAI_API_KEY", "")
    keys = keys or {}

    try:
        import concurrent.futures

        def _worker():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                items = loop.run_until_complete(
                    _gather_all_async(topic, keys, max_items_per_source)
                )
                briefing = ""
                if items and effective_key:
                    briefing = loop.run_until_complete(
                        _synthesize_async(topic, items, effective_key, text_model)
                    )
                return items, briefing
            finally:
                loop.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            items, briefing = pool.submit(_worker).result(timeout=90)

        result["briefing"] = briefing
        result["items"] = [
            {"source": i.source, "title": i.title, "body": i.body,
             "score": i.score, "url": i.url}
            for i in items
        ]
        result["sources_used"] = list({i.source for i in items})

    except Exception as exc:
        logger.error("gather_and_synthesize failed: %s", exc)
        result["error"] = str(exc)

    return result
