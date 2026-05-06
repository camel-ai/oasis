"""
Simulation runner for all three modes.

Mode 1 – Content Simulation: personas react to social media copy
Mode 2 – Browser-Action Simulation: personas navigate the live website
Mode 3 – Visual Input Simulation: personas analyse brand imagery
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import httpx
from bs4 import BeautifulSoup

from ux_sim_app.core import llm as _llm
from ux_sim_app.core.llm import chat, tool_args, text_content
from ux_sim_app.core.personas import Persona
from ux_sim_app.core.config import MAX_BROWSER_SESSIONS, DATA_DIR

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# ── Shared data models ─────────────────────────────────────────────────────────

@dataclass
class PersonaResponse:
    persona_name: str
    persona_type: str
    mode: int
    # Mode 1
    action: Optional[str] = None
    sentiment: Optional[str] = None
    reasoning: Optional[str] = None
    comment_text: Optional[str] = None
    # Mode 2
    browser_steps: List[str] = field(default_factory=list)
    usability_summary: Optional[str] = None
    dish_or_task_chosen: Optional[str] = None
    would_convert: Optional[bool] = None
    # Mode 3
    first_impression: Optional[str] = None
    resonance_score: Optional[int] = None
    engagement_likelihood: Optional[str] = None
    visual_feedback: Optional[str] = None
    attention_elements: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SimulationResult:
    mode: int
    mode_name: str
    responses: List[PersonaResponse] = field(default_factory=list)
    aggregate: Dict[str, Any] = field(default_factory=dict)
    content_tested: Optional[str] = None
    images_analysed: List[str] = field(default_factory=list)
    # Mode 2 video recordings — list of {persona_name, video_path} dicts
    video_recordings: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ── Tool schemas ───────────────────────────────────────────────────────────────

_SOCIAL_ACTION_TOOL = {
    "type": "function",
    "function": {
        "name": "social_action",
        "description": "Choose how this persona reacts to the content.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["like_post", "repost", "create_comment", "quote_post",
                             "dislike_post", "report_post", "do_nothing"],
                },
                "reasoning": {"type": "string"},
                "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]},
                "comment_text": {"type": "string", "description": "Only if action is create_comment or quote_post"},
            },
            "required": ["action", "reasoning", "sentiment"],
        },
    },
}

_VISUAL_FEEDBACK_TOOL = {
    "type": "function",
    "function": {
        "name": "visual_feedback",
        "description": "Structured visual/brand feedback from a persona.",
        "parameters": {
            "type": "object",
            "properties": {
                "first_impression": {"type": "string"},
                "attention_elements": {"type": "array", "items": {"type": "string"}},
                "resonance_score": {"type": "integer", "minimum": 1, "maximum": 10},
                "engagement_likelihood": {
                    "type": "string",
                    "enum": ["definitely", "probably", "maybe", "unlikely", "no"],
                },
                "feedback": {"type": "string"},
                "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]},
            },
            "required": ["first_impression", "resonance_score", "engagement_likelihood",
                         "feedback", "sentiment"],
        },
    },
}


# ── Mode 1: Content Simulation ─────────────────────────────────────────────────

async def _mode1_one(persona: Persona, content: str) -> PersonaResponse:
    resp = await chat(
        messages=[
            {"role": "system", "content": persona.system_prompt("content")},
            {"role": "user", "content":
                f"You just saw this post on your social media feed:\n\n---\n{content}\n---\n\n"
                "How do you react? Use the social_action tool."},
        ],
        tools=[_SOCIAL_ACTION_TOOL],
        tool_choice={"type": "function", "function": {"name": "social_action"}},
    )
    args = tool_args(resp)
    if not args:
        args = {"action": "do_nothing", "reasoning": text_content(resp), "sentiment": "neutral"}
    return PersonaResponse(
        persona_name=persona.name,
        persona_type=persona.persona_type,
        mode=1,
        action=args.get("action", "do_nothing"),
        sentiment=args.get("sentiment", "neutral"),
        reasoning=args.get("reasoning", ""),
        comment_text=args.get("comment_text"),
    )


async def run_mode1(personas: List[Persona], content_items: List[str]) -> List[SimulationResult]:
    """Run Mode 1 for each content item. Returns one SimulationResult per item."""
    results = []
    for content in content_items:
        responses = await asyncio.gather(*[_mode1_one(p, content) for p in personas])
        agg = _aggregate_mode1(list(responses))
        results.append(SimulationResult(
            mode=1,
            mode_name="Content Simulation",
            responses=list(responses),
            aggregate=agg,
            content_tested=content,
        ))
    return results


def _aggregate_mode1(responses: List[PersonaResponse]) -> Dict:
    actions: Dict[str, int] = {}
    sentiments = {"positive": 0, "neutral": 0, "negative": 0}
    for r in responses:
        a = r.action or "do_nothing"
        actions[a] = actions.get(a, 0) + 1
        s = r.sentiment or "neutral"
        sentiments[s] = sentiments.get(s, 0) + 1
    engaged = sum(1 for r in responses if r.action != "do_nothing")
    return {
        "action_distribution": actions,
        "sentiment_distribution": sentiments,
        "engagement_rate": round(engaged / max(len(responses), 1), 2),
    }


# ── Mode 2: Cookie-banner dismissal ──────────────────────────────────────────

# Common consent-button text patterns (case-insensitive, partial match)
_CONSENT_TEXTS = [
    "accept all", "accept cookies", "i accept", "agree", "allow all",
    "allow cookies", "consent", "got it", "i understand", "ok", "okay",
    "continue", "close", "dismiss", "decline all",  # 'decline all' still clears the banner
    "reject all",   # some banners disappear on reject too
    "save preferences", "save settings",
]

# CSS selectors that commonly wrap consent overlays
_CONSENT_SELECTORS = [
    "[id*='cookie']", "[class*='cookie']",
    "[id*='consent']", "[class*='consent']",
    "[id*='gdpr']", "[class*='gdpr']",
    "[id*='banner']", "[class*='banner']",
    "[id*='notice']", "[class*='notice']",
    "[id*='privacy']", "[class*='privacy']",
    "[aria-label*='cookie']", "[aria-label*='consent']",
    ".cc-banner", ".cc-window", "#onetrust-accept-btn-handler",
    "#CybotCookiebotDialogBodyButtonAccept",
    "[data-testid*='cookie']", "[data-testid*='consent']",
]


async def _dismiss_cookie_banners(page: Any, steps: List[str]) -> None:
    """Three-tier cookie banner dismissal strategy.

    Tier 1 – JS injection: set common consent cookies directly so the banner
             never renders on subsequent navigations.
    Tier 2 – Playwright click on known consent button text patterns.
    Tier 3 – Playwright click on known consent CSS selector patterns.
    """
    # ── Tier 1: Pre-set consent cookies via JS ─────────────────────────────────
    try:
        await page.evaluate("""
            () => {
                const pairs = [
                    ['cookieconsent_status', 'dismiss'],
                    ['cookie_consent', '1'],
                    ['cookies_accepted', 'true'],
                    ['gdpr_consent', '1'],
                    ['CookieConsent', '{stamp:\"accepted\",necessary:true,preferences:true,statistics:true,marketing:true}'],
                    ['OptanonAlertBoxClosed', new Date().toISOString()],
                    ['OptanonConsent', 'isGpcEnabled=0&datestamp=&version=6.10&isIABGlobal=false&hosts=&consentId=&interactionCount=1&landingPath=NotLandingPage&groups=C0001%3A1%2CC0002%3A1%2CC0003%3A1%2CC0004%3A1'],
                    ['__cfduid', '1'],
                    ['_ga_consent', 'granted'],
                ];
                pairs.forEach(([k, v]) => {
                    document.cookie = k + '=' + v + '; path=/; max-age=31536000; SameSite=Lax';
                    try { localStorage.setItem(k, v); } catch(e) {}
                });
            }
        """)
        steps.append("Cookie consent: pre-set consent cookies via JS")
    except Exception:
        pass

    # Short wait for banner to potentially render
    await page.wait_for_timeout(800)

    # ── Tier 2: Click by visible text ─────────────────────────────────────────
    for text in _CONSENT_TEXTS:
        try:
            # Use Playwright's text locator with exact=False for partial match
            btn = page.get_by_role("button", name=text)
            if await btn.count() > 0:
                await btn.first.click(timeout=2000)
                await page.wait_for_timeout(500)
                steps.append(f"Cookie consent: clicked button '{text}'")
                return  # banner dismissed — done
        except Exception:
            pass
        try:
            # Fallback: text selector
            loc = page.locator(f"text=/{text}/i")
            if await loc.count() > 0:
                await loc.first.click(timeout=2000)
                await page.wait_for_timeout(500)
                steps.append(f"Cookie consent: clicked text '{text}'")
                return
        except Exception:
            pass

    # ── Tier 3: Click by CSS selector patterns ─────────────────────────────────
    for sel in _CONSENT_SELECTORS:
        try:
            loc = page.locator(sel)
            if await loc.count() > 0:
                # Try to find a button or link inside the banner
                inner_btn = loc.locator("button, a, [role='button']")
                if await inner_btn.count() > 0:
                    await inner_btn.first.click(timeout=2000)
                    await page.wait_for_timeout(500)
                    steps.append(f"Cookie consent: clicked element inside '{sel}'")
                    return
        except Exception:
            pass

    steps.append("Cookie consent: no banner detected or already dismissed")


# ── Mode 2: Browser-Action Simulation ─────────────────────────────────────────

async def _mode2_one(
    persona: Persona,
    url: str,
    video_dir: Optional[str] = None,
    storage_state_path: Optional[str] = None,
) -> tuple:
    """Single persona browser session. Returns (PersonaResponse, video_path_or_None).

    storage_state_path: optional Playwright storage state JSON from captcha_guard.
    When provided the browser context inherits the cleared CAPTCHA session so the
    simulation does not hit the challenge again.
    """
    import os
    from pathlib import Path as _Path
    steps: List[str] = []
    page_text = ""
    video_path: Optional[str] = None

    try:
        from playwright.async_api import async_playwright
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"],
            )
            # Build context kwargs — add video recording when a dir is provided
            ctx_kwargs: Dict[str, Any] = {
                "user_agent": HEADERS["User-Agent"],
                "viewport": {"width": 1280, "height": 800},
            }
            if storage_state_path and os.path.isfile(storage_state_path):
                ctx_kwargs["storage_state"] = storage_state_path
            if video_dir:
                _Path(video_dir).mkdir(parents=True, exist_ok=True)
                ctx_kwargs["record_video_dir"] = video_dir
                ctx_kwargs["record_video_size"] = {"width": 1280, "height": 800}

            ctx = await browser.new_context(**ctx_kwargs)

            # ── Pre-consent injection: set localStorage/cookie flags BEFORE any page loads
            # This prevents Squarespace, OneTrust, CookieBot and similar banners from
            # ever rendering, since they check these flags at script init time.
            await ctx.add_init_script("""
                (() => {
                    const LS_CONSENT_KEYS = [
                        ['sqs-cookie-banner-v2-accepted', 'true'],
                        ['squarespace-cookie-banner', 'accepted'],
                        ['squarespace-popup-overlay', JSON.stringify({version:1,dismissed:true})],
                        ['cookieconsent_status', 'dismiss'],
                        ['cookie_consent', '1'],
                        ['cookies_accepted', 'true'],
                        ['gdpr_consent', '1'],
                        ['CookieConsent', JSON.stringify({stamp:'accepted',necessary:true,preferences:true,statistics:true,marketing:true})],
                        ['OptanonAlertBoxClosed', new Date().toISOString()],
                        ['OptanonConsent', 'isGpcEnabled=0&datestamp=&version=6.10&isIABGlobal=false&hosts=&consentId=&interactionCount=1&landingPath=NotLandingPage&groups=C0001%3A1%2CC0002%3A1%2CC0003%3A1%2CC0004%3A1'],
                        ['euconsent-v2', 'CPXxRfAPXxRfAAfKABENB-CgAAAAAAAAAAYgAAAAAAAA'],
                        ['cmplz_functional', '1'],
                        ['cmplz_marketing', '1'],
                        ['cmplz_statistics', '1'],
                        ['cmplz_preferences', '1'],
                        ['cookieyes-consent', 'consentid:accepted,consent:yes,action:yes,necessary:yes,functional:yes,analytics:yes,performance:yes,advertisement:yes'],
                    ];
                    try {
                        LS_CONSENT_KEYS.forEach(([k, v]) => {
                            try { localStorage.setItem(k, v); } catch(e) {}
                            document.cookie = k + '=' + encodeURIComponent(v) + '; path=/; max-age=31536000; SameSite=Lax';
                        });
                    } catch(e) {}
                })();
            """)

            page = await ctx.new_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            steps.append("Navigated to homepage")

            # Also run post-load dismissal as belt-and-braces for JS-rendered banners
            await _dismiss_cookie_banners(page, steps)

            content = await page.content()
            soup = BeautifulSoup(content, "html.parser")
            for t in soup(["script", "style"]): t.decompose()
            page_text = " ".join(soup.get_text(separator=" ").split())[:3000]
            steps.append(f"Read page content ({len(page_text)} chars)")

            # Try to find and click a menu/product link
            for kw in ["menu", "food", "product", "service", "shop", "order"]:
                try:
                    await page.click(f"text=/{kw}/i", timeout=3000)
                    await page.wait_for_timeout(1500)
                    extra = await page.content()
                    extra_soup = BeautifulSoup(extra, "html.parser")
                    for t in extra_soup(["script", "style"]): t.decompose()
                    extra_text = " ".join(extra_soup.get_text(separator=" ").split())[:1500]
                    page_text += f"\n\n[After clicking '{kw}']: {extra_text}"
                    steps.append(f"Clicked '{kw}' link successfully")
                    break
                except Exception:
                    pass
            else:
                steps.append("Could not click any menu/product link")

            # MUST close context (not just browser) to flush the .webm video file
            await ctx.close()
            await browser.close()

            # Retrieve the saved video path
            if video_dir:
                try:
                    vp = await page.video.path()
                    video_path = str(vp)
                    steps.append(f"Session video saved: {os.path.basename(video_path)}")
                except Exception:
                    # Find the most recently written .webm in the dir
                    webms = sorted(
                        _Path(video_dir).glob("*.webm"),
                        key=lambda f: f.stat().st_mtime,
                        reverse=True,
                    )
                    if webms:
                        video_path = str(webms[0])
                        steps.append(f"Session video saved: {webms[0].name}")

    except Exception as exc:
        steps.append(f"Browser error: {exc}")
        # Fallback: fetch via httpx
        try:
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                r = await client.get(url, headers=HEADERS)
                soup = BeautifulSoup(r.text, "html.parser")
                for t in soup(["script", "style"]): t.decompose()
                page_text = " ".join(soup.get_text(separator=" ").split())[:3000]
                steps.append("Fallback: fetched via HTTP")
        except Exception as e2:
            steps.append(f"Fallback also failed: {e2}")

    # Ask LLM to simulate the persona's experience
    resp = await chat(
        messages=[
            {"role": "system", "content": persona.system_prompt("browser")},
            {"role": "user", "content":
                f"You have just browsed this website: {url}\n\n"
                f"Page content you saw:\n{page_text}\n\n"
                f"Steps taken: {'; '.join(steps)}\n\n"
                "As this persona, answer:\n"
                "1. How easy was it to find what you were looking for? (1-10 and why)\n"
                "2. What specific task/item/dish/product would you choose and why?\n"
                "3. What confused or frustrated you on the website?\n"
                "4. What did you like about the website?\n"
                "5. Would you complete a purchase/booking? Why or why not?\n\n"
                "Be specific, in-character, and honest."},
        ],
        max_tokens=700,
    )
    summary = text_content(resp)

    # Extract would_convert signal
    convert = None
    if any(w in summary.lower() for w in ["yes, i would", "definitely book", "would book", "would order", "would buy"]):
        convert = True
    elif any(w in summary.lower() for w in ["would not", "wouldn't", "no, i"]):
        convert = False

    pr = PersonaResponse(
        persona_name=persona.name,
        persona_type=persona.persona_type,
        mode=2,
        browser_steps=steps,
        usability_summary=summary,
        would_convert=convert,
    )
    return pr, video_path


async def run_mode2(
    personas: List[Persona],
    url: str,
    record_video: bool = True,
    storage_state_path: Optional[str] = None,
) -> SimulationResult:
    """Run Mode 2 for all personas in parallel (up to MAX_BROWSER_SESSIONS).

    When record_video=True, each persona's browser session is recorded to a
    .webm file in DATA_DIR/recordings/<run_id>/ and stored in video_recordings.

    storage_state_path: optional Playwright storage state JSON from captcha_guard.
    Passed to each persona session so they inherit the CAPTCHA-cleared session.
    """
    import uuid as _uuid
    from pathlib import Path as _Path

    run_id = _uuid.uuid4().hex[:8]
    video_dir: Optional[str] = None
    if record_video:
        video_dir = str(_Path(DATA_DIR) / "recordings" / run_id)

    sem = asyncio.Semaphore(MAX_BROWSER_SESSIONS)

    async def _bounded(p: Persona):
        async with sem:
            return await _mode2_one(
                p, url,
                video_dir=video_dir,
                storage_state_path=storage_state_path,
            )

    raw = await asyncio.gather(*[_bounded(p) for p in personas], return_exceptions=True)

    clean: List[PersonaResponse] = []
    video_recordings: List[Dict[str, str]] = []

    for item in raw:
        if isinstance(item, Exception):
            clean.append(PersonaResponse(
                persona_name="Unknown", persona_type="Unknown", mode=2,
                usability_summary=f"Error: {item}",
            ))
        else:
            pr, vp = item
            clean.append(pr)
            if vp:
                video_recordings.append({
                    "persona_name": pr.persona_name,
                    "video_path": vp,
                })

    conversions = [r for r in clean if r.would_convert is True]
    return SimulationResult(
        mode=2,
        mode_name="Browser-Action Simulation",
        responses=clean,
        aggregate={
            "conversion_intent_rate": round(len(conversions) / max(len(clean), 1), 2),
            "personas_would_convert": len(conversions),
            "total_personas": len(clean),
        },
        video_recordings=video_recordings,
    )


# ── Mode 3: Visual Input Simulation ───────────────────────────────────────────

async def _mode3_one(persona: Persona, image_urls: List[str]) -> PersonaResponse:
    content: List[Dict] = [
        {"type": "text", "text":
            "These are images from the brand's website and marketing materials. "
            "Analyse them as this persona and provide your honest reaction using the visual_feedback tool."},
    ]
    for url in image_urls[:3]:
        content.append({"type": "image_url", "image_url": {"url": url, "detail": "low"}})

    resp = await _llm.chat(
        messages=[
            {"role": "system", "content": persona.system_prompt("visual")},
            {"role": "user", "content": content},
        ],
        tools=[_VISUAL_FEEDBACK_TOOL],
        tool_choice={"type": "function", "function": {"name": "visual_feedback"}},
        max_tokens=800,
        temperature=0.7,
        vision=True,
    )
    tc = resp["choices"][0]["message"].get("tool_calls", [])
    if tc:
        args = json.loads(tc[0]["function"]["arguments"])
    else:
        args = {
            "first_impression": resp["choices"][0]["message"].get("content", ""),
            "resonance_score": 5,
            "engagement_likelihood": "maybe",
            "feedback": "",
            "sentiment": "neutral",
        }

    return PersonaResponse(
        persona_name=persona.name,
        persona_type=persona.persona_type,
        mode=3,
        first_impression=args.get("first_impression", ""),
        resonance_score=args.get("resonance_score", 5),
        engagement_likelihood=args.get("engagement_likelihood", "maybe"),
        visual_feedback=args.get("feedback", ""),
        attention_elements=args.get("attention_elements", []),
        sentiment=args.get("sentiment", "neutral"),
    )


async def run_mode3(personas: List[Persona], image_urls: List[str]) -> SimulationResult:
    """Run Mode 3 for all personas."""
    responses = await asyncio.gather(
        *[_mode3_one(p, image_urls) for p in personas],
        return_exceptions=True,
    )
    clean: List[PersonaResponse] = []
    for r in responses:
        if isinstance(r, Exception):
            clean.append(PersonaResponse(
                persona_name="Unknown", persona_type="Unknown", mode=3,
                visual_feedback=f"Error: {r}",
            ))
        else:
            clean.append(r)

    scores = [r.resonance_score for r in clean if r.resonance_score is not None]
    sentiments = {"positive": 0, "neutral": 0, "negative": 0}
    engagement: Dict[str, int] = {}
    for r in clean:
        s = r.sentiment or "neutral"
        sentiments[s] = sentiments.get(s, 0) + 1
        e = r.engagement_likelihood or "maybe"
        engagement[e] = engagement.get(e, 0) + 1

    return SimulationResult(
        mode=3,
        mode_name="Visual Input Simulation",
        responses=clean,
        aggregate={
            "average_resonance": round(sum(scores) / max(len(scores), 1), 2),
            "sentiment_distribution": sentiments,
            "engagement_distribution": engagement,
        },
        images_analysed=image_urls,
    )
