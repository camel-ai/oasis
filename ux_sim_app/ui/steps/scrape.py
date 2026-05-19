"""steps/scrape.py — Step 1: Scrape website and generate personas."""
from __future__ import annotations

import json
import os
import traceback

from .shared import (
    _run, _captcha_sessions,
    scrape, generate_personas,
    preflight_check, CaptchaGuardResult,
    OPENAI_API_KEY,
    logger,
)


def step_scrape_and_generate(
    url: str,
    business_context: str,
    customer_profile: str,
    num_personas: int,
    api_key_override: str,
    rwd_briefing: str = "",
):
    """Scrape the website and generate personas.
    MUST yield exactly 4 values on every yield (matches outputs wiring).
    rwd_briefing: optional real-world community briefing from Tab 2.
    """
    _EMPTY = ("{}", "", "[]")  # personas_json, personas_display, image_urls_json

    url = url or ""
    if not url.strip():
        yield "❌ Please enter a website URL.", *_EMPTY
        return

    # Allow runtime API key override
    if api_key_override and api_key_override.strip():
        os.environ["OPENAI_API_KEY"] = api_key_override.strip()
        import ux_sim_app.core.config as cfg
        cfg.OPENAI_API_KEY = api_key_override.strip()

    effective_key = os.environ.get("OPENAI_API_KEY") or OPENAI_API_KEY
    if not effective_key:
        yield "❌ OpenAI API key is required. Enter it in the ⚙️ Settings tab.", *_EMPTY
        return

    try:
        # ── Pre-flight CAPTCHA check ──────────────────────────────────────────
        yield "⏳ Running pre-flight CAPTCHA check...", *_EMPTY
        guard: CaptchaGuardResult = _run(preflight_check(url.strip()))

        if guard.status == "aborted":
            yield f"{guard.message}", *_EMPTY
            return

        # Store the session state path so downstream Mode 2 and UX scan
        # can inherit the cleared session.
        _captcha_sessions[url.strip()] = guard.storage_state_path

        status_prefix = ""
        if guard.status == "solved":
            status_prefix = f"✅ CAPTCHA solved ({guard.captcha_type}). "
        elif guard.status == "clear":
            status_prefix = "✅ No CAPTCHA detected. "

        yield f"{status_prefix}⏳ Scraping website...", *_EMPTY

        scrape_result = _run(scrape(
            url.strip(),
            follow_links=2,
            storage_state_path=guard.storage_state_path,
        ))
        if scrape_result.error:
            yield (
                f"⚠️ Scrape warning: {scrape_result.error}. Continuing with partial data.",
                *_EMPTY,
            )

        website_text = (
            f"Title: {scrape_result.title}\n"
            f"Description: {scrape_result.description}\n\n"
            f"{scrape_result.body_text}"
        )

        yield f"⏳ Generating {num_personas} personas...", *_EMPTY

        personas = _run(generate_personas(
            website_text=website_text,
            website_url=url.strip(),
            business_context=business_context,
            customer_profile=customer_profile,
            num_personas=int(num_personas),
            real_world_context=(rwd_briefing or "").strip(),
        ))

        personas_json = json.dumps([p.to_dict() for p in personas], indent=2)

        display = f"### 👥 Focus Group ({len(personas)} personas)\n\n"
        for p in personas:
            display += (
                f"**{p.name}** (@{p.username}) · {p.age}yo · {p.gender} · "
                f"{p.country} · {p.mbti}  \n"
                f"*{p.persona_type}*  \n"
                f"{p.bio}  \n"
                f"Goals: {', '.join(p.goals[:2])}  \n\n"
            )

        image_urls_json = json.dumps(scrape_result.image_urls[:6])

        yield (
            f"✅ Scraped website and generated {len(personas)} personas.",
            personas_json,
            display,
            image_urls_json,
        )

    except Exception as exc:
        yield f"❌ Error: {exc}\n{traceback.format_exc()}", "{}", "", "[]"
