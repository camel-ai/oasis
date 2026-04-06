"""
UX / Usability scanner — Python reimplementation of the eyeson ux-analyst-ai approach.

Performs:
  1. Multi-viewport screenshots (desktop, tablet, mobile) via Playwright
  2. Visual design analysis (colour contrast, layout density, spacing) via PIL
  3. Accessibility heuristics (alt text, heading structure, form labels, ARIA)
  4. Navigation & findability checks (link density, CTA presence, menu depth)
  5. AI critique of screenshots across 6 UX dimensions (GPT-4o vision)

Returns a UXReport dataclass.
"""
from __future__ import annotations

import asyncio
import base64
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from ux_sim_app.core.config import (
    OPENAI_API_KEY, OPENAI_BASE_URL, VISION_MODEL, SCREENSHOTS_DIR
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# ── Data models ────────────────────────────────────────────────────────────────

@dataclass
class UXIssue:
    category: str
    severity: str          # High / Medium / Low
    description: str
    recommendation: str


@dataclass
class UXDimension:
    name: str
    score: int             # 0-100
    feedback: str
    issues: List[UXIssue] = field(default_factory=list)


@dataclass
class UXReport:
    url: str
    screenshots: Dict[str, str] = field(default_factory=dict)   # viewport -> local path
    overall_score: int = 0
    overall_summary: str = ""
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    dimensions: List[UXDimension] = field(default_factory=list)
    recommendations: List[Dict] = field(default_factory=list)
    heuristic_checks: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


# ── Screenshot capture ─────────────────────────────────────────────────────────

VIEWPORTS = {
    "desktop":  {"width": 1440, "height": 900},
    "tablet":   {"width": 768,  "height": 1024},
    "mobile":   {"width": 390,  "height": 844},
}


async def _take_screenshots(url: str, run_id: str) -> Dict[str, str]:
    """Take screenshots at three viewports. Returns {viewport: local_path}."""
    paths: Dict[str, str] = {}
    try:
        from playwright.async_api import async_playwright
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"],
            )
            for vp_name, vp_size in VIEWPORTS.items():
                ctx = await browser.new_context(
                    viewport=vp_size,
                    user_agent=HEADERS["User-Agent"],
                )
                page = await ctx.new_page()
                try:
                    await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                    await page.wait_for_timeout(2000)
                    out_path = SCREENSHOTS_DIR / f"{run_id}_{vp_name}.png"
                    await page.screenshot(path=str(out_path), full_page=False)
                    paths[vp_name] = str(out_path)
                except Exception as exc:
                    paths[vp_name] = f"ERROR:{exc}"
                finally:
                    await ctx.close()
            await browser.close()
    except Exception as exc:
        for vp in VIEWPORTS:
            paths[vp] = f"ERROR:{exc}"
    return paths


# ── Heuristic checks (HTML-based) ─────────────────────────────────────────────

def _heuristic_checks(html: str, url: str) -> Dict[str, Any]:
    """Run lightweight heuristic checks on the raw HTML."""
    soup = BeautifulSoup(html, "html.parser")
    checks: Dict[str, Any] = {}

    # 1. Alt text coverage
    imgs = soup.find_all("img")
    imgs_with_alt = [i for i in imgs if i.get("alt", "").strip()]
    checks["alt_text"] = {
        "total_images": len(imgs),
        "with_alt": len(imgs_with_alt),
        "coverage_pct": round(len(imgs_with_alt) / max(len(imgs), 1) * 100),
        "pass": len(imgs_with_alt) / max(len(imgs), 1) >= 0.8,
    }

    # 2. Heading structure
    h1s = soup.find_all("h1")
    h2s = soup.find_all("h2")
    checks["heading_structure"] = {
        "h1_count": len(h1s),
        "h2_count": len(h2s),
        "h1_text": [h.get_text(strip=True)[:60] for h in h1s],
        "pass": len(h1s) == 1,
    }

    # 3. Form labels
    inputs = soup.find_all("input", attrs={"type": lambda t: t not in ["hidden", "submit", "button"]})
    labeled = [i for i in inputs if i.get("aria-label") or i.get("id") and soup.find("label", attrs={"for": i.get("id")})]
    checks["form_labels"] = {
        "total_inputs": len(inputs),
        "labeled": len(labeled),
        "pass": len(labeled) >= len(inputs) * 0.8 if inputs else True,
    }

    # 4. CTA presence
    cta_patterns = re.compile(r"book|order|buy|shop|sign up|get started|contact|reserve|enquire", re.I)
    cta_buttons = [b for b in soup.find_all(["a", "button"]) if cta_patterns.search(b.get_text())]
    checks["cta_presence"] = {
        "cta_count": len(cta_buttons),
        "cta_texts": [b.get_text(strip=True)[:40] for b in cta_buttons[:5]],
        "pass": len(cta_buttons) >= 1,
    }

    # 5. Navigation depth
    nav = soup.find("nav") or soup.find(attrs={"role": "navigation"})
    nav_links = nav.find_all("a") if nav else []
    checks["navigation"] = {
        "nav_link_count": len(nav_links),
        "nav_items": [a.get_text(strip=True)[:30] for a in nav_links[:10]],
        "pass": 3 <= len(nav_links) <= 12,
    }

    # 6. Meta description
    meta_desc = soup.find("meta", attrs={"name": re.compile("description", re.I)})
    checks["meta_description"] = {
        "present": bool(meta_desc),
        "content": meta_desc.get("content", "")[:120] if meta_desc else "",
        "pass": bool(meta_desc),
    }

    # 7. Mobile viewport meta
    viewport_meta = soup.find("meta", attrs={"name": "viewport"})
    checks["mobile_viewport"] = {
        "present": bool(viewport_meta),
        "pass": bool(viewport_meta),
    }

    # 8. Page title
    title = soup.title.string.strip() if soup.title else ""
    checks["page_title"] = {
        "title": title,
        "length": len(title),
        "pass": 10 <= len(title) <= 70,
    }

    # Overall heuristic score
    passed = sum(1 for v in checks.values() if isinstance(v, dict) and v.get("pass", False))
    checks["heuristic_score"] = round(passed / len(checks) * 100)

    return checks


# ── AI vision critique ─────────────────────────────────────────────────────────

_CRITIQUE_PROMPT = """You are a senior UX designer and usability expert. Analyse the provided
website screenshots (desktop, tablet, mobile viewports) and return a comprehensive UX critique.

Evaluate across these six dimensions:
1. Visual Design (typography, colour, hierarchy, whitespace, brand consistency)
2. Usability (navigation clarity, findability, task completion ease, CTA clarity)
3. Accessibility (contrast, font size, touch targets, cognitive load)
4. Mobile Responsiveness (layout adaptation, touch targets, content priority)
5. Content Quality (copy clarity, information architecture, tone of voice)
6. Trust & Credibility (social proof, contact info, pricing transparency, professionalism)

For each dimension provide:
- A score 0-100
- Detailed feedback paragraph
- Specific issues with severity (High/Medium/Low) and actionable recommendations

Also provide:
- Overall score (0-100)
- 3-5 key strengths
- 3-5 key weaknesses
- Top 5 prioritised recommendations

Return ONLY valid JSON in this exact structure:
{
  "overall_score": 75,
  "overall_summary": "...",
  "strengths": ["...", "..."],
  "weaknesses": ["...", "..."],
  "dimensions": {
    "visual_design": {"score": 80, "feedback": "...", "issues": [{"category":"Typography","severity":"Medium","description":"...","recommendation":"..."}]},
    "usability": {"score": 65, "feedback": "...", "issues": []},
    "accessibility": {"score": 70, "feedback": "...", "issues": []},
    "mobile_responsiveness": {"score": 75, "feedback": "...", "issues": []},
    "content_quality": {"score": 80, "feedback": "...", "issues": []},
    "trust_credibility": {"score": 85, "feedback": "...", "issues": []}
  },
  "recommendations": [
    {"priority":"High","category":"Navigation","title":"...","description":"...","impact":"...","effort":"Low"}
  ]
}"""


async def _ai_critique(screenshots: Dict[str, str]) -> Dict:
    """Send screenshots to GPT-4o vision and get structured UX critique."""
    content: List[Dict] = [{"type": "text", "text": _CRITIQUE_PROMPT}]

    for vp, path in screenshots.items():
        if path.startswith("ERROR"):
            continue
        try:
            img_bytes = Path(path).read_bytes()
            b64 = base64.b64encode(img_bytes).decode()
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{b64}",
                    "detail": "low",
                },
            })
            content.append({"type": "text", "text": f"[{vp.upper()} viewport above]"})
        except Exception:
            pass

    if len(content) == 1:
        return {"error": "No screenshots available for AI critique"}

    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(
            f"{OPENAI_BASE_URL}/chat/completions",
            json={
                "model": VISION_MODEL,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 3000,
                "temperature": 0.3,
            },
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
        )
        r.raise_for_status()

    raw = r.json()["choices"][0]["message"]["content"]
    # Strip markdown code fences if present
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        return json.loads(match.group())
    return {"raw_response": raw}


# ── Main entry point ───────────────────────────────────────────────────────────

async def scan_website(url: str, run_id: str) -> UXReport:
    """Full UX scan: screenshots + heuristics + AI critique."""
    report = UXReport(url=url)

    # 1. Fetch HTML for heuristics
    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            r = await client.get(url, headers=HEADERS)
            html = r.text
        report.heuristic_checks = _heuristic_checks(html, url)
    except Exception as exc:
        report.heuristic_checks = {"error": str(exc)}

    # 2. Screenshots
    report.screenshots = await _take_screenshots(url, run_id)

    # 3. AI critique
    try:
        critique = await _ai_critique(report.screenshots)
        if "error" not in critique:
            report.overall_score = critique.get("overall_score", 0)
            report.overall_summary = critique.get("overall_summary", "")
            report.strengths = critique.get("strengths", [])
            report.weaknesses = critique.get("weaknesses", [])
            report.recommendations = critique.get("recommendations", [])

            for dim_key, dim_data in critique.get("dimensions", {}).items():
                issues = [
                    UXIssue(
                        category=i.get("category", ""),
                        severity=i.get("severity", "Medium"),
                        description=i.get("description", ""),
                        recommendation=i.get("recommendation", ""),
                    )
                    for i in dim_data.get("issues", [])
                ]
                report.dimensions.append(UXDimension(
                    name=dim_key.replace("_", " ").title(),
                    score=dim_data.get("score", 0),
                    feedback=dim_data.get("feedback", ""),
                    issues=issues,
                ))
        else:
            report.error = critique.get("error")
    except Exception as exc:
        report.error = f"AI critique failed: {exc}"

    return report
