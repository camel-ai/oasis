"""
slide_generator.py
==================
Generates a slide-style HTML report (and optionally PDF via Playwright)
using the Marpit rendering engine.

Pipeline:
  ScrapeResult + UXReport + list[SimulationResult] + list[Persona]
  → build_report_data(...)
  → SlideReportData
  → build_markdown(SlideReportData)   # Markdown AST (string)
  → render_html(markdown)             # calls Node.js Marpit via subprocess
  → HTML string
  → (optional) html_to_pdf(html_str, output_path)

Marpit slide separators:
  ---          new slide
  <!-- _class: <name> -->   apply CSS class to the section

Slide classes defined in the OASIS theme (marpit_render.js):
  cover         front cover
  toc           table of contents (grid layout)
  divider       teal section divider
  divider-light linen section divider
  issue         40/60 split: .left panel + .right panel
  strength      50/50 split: .left panel + .right panel
  back          back cover

v3 — Marpit rendering engine (replaces hand-crafted HTML string templates)
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Path to the Node.js Marpit render script (sibling of this file)
# ---------------------------------------------------------------------------
_RENDER_SCRIPT = Path(__file__).parent / "marpit_render.js"

# ---------------------------------------------------------------------------
# Content constraints
# ---------------------------------------------------------------------------
_TITLE_MAX     = 60
_SUBTITLE_MAX  = 80
_BODY_MAX      = 220
_QUOTE_MAX     = 140
_ANALYSIS_MAX  = 180
_TOC_LABEL_MAX = 50


def clamp_text(text: str, max_chars: int) -> str:
    """Truncate text to max_chars at a word boundary, appending ellipsis."""
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rsplit(" ", 1)[0]
    return truncated + "…"


# ---------------------------------------------------------------------------
# Data structures (unchanged public API)
# ---------------------------------------------------------------------------

@dataclass
class IssueSlide:
    number: str
    category: str
    title: str
    issue_text: str
    root_cause: str
    recommendation: str
    screenshot_url: str = ""
    persona_quotes: list[str] = field(default_factory=list)
    persona_names: list[str] = field(default_factory=list)
    redesign_analysis: str = ""
    redesign_html: str = ""
    redesign_html_sanitised: str = ""


@dataclass
class StrengthSlide:
    title: str
    brand_name: str
    description: str
    screenshot_url: str = ""


@dataclass
class SlideReportData:
    website_url: str
    website_title: str
    brand_name: str
    logo_url: str
    overall_score: float
    overall_summary: str
    methodology: str
    focus_areas: list[str]
    issues: list[IssueSlide]
    strengths: list[StrengthSlide]
    generated_by: str = "OASIS UX Simulation App"
    generated_date: str = field(default_factory=lambda: datetime.utcnow().strftime("%Y"))


# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------

def _md_esc(text: str) -> str:
    """Escape characters that are special in Markdown inline context."""
    # Escape backtick, asterisk, underscore, pipe, angle brackets
    return re.sub(r'([`*_|<>\\])', r'\\\1', text or "")


def _path_to_data_uri(path_or_url: str) -> str:
    """Convert a local file path to a base64 data URI; remote URLs pass through."""
    if not path_or_url:
        return ""
    if path_or_url.startswith("data:"):
        return path_or_url
    if path_or_url.startswith(("http://", "https://")):
        return path_or_url
    p = Path(path_or_url)
    if p.exists() and p.is_file():
        suffix = p.suffix.lower().lstrip(".")
        mime_map = {
            "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "webp": "image/webp", "gif": "image/gif", "svg": "image/svg+xml",
        }
        mime = mime_map.get(suffix, "image/png")
        data = base64.b64encode(p.read_bytes()).decode()
        return f"data:{mime};base64,{data}"
    return path_or_url


def _img_html(url: str, alt: str = "", style: str = "") -> str:
    """Return an HTML <img> tag with base64 src, or a placeholder div."""
    if not url:
        return (
            '<div style="width:100%;height:100%;background:#ddd;border-radius:8px;'
            'display:flex;align-items:center;justify-content:center;'
            'font-size:12px;color:#888;">No screenshot</div>'
        )
    data_uri = _path_to_data_uri(url)
    # Escape double-quotes for HTML attribute
    safe_src = data_uri.replace('"', "&quot;")
    safe_alt = (alt or "").replace('"', "&quot;")
    style_attr = f' style="{style}"' if style else ""
    return f'<img src="{safe_src}" alt="{safe_alt}"{style_attr} loading="eager">'


def _panel_html(content: str, label: str = "") -> str:
    """Wrap content in a .panel div with an optional label."""
    label_html = f'<div class="panel-label">{label}</div>' if label else ""
    return f'<div class="panel">{content}{label_html}</div>'


def _placeholder_panel(label: str = "") -> str:
    inner = (
        '<div style="width:100%;height:100%;background:#f0ece6;border-radius:8px;'
        'display:flex;align-items:center;justify-content:center;'
        'font-size:11px;color:#999;text-align:center;padding:12px;">'
        'Re-design mockup<br>(to be added)</div>'
    )
    return _panel_html(inner, label)


# ---------------------------------------------------------------------------
# Slide Markdown builders
# Each function returns a Markdown string for one slide (no trailing ---)
# ---------------------------------------------------------------------------

def _slide_cover(data: SlideReportData) -> str:
    logo_html = (
        _img_html(data.logo_url, data.brand_name,
                  "max-height:70px;max-width:260px;object-fit:contain;")
        if data.logo_url
        else f'<span class="brand-text">{clamp_text(data.brand_name, _TITLE_MAX)}</span>'
    )
    return f"""\
<!-- _class: cover -->

# Usability Test Report

<div class="logo-card">{logo_html}</div>
"""


def _slide_toc(sections: list[str]) -> str:
    items = ""
    for i, s in enumerate(sections, 1):
        items += (
            f'<div class="toc-item">'
            f'<span class="toc-num">{i:02d}</span>'
            f'<span>{clamp_text(s, _TOC_LABEL_MAX)}</span>'
            f'</div>\n'
        )
    return f"""\
<!-- _class: toc -->

# Table of<br>contents

<div class="toc-list">
{items}</div>
"""


def _slide_divider(number: str, title: str, subtitle: str = "", light: bool = False) -> str:
    cls = "divider-light" if light else "divider"
    sub_line = f"\n## {clamp_text(subtitle, _SUBTITLE_MAX)}" if subtitle else ""
    return f"""\
<!-- _class: {cls} -->

<div class="sec-number ghost">{number}</div>

# {clamp_text(title, _TITLE_MAX)}{sub_line}
"""


def _slide_intro(data: SlideReportData) -> str:
    focus_items = "\n".join(
        f"- {clamp_text(a, _SUBTITLE_MAX)}" for a in data.focus_areas[:6]
    )
    return f"""\
<div class="ghost">1.</div>

**{clamp_text(data.website_title, _TITLE_MAX)}** — {clamp_text(data.overall_summary, _BODY_MAX)}

{clamp_text(data.methodology, _BODY_MAX)}

**The UX review will focus on:**

{focus_items}
"""


def _slide_issue(issue: IssueSlide) -> str:
    # Persona quotes
    quotes_html = ""
    for name, quote in zip(issue.persona_names[:2], issue.persona_quotes[:2]):
        quotes_html += (
            f'<div class="quote"><strong>{clamp_text(name, 40)}:</strong> '
            f'"{clamp_text(quote, _QUOTE_MAX)}"</div>\n'
        )

    issue_body = clamp_text(issue.issue_text, _BODY_MAX)
    root_body  = clamp_text(issue.root_cause, _BODY_MAX)
    rec_body   = clamp_text(issue.recommendation, _BODY_MAX)

    # Right panel: Current Design | AI Redesign (or placeholder)
    current_panel = _panel_html(
        _img_html(issue.screenshot_url, "Current design"),
        "Current Design",
    )
    if issue.redesign_html_sanitised:
        srcdoc = issue.redesign_html_sanitised.replace('"', "&quot;")
        redesign_content = (
            f'<iframe srcdoc="{srcdoc}" sandbox="allow-same-origin" loading="lazy"'
            f' style="width:100%;height:100%;border:none;"></iframe>'
        )
        redesign_panel = _panel_html(redesign_content, "AI Redesign")
        analysis_strip = ""
        if issue.redesign_analysis:
            analysis_strip = (
                f'<div style="font-size:9.5px;color:#777;font-style:italic;'
                f'padding:4px 0 0;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;">'
                f'{clamp_text(issue.redesign_analysis, _ANALYSIS_MAX)}</div>\n'
            )
    else:
        redesign_panel = _placeholder_panel("Re-design")
        analysis_strip = ""

    return f"""\
<!-- _class: issue -->

<div class="left">
<div class="ghost">{clamp_text(issue.title, _TITLE_MAX)}</div>

### Predicted User Issue
{issue_body}
{quotes_html}
### Root Cause Analysis
{root_body}

### Recommendations
{rec_body}
</div>
<div class="right">
{current_panel}
{redesign_panel}
{analysis_strip}</div>
"""


def _slide_strength(s: StrengthSlide) -> str:
    body = (
        f"**{clamp_text(s.brand_name, 40)}** {clamp_text(s.description, _BODY_MAX)}"
        if s.brand_name else clamp_text(s.description, _BODY_MAX)
    )
    img_panel = _panel_html(_img_html(s.screenshot_url, s.title))
    return f"""\
<!-- _class: strength -->

<div class="left">
<div class="ghost">{clamp_text(s.title, _TITLE_MAX)}</div>

{body}
</div>
<div class="right">
{img_panel}
</div>
"""


def _slide_back_cover(data: SlideReportData) -> str:
    logo_html = (
        _img_html(data.logo_url, data.brand_name,
                  "max-height:60px;max-width:220px;object-fit:contain;")
        if data.logo_url
        else f'<span class="brand-text">{clamp_text(data.brand_name, _TITLE_MAX)}</span>'
    )
    return f"""\
<!-- _class: back -->

<div class="logo-card">{logo_html}</div>

*Usability Test Report*

By {data.generated_by} — {data.generated_date}
"""


# ---------------------------------------------------------------------------
# Markdown AST builder
# ---------------------------------------------------------------------------

def build_markdown(data: SlideReportData) -> str:
    """
    Convert a SlideReportData into a Marpit-compatible Markdown string.

    Each slide is separated by `---` (Marpit horizontal rule = new slide).
    Slide classes are set via Marpit's `<!-- _class: name -->` directive.
    Raw HTML is used for complex layout elements (panels, iframes, etc.)
    because Marpit passes html:true to markdown-it.
    """
    slides: list[str] = []

    # 1. Cover
    slides.append(_slide_cover(data))

    # 2. Table of Contents
    toc_sections = ["Introduction", "Predicted User Issues"]
    if data.strengths:
        toc_sections.append("Elements to Preserve")
    slides.append(_slide_toc(toc_sections))

    # 3. Section 01 – Introduction
    slides.append(_slide_divider("01", "Introduction", light=True))
    slides.append(_slide_intro(data))

    # 4. Section 02 – Predicted User Issues
    slides.append(_slide_divider("02", "Predicted User Issues"))

    seen_categories: list[str] = []
    for issue in data.issues:
        if issue.category not in seen_categories:
            seen_categories.append(issue.category)
            sub_num = f"02.{len(seen_categories)}"
            slides.append(_slide_divider(sub_num, issue.title, subtitle=issue.category))
        slides.append(_slide_issue(issue))

    # 5. Section 03 – Elements to Preserve
    if data.strengths:
        slides.append(_slide_divider("03", "Elements to preserve"))
        for s in data.strengths:
            slides.append(_slide_strength(s))

    # 6. Back Cover
    slides.append(_slide_back_cover(data))

    return "\n\n---\n\n".join(slides)


# ---------------------------------------------------------------------------
# Marpit renderer (calls Node.js script)
# ---------------------------------------------------------------------------

def render_html(data: SlideReportData) -> str:
    """
    Full pipeline:
      SlideReportData → Markdown → Node.js Marpit → self-contained HTML string

    The Node.js script (marpit_render.js) is a sibling of this file.
    It reads Markdown from a temp file and writes the full HTML to stdout.
    """
    markdown = build_markdown(data)

    # Write Markdown to a temp file (avoids shell escaping issues)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(markdown)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            ["node", str(_RENDER_SCRIPT), tmp_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if result.returncode != 0:
        raise RuntimeError(
            f"Marpit render failed (exit {result.returncode}):\n{result.stderr}"
        )

    if result.stderr:
        # Non-fatal warnings (e.g. unknown directives) — log but continue
        import logging
        logging.getLogger(__name__).warning("Marpit stderr: %s", result.stderr.strip())

    return result.stdout


# ---------------------------------------------------------------------------
# PDF export via Playwright
# ---------------------------------------------------------------------------

async def _html_to_pdf_async(html_str: str, output_path: str) -> str:
    """Render HTML to PDF using Playwright headless Chromium."""
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        raise RuntimeError("Playwright is not installed. Run: playwright install chromium")

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 960, "height": 540})
        await page.set_content(html_str, wait_until="networkidle")
        # Ensure background colors and images are printed faithfully
        await page.add_style_tag(content="""
            * {
                -webkit-print-color-adjust: exact !important;
                print-color-adjust: exact !important;
            }
        """)
        await page.pdf(
            path=output_path,
            width="960px",
            height="540px",
            print_background=True,
            margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
        )
        await browser.close()
    return output_path


def html_to_pdf(html_str: str, output_path: str) -> str:
    """Synchronous wrapper around the async PDF export."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _html_to_pdf_async(html_str, output_path))
                return future.result(timeout=60)
        else:
            return loop.run_until_complete(_html_to_pdf_async(html_str, output_path))
    except Exception as exc:
        raise RuntimeError(f"PDF export failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Data builder: converts raw app state dicts → SlideReportData
# ---------------------------------------------------------------------------

def build_report_data(
    url: str,
    scrape_data: dict,
    ux_data: dict,
    sim_results: list[dict],
    personas: list[dict],
) -> SlideReportData:
    """
    Convert raw JSON dicts from the Gradio state stores into a SlideReportData
    object ready for rendering.
    """
    website_title = scrape_data.get("title") or url
    brand_name = (website_title.split("|")[0].split("–")[0].split("-")[0]).strip()
    logo_url = scrape_data.get("logo_url") or ""

    overall_score   = ux_data.get("overall_score", 0)
    overall_summary = ux_data.get("overall_summary", "")
    strengths_raw   = ux_data.get("strengths", [])
    dimensions      = ux_data.get("dimensions", [])
    heuristics      = ux_data.get("heuristic_checks", {})
    screenshots     = ux_data.get("screenshots", {})

    desktop_ss = screenshots.get("desktop") or screenshots.get("mobile") or ""

    focus_areas = [d.get("name", "") for d in dimensions if d.get("name")]
    if not focus_areas:
        focus_areas = [
            "Navigation", "Visual Design", "Accessibility",
            "Content Clarity", "Mobile Responsiveness",
        ]

    methodology = clamp_text(
        f"This report was generated by the OASIS UX Simulation App using {len(personas)} "
        f"AI personas who browsed {url} and provided usability feedback. "
        f"The analysis combines automated heuristic checks, AI visual critique, and "
        f"simulated persona interactions to predict real user issues.",
        _BODY_MAX,
    )

    # ── Build Issue slides ──────────────────────────────────────────────────
    issues: list[IssueSlide] = []
    issue_counter = 1

    # From failed heuristic checks
    for check_name, check_data in heuristics.items():
        if isinstance(check_data, dict) and not check_data.get("passed", True):
            detail = check_data.get("detail", "")
            issues.append(IssueSlide(
                number=f"02.{issue_counter}",
                category="Heuristic Check",
                title=clamp_text(check_name.replace("_", " ").title(), _TITLE_MAX),
                issue_text=clamp_text(
                    f"Users might encounter difficulties because: {detail}", _BODY_MAX
                ),
                root_cause=clamp_text(
                    f"The page {check_name.replace('_', ' ')} check failed. {detail}", _BODY_MAX
                ),
                recommendation=clamp_text(
                    check_data.get("recommendation", "Review and fix this element."), _BODY_MAX
                ),
                screenshot_url=desktop_ss,
            ))
            issue_counter += 1

    # From UX dimension weaknesses
    for dim in dimensions:
        dim_name = dim.get("name", "")
        dim_issues = dim.get("issues", [])
        dim_recs   = dim.get("recommendations", [])
        for i, iss in enumerate(dim_issues[:2]):
            iss_text = iss if isinstance(iss, str) else (
                iss.get("issue") or iss.get("text") or iss.get("description") or str(iss)
            )
            rec_raw = dim_recs[i] if i < len(dim_recs) else "Review and improve this aspect."
            rec_text = rec_raw if isinstance(rec_raw, str) else (
                rec_raw.get("recommendation") or rec_raw.get("text") or str(rec_raw)
            )
            issues.append(IssueSlide(
                number=f"02.{issue_counter}",
                category=dim_name,
                title=clamp_text(f"{dim_name}: {iss_text[:40]}", _TITLE_MAX),
                issue_text=clamp_text(f"Users might struggle because {iss_text}", _BODY_MAX),
                root_cause=clamp_text(
                    f"The {dim_name} dimension scored {dim.get('score', 'N/A')}/10. {iss_text}",
                    _BODY_MAX,
                ),
                recommendation=clamp_text(rec_text, _BODY_MAX),
                screenshot_url=desktop_ss,
            ))
            issue_counter += 1

    # Enrich issues with persona quotes from Mode 2 browser simulation
    mode2_results = [r for r in sim_results if r.get("mode") == "browser_usability"]
    if mode2_results:
        responses = mode2_results[0].get("responses", [])
        for resp in responses:
            pain_points = resp.get("pain_points", [])
            if pain_points and issues:
                for issue in issues:
                    if len(issue.persona_quotes) < 2:
                        issue.persona_quotes.append(clamp_text(pain_points[0], _QUOTE_MAX))
                        issue.persona_names.append(
                            clamp_text(resp.get("persona_name", "Persona"), 40)
                        )
                        break

    # ── Build Strength slides ───────────────────────────────────────────────
    strength_slides: list[StrengthSlide] = []

    for s in strengths_raw[:3]:
        s_text = s if isinstance(s, str) else (
            s.get("strength") or s.get("text") or s.get("description") or str(s)
        )
        s_text = clamp_text(s_text, _BODY_MAX)
        strength_slides.append(StrengthSlide(
            title=clamp_text(s_text, _TITLE_MAX),
            brand_name=brand_name,
            description=s_text,
            screenshot_url=desktop_ss,
        ))

    mode3_results = [r for r in sim_results if r.get("mode") == "visual_branding"]
    if mode3_results:
        responses = mode3_results[0].get("responses", [])
        high_resonance = [r for r in responses if r.get("resonance_score", 0) >= 7]
        for resp in high_resonance[:2]:
            comment = clamp_text(resp.get("in_character_comment", ""), _BODY_MAX)
            if comment:
                strength_slides.append(StrengthSlide(
                    title="Visual Branding",
                    brand_name=brand_name,
                    description=f"{comment} (Resonance score: {resp.get('resonance_score', 'N/A')}/10)",
                    screenshot_url=mode3_results[0].get("content_tested", ""),
                ))

    return SlideReportData(
        website_url=url,
        website_title=website_title,
        brand_name=brand_name,
        logo_url=logo_url,
        overall_score=overall_score,
        overall_summary=overall_summary,
        methodology=methodology,
        focus_areas=focus_areas,
        issues=issues,
        strengths=strength_slides,
        generated_date=datetime.utcnow().strftime("%Y"),
    )
