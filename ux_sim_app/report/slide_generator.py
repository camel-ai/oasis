"""
slide_generator.py
==================
Generates a slide-style HTML report (and optionally PDF via Playwright)
that mirrors the design language of the reference "Usability Test Report"
PDF: beige/teal palette, 16:9 slides, issue-detail 40/60 split template,
strengths 50/50 template, and teal section/sub-section dividers.

Data flow:
  ScrapeResult + UXReport + list[SimulationResult] + list[Persona]
  → build_report_data(...)
  → SlideReportData
  → render_html(SlideReportData)
  → HTML string
  → (optional) html_to_pdf(html_str, output_path)

v2 improvements (per design-system review):
  - Design token layer (--font-*, --space-*, --color-*)
  - Strict flex/grid slide layouts (no free-form positioning)
  - Content clamping (clamp_text) applied to all user-supplied text
  - Clear typography hierarchy (.title / .subtitle / .body / .label)
  - Fixed image/iframe containers (overflow:hidden, object-fit:cover)
  - Playwright print-color-adjust fix injected before pdf()
  - DEBUG_LAYOUT flag for layout debugging
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Debug flag — set True to outline every element in red for layout debugging
# ---------------------------------------------------------------------------
DEBUG_LAYOUT = False

# ---------------------------------------------------------------------------
# Content constraints
# ---------------------------------------------------------------------------
_TITLE_MAX      = 60
_SUBTITLE_MAX   = 80
_BODY_MAX       = 220
_QUOTE_MAX      = 140
_ANALYSIS_MAX   = 180
_TOC_LABEL_MAX  = 50


def clamp_text(text: str, max_chars: int) -> str:
    """Truncate text to max_chars, appending ellipsis if needed."""
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    # Try to break at a word boundary
    truncated = text[:max_chars].rsplit(" ", 1)[0]
    return truncated + "…"


# ---------------------------------------------------------------------------
# Design tokens (CSS)
# ---------------------------------------------------------------------------
_DEBUG_CSS = """
/* DEBUG: outline every element */
.slide * { outline: 1px solid rgba(255,0,0,0.35) !important; }
""" if DEBUG_LAYOUT else ""

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

/* ── Reset ───────────────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

/* ── Design tokens ───────────────────────────────────────────────────────── */
:root {
  /* Palette */
  --color-bg:       #E8E0D5;
  --color-teal:     #7DD9D0;
  --color-navy:     #1A2280;
  --color-text:     #2B2B2B;
  --color-muted:    #7A7A9A;
  --color-white:    #FFFFFF;
  --color-card:     #FAFAC8;
  --color-panel:    #F0ECE6;

  /* Typography */
  --font-family:    'Inter', 'Segoe UI', Arial, sans-serif;
  --font-hero:      52px;
  --font-title:     32px;
  --font-subtitle:  20px;
  --font-body:      14px;
  --font-label:     11px;
  --font-ghost:     80px;

  /* Spacing */
  --space-xs:  6px;
  --space-sm:  12px;
  --space-md:  20px;
  --space-lg:  32px;
  --space-xl:  48px;

  /* Slide dimensions */
  --slide-w: 960px;
  --slide-h: 540px;
  --slide-pad: var(--space-lg);
}

/* ── Base ─────────────────────────────────────────────────────────────────── */
body {
  background: #999;
  font-family: var(--font-family);
  color: var(--color-text);
}

/* ── Typography helpers ──────────────────────────────────────────────────── */
.title {
  font-size: var(--font-title);
  font-weight: 700;
  color: var(--color-text);
  line-height: 1.2;
}
.subtitle {
  font-size: var(--font-subtitle);
  font-weight: 600;
  color: var(--color-muted);
  line-height: 1.3;
}
.body {
  font-size: var(--font-body);
  font-weight: 400;
  line-height: 1.65;
  color: var(--color-text);
}
.body strong { font-weight: 700; }
.label {
  font-size: var(--font-label);
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--color-navy);
}
.ghost {
  font-size: var(--font-ghost);
  font-weight: 800;
  color: rgba(255,255,255,0.55);
  line-height: 1;
}

/* ── Slide container ─────────────────────────────────────────────────────── */
.slide {
  width: var(--slide-w);
  height: var(--slide-h);
  overflow: hidden;
  margin: 0 auto 24px;
  background: var(--color-bg);
  page-break-after: always;
  break-after: page;
  display: flex;
  flex-direction: column;
}

.slide.bg-teal   { background: var(--color-teal); }
.slide.bg-linen  { background: var(--color-bg); }

/* ── Image / iframe panel ────────────────────────────────────────────────── */
.panel {
  border-radius: 8px;
  overflow: hidden;
  background: var(--color-white);
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
}
.panel img,
.panel iframe {
  width: 100%;
  height: 100%;
  object-fit: cover;
  object-position: top;
  border: none;
  display: block;
}
.panel-label {
  font-size: var(--font-label);
  font-weight: 600;
  color: var(--color-navy);
  text-align: center;
  padding: var(--space-xs) 0 0;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  flex-shrink: 0;
}

/* ── COVER ───────────────────────────────────────────────────────────────── */
.cover-inner {
  flex: 1;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  padding: var(--space-xl) var(--space-xl) var(--space-lg);
}
.cover-title {
  font-size: var(--font-hero);
  font-weight: 800;
  color: var(--color-text);
  line-height: 1.1;
  max-width: 580px;
}
.cover-logo-card {
  align-self: center;
  background: var(--color-card);
  padding: var(--space-md) var(--space-lg);
  border-radius: 6px;
  display: flex;
  align-items: center;
  justify-content: center;
  min-width: 180px;
  min-height: 80px;
}
.cover-logo-card img { max-height: 70px; max-width: 260px; object-fit: contain; }
.cover-logo-card .brand-text {
  font-size: 30px; font-weight: 800; color: var(--color-text);
}

/* ── TABLE OF CONTENTS ───────────────────────────────────────────────────── */
.toc-inner {
  flex: 1;
  display: grid;
  grid-template-columns: 260px 1fr;
  gap: var(--space-lg);
  padding: var(--space-xl);
  align-items: start;
}
.toc-heading {
  font-size: 38px;
  font-weight: 800;
  color: var(--color-navy);
  line-height: 1.2;
}
.toc-list { display: flex; flex-direction: column; gap: 0; }
.toc-item {
  display: flex;
  align-items: baseline;
  gap: var(--space-md);
  padding: var(--space-sm) 0;
  border-bottom: 1px dashed #B0A89A;
}
.toc-item:last-child { border-bottom: none; }
.toc-num { font-size: 13px; font-weight: 700; color: var(--color-navy); min-width: 28px; }
.toc-label { font-size: 18px; font-weight: 600; color: var(--color-text); }

/* ── SECTION DIVIDER ─────────────────────────────────────────────────────── */
.sec-inner {
  flex: 1;
  display: flex;
  flex-direction: column;
  justify-content: center;
  padding: var(--space-xl);
  gap: var(--space-sm);
}
.sec-number {
  font-size: 100px;
  font-weight: 800;
  color: var(--color-white);
  line-height: 1;
}
.sec-title {
  font-size: 56px;
  font-weight: 800;
  color: var(--color-navy);
  line-height: 1.1;
  max-width: 700px;
}
.sec-subtitle {
  font-size: 20px;
  font-weight: 600;
  color: var(--color-muted);
}

/* ── INTRO TEXT SLIDE ────────────────────────────────────────────────────── */
.intro-inner {
  flex: 1;
  display: flex;
  flex-direction: column;
  padding: var(--space-lg) var(--space-xl);
  gap: var(--space-sm);
  overflow: hidden;
}
.intro-ghost { flex-shrink: 0; }
.intro-body { flex: 1; overflow: hidden; display: flex; flex-direction: column; gap: var(--space-xs); }
.focus-label { font-size: var(--font-body); font-weight: 700; text-decoration: underline; margin-top: var(--space-sm); }
.focus-item { font-size: 13px; margin-left: var(--space-md); }
.focus-item::before { content: "— "; }

/* ── ISSUE SLIDE (40/60 split) ───────────────────────────────────────────── */
.issue-inner {
  flex: 1;
  display: grid;
  grid-template-columns: 40% 60%;
  min-height: 0;
}
.issue-left {
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: var(--space-xs);
  padding: var(--space-lg) var(--space-md) var(--space-lg) var(--space-xl);
  overflow: hidden;
}
.issue-ghost { flex-shrink: 0; }
.issue-section { display: flex; flex-direction: column; gap: 3px; flex-shrink: 0; }
.issue-section + .issue-section { margin-top: var(--space-xs); }

.issue-right {
  display: flex;
  align-items: stretch;
  gap: var(--space-sm);
  padding: var(--space-md) var(--space-md) var(--space-lg) var(--space-xs);
  min-height: 0;
}

/* ── PERSONA QUOTE ───────────────────────────────────────────────────────── */
.persona-quote {
  background: rgba(125,217,208,0.18);
  border-left: 3px solid var(--color-teal);
  padding: 5px 8px;
  margin-top: 5px;
  font-size: 11.5px;
  font-style: italic;
  color: #444;
  border-radius: 0 4px 4px 0;
  overflow: hidden;
}
.persona-quote .persona-name { font-weight: 700; font-style: normal; color: var(--color-navy); }

/* ── STRENGTH SLIDE (50/50 split) ────────────────────────────────────────── */
.strength-inner {
  flex: 1;
  display: grid;
  grid-template-columns: 1fr 1fr;
  min-height: 0;
}
.strength-left {
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: var(--space-sm);
  padding: var(--space-xl) var(--space-md) var(--space-xl) var(--space-xl);
  overflow: hidden;
}
.strength-ghost { flex-shrink: 0; }
.strength-right {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: var(--space-md);
  min-height: 0;
}
.strength-right .panel { width: 100%; }

/* ── BACK COVER ──────────────────────────────────────────────────────────── */
.back-inner {
  flex: 1;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  padding: var(--space-xl);
}
.back-logo-card {
  align-self: flex-start;
  background: var(--color-card);
  padding: var(--space-sm) var(--space-lg);
  border-radius: 6px;
  display: flex;
  align-items: center;
  justify-content: center;
}
.back-logo-card img { max-height: 60px; max-width: 220px; object-fit: contain; }
.back-logo-card .brand-text { font-size: 28px; font-weight: 800; color: var(--color-text); }
.back-report-title { font-size: 18px; font-style: italic; color: var(--color-text); }
.back-author { align-self: flex-end; font-size: 14px; color: var(--color-text); }

/* ── PRINT ───────────────────────────────────────────────────────────────── */
@media print {
  * { -webkit-print-color-adjust: exact !important; print-color-adjust: exact !important; }
  body { background: white; }
  .slide { margin: 0; box-shadow: none; }
  @page { size: 960px 540px landscape; margin: 0; }
}
""" + _DEBUG_CSS


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class IssueSlide:
    number: str          # e.g. "02.1"
    category: str        # e.g. "Navigation"
    title: str           # e.g. "Menu Findability"
    issue_text: str
    root_cause: str
    recommendation: str
    screenshot_url: str = ""
    persona_quotes: list[str] = field(default_factory=list)
    persona_names: list[str] = field(default_factory=list)
    # Redesign fields (populated by redesign_client pipeline)
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
# HTML helpers
# ---------------------------------------------------------------------------

def _esc(text: str) -> str:
    """Minimal HTML escaping."""
    return (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _bold_probability(text: str) -> str:
    """Bold probability qualifiers like 'might', 'likely', 'may', 'could'."""
    for word in ("might", "likely", "may", "could", "probably", "possibly"):
        text = re.sub(rf'\b({word})\b', r'<strong>\1</strong>', text, flags=re.IGNORECASE)
    return text


def _path_to_data_uri(path_or_url: str) -> str:
    """Convert a local file path to a base64 data URI for reliable HTML embedding.
    If it's already a data URI or a remote URL, return as-is.
    """
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


def _img_tag(url: str, alt: str = "") -> str:
    """Return an <img> tag with base64-embedded src, or a placeholder div."""
    if not url:
        return (
            '<div style="width:100%;height:100%;background:var(--color-panel);border-radius:8px;'
            'display:flex;align-items:center;justify-content:center;'
            'font-size:12px;color:#888;">No screenshot available</div>'
        )
    data_uri = _path_to_data_uri(url)
    return f'<img src="{_esc(data_uri)}" alt="{_esc(alt)}" loading="eager">'


def _panel(content_html: str, label: str = "") -> str:
    """Wrap content in a .panel container with an optional label below."""
    label_html = f'<div class="panel-label">{_esc(label)}</div>' if label else ""
    return f'<div class="panel">{content_html}{label_html}</div>'


# ---------------------------------------------------------------------------
# Slide template functions
# ---------------------------------------------------------------------------

def slide_cover(title: str, logo_url: str = "", brand_name: str = "") -> str:
    logo_content = (
        _img_tag(logo_url, brand_name) if logo_url
        else f'<span class="brand-text">{_esc(clamp_text(brand_name, _TITLE_MAX))}</span>'
    )
    return f"""
<div class="slide bg-linen">
  <div class="cover-inner">
    <div class="cover-title">{_esc(title)}</div>
    <div class="cover-logo-card">{logo_content}</div>
  </div>
</div>"""


def slide_toc(sections: list[str]) -> str:
    items_html = ""
    for i, s in enumerate(sections, 1):
        items_html += f"""
    <div class="toc-item">
      <span class="toc-num">{i:02d}</span>
      <span class="toc-label">{_esc(clamp_text(s, _TOC_LABEL_MAX))}</span>
    </div>"""
    return f"""
<div class="slide bg-linen">
  <div class="toc-inner">
    <div class="toc-heading">Table of<br>contents</div>
    <div class="toc-list">{items_html}</div>
  </div>
</div>"""


def slide_section_divider(number: str, title: str, subtitle: str = "", bg: str = "teal") -> str:
    bg_class = "bg-teal" if bg == "teal" else "bg-linen"
    sub_html = (
        f'<div class="sec-subtitle subtitle">{_esc(clamp_text(subtitle, _SUBTITLE_MAX))}</div>'
        if subtitle else ""
    )
    return f"""
<div class="slide {bg_class}">
  <div class="sec-inner">
    <div class="sec-number ghost">{_esc(number)}</div>
    <div class="sec-title">{_esc(clamp_text(title, _TITLE_MAX))}</div>
    {sub_html}
  </div>
</div>"""


def slide_intro(
    website_title: str,
    methodology: str,
    focus_areas: list[str],
    overall_summary: str,
) -> str:
    focus_items = "".join(
        f'<div class="focus-item body">{_esc(clamp_text(a, _SUBTITLE_MAX))}</div>'
        for a in focus_areas[:6]
    )
    return f"""
<div class="slide bg-linen">
  <div class="intro-inner">
    <div class="intro-ghost ghost">1.</div>
    <div class="intro-body">
      <p class="body"><strong>{_esc(clamp_text(website_title, _TITLE_MAX))}</strong> — {_esc(clamp_text(overall_summary, _BODY_MAX))}</p>
      <p class="body">{_esc(clamp_text(methodology, _BODY_MAX))}</p>
      <div class="focus-label">The UX review will focus on:</div>
      {focus_items}
    </div>
  </div>
</div>"""


def slide_issue(issue: IssueSlide) -> str:
    # Persona quote callouts (max 2, clamped)
    quotes_html = ""
    for name, quote in zip(issue.persona_names[:2], issue.persona_quotes[:2]):
        quotes_html += f"""
      <div class="persona-quote">
        <span class="persona-name">{_esc(clamp_text(name, 40))}</span>: "{_esc(clamp_text(quote, _QUOTE_MAX))}"
      </div>"""

    issue_body = _bold_probability(_esc(clamp_text(issue.issue_text, _BODY_MAX)))
    root_body  = _bold_probability(_esc(clamp_text(issue.root_cause, _BODY_MAX)))
    rec_body   = _esc(clamp_text(issue.recommendation, _BODY_MAX))

    # Right panel: Current | AI Redesign split, or single screenshot
    if issue.redesign_html_sanitised:
        _srcdoc = issue.redesign_html_sanitised.replace('"', '&quot;')
        right_panel = f"""
  <div class="issue-right">
    {_panel(_img_tag(issue.screenshot_url, 'Current design'), 'Current Design')}
    {_panel(f'<iframe srcdoc="{_srcdoc}" sandbox="allow-same-origin" loading="lazy"></iframe>', 'AI Redesign')}
    {f'<div style="position:absolute;bottom:var(--space-xs);left:40%;right:0;font-size:9.5px;color:#777;font-style:italic;padding:0 var(--space-md);overflow:hidden;white-space:nowrap;text-overflow:ellipsis;">{_esc(clamp_text(issue.redesign_analysis, _ANALYSIS_MAX))}</div>' if issue.redesign_analysis else ''}
  </div>"""
    else:
        placeholder = (
            '<div style="width:100%;height:100%;background:var(--color-panel);border-radius:8px;'
            'display:flex;align-items:center;justify-content:center;'
            'font-size:11px;color:#999;text-align:center;padding:12px;">'
            'Re-design mockup<br>(to be added)</div>'
        )
        right_panel = f"""
  <div class="issue-right">
    {_panel(_img_tag(issue.screenshot_url, 'Current design'), 'Current Design')}
    {_panel(placeholder, 'Re-design')}
  </div>"""

    return f"""
<div class="slide bg-linen" style="position:relative;">
  <div class="issue-inner">
    <div class="issue-left">
      <div class="issue-ghost ghost">{_esc(clamp_text(issue.title, _TITLE_MAX))}</div>
      <div class="issue-section">
        <div class="label">Predicted User Issue</div>
        <div class="body">{issue_body}{quotes_html}</div>
      </div>
      <div class="issue-section">
        <div class="label">Root Cause Analysis</div>
        <div class="body">{root_body}</div>
      </div>
      <div class="issue-section">
        <div class="label">Recommendations</div>
        <div class="body">{rec_body}</div>
      </div>
    </div>
    {right_panel}
  </div>
</div>"""


def slide_strength(s: StrengthSlide) -> str:
    body = (
        f"<strong>{_esc(clamp_text(s.brand_name, 40))}</strong> {_esc(clamp_text(s.description, _BODY_MAX))}"
        if s.brand_name else _esc(clamp_text(s.description, _BODY_MAX))
    )
    return f"""
<div class="slide bg-linen">
  <div class="strength-inner">
    <div class="strength-left">
      <div class="strength-ghost ghost">{_esc(clamp_text(s.title, _TITLE_MAX))}</div>
      <div class="body">{body}</div>
    </div>
    <div class="strength-right">
      {_panel(_img_tag(s.screenshot_url, s.title))}
    </div>
  </div>
</div>"""


def slide_back_cover(logo_url: str, brand_name: str, report_title: str, author: str, year: str) -> str:
    logo_content = (
        _img_tag(logo_url, brand_name) if logo_url
        else f'<span class="brand-text">{_esc(clamp_text(brand_name, _TITLE_MAX))}</span>'
    )
    return f"""
<div class="slide bg-linen">
  <div class="back-inner">
    <div class="back-logo-card">{logo_content}</div>
    <div class="back-report-title">{_esc(report_title)}</div>
    <div class="back-author">By {_esc(author)} — {_esc(year)}</div>
  </div>
</div>"""


# ---------------------------------------------------------------------------
# Full report renderer
# ---------------------------------------------------------------------------

def render_html(data: SlideReportData) -> str:
    """Render all slides into a single self-contained HTML document."""
    slides: list[str] = []

    # 1. Cover
    slides.append(slide_cover("Usability Test Report", data.logo_url, data.brand_name))

    # 2. Table of Contents
    toc_sections = ["Introduction", "Predicted User Issues"]
    if data.strengths:
        toc_sections.append("Elements to Preserve")
    slides.append(slide_toc(toc_sections))

    # 3. Section 01 – Introduction
    slides.append(slide_section_divider("01", "Introduction", bg="linen"))
    slides.append(slide_intro(
        website_title=data.website_title,
        methodology=data.methodology,
        focus_areas=data.focus_areas,
        overall_summary=data.overall_summary,
    ))

    # 4. Section 02 – Predicted User Issues
    slides.append(slide_section_divider("02", "Predicted User Issues"))

    seen_categories: list[str] = []
    for issue in data.issues:
        if issue.category not in seen_categories:
            seen_categories.append(issue.category)
            sub_num = f"02.{len(seen_categories)}"
            slides.append(slide_section_divider(sub_num, issue.title, subtitle=issue.category))
        slides.append(slide_issue(issue))

    # 5. Section 03 – Elements to Preserve
    if data.strengths:
        slides.append(slide_section_divider("03", "Elements to preserve"))
        for s in data.strengths:
            slides.append(slide_strength(s))

    # 6. Back Cover
    slides.append(slide_back_cover(
        logo_url=data.logo_url,
        brand_name=data.brand_name,
        report_title="Usability Test Report",
        author=data.generated_by,
        year=data.generated_date,
    ))

    body = "\n".join(slides)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Usability Test Report — {_esc(data.website_title)}</title>
  <style>{CSS}</style>
</head>
<body>
{body}
</body>
</html>"""


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
        focus_areas = ["Navigation", "Visual Design", "Accessibility", "Content Clarity", "Mobile Responsiveness"]

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
                issue_text=clamp_text(f"Users might encounter difficulties because: {detail}", _BODY_MAX),
                root_cause=clamp_text(f"The page {check_name.replace('_', ' ')} check failed. {detail}", _BODY_MAX),
                recommendation=clamp_text(check_data.get("recommendation", "Review and fix this element."), _BODY_MAX),
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
                        issue.persona_names.append(clamp_text(resp.get("persona_name", "Persona"), 40))
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
