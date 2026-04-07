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
# Design tokens
# ---------------------------------------------------------------------------
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --bg-linen:    #E8E0D5;
    --bg-teal:     #7DD9D0;
    --navy:        #1A2280;
    --charcoal:    #2B2B2B;
    --white:       #FFFFFF;
    --grey-sub:    #7A7A9A;
    --yellow-card: #FAFAC8;
    --font:        'Inter', 'Segoe UI', Arial, sans-serif;
}

body {
    background: #999;
    font-family: var(--font);
    color: var(--charcoal);
}

/* ── Slide container ─────────────────────────────────────────────────────── */
.slide {
    width: 960px;
    height: 540px;
    position: relative;
    overflow: hidden;
    margin: 0 auto 24px;
    background: var(--bg-linen);
    page-break-after: always;
    break-after: page;
}

/* ── Backgrounds ─────────────────────────────────────────────────────────── */
.slide.bg-teal   { background: var(--bg-teal); }
.slide.bg-linen  { background: var(--bg-linen); }

/* ── COVER ───────────────────────────────────────────────────────────────── */
.cover-title {
    position: absolute;
    top: 60px; left: 60px;
    font-size: 52px;
    font-weight: 800;
    color: var(--charcoal);
    line-height: 1.15;
    max-width: 600px;
}
.cover-logo-card {
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    background: var(--yellow-card);
    padding: 24px 36px;
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    min-width: 200px;
    min-height: 100px;
}
.cover-logo-card img { max-height: 80px; max-width: 280px; object-fit: contain; }
.cover-logo-card .brand-text {
    font-size: 36px; font-weight: 800; color: var(--charcoal);
}

/* ── TABLE OF CONTENTS ───────────────────────────────────────────────────── */
.toc-left {
    position: absolute;
    top: 60px; left: 60px;
    font-size: 38px; font-weight: 800; color: var(--navy);
    line-height: 1.2; max-width: 260px;
}
.toc-right {
    position: absolute;
    top: 60px; left: 360px; right: 60px;
}
.toc-item {
    display: flex;
    align-items: baseline;
    gap: 16px;
    padding: 14px 0;
    border-bottom: 1px dashed #B0A89A;
}
.toc-item:last-child { border-bottom: none; }
.toc-num { font-size: 13px; font-weight: 700; color: var(--navy); min-width: 28px; }
.toc-label { font-size: 18px; font-weight: 600; color: var(--charcoal); }

/* ── SECTION DIVIDER ─────────────────────────────────────────────────────── */
.sec-number {
    position: absolute;
    top: 50px; left: 60px;
    font-size: 100px; font-weight: 800; color: var(--white);
    line-height: 1;
}
.sec-title {
    position: absolute;
    top: 170px; left: 60px;
    font-size: 56px; font-weight: 800; color: var(--navy);
    line-height: 1.1; max-width: 700px;
}
.sec-subtitle {
    position: absolute;
    top: 280px; left: 60px;
    font-size: 20px; font-weight: 600; color: var(--grey-sub);
}

/* ── INTRO TEXT SLIDE ────────────────────────────────────────────────────── */
.intro-ghost {
    position: absolute;
    top: 30px; left: 50px;
    font-size: 80px; font-weight: 800;
    color: rgba(255,255,255,0.55);
    line-height: 1;
}
.intro-body {
    position: absolute;
    top: 110px; left: 60px; right: 60px; bottom: 40px;
    overflow: hidden;
}
.intro-body p { font-size: 15px; line-height: 1.65; margin-bottom: 12px; }
.intro-body .focus-label {
    font-size: 15px; font-weight: 700; text-decoration: underline; margin-top: 14px; margin-bottom: 6px;
}
.intro-body .focus-item { font-size: 14px; margin-left: 16px; margin-bottom: 4px; }
.intro-body .focus-item::before { content: "— "; }

/* ── ISSUE DETAIL (40/60 split) ──────────────────────────────────────────── */
.issue-left {
    position: absolute;
    top: 0; left: 0;
    width: 40%; height: 100%;
    padding: 40px 36px 40px 56px;
    display: flex; flex-direction: column; justify-content: center;
}
.issue-ghost-title {
    font-size: 48px; font-weight: 800;
    color: rgba(255,255,255,0.6);
    line-height: 1.1; margin-bottom: 18px;
}
.issue-section-label {
    font-size: 13px; font-weight: 700; color: var(--navy);
    margin-bottom: 5px; margin-top: 14px;
}
.issue-section-label:first-of-type { margin-top: 0; }
.issue-section-body {
    font-size: 13px; line-height: 1.6; color: var(--charcoal);
}
.issue-section-body strong { font-weight: 700; }

.issue-right {
    position: absolute;
    top: 0; left: 40%; right: 0; height: 100%;
    display: flex; align-items: center; justify-content: center;
    gap: 16px; padding: 24px 24px 50px 16px;
}
.screenshot-box {
    flex: 1;
    display: flex; flex-direction: column; align-items: center;
    height: 100%;
}
.screenshot-box img {
    flex: 1;
    width: 100%; object-fit: contain; object-position: top;
    border-radius: 6px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.15);
}
.screenshot-label {
    font-size: 12px; font-weight: 500; color: var(--charcoal);
    margin-top: 8px; text-align: center;
}

/* ── STRENGTH / PRESERVE (50/50 split) ───────────────────────────────────── */
.strength-left {
    position: absolute;
    top: 0; left: 0;
    width: 50%; height: 100%;
    padding: 50px 36px 50px 56px;
    display: flex; flex-direction: column; justify-content: center;
}
.strength-ghost-title {
    font-size: 52px; font-weight: 800;
    color: rgba(255,255,255,0.55);
    line-height: 1.1; margin-bottom: 24px;
}
.strength-body {
    font-size: 15px; line-height: 1.7; color: var(--charcoal);
}
.strength-body strong { font-weight: 700; }

.strength-right {
    position: absolute;
    top: 0; left: 50%; right: 0; height: 100%;
    display: flex; align-items: center; justify-content: center;
    padding: 24px;
}
.strength-right img {
    max-height: 100%; max-width: 100%;
    object-fit: contain; object-position: center;
    border-radius: 6px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.15);
}

/* ── BACK COVER ──────────────────────────────────────────────────────────── */
.back-logo-card {
    position: absolute;
    top: 80px; left: 60px;
    background: var(--yellow-card);
    padding: 20px 30px;
    border-radius: 4px;
    display: flex; align-items: center; justify-content: center;
}
.back-logo-card img { max-height: 70px; max-width: 240px; object-fit: contain; }
.back-logo-card .brand-text { font-size: 30px; font-weight: 800; color: var(--charcoal); }
.back-report-title {
    position: absolute;
    top: 230px; left: 60px;
    font-size: 18px; font-style: italic; color: var(--charcoal);
}
.back-author {
    position: absolute;
    bottom: 50px; right: 60px;
    font-size: 14px; color: var(--charcoal);
}

/* ── PERSONA QUOTE CALLOUT (used inside issue slides) ────────────────────── */
.persona-quote {
    background: rgba(125,217,208,0.18);
    border-left: 3px solid var(--bg-teal);
    padding: 6px 10px;
    margin-top: 8px;
    font-size: 11.5px;
    font-style: italic;
    color: #444;
    border-radius: 0 4px 4px 0;
}
.persona-quote .persona-name { font-weight: 700; font-style: normal; color: var(--navy); }

/* ── REDESIGN SPLIT (screenshot left | HTML iframe right) ──────────────── */
.redesign-slide {
    display: flex;
    width: 100%; height: 100%;
}
.redesign-old {
    width: 50%; height: 100%;
    display: flex; flex-direction: column;
    padding: 20px 12px 20px 24px;
}
.redesign-old img {
    flex: 1;
    width: 100%; object-fit: cover; object-position: top;
    border-radius: 6px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.15);
}
.redesign-new {
    width: 50%; height: 100%;
    display: flex; flex-direction: column;
    padding: 20px 24px 20px 12px;
}
.redesign-new iframe {
    flex: 1;
    width: 100%; border: none;
    border-radius: 6px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.15);
    background: #fff;
}
.redesign-label {
    font-size: 11px; font-weight: 600;
    color: var(--navy); text-align: center;
    margin-top: 6px; text-transform: uppercase; letter-spacing: 0.05em;
}
.redesign-analysis {
    position: absolute;
    bottom: 0; left: 0; right: 0;
    background: rgba(26,34,128,0.08);
    padding: 6px 24px;
    font-size: 10.5px; color: #555;
    border-top: 1px solid rgba(26,34,128,0.1);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}

/* ── PRINT ───────────────────────────────────────────────────────────────── */
@media print {
    body { background: white; }
    .slide { margin: 0; box-shadow: none; }
    @page { size: 960px 540px landscape; margin: 0; }
}
"""

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
    redesign_analysis: str = ""          # AI analysis of the screenshot
    redesign_html: str = ""              # UX-improved HTML code
    redesign_html_sanitised: str = ""    # srcdoc-safe version for iframe embed


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


def _img_tag(url: str, alt: str = "") -> str:
    """Return an <img> tag. If url is empty, return a placeholder div."""
    if not url:
        return (
            '<div style="width:100%;height:100%;background:#ddd;border-radius:6px;'
            'display:flex;align-items:center;justify-content:center;'
            'font-size:12px;color:#888;">No screenshot</div>'
        )
    return f'<img src="{_esc(url)}" alt="{_esc(alt)}" loading="lazy">'


# ---------------------------------------------------------------------------
# Slide template functions
# ---------------------------------------------------------------------------

def slide_cover(title: str, logo_url: str = "", brand_name: str = "") -> str:
    logo_content = _img_tag(logo_url, brand_name) if logo_url else f'<span class="brand-text">{_esc(brand_name)}</span>'
    return f"""
<div class="slide bg-linen">
  <div class="cover-title">{_esc(title)}</div>
  <div class="cover-logo-card">{logo_content}</div>
</div>"""


def slide_toc(sections: list[str]) -> str:
    items_html = ""
    for i, s in enumerate(sections, 1):
        items_html += f"""
    <div class="toc-item">
      <span class="toc-num">{i:02d}</span>
      <span class="toc-label">{_esc(s)}</span>
    </div>"""
    return f"""
<div class="slide bg-linen">
  <div class="toc-left">Table of<br>contents</div>
  <div class="toc-right">{items_html}</div>
</div>"""


def slide_section_divider(number: str, title: str, subtitle: str = "", bg: str = "teal") -> str:
    bg_class = "bg-teal" if bg == "teal" else "bg-linen"
    sub_html = f'<div class="sec-subtitle">{_esc(subtitle)}</div>' if subtitle else ""
    return f"""
<div class="slide {bg_class}">
  <div class="sec-number">{_esc(number)}</div>
  <div class="sec-title">{_esc(title)}</div>
  {sub_html}
</div>"""


def slide_intro(
    website_title: str,
    methodology: str,
    focus_areas: list[str],
    overall_summary: str,
) -> str:
    focus_items = "".join(f'<div class="focus-item">{_esc(a)}</div>' for a in focus_areas)
    return f"""
<div class="slide bg-linen">
  <div class="intro-ghost">1.</div>
  <div class="intro-body">
    <p><strong>{_esc(website_title)}</strong> — {_esc(overall_summary)}</p>
    <p>{_esc(methodology)}</p>
    <div class="focus-label">The UX review will focus on:</div>
    {focus_items}
  </div>
</div>"""


def slide_issue(issue: IssueSlide) -> str:
    # Build persona quote callouts
    quotes_html = ""
    for name, quote in zip(issue.persona_names, issue.persona_quotes):
        quotes_html += f"""
      <div class="persona-quote">
        <span class="persona-name">{_esc(name)}:</span> "{_esc(quote)}"
      </div>"""

    issue_body = _bold_probability(_esc(issue.issue_text))
    root_body = _bold_probability(_esc(issue.root_cause))
    rec_body = _esc(issue.recommendation)

    # Right panel: if a redesign exists, show Current | Redesign side-by-side;
    # otherwise fall back to the single stacked screenshot layout.
    if issue.redesign_html_sanitised:
        _srcdoc = issue.redesign_html_sanitised.replace('"', '&quot;')
        right_panel = f"""
  <div class="issue-right" style="display:flex;flex-direction:column;gap:6px;padding:16px 20px 16px 0;">
    <div style="display:flex;gap:8px;flex:1;min-height:0;">
      <div style="flex:1;display:flex;flex-direction:column;min-width:0;">
        <div class="screenshot-box" style="flex:1;">
          {_img_tag(issue.screenshot_url, 'Current design')}
        </div>
        <div class="screenshot-label">Current Design</div>
      </div>
      <div style="flex:1;display:flex;flex-direction:column;min-width:0;">
        <div class="screenshot-box" style="flex:1;">
          <iframe srcdoc="{_srcdoc}" sandbox="allow-same-origin" loading="lazy"
            style="width:100%;height:100%;border:none;border-radius:6px;"></iframe>
        </div>
        <div class="screenshot-label">AI Redesign</div>
      </div>
    </div>
    {f'<div style="font-size:9.5px;color:#777;font-style:italic;line-height:1.4;">{_esc(issue.redesign_analysis[:160])}</div>' if issue.redesign_analysis else ''}
  </div>"""
    else:
        right_panel = f"""
  <div class="issue-right">
    <div class="screenshot-box">
      {_img_tag(issue.screenshot_url, 'Current design')}
      <div class="screenshot-label">Current design</div>
    </div>
    <div class="screenshot-box">
      <div style="width:100%;height:100%;background:#f0ece6;border-radius:6px;
                  display:flex;align-items:center;justify-content:center;
                  font-size:11px;color:#999;text-align:center;padding:12px;">
        Re-design mockup<br>(to be added)
      </div>
      <div class="screenshot-label">Re-design</div>
    </div>
  </div>"""

    return f"""
<div class="slide bg-linen">
  <div class="issue-left">
    <div class="issue-ghost-title">{_esc(issue.title)}</div>
    <div class="issue-section-label">Predicted User Issue</div>
    <div class="issue-section-body">{issue_body}{quotes_html}</div>
    <div class="issue-section-label">Root Cause Analysis</div>
    <div class="issue-section-body">{root_body}</div>
    <div class="issue-section-label">Recommendations: Design Solutions</div>
    <div class="issue-section-body">{rec_body}</div>
  </div>
  {right_panel}
</div>"""


def slide_strength(s: StrengthSlide) -> str:
    body = f"<strong>{_esc(s.brand_name)}</strong> {_esc(s.description)}" if s.brand_name else _esc(s.description)
    return f"""
<div class="slide bg-linen">
  <div class="strength-left">
    <div class="strength-ghost-title">{_esc(s.title)}</div>
    <div class="strength-body">{body}</div>
  </div>
  <div class="strength-right">
    {_img_tag(s.screenshot_url, s.title)}
  </div>
</div>"""


def slide_redesign(issue: IssueSlide) -> str:
    """
    A full-width 'Old vs New' slide that shows:
      - Left half: current-design screenshot
      - Right half: AI-generated HTML redesign in an iframe
    Only rendered when issue.redesign_html_sanitised is populated.
    """
    old_content = _img_tag(issue.screenshot_url, "Current design") if issue.screenshot_url else (
        '<div style="width:100%;height:100%;background:#ddd;border-radius:6px;'
        'display:flex;align-items:center;justify-content:center;font-size:12px;color:#888;">'
        'No screenshot</div>'
    )
    if issue.redesign_html_sanitised:
        _srcdoc = issue.redesign_html_sanitised.replace('"', '&quot;')
        new_content = (
            f'<iframe srcdoc="{_srcdoc}" sandbox="allow-same-origin" loading="lazy"></iframe>'
        )
    else:
        new_content = (
            '<div style="width:100%;height:100%;background:#f0ece6;border-radius:6px;'
            'display:flex;align-items:center;justify-content:center;font-size:11px;'
            'color:#999;text-align:center;padding:12px;">'
            'Redesign being generated…</div>'
        )
    analysis_strip = f'<div class="redesign-analysis">AI Analysis: {_esc(issue.redesign_analysis[:180])}</div>' if issue.redesign_analysis else ""
    return f"""
<div class="slide bg-linen" style="overflow:hidden;">
  <div class="redesign-slide">
    <div class="redesign-old">
      {old_content}
      <div class="redesign-label">Current Design</div>
    </div>
    <div class="redesign-new">
      {new_content}
      <div class="redesign-label">AI Redesign</div>
    </div>
  </div>
  {analysis_strip}
</div>"""


def slide_back_cover(logo_url: str, brand_name: str, report_title: str, author: str, year: str) -> str:
    logo_content = _img_tag(logo_url, brand_name) if logo_url else f'<span class="brand-text">{_esc(brand_name)}</span>'
    return f"""
<div class="slide bg-linen">
  <div class="back-logo-card">{logo_content}</div>
  <div class="back-report-title">{_esc(report_title)}</div>
  <div class="back-author">By {_esc(author)} — {_esc(year)}</div>
</div>"""


# ---------------------------------------------------------------------------
# Full report renderer
# ---------------------------------------------------------------------------

def render_html(data: SlideReportData) -> str:
    """Render all slides into a single HTML document."""
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

    # Group issues by category for sub-section dividers
    seen_categories: list[str] = []
    issue_counter = 1
    for issue in data.issues:
        if issue.category not in seen_categories:
            seen_categories.append(issue.category)
            sub_num = f"02.{len(seen_categories)}"
            slides.append(slide_section_divider(sub_num, issue.title, subtitle=issue.category))
        slides.append(slide_issue(issue))
        issue_counter += 1

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

    overall_score = ux_data.get("overall_score", 0)
    overall_summary = ux_data.get("overall_summary", "")
    strengths_raw = ux_data.get("strengths", [])
    weaknesses_raw = ux_data.get("weaknesses", [])
    dimensions = ux_data.get("dimensions", [])
    heuristics = ux_data.get("heuristic_checks", {})
    screenshots = ux_data.get("screenshots", {})

    # Pick the best available screenshot URL
    desktop_ss = screenshots.get("desktop") or screenshots.get("mobile") or ""

    # Focus areas from UX dimensions
    focus_areas = [d.get("name", "") for d in dimensions if d.get("name")]
    if not focus_areas:
        focus_areas = ["Navigation", "Visual Design", "Accessibility", "Content Clarity", "Mobile Responsiveness"]

    # Methodology blurb
    methodology = (
        f"This report was generated by the OASIS UX Simulation App using {len(personas)} "
        f"AI personas who browsed {url} and provided usability feedback. "
        f"The analysis combines automated heuristic checks, AI visual critique, and "
        f"simulated persona interactions to predict real user issues."
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
                title=check_name.replace("_", " ").title(),
                issue_text=f"Users might encounter difficulties because: {detail}",
                root_cause=f"The page {check_name.replace('_', ' ')} check failed. {detail}",
                recommendation=check_data.get("recommendation", "Review and fix this element."),
                screenshot_url=desktop_ss,
            ))
            issue_counter += 1

    # From UX dimension weaknesses
    for dim in dimensions:
        dim_name = dim.get("name", "")
        dim_issues = dim.get("issues", [])
        dim_recs = dim.get("recommendations", [])
        for i, iss in enumerate(dim_issues[:2]):  # max 2 per dimension
            # iss / rec may be a dict (from LLM JSON) or a plain string — normalise
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
                title=f"{dim_name}: {iss_text[:40]}",
                issue_text=f"Users might struggle because {iss_text}",
                root_cause=f"The {dim_name} dimension scored {dim.get('score', 'N/A')}/10. {iss_text}",
                recommendation=rec_text,
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
                # Attach the first pain point to the most relevant issue
                for issue in issues:
                    if len(issue.persona_quotes) < 2:
                        issue.persona_quotes.append(pain_points[0])
                        issue.persona_names.append(resp.get("persona_name", "Persona"))
                        break

    # ── Build Strength slides ───────────────────────────────────────────────
    strength_slides: list[StrengthSlide] = []

    # From UX strengths
    for s in strengths_raw[:3]:
        s_text = s if isinstance(s, str) else (
            s.get("strength") or s.get("text") or s.get("description") or str(s)
        )
        strength_slides.append(StrengthSlide(
            title=s_text[:40] if len(s_text) > 40 else s_text,
            brand_name=brand_name,
            description=s_text,
            screenshot_url=desktop_ss,
        ))

    # From Mode 3 visual resonance highlights
    mode3_results = [r for r in sim_results if r.get("mode") == "visual_branding"]
    if mode3_results:
        responses = mode3_results[0].get("responses", [])
        high_resonance = [r for r in responses if r.get("resonance_score", 0) >= 7]
        for resp in high_resonance[:2]:
            comment = resp.get("in_character_comment", "")
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
