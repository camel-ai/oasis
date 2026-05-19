"""
quarkdown_generator.py
======================
Quarkdown-based slide report generator for the OASIS UX Simulation App.

Pipeline:
  SlideReportData + DesignKit
  → build_qd_source(data, kit)   # Quarkdown .qd source string
  → compile_qd(source, out_dir)  # calls `quarkdown compile` via subprocess → HTML dir
  → inject_design_kit(out_dir, kit)  # injects CSS design-token overrides
  → html_dir_to_pdf(out_dir, pdf_path)  # Playwright screenshots each slide → PDF

Quarkdown slide syntax:
  .doctype {slides}              sets reveal.js output
  ---                            slide separator (same as Marpit)
  .theme {oasis}                 custom theme name (maps to theme/theme.css)
  .var {key} {value}             set a template variable

Design kit injection:
  After Quarkdown compiles the HTML, we inject a <style> block containing
  the kit's CSS custom properties into the <head> of index.html.
  This overrides the default OASIS theme tokens without touching the .qd source.

Requires:
  - Java 17 at /opt/jdk17 (installed by setup script)
  - Quarkdown v2.0.1 at /opt/quarkdown (installed by setup script)
  - Playwright (already a project dependency)
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Quarkdown binary (wrapper that sets JAVA_HOME=/opt/jdk17)
# ---------------------------------------------------------------------------
_QD_BIN = Path("/usr/local/bin/quarkdown")
_QD_JAVA_HOME = Path("/opt/jdk17")

# ---------------------------------------------------------------------------
# Content constraints (same as Marpit pipeline for consistency)
# ---------------------------------------------------------------------------
_TITLE_MAX     = 60
_SUBTITLE_MAX  = 80
_BODY_MAX      = 220
_QUOTE_MAX     = 140
_ANALYSIS_MAX  = 180
_TOC_LABEL_MAX = 50


def clamp_text(text: str, max_chars: int) -> str:
    """Truncate text to max_chars, preferring sentence boundaries."""
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    window = text[:max_chars]
    for punct in (". ", "! ", "? ", ".\n", "!\n", "?\n"):
        idx = window.rfind(punct)
        if idx > max_chars // 3:
            return text[:idx + 1].rstrip()
    word_cut = window.rsplit(" ", 1)[0]
    if word_cut:
        return word_cut + "\u2026"
    return window + "\u2026"


def _qd_escape(text: str) -> str:
    """Escape characters that are special in Quarkdown inline context."""
    # Escape leading dots (function calls), backticks, and braces
    text = (text or "").replace("\\", "\\\\")
    text = text.replace("{", "\\{").replace("}", "\\}")
    # Escape a dot at the start of a line (would be parsed as a function call)
    text = re.sub(r"^(\s*)\.", r"\1\\.", text, flags=re.MULTILINE)
    return text


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


# ---------------------------------------------------------------------------
# DesignKit dataclass (populated by design_chooser.py)
# ---------------------------------------------------------------------------

@dataclass
class DesignKit:
    """Represents a slide design kit with CSS custom properties."""
    id: str                          # e.g. "bold-signal", "stripe", "dk/classic"
    name: str                        # Human-readable name
    source: str                      # "frontend-slides" | "designkits.sh" | "oasis-default"
    vibe: str = ""                   # Short description
    css_vars: dict[str, str] = field(default_factory=dict)   # CSS custom properties
    font_imports: list[str] = field(default_factory=list)    # Google Fonts @import URLs
    extra_css: str = ""              # Any additional CSS rules

    def to_css_block(self) -> str:
        """Render as a <style> block for injection into the compiled HTML."""
        lines: list[str] = []
        for url in self.font_imports:
            lines.append(f"@import url('{url}');")
        if lines:
            lines.append("")
        lines.append(":root {")
        for k, v in self.css_vars.items():
            lines.append(f"    {k}: {v};")
        lines.append("}")
        if self.extra_css:
            lines.append("")
            lines.append(self.extra_css)
        return "\n".join(lines)


# Default OASIS design kit (matches the old Marpit theme tokens)
OASIS_DEFAULT_KIT = DesignKit(
    id="oasis-default",
    name="OASIS Default",
    source="oasis-default",
    vibe="Clean, professional, teal-accented",
    css_vars={
        "--oasis-bg":          "#FAFAF8",
        "--oasis-bg-dark":     "#1A2332",
        "--oasis-accent":      "#00897B",
        "--oasis-accent-2":    "#00695C",
        "--oasis-text":        "#1A1A1A",
        "--oasis-text-muted":  "#555555",
        "--oasis-border":      "#E0E0E0",
        "--oasis-card":        "#FFFFFF",
        "--oasis-font-display":"'Inter', 'Helvetica Neue', sans-serif",
        "--oasis-font-body":   "'Inter', 'Helvetica Neue', sans-serif",
        "--oasis-font-mono":   "'JetBrains Mono', 'Fira Code', monospace",
        "--oasis-radius":      "8px",
        "--oasis-shadow":      "0 2px 8px rgba(0,0,0,0.08)",
    },
    font_imports=[
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap",
    ],
)


# ---------------------------------------------------------------------------
# Quarkdown .qd source builder
# ---------------------------------------------------------------------------

def _slide_sep() -> str:
    return "\n\n---\n\n"


def _cover_slide(data: "SlideReportData") -> str:
    logo_line = ""
    if data.logo_url:
        uri = _path_to_data_uri(data.logo_url)
        if uri:
            logo_line = f'\n<img src="{uri}" alt="{_qd_escape(data.brand_name)}" class="oasis-logo">\n'
    brand = _qd_escape(clamp_text(data.brand_name, _TITLE_MAX))
    return f"""\
<!-- _class: oasis-cover -->

# Usability Test Report

<div class="oasis-cover-brand">{logo_line or brand}</div>

<div class="oasis-cover-meta">Generated by OASIS UX Simulation App &mdash; {data.generated_date}</div>
"""


def _toc_slide(sections: list[str]) -> str:
    items = ""
    for i, s in enumerate(sections, 1):
        items += f'<div class="oasis-toc-item"><span class="oasis-toc-num">{i:02d}</span><span>{_qd_escape(clamp_text(s, _TOC_LABEL_MAX))}</span></div>\n'
    return f"""\
<!-- _class: oasis-toc -->

# Table of Contents

<div class="oasis-toc-grid">
{items}</div>
"""


def _divider_slide(number: str, title: str, subtitle: str = "", dark: bool = True) -> str:
    cls = "oasis-divider" if dark else "oasis-divider-light"
    sub_line = f'\n<p class="oasis-divider-sub">{_qd_escape(clamp_text(subtitle, _SUBTITLE_MAX))}</p>' if subtitle else ""
    return f"""\
<!-- _class: {cls} -->

<div class="oasis-sec-num">{number}</div>

# {_qd_escape(clamp_text(title, _TITLE_MAX))}{sub_line}
"""


def _intro_slide(data: "SlideReportData") -> str:
    focus_items = "\n".join(
        f"- {_qd_escape(clamp_text(a, _SUBTITLE_MAX))}" for a in data.focus_areas[:6]
    )
    return f"""\
<!-- _class: oasis-intro -->

**{_qd_escape(clamp_text(data.website_title, _TITLE_MAX))}** — {_qd_escape(clamp_text(data.overall_summary, _BODY_MAX))}

{_qd_escape(clamp_text(data.methodology, _BODY_MAX))}

**Focus areas:**

{focus_items}
"""


def _issue_slide(issue: "IssueSlide") -> str:
    quotes_html = ""
    for name, quote in zip(issue.persona_names[:2], issue.persona_quotes[:2]):
        quotes_html += (
            f'<blockquote class="oasis-quote"><strong>{_qd_escape(clamp_text(name, 40))}:</strong> '
            f'&ldquo;{_qd_escape(clamp_text(quote, _QUOTE_MAX))}&rdquo;</blockquote>\n'
        )

    # Screenshot panel
    screenshot_html = ""
    if issue.screenshot_url:
        uri = _path_to_data_uri(issue.screenshot_url)
        screenshot_html = f'<img src="{uri}" alt="Current design" class="oasis-panel-img">'
    else:
        screenshot_html = '<div class="oasis-panel-placeholder">No screenshot</div>'

    # Redesign panel
    redesign_html = ""
    if issue.redesign_screenshot_url:
        uri = _path_to_data_uri(issue.redesign_screenshot_url)
        redesign_html = f'<img src="{uri}" alt="AI Redesign" class="oasis-panel-img">'
        if issue.redesign_analysis:
            redesign_html += f'<p class="oasis-panel-caption">{_qd_escape(clamp_text(issue.redesign_analysis, _ANALYSIS_MAX))}</p>'
    elif issue.redesign_html_sanitised:
        srcdoc = issue.redesign_html_sanitised.replace('"', "&quot;")
        redesign_html = f'<iframe srcdoc="{srcdoc}" sandbox="allow-same-origin" class="oasis-panel-iframe"></iframe>'
    else:
        redesign_html = '<div class="oasis-panel-placeholder">Re-design mockup (to be added)</div>'

    return f"""\
<!-- _class: oasis-issue -->

<div class="oasis-issue-layout">
<div class="oasis-issue-left">

<h3 class="oasis-issue-title">{_qd_escape(clamp_text(issue.title, _TITLE_MAX))}</h3>

**Predicted User Issue**

{_qd_escape(clamp_text(issue.issue_text, _BODY_MAX))}

{quotes_html}

**Root Cause**

{_qd_escape(clamp_text(issue.root_cause, _BODY_MAX))}

**Recommendation**

{_qd_escape(clamp_text(issue.recommendation, _BODY_MAX))}

</div>
<div class="oasis-issue-right">
<div class="oasis-panel">
<div class="oasis-panel-label">Current Design</div>
{screenshot_html}
</div>
<div class="oasis-panel">
<div class="oasis-panel-label">AI Redesign</div>
{redesign_html}
</div>
</div>
</div>
"""


def _strength_slide(s: "StrengthSlide") -> str:
    body = (
        f"**{_qd_escape(clamp_text(s.brand_name, 40))}** {_qd_escape(clamp_text(s.description, _BODY_MAX))}"
        if s.brand_name else _qd_escape(clamp_text(s.description, _BODY_MAX))
    )
    img_html = ""
    if s.screenshot_url:
        uri = _path_to_data_uri(s.screenshot_url)
        img_html = f'<img src="{uri}" alt="{_qd_escape(s.title)}" class="oasis-panel-img">'
    else:
        img_html = '<div class="oasis-panel-placeholder">No screenshot</div>'

    return f"""\
<!-- _class: oasis-strength -->

<div class="oasis-strength-layout">
<div class="oasis-strength-left">

<h3 class="oasis-strength-title">{_qd_escape(clamp_text(s.title, _TITLE_MAX))}</h3>

{body}

</div>
<div class="oasis-strength-right">
<div class="oasis-panel">
{img_html}
</div>
</div>
</div>
"""


def _back_cover_slide(data: "SlideReportData") -> str:
    logo_line = ""
    if data.logo_url:
        uri = _path_to_data_uri(data.logo_url)
        if uri:
            logo_line = f'<img src="{uri}" alt="{_qd_escape(data.brand_name)}" class="oasis-logo">'
    brand = logo_line or f'<span class="oasis-brand-text">{_qd_escape(clamp_text(data.brand_name, _TITLE_MAX))}</span>'
    return f"""\
<!-- _class: oasis-back -->

{brand}

*Usability Test Report*

By {data.generated_by} &mdash; {data.generated_date}
"""


def build_qd_source(data: "SlideReportData", kit: Optional[DesignKit] = None) -> str:
    """
    Convert a SlideReportData into a Quarkdown .qd source string.

    The source uses `.doctype {slides}` to target reveal.js output.
    Design kit CSS is injected post-compilation (see inject_design_kit()),
    but the kit name is embedded as a comment for traceability.
    """
    kit = kit or OASIS_DEFAULT_KIT
    slides: list[str] = []

    # Quarkdown document header
    header = f"""\
.doctype {{slides}}

<!-- OASIS design kit: {kit.id} ({kit.source}) -->

"""
    # 1. Cover
    slides.append(_cover_slide(data))

    # 2. TOC
    toc_sections = ["Introduction", "Predicted User Issues"]
    if data.strengths:
        toc_sections.append("Elements to Preserve")
    slides.append(_toc_slide(toc_sections))

    # 3. Section 01 – Introduction
    slides.append(_divider_slide("01", "Introduction", dark=False))
    slides.append(_intro_slide(data))

    # 4. Section 02 – Issues
    slides.append(_divider_slide("02", "Predicted User Issues"))

    seen_categories: list[str] = []
    for issue in data.issues:
        if issue.category not in seen_categories:
            seen_categories.append(issue.category)
            sub_num = f"02.{len(seen_categories)}"
            slides.append(_divider_slide(sub_num, issue.title, subtitle=issue.category))
        slides.append(_issue_slide(issue))

    # 5. Section 03 – Strengths
    if data.strengths:
        slides.append(_divider_slide("03", "Elements to Preserve"))
        for s in data.strengths:
            slides.append(_strength_slide(s))

    # 6. Back cover
    slides.append(_back_cover_slide(data))

    return header + _slide_sep().join(slides)


# ---------------------------------------------------------------------------
# Quarkdown compiler
# ---------------------------------------------------------------------------

def compile_qd(source: str, out_dir: Path) -> Path:
    """
    Write source to a temp .qd file and compile it with Quarkdown.

    Returns the path to the compiled output directory (contains index.html).
    Raises RuntimeError on compilation failure.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".qd", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(source)
        tmp_path = Path(tmp.name)

    env = os.environ.copy()
    env["JAVA_HOME"] = str(_QD_JAVA_HOME)
    # Quarkdown needs Node.js for reveal.js post-processing
    env["QD_NO_SANDBOX"] = "true"

    try:
        result = subprocess.run(
            [
                str(_QD_BIN),
                "compile",
                "--strict",
                "-o", str(out_dir),
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass

    if result.returncode != 0:
        raise RuntimeError(
            f"Quarkdown compile failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    if result.stderr:
        logger.warning("Quarkdown stderr: %s", result.stderr.strip())

    # Find the output subdirectory (Quarkdown creates a named subdir)
    subdirs = [p for p in out_dir.iterdir() if p.is_dir()]
    if subdirs:
        return subdirs[0]
    return out_dir


def inject_design_kit(compiled_dir: Path, kit: DesignKit) -> None:
    """
    Inject the design kit's CSS custom properties into the compiled index.html.

    Quarkdown's reveal.js output uses CSS variables for theming; we override
    them by inserting a <style> block before </head>.
    """
    index_html = compiled_dir / "index.html"
    if not index_html.exists():
        logger.warning("inject_design_kit: index.html not found in %s", compiled_dir)
        return

    css_block = kit.to_css_block()
    oasis_layout_css = _oasis_layout_css()

    injection = f"""
<!-- OASIS design kit injection: {kit.id} -->
<style>
{css_block}
</style>
<style>
{oasis_layout_css}
</style>
"""
    html = index_html.read_text(encoding="utf-8")
    html = html.replace("</head>", injection + "</head>", 1)
    index_html.write_text(html, encoding="utf-8")


def _oasis_layout_css() -> str:
    """
    Core OASIS layout CSS that maps our custom properties onto reveal.js.
    This is injected alongside the design kit CSS.
    """
    return """
/* ── OASIS reveal.js theme overrides ─────────────────────────────────────── */
.reveal {
    font-family: var(--oasis-font-body, 'Inter', sans-serif);
    color: var(--oasis-text, #1a1a1a);
    background: var(--oasis-bg, #FAFAF8);
}
.reveal .slides section {
    background: var(--oasis-bg, #FAFAF8);
    color: var(--oasis-text, #1a1a1a);
    padding: 40px 48px;
    box-sizing: border-box;
    text-align: left;
}
.reveal h1, .reveal h2, .reveal h3 {
    font-family: var(--oasis-font-display, 'Inter', sans-serif);
    color: var(--oasis-text, #1a1a1a);
    text-transform: none;
    letter-spacing: -0.02em;
}
.reveal h1 { font-size: clamp(1.6rem, 3.5vw, 2.4rem); }
.reveal h2 { font-size: clamp(1.2rem, 2.5vw, 1.8rem); }
.reveal h3 { font-size: clamp(1rem, 2vw, 1.4rem); }
.reveal p, .reveal li { font-size: clamp(0.75rem, 1.4vw, 1rem); line-height: 1.6; }

/* Cover slide */
.oasis-cover .reveal .slides section,
section.oasis-cover {
    background: var(--oasis-bg-dark, #1A2332) !important;
    color: #fff !important;
    display: flex !important;
    flex-direction: column;
    align-items: flex-start;
    justify-content: center;
}
.oasis-cover h1 { color: #fff; font-size: clamp(2rem, 4vw, 3rem); }
.oasis-cover-brand { margin: 1.5rem 0 0.5rem; }
.oasis-cover-meta { font-size: 0.75rem; opacity: 0.6; margin-top: 1rem; }
.oasis-logo { max-height: 60px; max-width: 240px; object-fit: contain; }

/* TOC slide */
.oasis-toc-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 12px;
    margin-top: 1.5rem;
}
.oasis-toc-item {
    background: var(--oasis-card, #fff);
    border: 1px solid var(--oasis-border, #e0e0e0);
    border-radius: var(--oasis-radius, 8px);
    padding: 12px 16px;
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 0.85rem;
    box-shadow: var(--oasis-shadow, 0 2px 8px rgba(0,0,0,0.06));
}
.oasis-toc-num {
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--oasis-accent, #00897B);
    line-height: 1;
    min-width: 2rem;
}

/* Divider slides */
section.oasis-divider {
    background: var(--oasis-bg-dark, #1A2332) !important;
    color: #fff !important;
}
section.oasis-divider h1 { color: #fff; font-size: clamp(2rem, 4vw, 3rem); }
.oasis-sec-num {
    font-size: clamp(4rem, 10vw, 7rem);
    font-weight: 900;
    color: var(--oasis-accent, #00897B);
    opacity: 0.15;
    line-height: 1;
    margin-bottom: -1rem;
}
section.oasis-divider-light { background: var(--oasis-bg, #FAFAF8) !important; }
.oasis-divider-sub { font-size: 0.9rem; opacity: 0.7; margin-top: 0.5rem; }

/* Issue layout */
.oasis-issue-layout {
    display: grid;
    grid-template-columns: 55% 45%;
    gap: 24px;
    height: calc(100% - 2rem);
    overflow: hidden;
}
.oasis-issue-left { overflow: hidden; font-size: 0.8rem; }
.oasis-issue-right {
    display: flex;
    flex-direction: column;
    gap: 12px;
    overflow: hidden;
}
.oasis-issue-title {
    font-size: 1rem !important;
    font-weight: 700;
    color: var(--oasis-accent, #00897B);
    margin-bottom: 0.5rem;
}
.oasis-quote {
    background: var(--oasis-card, #fff);
    border-left: 3px solid var(--oasis-accent, #00897B);
    padding: 6px 10px;
    margin: 6px 0;
    font-size: 0.75rem;
    font-style: italic;
    border-radius: 0 4px 4px 0;
}

/* Strength layout */
.oasis-strength-layout {
    display: grid;
    grid-template-columns: 50% 50%;
    gap: 24px;
    height: calc(100% - 2rem);
    overflow: hidden;
}
.oasis-strength-left { overflow: hidden; font-size: 0.85rem; }
.oasis-strength-right { overflow: hidden; }
.oasis-strength-title {
    font-size: 1.1rem !important;
    font-weight: 700;
    color: var(--oasis-accent, #00897B);
    margin-bottom: 0.75rem;
}

/* Panel */
.oasis-panel {
    flex: 1;
    background: var(--oasis-card, #fff);
    border: 1px solid var(--oasis-border, #e0e0e0);
    border-radius: var(--oasis-radius, 8px);
    overflow: hidden;
    display: flex;
    flex-direction: column;
    min-height: 0;
}
.oasis-panel-label {
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--oasis-text-muted, #555);
    padding: 6px 10px 4px;
    border-bottom: 1px solid var(--oasis-border, #e0e0e0);
    background: var(--oasis-bg, #FAFAF8);
}
.oasis-panel-img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    flex: 1;
    min-height: 0;
    display: block;
}
.oasis-panel-iframe {
    width: 100%;
    flex: 1;
    min-height: 0;
    border: none;
}
.oasis-panel-placeholder {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.7rem;
    color: var(--oasis-text-muted, #999);
    background: var(--oasis-bg, #f5f5f5);
    padding: 12px;
    text-align: center;
}
.oasis-panel-caption {
    font-size: 0.65rem;
    color: var(--oasis-text-muted, #555);
    font-style: italic;
    padding: 4px 8px;
    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;
}

/* Back cover */
section.oasis-back {
    background: var(--oasis-bg-dark, #1A2332) !important;
    color: #fff !important;
    display: flex !important;
    flex-direction: column;
    align-items: flex-start;
    justify-content: center;
}
section.oasis-back h1, section.oasis-back p { color: #fff; }
.oasis-brand-text { font-size: 1.5rem; font-weight: 700; color: #fff; }

/* Print / PDF */
@media print {
    * { -webkit-print-color-adjust: exact !important; print-color-adjust: exact !important; }
}
"""


# ---------------------------------------------------------------------------
# PDF export via Playwright (screenshots each slide)
# ---------------------------------------------------------------------------

async def _html_dir_to_pdf_async(compiled_dir: Path, pdf_path: str) -> str:
    """
    Open the Quarkdown reveal.js output in Playwright, navigate through each
    slide, screenshot each one, and combine into a PDF.
    """
    try:
        from playwright.async_api import async_playwright
        from PIL import Image
        import io
    except ImportError:
        raise RuntimeError("Playwright and Pillow are required for PDF export.")

    index_html = compiled_dir / "index.html"
    if not index_html.exists():
        raise RuntimeError(f"Compiled index.html not found in {compiled_dir}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(args=["--no-sandbox", "--disable-setuid-sandbox"])
        page = await browser.new_page(viewport={"width": 1280, "height": 720})

        file_url = f"file://{index_html.resolve()}"
        await page.goto(file_url, wait_until="networkidle")
        await page.wait_for_timeout(1000)  # let reveal.js initialize

        # Inject print-color-adjust for accurate backgrounds
        await page.add_style_tag(content="""
            * { -webkit-print-color-adjust: exact !important; print-color-adjust: exact !important; }
        """)

        # Count slides via reveal.js API
        total_slides = await page.evaluate("""
            () => {
                if (window.Reveal) return Reveal.getTotalSlides();
                return document.querySelectorAll('.reveal .slides section:not(.stack)').length || 1;
            }
        """)
        total_slides = max(int(total_slides), 1)

        images: list[Image.Image] = []
        for i in range(total_slides):
            # Navigate to slide i
            await page.evaluate(f"() => {{ if (window.Reveal) Reveal.slide({i}); }}")
            await page.wait_for_timeout(300)

            screenshot_bytes = await page.screenshot(full_page=False, type="png")
            img = Image.open(io.BytesIO(screenshot_bytes))
            images.append(img)

        await browser.close()

    if not images:
        raise RuntimeError("No slide screenshots captured.")

    # Save as multi-page PDF
    first = images[0].convert("RGB")
    rest = [img.convert("RGB") for img in images[1:]]
    first.save(
        pdf_path,
        save_all=True,
        append_images=rest,
        resolution=150,
    )
    return pdf_path


def html_dir_to_pdf(compiled_dir: Path, pdf_path: str) -> str:
    """Synchronous wrapper around the async PDF export."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _html_dir_to_pdf_async(compiled_dir, pdf_path))
                return future.result(timeout=120)
        else:
            return loop.run_until_complete(_html_dir_to_pdf_async(compiled_dir, pdf_path))
    except Exception as exc:
        raise RuntimeError(f"PDF export failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Compatibility shim: keep the old render_html / html_to_pdf API working
# so existing callers (ui/steps/report.py) need minimal changes.
# ---------------------------------------------------------------------------

def render_html_quarkdown(
    data: "SlideReportData",
    kit: Optional[DesignKit] = None,
    out_dir: Optional[Path] = None,
) -> tuple[str, Path]:
    """
    Full pipeline: SlideReportData + DesignKit → (HTML string, compiled_dir Path)

    Returns the HTML string (for preview) and the compiled directory path
    (for PDF export).
    """
    if out_dir is None:
        out_dir = Path(tempfile.mkdtemp(prefix="oasis_qd_"))

    source = build_qd_source(data, kit)
    compiled_dir = compile_qd(source, out_dir)
    inject_design_kit(compiled_dir, kit or OASIS_DEFAULT_KIT)

    index_html = compiled_dir / "index.html"
    html_str = index_html.read_text(encoding="utf-8") if index_html.exists() else ""
    return html_str, compiled_dir


# ---------------------------------------------------------------------------
# Import compatibility: re-export SlideReportData etc. from slide_generator
# so callers can import from either module.
# ---------------------------------------------------------------------------

try:
    from ux_sim_app.report.slide_generator import (  # noqa: F401
        SlideReportData,
        IssueSlide,
        StrengthSlide,
        build_report_data,
        clamp_text as _clamp_text_orig,
    )
except ImportError:
    # Fallback stubs if slide_generator is not available
    from dataclasses import dataclass as _dc, field as _field

    @_dc
    class IssueSlide:  # type: ignore[no-redef]
        number: str = ""
        category: str = ""
        title: str = ""
        issue_text: str = ""
        root_cause: str = ""
        recommendation: str = ""
        screenshot_url: str = ""
        persona_quotes: list = _field(default_factory=list)
        persona_names: list = _field(default_factory=list)
        redesign_analysis: str = ""
        redesign_html: str = ""
        redesign_html_sanitised: str = ""
        redesign_screenshot_url: str = ""

    @_dc
    class StrengthSlide:  # type: ignore[no-redef]
        title: str = ""
        brand_name: str = ""
        description: str = ""
        screenshot_url: str = ""

    @_dc
    class SlideReportData:  # type: ignore[no-redef]
        website_url: str = ""
        website_title: str = ""
        brand_name: str = ""
        logo_url: str = ""
        overall_score: float = 0.0
        overall_summary: str = ""
        methodology: str = ""
        focus_areas: list = _field(default_factory=list)
        issues: list = _field(default_factory=list)
        strengths: list = _field(default_factory=list)
        generated_by: str = "OASIS UX Simulation App"
        generated_date: str = _field(default_factory=lambda: datetime.utcnow().strftime("%Y"))

    def build_report_data(*args, **kwargs):  # type: ignore[misc]
        return SlideReportData()
