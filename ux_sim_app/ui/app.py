"""
OASIS UX Simulation App – Gradio Interface
==========================================
A no-code interface for running multi-mode persona simulations + UX audits.

Fixes applied (v2):
- All generator functions now yield the EXACT same number of outputs as declared
  in their Gradio .click(outputs=[...]) wiring (Gradio 6 strict validation).
- step_scrape_and_generate: 4 outputs → every yield returns 4 values.
- step_run_simulations:     2 outputs → every yield returns 2 values.
- step_ux_scan:             2 outputs → every yield returns 2 values.
- step_generate_report:     3 outputs → every yield returns 3 values.
- Removed theme= from gr.Blocks() (moved to launch() for Gradio 6 compat).
- _run() now always creates a fresh event loop to avoid "loop already running"
  issues when Gradio's async server is active.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

import gradio as gr

# Ensure the app package is importable when run as a module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import ux_sim_app.core.config as cfg
from ux_sim_app.core.config import (
    OPENAI_API_KEY, BROWSERBASE_API_KEY, REPORTS_DIR, TEXT_MODEL, VISION_MODEL
)
from ux_sim_app.core.scraper import scrape
from ux_sim_app.core.personas import generate_personas, Persona
from ux_sim_app.modes.runner import run_mode1, run_mode2, run_mode3, SimulationResult
from ux_sim_app.ux.scanner import scan_website
from ux_sim_app.report.generator import generate_full_report, send_report_email
from ux_sim_app.report.slide_generator import build_report_data, render_html, html_to_pdf, IssueSlide
from ux_sim_app.report.redesign_client import generate_redesign, sanitise_for_embed
from ux_sim_app.integrations import notebooklm as nlm
from ux_sim_app.integrations.notebooklm import UX_CATEGORIES
import logging
logger = logging.getLogger(__name__)


# ── Async helper ───────────────────────────────────────────────────────────────

def _run(coro):
    """Run an async coroutine safely from a sync Gradio callback.

    Gradio 6 runs inside an async event loop (uvicorn/anyio). We must NOT call
    loop.run_until_complete() on the running loop. Instead we always spin up a
    fresh thread with its own loop.
    """
    import concurrent.futures
    def _worker():
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_worker).result()


# ── Step 1: Scrape & generate personas ────────────────────────────────────────
# Outputs (4): status_personas, state_personas_json, personas_display, state_image_urls_json

def step_scrape_and_generate(
    url: str,
    business_context: str,
    customer_profile: str,
    num_personas: int,
    api_key_override: str,
):
    """Scrape the website and generate personas.
    MUST yield exactly 4 values on every yield (matches outputs wiring).
    """
    _EMPTY = ("{}", "", "[]")  # personas_json, personas_display, image_urls_json

    url = url or ""  # guard against None
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
        yield "⏳ Scraping website...", *_EMPTY

        scrape_result = _run(scrape(url.strip(), follow_links=2))
        if scrape_result.error:
            yield f"⚠️ Scrape warning: {scrape_result.error}. Continuing with partial data.", *_EMPTY

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
        ))

        personas_json = json.dumps([p.to_dict() for p in personas], indent=2)

        # Build display markdown
        display = f"### 👥 Focus Group ({len(personas)} personas)\n\n"
        for p in personas:
            display += (
                f"**{p.name}** (@{p.username}) · {p.age}yo · {p.gender} · "
                f"{p.country} · {p.mbti}  \n"
                f"*{p.persona_type}*  \n"
                f"{p.bio}  \n"
                f"Goals: {', '.join(p.goals[:2])}  \n\n"
            )

        # Store image URLs for Mode 3
        image_urls_json = json.dumps(scrape_result.image_urls[:6])

        yield (
            f"✅ Scraped website and generated {len(personas)} personas.",
            personas_json,
            display,
            image_urls_json,
        )

    except Exception as exc:
        import traceback
        yield f"❌ Error: {exc}\n{traceback.format_exc()}", "{}", "", "[]"


# ── Step 2: Run simulations ────────────────────────────────────────────────────
# Outputs (2): status_sims, state_sim_results_json

def step_run_simulations(
    url: str,
    personas_json: str,
    image_urls_json: str,
    run_mode1_flag: bool,
    run_mode2_flag: bool,
    run_mode3_flag: bool,
    content_items_text: str,
):
    """Run selected simulation modes.
    MUST yield exactly 3 values on every yield:
      (status_text, sim_results_json, video_recordings_json)
    """
    _EMPTY_RESULTS = "{}"
    _EMPTY_VIDS = "[]"

    # Guard all string params against None
    url = url or ""
    personas_json = personas_json or ""
    image_urls_json = image_urls_json or ""
    content_items_text = content_items_text or ""

    if not personas_json or personas_json in ("{}", ""):
        yield "❌ Please generate personas first (Tab 1).", _EMPTY_RESULTS, _EMPTY_VIDS
        return

    try:
        persona_dicts = json.loads(personas_json)
        personas = [Persona(**p) for p in persona_dicts]
    except Exception as exc:
        yield f"❌ Failed to parse personas: {exc}", _EMPTY_RESULTS, _EMPTY_VIDS
        return

    try:
        image_urls = json.loads(image_urls_json) if image_urls_json else []
    except Exception:
        image_urls = []

    results: list[SimulationResult] = []
    all_video_recordings: list = []

    if run_mode2_flag:
        yield "⏳ Running Mode 2 – Browser Usability Simulation...", _EMPTY_RESULTS, _EMPTY_VIDS
        try:
            r = _run(run_mode2(personas, url.strip()))
            results.append(r)
            # Collect video recordings from this Mode 2 run
            all_video_recordings.extend(r.video_recordings or [])
            conv = int(r.aggregate.get("conversion_intent_rate", 0) * 100)
            n_vids = len(r.video_recordings or [])
            yield (
                f"✅ Mode 2 complete. Conversion intent: {conv}%. "
                f"{n_vids} session recording(s) saved.",
                _EMPTY_RESULTS,
                _EMPTY_VIDS,
            )
        except Exception as exc:
            yield f"⚠️ Mode 2 error: {exc}", _EMPTY_RESULTS, _EMPTY_VIDS

    if run_mode3_flag:
        if not image_urls:
            yield "⚠️ Mode 3: No images found on the website. Skipping.", _EMPTY_RESULTS, _EMPTY_VIDS
        else:
            yield f"⏳ Running Mode 3 – Visual Simulation ({len(image_urls)} images)...", _EMPTY_RESULTS, _EMPTY_VIDS
            try:
                r = _run(run_mode3(personas, image_urls))
                results.append(r)
                avg = r.aggregate.get("average_resonance", 0)
                yield f"✅ Mode 3 complete. Average resonance: {avg}/10", _EMPTY_RESULTS, _EMPTY_VIDS
            except Exception as exc:
                yield f"⚠️ Mode 3 error: {exc}", _EMPTY_RESULTS, _EMPTY_VIDS

    if run_mode1_flag:
        content_items_text = content_items_text or ""  # guard against None
        items = [c.strip() for c in content_items_text.strip().split("\n---\n") if c.strip()]
        if not items:
            items = [content_items_text.strip()] if content_items_text.strip() else []
        if not items:
            yield "⚠️ Mode 1: No content items provided. Skipping.", _EMPTY_RESULTS, _EMPTY_VIDS
        else:
            yield f"⏳ Running Mode 1 – Content Simulation ({len(items)} items)...", _EMPTY_RESULTS, _EMPTY_VIDS
            try:
                rs = _run(run_mode1(personas, items))
                results.extend(rs)
                avg_eng = sum(r.aggregate.get("engagement_rate", 0) for r in rs) / max(len(rs), 1)
                yield f"✅ Mode 1 complete. Avg engagement: {int(avg_eng * 100)}%", _EMPTY_RESULTS, _EMPTY_VIDS
            except Exception as exc:
                yield f"⚠️ Mode 1 error: {exc}", _EMPTY_RESULTS, _EMPTY_VIDS

    if not results:
        yield "⚠️ No simulation modes were run. Please select at least one mode.", _EMPTY_RESULTS, _EMPTY_VIDS
        return

    results_json = json.dumps([r.to_dict() for r in results], indent=2)
    vids_json = json.dumps(all_video_recordings, indent=2)
    yield (
        f"✅ All simulations complete. {len(results)} result set(s) ready.",
        results_json,
        vids_json,
    )


# ── NotebookLM helpers ────────────────────────────────────────────────────────

def nlm_authenticate():
    """Trigger NotebookLM OAuth in a background thread.

    Returns (status_message, notebook_label).
    The notebook_label shows the connected notebook title or auth status.
    """
    try:
        ok, msg = _run(nlm.authenticate_notebooklm())
        state = nlm.get_state()
        # Show notebook title if already connected, otherwise auth status
        nb_label = state.get("notebook_title") or ("✅ Authenticated" if ok else "⏳ Awaiting login")
        return msg, nb_label
    except Exception as e:
        return f"❌ Auth error: {e}", "—"


def nlm_connect(notebook_id_or_title: str):
    """Connect to or create a NotebookLM notebook."""
    try:
        ok, msg = _run(nlm.connect_notebook(notebook_id_or_title or "UX Best Practices"))
        state = nlm.get_state()
        nb_label = state.get("notebook_title") or "—"
        return msg, nb_label
    except Exception as e:
        return f"❌ Connect error: {e}", "—"


def nlm_disconnect():
    """Disconnect from NotebookLM."""
    try:
        _run(nlm.disconnect())
        return "Disconnected from NotebookLM.", "—"
    except Exception as e:
        return f"Error: {e}", "—"


def nlm_run_config_loop(business_context: str, personas_json: str):
    """Run the UX configuration loop: query NotebookLM (or built-in KB) for best practices
    across all 15 UX categories and cache the results.
    Yields progress updates.
    """
    persona_summary = ""
    try:
        pds = json.loads(personas_json or "[]")
        if isinstance(pds, list) and pds:
            names = [p.get("name", "") for p in pds[:3]]
            ages = [str(p.get("age", "")) for p in pds[:3]]
            persona_summary = f"Personas include: {', '.join(names)} (ages {', '.join(ages)})"
    except Exception:
        pass

    state = nlm.get_state()
    source = "NotebookLM" if (state["authenticated"] and state["notebook_id"]) else "built-in knowledge base"
    total = len(UX_CATEGORIES)

    yield f"⏳ Starting UX configuration loop ({total} categories) using {source}...", "{}"

    results: dict = {}

    def _on_progress(cat_name, idx, tot):
        pass  # progress tracked by loop below

    for i, cat in enumerate(UX_CATEGORIES):
        yield f"⏳ [{i+1}/{total}] Querying: {cat['name']}...", json.dumps(results)
        try:
            bp = _run(nlm.query_category_best_practices(
                cat,
                business_context=business_context or "",
                persona_summary=persona_summary,
            ))
            results[cat["id"]] = bp
        except Exception as exc:
            results[cat["id"]] = f"Error: {exc}"

    # Format a readable summary
    summary_lines = []
    for cat in UX_CATEGORIES:
        bp = results.get(cat["id"], "")
        summary_lines.append(f"### {cat['name']}\n{bp[:300]}..." if len(bp) > 300 else f"### {cat['name']}\n{bp}")
    summary_md = "\n\n".join(summary_lines)

    yield f"✅ Configuration loop complete. {total} categories loaded from {source}.", json.dumps(results)


# ── Step 3: UX Scan ────────────────────────────────────────────────────────────
# Outputs (2): status_ux, state_ux_json

def step_ux_scan(url: str):
    """Run the full UX scan.
    MUST yield exactly 2 values on every yield.
    """
    _EMPTY_UX = "{}"

    if not url or not url.strip():
        yield "❌ Please enter a URL first (Tab 1).", _EMPTY_UX
        return

    yield "⏳ Running UX scan (screenshots + heuristics + AI critique)...", _EMPTY_UX

    try:
        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        ux_report = _run(scan_website(url.strip(), run_id))

        ux_json = json.dumps({
            "url": ux_report.url,
            "run_id": run_id,
            "overall_score": ux_report.overall_score,
            "overall_summary": ux_report.overall_summary,
            "strengths": ux_report.strengths,
            "weaknesses": ux_report.weaknesses,
            "screenshots": ux_report.screenshots,
            "heuristic_checks": ux_report.heuristic_checks,
            "dimensions": [
                {
                    "name": d.name,
                    "score": d.score,
                    "feedback": d.feedback,
                    "issues": [
                        {
                            "category": i.category,
                            "severity": i.severity,
                            "description": i.description,
                            "recommendation": i.recommendation,
                        }
                        for i in d.issues
                    ],
                }
                for d in ux_report.dimensions
            ],
            "recommendations": ux_report.recommendations,
            "error": ux_report.error,
        }, indent=2)

        score = ux_report.overall_score
        heur = ux_report.heuristic_checks.get("heuristic_score", 0)
        dims = len(ux_report.dimensions)
        issues = sum(len(d.issues) for d in ux_report.dimensions)
        yield (
            f"✅ UX scan complete. Score: {score}/100 · Heuristic: {heur}/100 · "
            f"{dims} dimensions · {issues} issues found.",
            ux_json,
        )

    except Exception as exc:
        import traceback
        yield f"❌ UX scan error: {exc}\n{traceback.format_exc()}", _EMPTY_UX


# ── Step 4: Generate report ────────────────────────────────────────────────────
# Outputs (3): status_report, report_file, state_report_html

def step_generate_report(
    url: str,
    personas_json: str,
    sim_results_json: str,
    ux_json: str,
    generate_redesigns: bool,
):
    """Generate the slide-style HTML + PDF report with optional AI redesigns.
    MUST yield exactly 3 values on every yield.
    """
    _EMPTY_REPORT = (None, "")  # report_file, state_report_html

    if not personas_json or personas_json in ("{}", ""):
        yield "❌ Please complete Step 1 (generate personas) first.", *_EMPTY_REPORT
        return
    if not sim_results_json or sim_results_json in ("{}", ""):
        yield "❌ Please run at least one simulation mode (Tab 3) first.", *_EMPTY_REPORT
        return
    if not ux_json or ux_json in ("{}", ""):
        yield "❌ Please run the UX scan (Tab 4) first.", *_EMPTY_REPORT
        return

    yield "⏳ Building slide-style report...", *_EMPTY_REPORT

    try:
        import ux_sim_app.core.config as cfg

        persona_dicts = json.loads(personas_json)
        sim_dicts = json.loads(sim_results_json)
        ux_data = json.loads(ux_json)
        scrape_data = ux_data  # ux_data already contains title, screenshots, etc.

        # Build the structured slide data
        report_data = build_report_data(
            url=url or "",
            scrape_data=scrape_data,
            ux_data=ux_data,
            sim_results=sim_dicts,
            personas=persona_dicts,
        )

        # Optionally enrich each issue slide with an AI HTML redesign (OpenAI GPT-4o Vision)
        effective_openai = (os.environ.get("OPENAI_API_KEY") or "").strip()
        if generate_redesigns:
            total = len(report_data.issues)
            for idx, issue in enumerate(report_data.issues):
                yield (
                    f"⏳ Generating redesign {idx + 1}/{total}: {issue.title[:40]}...",
                    *_EMPTY_REPORT,
                )
                try:
                    rd = generate_redesign(
                        screenshot_url=issue.screenshot_url,
                        ux_issues=[issue.issue_text, issue.recommendation],
                        openai_key=effective_openai,
                        vision_model=cfg.VISION_MODEL,
                    )
                    if not rd.get("error"):
                        issue.redesign_analysis = rd.get("analysis", "")
                        issue.redesign_html = rd.get("html_code", "")
                        issue.redesign_html_sanitised = (
                            rd.get("html_sanitised")
                            or sanitise_for_embed(issue.redesign_html)
                        )
                        # Store the rendered screenshot of the redesign for the slide panel
                        if rd.get("redesign_screenshot"):
                            issue.redesign_screenshot_url = rd["redesign_screenshot"]
                    else:
                        logger.warning("Redesign failed for %s: %s", issue.title, rd.get("error"))
                except Exception as exc:
                    logger.warning("Redesign exception for %s: %s", issue.title, exc)

        yield "⏳ Rendering slides and exporting PDF...", *_EMPTY_REPORT

        # Render HTML
        html = render_html(report_data)

        # Save HTML report
        run_id = ux_data.get("run_id", uuid.uuid4().hex[:8])
        html_path = REPORTS_DIR / f"report_{run_id}.html"
        html_path.write_text(html, encoding="utf-8")

        # Export PDF via Playwright
        pdf_path = REPORTS_DIR / f"report_{run_id}.pdf"
        try:
            html_to_pdf(html, str(pdf_path))
            download_path = str(pdf_path)
            status_msg = f"✅ Slide report generated ({len(report_data.issues)} issues, {len(report_data.strengths)} strengths). PDF ready."
        except Exception as pdf_err:
            # PDF failed — fall back to HTML download
            download_path = str(html_path)
            status_msg = f"✅ Report generated (PDF export failed: {pdf_err}). Downloading HTML instead."

        yield status_msg, download_path, html

    except Exception as exc:
        import traceback
        yield f"❌ Report generation error: {exc}\n{traceback.format_exc()}", None, ""


# ── Step 5: Deliver report by email ───────────────────────────────────────────

def deliver_email(report_html: str, to_email: str, url: str) -> str:
    if not report_html:
        return "❌ No report to send. Generate the report first (Tab 5)."
    if not to_email or not to_email.strip():
        return "❌ Please enter a recipient email address."
    run_id = uuid.uuid4().hex[:8]
    ok, msg = send_report_email(report_html, to_email.strip(), url or "", run_id)
    return f"✅ {msg}" if ok else f"❌ {msg}"


# ── Session Recordings helpers ───────────────────────────────────────────────────

def _update_recordings_tab(vids_json: str):
    """Called after simulations finish. Updates the Recordings tab controls.
    Returns: (rec_status, rec_slider, rec_persona_label, rec_video, rec_download)
    """
    try:
        recs = json.loads(vids_json or "[]")
    except Exception:
        recs = []

    if not recs:
        return (
            "No recordings found. Make sure Mode 2 is selected and Playwright is installed.",
            gr.Slider(minimum=0, maximum=0, value=0),
            "—",
            None,
            None,
        )

    first = recs[0]
    vp = first.get("video_path") or ""
    return (
        f"✅ {len(recs)} recording(s) available. Use the slider to switch between personas.",
        gr.Slider(minimum=0, maximum=len(recs) - 1, step=1, value=0),
        first.get("persona_name", "—"),
        vp if vp and os.path.exists(vp) else None,
        vp if vp and os.path.exists(vp) else None,
    )


def _select_recording(vids_json: str, idx: int):
    """Called when the slider changes. Returns (persona_label, video, download)."""
    try:
        recs = json.loads(vids_json or "[]")
    except Exception:
        recs = []

    if not recs:
        return "—", None, None

    idx = max(0, min(int(idx), len(recs) - 1))
    rec = recs[idx]
    vp = rec.get("video_path") or ""
    return (
        rec.get("persona_name", "—"),
        vp if vp and os.path.exists(vp) else None,
        vp if vp and os.path.exists(vp) else None,
    )


# ── Gradio UI ──────────────────────────────────────────────────────────────────

THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
)

with gr.Blocks(title="OASIS UX Simulation App") as demo:

    # ── Shared state ───────────────────────────────────────────────────────────
    state_personas_json    = gr.State("{}")
    state_image_urls_json  = gr.State("[]")
    state_sim_results_json = gr.State("{}")
    state_ux_json          = gr.State("{}")
    state_report_html      = gr.State("")
    state_url              = gr.State("")
    # List of {persona_name, video_path} dicts from Mode 2 recordings
    state_video_recordings = gr.State("[]")  # JSON string

    # ── Header ─────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div style="background:linear-gradient(135deg,#2c3e50 0%,#3498db 100%);
         padding:28px 32px;border-radius:12px;margin-bottom:8px;color:#fff">
      <h1 style="margin:0;font-size:1.9em">🧠 OASIS UX Simulation App</h1>
      <p style="margin:6px 0 0 0;color:#aed6f1;font-size:1em">
        Automated persona generation &middot; Browser usability &middot;
        Visual branding &middot; Content simulation &middot; UX audit
      </p>
    </div>
    """)

    with gr.Tabs():

        # ══════════════════════════════════════════════════════════════════════
        # TAB 1 – Setup & Personas
        # ══════════════════════════════════════════════════════════════════════
        with gr.Tab("1 · Setup & Personas"):
            gr.Markdown("### Website & Business Context")
            gr.Markdown(
                "Enter the website you want to test. The app will automatically scrape it "
                "and generate a realistic customer focus group. You can add your own context "
                "to guide persona creation."
            )
            with gr.Row():
                with gr.Column(scale=2):
                    url_input = gr.Textbox(
                        label="Website URL",
                        placeholder="https://example.com",
                        info="The website to analyse and test",
                    )
                    business_context = gr.Textbox(
                        label="Business Context (optional)",
                        placeholder=(
                            "e.g. We are a family-owned bistro in Tootgarook, VIC. "
                            "We specialise in classic Australian pub food with a modern twist. "
                            "We run weekly Parma nights and cookery masterclasses."
                        ),
                        lines=4,
                        info="Tell us about your business — this enriches persona generation",
                    )
                    customer_profile = gr.Textbox(
                        label="Ideal Customer Profile (optional)",
                        placeholder=(
                            "e.g. Local families, couples aged 28-55, tourists visiting "
                            "the Mornington Peninsula, food lovers who appreciate value for money."
                        ),
                        lines=3,
                        info="Describe who your ideal customers are",
                    )
                with gr.Column(scale=1):
                    num_personas = gr.Slider(
                        minimum=2, maximum=10, value=5, step=1,
                        label="Number of Personas",
                        info="How many personas to generate",
                    )
                    gr.Markdown("---")
                    gr.Markdown("**Simulation Modes to Run**")
                    mode1_flag = gr.Checkbox(label="Mode 1 – Content Simulation", value=True)
                    mode2_flag = gr.Checkbox(label="Mode 2 – Browser Usability", value=True)
                    mode3_flag = gr.Checkbox(label="Mode 3 – Visual Branding", value=True)

            btn_generate = gr.Button("🚀 Scrape & Generate Personas", variant="primary", size="lg")
            status_personas = gr.Textbox(label="Status", interactive=False, lines=2)
            personas_display = gr.Markdown(label="Generated Personas")

        # ══════════════════════════════════════════════════════════════════════
        # TAB 2 – Content (Mode 1)
        # ══════════════════════════════════════════════════════════════════════
        with gr.Tab("2 · Content to Test (Mode 1)"):
            gr.Markdown("### Social Media / Marketing Content")
            gr.Markdown(
                "Enter the marketing copy, social media posts, or website text you want "
                "to test with your personas. Separate multiple items with `---` on its own line."
            )
            content_items = gr.Textbox(
                label="Content Items",
                placeholder=(
                    "Seven amazing Parmas, all for just $25. Bookings highly recommended!\n"
                    "---\n"
                    "A place that feels like home. Great food, great service, without the fuss.\n"
                    "---\n"
                    "Join us this Thursday for Parma Night! Full house every week — book early."
                ),
                lines=12,
                info="Separate multiple content items with --- on its own line",
            )
            gr.Markdown("_If you leave this empty, Mode 1 will be skipped even if selected._")

        # ══════════════════════════════════════════════════════════════════════
        # TAB 3 – Run Simulations
        # ══════════════════════════════════════════════════════════════════════
        with gr.Tab("3 · Run Simulations"):
            gr.Markdown("### Run Selected Simulation Modes")
            gr.Markdown(
                "This step runs the browser usability simulation (Mode 2), "
                "visual branding analysis (Mode 3), and content reaction simulation (Mode 1) "
                "using the personas generated in Step 1."
            )
            btn_run_sims = gr.Button("▶️ Run Simulations", variant="primary", size="lg")
            status_sims = gr.Textbox(label="Status", interactive=False, lines=3)

        # ══════════════════════════════════════════════════════════════════════
        # TAB 4 – UX Scan
        # ══════════════════════════════════════════════════════════════════════
        with gr.Tab("4 · UX Scan"):
            gr.Markdown("### Full UX / Usability Scan")
            gr.Markdown(
                "This step takes multi-viewport screenshots of the website, runs heuristic "
                "checks (alt text, headings, CTAs, navigation, accessibility), and uses "
                "persona-generated critique to improve the design across 15 UX dimensions."
            )

            # ── NotebookLM Knowledge Enrichment Panel ──────────────────────────
            with gr.Accordion("📓 NotebookLM Knowledge Enrichment (optional)", open=False):
                gr.Markdown(
                    "Connect to a Google NotebookLM notebook to enrich the UX scan with "
                    "your own curated knowledge base. The agent will query the notebook for "
                    "best practices across **15 UX heuristic categories** and personalise "
                    "the scan based on your business context and personas.\n\n"
                    "**Without a notebook**, the app uses a built-in UX best-practices knowledge base."
                )

                with gr.Row():
                    nlm_status_box = gr.Textbox(
                        label="NotebookLM Status",
                        interactive=False,
                        value="Not connected — using built-in knowledge base",
                        scale=3,
                    )
                    nlm_notebook_label = gr.Textbox(
                        label="Connected Notebook",
                        interactive=False,
                        value="—",
                        scale=1,
                    )

                with gr.Row():
                    btn_nlm_auth = gr.Button(
                        "🔐 Authenticate with Google",
                        variant="secondary",
                        scale=1,
                    )
                    nlm_notebook_input = gr.Textbox(
                        label="Notebook ID or Title",
                        placeholder="UX Best Practices",
                        info="Leave blank to create a new notebook",
                        scale=2,
                    )
                    btn_nlm_connect = gr.Button(
                        "🔗 Connect / Create Notebook",
                        variant="secondary",
                        scale=1,
                    )
                    btn_nlm_disconnect = gr.Button(
                        "⏏️ Disconnect",
                        variant="stop",
                        scale=1,
                    )

                gr.Markdown("#### 🤖 Agent Configuration Loop")
                gr.Markdown(
                    "Run the agent loop to query the notebook (or built-in KB) for best practices "
                    "across all 15 UX categories. Results are cached and used to personalise "
                    "the UX scan and report."
                )
                btn_nlm_config_loop = gr.Button(
                    "▶️ Run UX Configuration Loop",
                    variant="primary",
                )
                nlm_loop_status = gr.Textbox(
                    label="Configuration Loop Status",
                    interactive=False,
                    lines=2,
                )
                nlm_bp_json = gr.State("{}")  # cache only — not displayed to user

            btn_ux_scan = gr.Button("🔍 Run UX Scan", variant="primary", size="lg")
            status_ux = gr.Textbox(label="Status", interactive=False, lines=2)

        # ══════════════════════════════════════════════════════════════════════
        # TAB 5 – Report
        # ══════════════════════════════════════════════════════════════════════
        with gr.Tab("5 · Report"):
            gr.Markdown("### Generate & Deliver Report")
            gr.Markdown(
                "Generates a **slide-style HTML + PDF report** (16:9, matching the reference design) "
                "combining the UX audit and persona simulation results. "
                "Each issue slide shows the current design screenshot alongside an **AI-generated HTML redesign** "
                "powered by GPT-4o Vision. "
                "Persona feedback is highlighted in teal callout boxes."
            )
            with gr.Row():
                generate_redesigns_flag = gr.Checkbox(
                    label="🎨 Generate AI HTML Redesigns (GPT-4o Vision)",
                    value=False,
                    info="For each issue, captures a screenshot and uses GPT-4o Vision to generate a side-by-side improved HTML redesign.",
                )
            btn_gen_report = gr.Button("📄 Generate Slide Report + PDF", variant="primary", size="lg")
            status_report = gr.Textbox(label="Status", interactive=False, lines=2)

            with gr.Row():
                report_file = gr.File(label="⬇️ Download Report", interactive=False)

            with gr.Accordion("📧 Send Report by Email", open=False):
                with gr.Row():
                    email_to = gr.Textbox(
                        label="Recipient Email",
                        placeholder="client@example.com",
                        scale=3,
                    )
                    btn_send_email = gr.Button("Send", variant="secondary", scale=1)
                email_status = gr.Textbox(label="Email Status", interactive=False)

            with gr.Accordion("👁️ Preview & Edit Report", open=False):
                report_preview = gr.HTML(label="Report Preview")
                report_editor = gr.Code(
                    label="HTML Source (editable)",
                    language="html",
                    lines=30,
                    interactive=True,
                )
                btn_refresh_preview = gr.Button("🔄 Refresh Preview from Editor")

        # ══════════════════════════════════════════════════════════════════════
        # TAB 6 – Session Recordings
        # ══════════════════════════════════════════════════════════════════════
        with gr.Tab("6 · Session Recordings"):
            gr.Markdown("### 🎬 Persona Browser Session Recordings")
            gr.Markdown(
                "Each Mode 2 browser session is recorded automatically. "
                "Use the slider below to switch between personas. "
                "Videos are saved as `.webm` files and can be downloaded directly."
            )
            with gr.Row():
                rec_status = gr.Textbox(
                    label="Recording Status",
                    interactive=False,
                    value="Run Mode 2 simulations first to generate recordings.",
                )
            with gr.Row():
                rec_slider = gr.Slider(
                    minimum=0, maximum=0, step=1, value=0,
                    label="Persona (use slider to switch)",
                    interactive=True,
                )
            with gr.Row():
                rec_persona_label = gr.Textbox(
                    label="Current Persona",
                    interactive=False,
                    value="—",
                )
            with gr.Row():
                rec_video = gr.Video(
                    label="Session Recording",
                    interactive=False,
                )
            with gr.Row():
                rec_download = gr.File(
                    label="⬇️ Download Recording",
                    interactive=False,
                )

        # ══════════════════════════════════════════════════════════════════════
        # TAB 7 – Settings
        # ══════════════════════════════════════════════════════════════════════
        with gr.Tab("⚙️ Settings"):
            gr.Markdown("### API Keys, Provider & Model Configuration")
            gr.Markdown(
                "These settings override the `.env` file values for the current session. "
                "Changes take effect immediately without restarting."
            )

            # ── Provider selection (separate for text and vision) ──────────────────
            gr.Markdown("#### Provider Selection")
            gr.Markdown("Set providers independently for **text** tasks (persona generation, content analysis) and **vision** tasks (Mode 3, AI redesigns).")
            with gr.Row():
                text_provider_radio = gr.Radio(
                    choices=["openai", "openrouter", "custom"],
                    value=cfg.TEXT_PROVIDER,
                    label="Text Provider",
                    info="Provider for all non-vision LLM calls.",
                )
                vision_provider_radio = gr.Radio(
                    choices=["openai", "openrouter", "custom"],
                    value=cfg.VISION_PROVIDER,
                    label="Vision Provider",
                    info="Provider for vision/image LLM calls (Mode 3, redesigns).",
                )
            # legacy single provider_radio alias for wiring compatibility
            provider_radio = text_provider_radio

            # ── API Keys ────────────────────────────────────────────────────────────────
            with gr.Row():
                with gr.Column():
                    api_key_input = gr.Textbox(
                        label="OpenAI API Key",
                        placeholder="sk-...",
                        type="password",
                        value=OPENAI_API_KEY,
                        info="Used when provider = openai",
                    )
                    openrouter_key_input = gr.Textbox(
                        label="OpenRouter API Key",
                        placeholder="sk-or-v1-...",
                        type="password",
                        value=cfg.OPENROUTER_API_KEY,
                        info="Used when provider = openrouter. Get one at openrouter.ai",
                    )
                with gr.Column():
                    custom_api_key_input = gr.Textbox(
                        label="Custom Provider API Key",
                        placeholder="your-api-key",
                        type="password",
                        value=cfg.CUSTOM_API_KEY,
                        info="Used when provider = custom",
                    )
                    custom_base_url_input = gr.Textbox(
                        label="Custom Base URL",
                        placeholder="https://your-provider.com/v1",
                        value=cfg.CUSTOM_BASE_URL,
                        info="OpenAI-compatible base URL for custom provider",
                    )

            # ── Model selection ─────────────────────────────────────────────────
            gr.Markdown("#### Model Selection")
            gr.Markdown(
                "**Text model** is used for persona generation, content simulation, and UX analysis. "
                "**Vision model** is used for Mode 3 visual simulation and redesign screenshot analysis."
            )
            with gr.Row():
                with gr.Column():
                    text_model_input = gr.Dropdown(
                        label="Text Model",
                        choices=cfg.OPENAI_TEXT_MODELS + cfg.OPENROUTER_TEXT_MODELS,
                        value=cfg.TEXT_MODEL,
                        allow_custom_value=True,
                        info="Used for all non-vision LLM calls. Type any model name for custom providers.",
                    )
                with gr.Column():
                    vision_model_input = gr.Dropdown(
                        label="Vision Model",
                        choices=cfg.OPENAI_VISION_MODELS + cfg.OPENROUTER_VISION_MODELS,
                        value=cfg.VISION_MODEL,
                        allow_custom_value=True,
                        info="Must support vision/image input. Used for Mode 3 and AI redesigns.",
                    )

            # ── OpenRouter model reference ──────────────────────────────────────
            with gr.Accordion("📋 OpenRouter Model Reference", open=False):
                gr.Markdown(
                    "**Recommended text models for this app:**\n\n"
                    + "\n".join(f"- `{m}`" for m in cfg.OPENROUTER_TEXT_MODELS)
                    + "\n\n**Recommended vision models:**\n\n"
                    + "\n".join(f"- `{m}`" for m in cfg.OPENROUTER_VISION_MODELS)
                    + "\n\nFull model list: [openrouter.ai/models](https://openrouter.ai/models)"
                )

            # ── Browserbase ─────────────────────────────────────────────────────
            with gr.Row():
                bb_key_input = gr.Textbox(
                    label="Browserbase API Key (optional)",
                    placeholder="bb_live_...",
                    type="password",
                    value=BROWSERBASE_API_KEY,
                    info="Only needed as fallback for Mode 2 if Playwright fails entirely",
                )

            # ── Email / SMTP ─────────────────────────────────────────────────────
            gr.Markdown("#### Email Report Delivery (SMTP)")
            with gr.Row():
                with gr.Column():
                    smtp_host_input = gr.Textbox(
                        label="SMTP Host",
                        placeholder="smtp.gmail.com",
                        info="For email delivery of reports",
                    )
                    smtp_port_input = gr.Number(label="SMTP Port", value=587)
                with gr.Column():
                    smtp_user_input = gr.Textbox(
                        label="SMTP Username / Email", placeholder="you@gmail.com"
                    )
                    smtp_pass_input = gr.Textbox(
                        label="SMTP Password / App Password", type="password"
                    )

            btn_save_settings = gr.Button("💾 Save Settings", variant="primary")
            settings_status = gr.Textbox(label="Status", interactive=False)

            def save_settings(
                text_provider, vision_provider, api_key, or_key, custom_key, custom_url,
                text_model, vision_model,
                bb_key, smtp_host, smtp_port, smtp_user, smtp_pass
            ):
                import ux_sim_app.core.config as cfg
                from ux_sim_app.core.llm import set_runtime_config

                saved = []

                # Providers
                if text_provider:
                    cfg.TEXT_PROVIDER = text_provider.lower()
                    saved.append(f"text_provider={text_provider}")
                if vision_provider:
                    cfg.VISION_PROVIDER = vision_provider.lower()
                    saved.append(f"vision_provider={vision_provider}")

                # API keys
                if api_key and api_key.strip():
                    cfg.OPENAI_API_KEY = api_key.strip()
                    os.environ["OPENAI_API_KEY"] = api_key.strip()
                if or_key and or_key.strip():
                    cfg.OPENROUTER_API_KEY = or_key.strip()
                if custom_key and custom_key.strip():
                    cfg.CUSTOM_API_KEY = custom_key.strip()
                if custom_url and custom_url.strip():
                    cfg.CUSTOM_BASE_URL = custom_url.strip()

                # Resolve effective key/url after provider change
                cfg.EFFECTIVE_API_KEY = cfg._resolve_api_key()
                cfg.EFFECTIVE_BASE_URL = cfg._resolve_base_url()

                # Models
                if text_model and text_model.strip():
                    cfg.TEXT_MODEL = text_model.strip()
                    saved.append(f"text_model={text_model.strip()}")
                if vision_model and vision_model.strip():
                    cfg.VISION_MODEL = vision_model.strip()
                    saved.append(f"vision_model={vision_model.strip()}")

                # Push to llm runtime
                set_runtime_config(
                    provider=cfg.TEXT_PROVIDER,
                    vision_provider=cfg.VISION_PROVIDER,
                    api_key=cfg.EFFECTIVE_API_KEY,
                    base_url=cfg.EFFECTIVE_BASE_URL,
                    text_model=cfg.TEXT_MODEL,
                    vision_model=cfg.VISION_MODEL,
                )

                # Browserbase
                if bb_key and bb_key.strip():
                    cfg.BROWSERBASE_API_KEY = bb_key.strip()

                # SMTP
                if smtp_host and smtp_host.strip():
                    cfg.SMTP_HOST = smtp_host.strip()
                if smtp_port:
                    cfg.SMTP_PORT = int(smtp_port)
                if smtp_user and smtp_user.strip():
                    cfg.SMTP_USER = smtp_user.strip()
                    cfg.SMTP_FROM = smtp_user.strip()
                if smtp_pass and smtp_pass.strip():
                    cfg.SMTP_PASS = smtp_pass.strip()

                detail = ", ".join(saved) if saved else "no changes"
                return f"✅ Settings saved for this session ({detail})."

            btn_save_settings.click(
                save_settings,
                inputs=[
                    text_provider_radio, vision_provider_radio,
                    api_key_input, openrouter_key_input,
                    custom_api_key_input, custom_base_url_input,
                    text_model_input, vision_model_input,
                    bb_key_input, smtp_host_input, smtp_port_input,
                    smtp_user_input, smtp_pass_input,
                ],
                outputs=[settings_status],
            )

    # ── Wiring ─────────────────────────────────────────────────────────────────

    # Step 1: Scrape + personas → 4 outputs
    btn_generate.click(
        fn=step_scrape_and_generate,
        inputs=[url_input, business_context, customer_profile, num_personas, api_key_input],
        outputs=[status_personas, state_personas_json, personas_display, state_image_urls_json],
    ).then(
        fn=lambda u: u,
        inputs=[url_input],
        outputs=[state_url],
    )

    # Step 2: Run simulations → 3 outputs (status, sim_results_json, video_recordings_json)
    btn_run_sims.click(
        fn=step_run_simulations,
        inputs=[
            state_url, state_personas_json, state_image_urls_json,
            mode1_flag, mode2_flag, mode3_flag, content_items,
        ],
        outputs=[status_sims, state_sim_results_json, state_video_recordings],
    ).then(
        fn=_update_recordings_tab,
        inputs=[state_video_recordings],
        outputs=[rec_status, rec_slider, rec_persona_label, rec_video, rec_download],
    )

    # Recordings tab – slider change → switch video
    rec_slider.change(
        fn=_select_recording,
        inputs=[state_video_recordings, rec_slider],
        outputs=[rec_persona_label, rec_video, rec_download],
    )

    # NotebookLM auth
    btn_nlm_auth.click(
        fn=nlm_authenticate,
        inputs=[],
        outputs=[nlm_status_box, nlm_notebook_label],
    )

    # NotebookLM connect
    btn_nlm_connect.click(
        fn=nlm_connect,
        inputs=[nlm_notebook_input],
        outputs=[nlm_status_box, nlm_notebook_label],
    )

    # NotebookLM disconnect
    btn_nlm_disconnect.click(
        fn=nlm_disconnect,
        inputs=[],
        outputs=[nlm_status_box, nlm_notebook_label],
    )

    # NotebookLM config loop → yields (status, bp_json)
    btn_nlm_config_loop.click(
        fn=nlm_run_config_loop,
        inputs=[business_context, state_personas_json],
        outputs=[nlm_loop_status, nlm_bp_json],
        # Best practices are cached internally and silently injected into the
        # UX scanner prompt — they are never displayed to the user.
    )

    # Step 3: UX scan → 2 outputs
    btn_ux_scan.click(
        fn=step_ux_scan,
        inputs=[state_url],
        outputs=[status_ux, state_ux_json],
    )

    # Step 4: Generate report → 3 outputs
    btn_gen_report.click(
        fn=step_generate_report,
        inputs=[
            state_url, state_personas_json, state_sim_results_json, state_ux_json,
            generate_redesigns_flag,
        ],
        outputs=[status_report, report_file, state_report_html],
    ).then(
        fn=lambda h: (h, h),
        inputs=[state_report_html],
        outputs=[report_preview, report_editor],
    )

    # Email delivery
    btn_send_email.click(
        fn=deliver_email,
        inputs=[state_report_html, email_to, state_url],
        outputs=[email_status],
    )

    # Refresh preview from editor
    btn_refresh_preview.click(
        fn=lambda h: h,
        inputs=[report_editor],
        outputs=[report_preview],
    )


# ── Entry point ────────────────────────────────────────────────────────────────

def launch(port: int = 7860, share: bool = False):
    demo.queue(max_size=5).launch(
        server_name="0.0.0.0",
        server_port=port,
        share=share,
        show_error=True,
        theme=THEME,
    )


if __name__ == "__main__":
    launch()
