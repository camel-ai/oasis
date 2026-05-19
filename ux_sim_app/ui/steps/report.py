"""steps/report.py — Step 4: Generate slide report + email delivery.

Slide engine: Quarkdown (iamgio/quarkdown) with design kit injection.
Design kits: frontend-slides presets + designkits.sh official/community kits.
Fallback: Marpit (Node.js) if Quarkdown binary is unavailable.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import traceback
import uuid
from pathlib import Path

from .shared import (
    cfg, REPORTS_DIR,
    # Quarkdown pipeline
    build_report_data, render_html_quarkdown, html_dir_to_pdf,
    get_kit, OASIS_DEFAULT_KIT,
    # Marpit fallback
    render_html as render_html_marpit, html_to_pdf as html_to_pdf_marpit,
    IssueSlide,
    generate_redesign, sanitise_for_embed,
    logger,
)

# Check whether Quarkdown is available at startup
import subprocess as _sp
_QD_AVAILABLE = False
try:
    _r = _sp.run(
        ["/usr/local/bin/quarkdown", "--version"],
        capture_output=True, text=True, timeout=10,
        env={**os.environ, "JAVA_HOME": "/opt/jdk17"},
    )
    _QD_AVAILABLE = _r.returncode == 0
except Exception:
    pass


def step_generate_report(
    url: str,
    personas_json: str,
    sim_results_json: str,
    ux_json: str,
    generate_redesigns: bool,
    design_kit_id: str = "oasis-default",
):
    """Generate the Quarkdown slide-style HTML + PDF report with optional AI redesigns.

    Parameters
    ----------
    design_kit_id : str
        ID of the design kit to use for slide styling.
        Comes from the Gradio dropdown (KIT_CHOICES).
        Falls back to "oasis-default" if not found.

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

    kit = get_kit(design_kit_id or "oasis-default")
    engine = "Quarkdown" if _QD_AVAILABLE else "Marpit (fallback)"
    yield f"⏳ Building slide report ({engine}, design: {kit.name})...", *_EMPTY_REPORT

    try:
        persona_dicts = json.loads(personas_json)
        sim_dicts = json.loads(sim_results_json)
        ux_data = json.loads(ux_json)
        scrape_data = ux_data

        report_data = build_report_data(
            url=url or "",
            scrape_data=scrape_data,
            ux_data=ux_data,
            sim_results=sim_dicts,
            personas=persona_dicts,
        )

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
                        if rd.get("redesign_screenshot"):
                            issue.redesign_screenshot_url = rd["redesign_screenshot"]
                    else:
                        logger.warning("Redesign failed for %s: %s", issue.title, rd.get("error"))
                except Exception as exc:
                    logger.warning("Redesign exception for %s: %s", issue.title, exc)

        run_id = ux_data.get("run_id", uuid.uuid4().hex[:8])

        # ── Quarkdown path ────────────────────────────────────────────────────
        if _QD_AVAILABLE:
            yield f"⏳ Rendering Quarkdown slides (design: {kit.name})...", *_EMPTY_REPORT

            qd_out_dir = REPORTS_DIR / f"qd_{run_id}"
            try:
                html, compiled_dir = render_html_quarkdown(
                    report_data, kit=kit, out_dir=qd_out_dir
                )
            except Exception as qd_err:
                logger.warning("Quarkdown render failed, falling back to Marpit: %s", qd_err)
                yield f"⚠️ Quarkdown failed ({qd_err}), falling back to Marpit...", *_EMPTY_REPORT
                _use_marpit = True
            else:
                _use_marpit = False

            if not _use_marpit:
                # Save the HTML preview copy
                html_path = REPORTS_DIR / f"report_{run_id}.html"
                html_path.write_text(html, encoding="utf-8")

                yield "⏳ Exporting PDF via Playwright...", *_EMPTY_REPORT
                pdf_path = REPORTS_DIR / f"report_{run_id}.pdf"
                try:
                    html_dir_to_pdf(compiled_dir, str(pdf_path))
                    download_path = str(pdf_path)
                    status_msg = (
                        f"✅ Slide report generated with Quarkdown + {kit.name} design "
                        f"({len(report_data.issues)} issues, {len(report_data.strengths)} strengths). "
                        f"PDF ready."
                    )
                except Exception as pdf_err:
                    download_path = str(html_path)
                    status_msg = (
                        f"✅ Report generated (PDF export failed: {pdf_err}). "
                        f"Downloading HTML instead."
                    )

                yield status_msg, download_path, html
                return

        # ── Marpit fallback path ──────────────────────────────────────────────
        yield "⏳ Rendering Marpit slides (fallback)...", *_EMPTY_REPORT

        html = render_html_marpit(report_data)

        html_path = REPORTS_DIR / f"report_{run_id}.html"
        html_path.write_text(html, encoding="utf-8")

        yield "⏳ Exporting PDF via Playwright (this may take 10-15 seconds)...", *_EMPTY_REPORT

        pdf_path = REPORTS_DIR / f"report_{run_id}.pdf"
        try:
            html_to_pdf_marpit(html, str(pdf_path))
            download_path = str(pdf_path)
            status_msg = (
                f"✅ Slide report generated with Marpit "
                f"({len(report_data.issues)} issues, {len(report_data.strengths)} strengths). "
                f"PDF ready."
            )
        except Exception as pdf_err:
            download_path = str(html_path)
            status_msg = f"✅ Report generated (PDF export failed: {pdf_err}). Downloading HTML instead."

        yield status_msg, download_path, html

    except Exception as exc:
        yield f"❌ Report generation error: {exc}\n{traceback.format_exc()}", None, ""


def deliver_email(report_html: str, to_email: str, url: str) -> str:
    if not report_html:
        return "❌ No report to send. Generate the report first (Tab 5)."
    if not to_email or not to_email.strip():
        return "❌ Please enter a recipient email address."
    run_id = uuid.uuid4().hex[:8]
    try:
        from ux_sim_app.report.generator import send_report_email as _send_email
        ok, msg = _send_email(report_html, to_email.strip(), url or "", run_id)
        return f"✅ {msg}" if ok else f"❌ {msg}"
    except ImportError:
        return "❌ Email delivery is not available (generator module not installed)."
