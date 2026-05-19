"""steps/report.py — Step 4: Generate slide report + email delivery."""
from __future__ import annotations

import json
import os
import traceback
import uuid

from .shared import (
    cfg, REPORTS_DIR,
    build_report_data, render_html, html_to_pdf, IssueSlide,
    generate_redesign, sanitise_for_embed,
    logger,
)


def step_generate_report(
    url: str,
    personas_json: str,
    sim_results_json: str,
    ux_json: str,
    generate_redesigns: bool,
):
    """Generate the Marpit slide-style HTML + PDF report with optional AI redesigns.
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

        yield "⏳ Rendering Marpit slides...", *_EMPTY_REPORT

        html = render_html(report_data)

        run_id = ux_data.get("run_id", uuid.uuid4().hex[:8])
        html_path = REPORTS_DIR / f"report_{run_id}.html"
        html_path.write_text(html, encoding="utf-8")

        yield "⏳ Exporting PDF via Playwright (this may take 10-15 seconds)...", *_EMPTY_REPORT

        pdf_path = REPORTS_DIR / f"report_{run_id}.pdf"
        try:
            html_to_pdf(html, str(pdf_path))
            download_path = str(pdf_path)
            status_msg = (
                f"✅ Slide report generated ({len(report_data.issues)} issues, "
                f"{len(report_data.strengths)} strengths). PDF ready."
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
