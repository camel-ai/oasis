"""
Report generator: produces a rich HTML report combining UX scan + persona simulation results.
Persona feedback is visually highlighted and drives the attention/priority of UX findings.
"""
from __future__ import annotations

import asyncio
import json
import re
import smtplib
import ssl
from datetime import datetime
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, List, Optional

from ux_sim_app.core.config import (
    REPORTS_DIR, SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, SMTP_FROM
)
from ux_sim_app.core.llm import chat, text_content
from ux_sim_app.core.personas import Persona
from ux_sim_app.modes.runner import SimulationResult
from ux_sim_app.ux.scanner import UXReport

# ── Severity colour mapping ────────────────────────────────────────────────────
SEV_COLOR = {"High": "#e74c3c", "Medium": "#f39c12", "Low": "#27ae60"}
SENTIMENT_COLOR = {"positive": "#27ae60", "neutral": "#7f8c8d", "negative": "#e74c3c"}
ACTION_ICON = {
    "like_post": "👍", "repost": "🔁", "create_comment": "💬",
    "quote_post": "🗣️", "dislike_post": "👎", "report_post": "🚩", "do_nothing": "😐",
}
ENGAGEMENT_ICON = {
    "definitely": "🔥", "probably": "✅", "maybe": "🤔", "unlikely": "❌", "no": "🚫",
}


# ── AI synthesis ───────────────────────────────────────────────────────────────

async def _synthesise_insights(
    ux_report: UXReport,
    sim_results: List[SimulationResult],
    personas: List[Persona],
) -> str:
    """Ask the LLM to synthesise persona feedback with UX findings into a narrative."""
    persona_summary = "\n".join(
        f"- {p.name} ({p.persona_type}, {p.age}yo): {p.bio}" for p in personas
    )

    ux_issues_summary = ""
    for dim in ux_report.dimensions:
        if dim.issues:
            ux_issues_summary += f"\n{dim.name} (score {dim.score}/100):\n"
            for issue in dim.issues[:3]:
                ux_issues_summary += f"  [{issue.severity}] {issue.description}\n"

    mode2_feedback = ""
    for sr in sim_results:
        if sr.mode == 2:
            for r in sr.responses[:3]:
                mode2_feedback += f"\n{r.persona_name}: {(r.usability_summary or '')[:300]}\n"

    mode1_feedback = ""
    for sr in sim_results:
        if sr.mode == 1:
            for r in sr.responses[:3]:
                mode1_feedback += (
                    f"\n{r.persona_name} → {r.action} ({r.sentiment}): {r.reasoning}\n"
                )

    prompt = f"""You are a UX strategist writing the executive summary of a combined
UX audit + persona simulation report.

Personas tested:
{persona_summary}

UX scan findings:
{ux_issues_summary}

Persona browser feedback:
{mode2_feedback}

Persona content reactions:
{mode1_feedback}

Write a 3-paragraph executive narrative (no bullet points) that:
1. Highlights where persona feedback CONFIRMS or CONTRADICTS the automated UX findings
2. Identifies the 2-3 most critical issues that real users actually noticed
3. Provides a clear priority recommendation for the website owner

Be specific, reference actual persona names and quotes where relevant. Be direct and actionable."""

    resp = await chat(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800,
        temperature=0.5,
    )
    return text_content(resp)


# ── HTML builder ───────────────────────────────────────────────────────────────

def _score_badge(score: int) -> str:
    color = "#27ae60" if score >= 75 else "#f39c12" if score >= 50 else "#e74c3c"
    return f'<span style="background:{color};color:#fff;padding:4px 10px;border-radius:20px;font-weight:700;font-size:1.1em">{score}/100</span>'


def _persona_card_mode1(r) -> str:
    sc = SENTIMENT_COLOR.get(r.sentiment or "neutral", "#7f8c8d")
    icon = ACTION_ICON.get(r.action or "do_nothing", "❓")
    comment_html = (
        f'<blockquote style="border-left:3px solid {sc};margin:8px 0;padding:6px 12px;'
        f'background:#f9f9f9;font-style:italic">{r.comment_text}</blockquote>'
        if r.comment_text else ""
    )
    return f"""
<div style="border:1px solid #e0e0e0;border-radius:8px;padding:14px;margin:8px 0;
     border-left:4px solid {sc}">
  <strong>{r.persona_name}</strong>
  <span style="color:#888;font-size:0.85em"> · {r.persona_type}</span>
  <span style="float:right;font-size:1.3em">{icon}</span>
  <div style="margin-top:6px">
    <span style="background:{sc};color:#fff;padding:2px 8px;border-radius:12px;font-size:0.8em">
      {r.action} · {r.sentiment}
    </span>
  </div>
  <p style="margin:8px 0;color:#444;font-size:0.9em">{r.reasoning}</p>
  {comment_html}
</div>"""


def _persona_card_mode2(r) -> str:
    convert_html = ""
    if r.would_convert is True:
        convert_html = '<span style="color:#27ae60;font-weight:700">✅ Would convert</span>'
    elif r.would_convert is False:
        convert_html = '<span style="color:#e74c3c;font-weight:700">❌ Would not convert</span>'
    summary_html = ""
    if r.usability_summary:
        # Highlight key UX pain-point phrases
        highlighted = r.usability_summary
        for phrase in ["couldn't find", "hard to find", "difficult", "confusing", "unclear",
                        "frustrat", "couldn't click", "not easy", "no direct link"]:
            highlighted = re.sub(
                f"({re.escape(phrase)})",
                r'<mark style="background:#fff3cd">\1</mark>',
                highlighted,
                flags=re.IGNORECASE,
            )
        summary_html = f'<p style="font-size:0.9em;color:#333;margin:8px 0">{highlighted}</p>'
    return f"""
<div style="border:1px solid #e0e0e0;border-radius:8px;padding:14px;margin:8px 0;
     border-left:4px solid #3498db">
  <strong>{r.persona_name}</strong>
  <span style="color:#888;font-size:0.85em"> · {r.persona_type}</span>
  <span style="float:right">{convert_html}</span>
  {summary_html}
</div>"""


def _persona_card_mode3(r) -> str:
    sc = SENTIMENT_COLOR.get(r.sentiment or "neutral", "#7f8c8d")
    icon = ENGAGEMENT_ICON.get(r.engagement_likelihood or "maybe", "❓")
    score = r.resonance_score or 0
    bar_color = "#27ae60" if score >= 7 else "#f39c12" if score >= 4 else "#e74c3c"
    return f"""
<div style="border:1px solid #e0e0e0;border-radius:8px;padding:14px;margin:8px 0;
     border-left:4px solid {sc}">
  <strong>{r.persona_name}</strong>
  <span style="color:#888;font-size:0.85em"> · {r.persona_type}</span>
  <span style="float:right;font-size:1.2em">{icon} {r.engagement_likelihood}</span>
  <div style="margin:8px 0">
    <span style="font-size:0.85em;color:#666">Resonance: </span>
    <span style="background:{bar_color};color:#fff;padding:2px 8px;border-radius:12px;font-size:0.85em">
      {score}/10
    </span>
  </div>
  <p style="font-size:0.85em;color:#555;margin:4px 0;font-style:italic">{r.first_impression}</p>
  <p style="font-size:0.9em;color:#333;margin:6px 0">{r.visual_feedback}</p>
</div>"""


def _ux_dimension_section(dim) -> str:
    color = "#27ae60" if dim.score >= 75 else "#f39c12" if dim.score >= 50 else "#e74c3c"
    issues_html = ""
    for issue in dim.issues:
        sev_c = SEV_COLOR.get(issue.severity, "#888")
        issues_html += f"""
<div style="margin:6px 0;padding:10px;background:#f8f9fa;border-radius:6px;
     border-left:3px solid {sev_c}">
  <span style="background:{sev_c};color:#fff;padding:1px 7px;border-radius:10px;
       font-size:0.78em;font-weight:700">{issue.severity}</span>
  <span style="font-size:0.85em;color:#666;margin-left:6px">{issue.category}</span>
  <p style="margin:6px 0;font-size:0.9em">{issue.description}</p>
  <p style="margin:0;font-size:0.85em;color:#2980b9">
    💡 <em>{issue.recommendation}</em>
  </p>
</div>"""
    return f"""
<div style="border:1px solid #e0e0e0;border-radius:10px;padding:18px;margin:12px 0">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px">
    <h3 style="margin:0;font-size:1.05em">{dim.name}</h3>
    {_score_badge(dim.score)}
  </div>
  <div style="background:#f0f0f0;border-radius:4px;height:8px;margin-bottom:10px">
    <div style="background:{color};width:{dim.score}%;height:8px;border-radius:4px"></div>
  </div>
  <p style="font-size:0.9em;color:#444;margin:0 0 10px 0">{dim.feedback}</p>
  {issues_html}
</div>"""


def _heuristic_table(checks: Dict) -> str:
    rows = ""
    for key, val in checks.items():
        if key == "heuristic_score":
            continue
        if not isinstance(val, dict):
            continue
        passed = val.get("pass", False)
        icon = "✅" if passed else "❌"
        label = key.replace("_", " ").title()
        detail = ""
        if "coverage_pct" in val:
            detail = f"{val['coverage_pct']}% coverage"
        elif "title" in val:
            detail = val["title"][:60]
        elif "cta_texts" in val:
            detail = ", ".join(val["cta_texts"][:3])
        elif "nav_items" in val:
            detail = ", ".join(val["nav_items"][:4])
        rows += f"<tr><td>{icon}</td><td><strong>{label}</strong></td><td style='color:#666;font-size:0.85em'>{detail}</td></tr>"
    score = checks.get("heuristic_score", 0)
    return f"""
<table style="width:100%;border-collapse:collapse;font-size:0.9em">
  <thead>
    <tr style="background:#f0f0f0">
      <th style="padding:8px;text-align:left;width:40px">Pass</th>
      <th style="padding:8px;text-align:left">Check</th>
      <th style="padding:8px;text-align:left">Detail</th>
    </tr>
  </thead>
  <tbody>
    {rows}
    <tr style="background:#f8f9fa;font-weight:700">
      <td colspan="2" style="padding:8px">Heuristic Score</td>
      <td style="padding:8px">{_score_badge(score)}</td>
    </tr>
  </tbody>
</table>"""


def build_html_report(
    url: str,
    personas: List[Persona],
    ux_report: UXReport,
    sim_results: List[SimulationResult],
    executive_summary: str,
    run_id: str,
) -> str:
    now = datetime.utcnow().strftime("%B %d, %Y at %H:%M UTC")

    # ── Persona section ────────────────────────────────────────────────────────
    persona_cards_html = ""
    for p in personas:
        persona_cards_html += f"""
<div style="display:inline-block;vertical-align:top;width:180px;margin:6px;
     border:1px solid #e0e0e0;border-radius:8px;padding:12px;text-align:center">
  <div style="width:50px;height:50px;border-radius:50%;background:#3498db;
       color:#fff;font-size:1.4em;line-height:50px;margin:0 auto 8px">
    {p.name[0]}
  </div>
  <strong style="font-size:0.9em">{p.name}</strong><br>
  <span style="color:#888;font-size:0.78em">{p.persona_type}</span><br>
  <span style="color:#aaa;font-size:0.75em">{p.age}yo · {p.mbti}</span>
</div>"""

    # ── Mode 1 sections ────────────────────────────────────────────────────────
    mode1_sections = ""
    for sr in sim_results:
        if sr.mode != 1:
            continue
        cards = "".join(_persona_card_mode1(r) for r in sr.responses)
        agg = sr.aggregate
        eng_rate = int(agg.get("engagement_rate", 0) * 100)
        mode1_sections += f"""
<div style="border:1px solid #e0e0e0;border-radius:10px;padding:18px;margin:14px 0">
  <h4 style="margin:0 0 6px 0;color:#2c3e50">Content Tested</h4>
  <blockquote style="border-left:4px solid #3498db;margin:0 0 14px 0;padding:8px 14px;
       background:#f0f7ff;font-style:italic;color:#333">
    {sr.content_tested}
  </blockquote>
  <div style="margin-bottom:12px">
    <span style="background:#3498db;color:#fff;padding:3px 10px;border-radius:12px;font-size:0.85em">
      Engagement Rate: {eng_rate}%
    </span>
    {"".join(f'<span style="background:#ecf0f1;padding:3px 8px;border-radius:12px;font-size:0.8em;margin-left:4px">{k}: {v}</span>' for k, v in agg.get("action_distribution", {}).items())}
  </div>
  {cards}
</div>"""

    # ── Mode 2 section ─────────────────────────────────────────────────────────
    mode2_section = ""
    for sr in sim_results:
        if sr.mode != 2:
            continue
        cards = "".join(_persona_card_mode2(r) for r in sr.responses)
        agg = sr.aggregate
        conv_rate = int(agg.get("conversion_intent_rate", 0) * 100)
        mode2_section = f"""
<div style="margin:14px 0">
  <div style="margin-bottom:12px">
    <span style="background:#9b59b6;color:#fff;padding:3px 10px;border-radius:12px;font-size:0.85em">
      Conversion Intent: {conv_rate}% ({agg.get('personas_would_convert',0)}/{agg.get('total_personas',0)} personas)
    </span>
  </div>
  {cards}
</div>"""

    # ── Mode 3 section ─────────────────────────────────────────────────────────
    mode3_section = ""
    for sr in sim_results:
        if sr.mode != 3:
            continue
        cards = "".join(_persona_card_mode3(r) for r in sr.responses)
        agg = sr.aggregate
        avg_res = agg.get("average_resonance", 0)
        mode3_section = f"""
<div style="margin:14px 0">
  <div style="margin-bottom:12px">
    <span style="background:#e67e22;color:#fff;padding:3px 10px;border-radius:12px;font-size:0.85em">
      Average Resonance: {avg_res}/10
    </span>
    {"".join(f'<span style="background:#ecf0f1;padding:3px 8px;border-radius:12px;font-size:0.8em;margin-left:4px">{k}: {v}</span>' for k, v in agg.get("engagement_distribution", {}).items())}
  </div>
  {cards}
</div>"""

    # ── UX dimensions ──────────────────────────────────────────────────────────
    dimensions_html = "".join(_ux_dimension_section(d) for d in ux_report.dimensions)

    # ── Recommendations ────────────────────────────────────────────────────────
    recs_html = ""
    for rec in ux_report.recommendations:
        prio = rec.get("priority", "Medium")
        prio_c = SEV_COLOR.get(prio, "#888")
        recs_html += f"""
<div style="border:1px solid #e0e0e0;border-radius:8px;padding:14px;margin:8px 0;
     border-left:4px solid {prio_c}">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
    <strong>{rec.get('title','')}</strong>
    <span style="background:{prio_c};color:#fff;padding:2px 8px;border-radius:10px;font-size:0.8em">
      {prio} Priority
    </span>
  </div>
  <span style="color:#888;font-size:0.8em">{rec.get('category','')} · Effort: {rec.get('effort','')}</span>
  <p style="margin:8px 0;font-size:0.9em">{rec.get('description','')}</p>
  <p style="margin:0;font-size:0.85em;color:#27ae60">Impact: {rec.get('impact','')}</p>
</div>"""

    # ── Screenshots ────────────────────────────────────────────────────────────
    screenshots_html = ""
    for vp, path in ux_report.screenshots.items():
        if not path.startswith("ERROR") and Path(path).exists():
            import base64
            b64 = base64.b64encode(Path(path).read_bytes()).decode()
            screenshots_html += f"""
<div style="display:inline-block;vertical-align:top;margin:8px;text-align:center">
  <p style="font-weight:700;margin:0 0 6px 0;text-transform:capitalize">{vp}</p>
  <img src="data:image/png;base64,{b64}"
       style="max-width:320px;border:1px solid #e0e0e0;border-radius:6px;box-shadow:0 2px 8px rgba(0,0,0,0.1)">
</div>"""

    # ── Full HTML ──────────────────────────────────────────────────────────────
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>UX + Persona Simulation Report – {url}</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         color: #2c3e50; margin: 0; padding: 0; background: #f5f6fa; }}
  .container {{ max-width: 960px; margin: 0 auto; padding: 24px; }}
  h1 {{ font-size: 1.8em; margin-bottom: 4px; }}
  h2 {{ font-size: 1.3em; border-bottom: 2px solid #3498db; padding-bottom: 6px;
       margin-top: 32px; color: #2980b9; }}
  h3 {{ font-size: 1.1em; color: #2c3e50; }}
  .header {{ background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
             color: #fff; padding: 32px; border-radius: 12px; margin-bottom: 24px; }}
  .header a {{ color: #aed6f1; }}
  .card {{ background: #fff; border-radius: 12px; padding: 24px; margin-bottom: 20px;
           box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
  .score-row {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 16px 0; }}
  .score-box {{ flex: 1; min-width: 120px; background: #f8f9fa; border-radius: 8px;
               padding: 14px; text-align: center; }}
  .score-box .label {{ font-size: 0.78em; color: #888; text-transform: uppercase; }}
  mark {{ background: #fff3cd; padding: 1px 2px; border-radius: 3px; }}
  @media print {{ body {{ background: #fff; }} .container {{ padding: 0; }} }}
</style>
</head>
<body>
<div class="container">

  <!-- Header -->
  <div class="header">
    <h1>UX + Persona Simulation Report</h1>
    <p style="margin:4px 0;font-size:1.05em">
      <a href="{url}" target="_blank">{url}</a>
    </p>
    <p style="margin:4px 0;color:#bdc3c7;font-size:0.85em">Generated {now} · Run ID: {run_id}</p>
    <div class="score-row" style="margin-top:20px">
      <div class="score-box">
        <div class="label">UX Score</div>
        <div style="font-size:2em;font-weight:700">{ux_report.overall_score}</div>
        <div style="font-size:0.8em;color:#aaa">/100</div>
      </div>
      <div class="score-box">
        <div class="label">Heuristic Score</div>
        <div style="font-size:2em;font-weight:700">{ux_report.heuristic_checks.get('heuristic_score',0)}</div>
        <div style="font-size:0.8em;color:#aaa">/100</div>
      </div>
      <div class="score-box">
        <div class="label">Personas Tested</div>
        <div style="font-size:2em;font-weight:700">{len(personas)}</div>
      </div>
      <div class="score-box">
        <div class="label">Simulation Modes</div>
        <div style="font-size:2em;font-weight:700">{len(set(sr.mode for sr in sim_results))}</div>
      </div>
    </div>
  </div>

  <!-- Executive Summary -->
  <div class="card">
    <h2>🧠 Executive Summary</h2>
    <p style="line-height:1.7;font-size:0.95em">{executive_summary.replace(chr(10), '<br>')}</p>
    <div style="margin-top:16px">
      <strong>Strengths:</strong>
      <ul>{"".join(f"<li>{s}</li>" for s in ux_report.strengths)}</ul>
      <strong>Weaknesses:</strong>
      <ul>{"".join(f"<li>{w}</li>" for w in ux_report.weaknesses)}</ul>
    </div>
  </div>

  <!-- Focus Group -->
  <div class="card">
    <h2>👥 Simulated Focus Group</h2>
    <div>{persona_cards_html}</div>
  </div>

  <!-- Screenshots -->
  {f'<div class="card"><h2>📸 Website Screenshots</h2>{screenshots_html}</div>' if screenshots_html else ''}

  <!-- UX Dimensions -->
  <div class="card">
    <h2>🔍 UX Analysis (6 Dimensions)</h2>
    {dimensions_html}
  </div>

  <!-- Heuristic Checks -->
  <div class="card">
    <h2>✅ Heuristic Checks</h2>
    {_heuristic_table(ux_report.heuristic_checks)}
  </div>

  <!-- Mode 2: Browser Simulation -->
  {f'<div class="card"><h2>🖥️ Mode 2 – Browser Usability (Persona Feedback)</h2>{mode2_section}</div>' if mode2_section else ''}

  <!-- Mode 3: Visual Simulation -->
  {f'<div class="card"><h2>🎨 Mode 3 – Visual & Branding Feedback</h2>{mode3_section}</div>' if mode3_section else ''}

  <!-- Mode 1: Content Simulation -->
  {f'<div class="card"><h2>📣 Mode 1 – Content & Marketing Simulation</h2>{mode1_sections}</div>' if mode1_sections else ''}

  <!-- Recommendations -->
  <div class="card">
    <h2>🚀 Prioritised Recommendations</h2>
    {recs_html}
  </div>

  <p style="text-align:center;color:#aaa;font-size:0.8em;margin-top:32px">
    Generated by OASIS UX Simulation Backend · {now}
  </p>
</div>
</body>
</html>"""


# ── Save & deliver ─────────────────────────────────────────────────────────────

def save_report(html: str, run_id: str) -> Path:
    path = REPORTS_DIR / f"report_{run_id}.html"
    path.write_text(html, encoding="utf-8")
    return path


def send_report_email(html: str, to_email: str, url: str, run_id: str) -> Tuple[bool, str]:
    """Send the HTML report as an email attachment."""
    from typing import Tuple
    if not SMTP_USER or not SMTP_PASS:
        return False, "SMTP credentials not configured. Set SMTP_USER and SMTP_PASS in .env"
    try:
        msg = MIMEMultipart("mixed")
        msg["From"] = SMTP_FROM or SMTP_USER
        msg["To"] = to_email
        msg["Subject"] = f"UX + Persona Simulation Report – {url}"

        body = MIMEText(
            f"Please find your UX + Persona Simulation Report for {url} attached.\n\n"
            f"Run ID: {run_id}",
            "plain",
        )
        msg.attach(body)

        attachment = MIMEApplication(html.encode("utf-8"), Name=f"report_{run_id}.html")
        attachment["Content-Disposition"] = f'attachment; filename="report_{run_id}.html"'
        msg.attach(attachment)

        context = ssl.create_default_context()
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.ehlo()
            server.starttls(context=context)
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_USER, to_email, msg.as_string())
        return True, f"Report sent to {to_email}"
    except Exception as exc:
        return False, f"Email failed: {exc}"


# ── Orchestrate full report generation ────────────────────────────────────────

async def generate_full_report(
    url: str,
    personas: List[Persona],
    ux_report: UXReport,
    sim_results: List[SimulationResult],
    run_id: str,
) -> Tuple[str, Path]:
    """Generate the executive summary and build the full HTML report. Returns (html, path)."""
    from typing import Tuple
    executive_summary = await _synthesise_insights(ux_report, sim_results, personas)
    html = build_html_report(url, personas, ux_report, sim_results, executive_summary, run_id)
    path = save_report(html, run_id)
    return html, path
