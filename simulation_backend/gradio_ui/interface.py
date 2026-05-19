"""
Gradio testing interface for the OASIS multi-mode simulation backend.

Tabs:
  1. Persona Generation  – generate and inspect focus groups
  2. Simulation Runner   – run any of the three modes and see live results
  3. Results Viewer      – browse past simulation results
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Optional, Tuple

import gradio as gr

log = logging.getLogger("simulation_backend.gradio_ui")

# ── Lazy imports to avoid circular dependencies ────────────────────────────────

def _get_pm():
    from simulation_backend.core.persona_manager import get_persona_manager
    return get_persona_manager()


def _get_js():
    from simulation_backend.core.job_store import get_job_store
    return get_job_store()


def _get_orch():
    from simulation_backend.core.orchestrator import get_orchestrator
    return get_orchestrator()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _run_async(coro):
    """Run an async coroutine from a sync Gradio callback."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def _format_json(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, default=str)
    except Exception:
        return str(obj)


# ── Tab 1: Persona Generation ──────────────────────────────────────────────────

def generate_personas_ui(
    business_desc: str,
    customer_profile: str,
    num_personas: int,
    api_key_override: str,
) -> Tuple[str, str]:
    if not business_desc.strip():
        return "❌ Please enter a business description.", ""
    try:
        pm = _get_pm()
        group = _run_async(
            pm.generate_personas(
                business_description=business_desc,
                customer_profile=customer_profile,
                num_personas=int(num_personas),
                api_key=api_key_override.strip() or None,
            )
        )
        summary = (
            f"✅ Created focus group **{group.name}**\n"
            f"- ID: `{group.id}`\n"
            f"- Personas: {len(group.personas)}"
        )
        detail = _format_json(group.model_dump())
        return summary, detail
    except Exception as exc:
        return f"❌ Error: {exc}", ""


# ── Tab 2: Simulation Runner ───────────────────────────────────────────────────

def _get_focus_group_choices() -> list[str]:
    try:
        pm = _get_pm()
        return [f"{g.name} [{g.id[:8]}…]" for g in pm.list_groups()]
    except Exception:
        return []


def _resolve_group_id(choice: str) -> Optional[str]:
    """Extract the full group ID from the dropdown label."""
    try:
        pm = _get_pm()
        for g in pm.list_groups():
            if g.id[:8] in choice:
                return g.id
    except Exception:
        pass
    return None


def run_simulation_ui(
    group_choice: str,
    mode_label: str,
    content_payload: str,
    target_url: str,
    image_url: str,
    use_browserbase: bool,
) -> Tuple[str, str]:
    mode_map = {
        "Mode 1 – Content Simulation": 1,
        "Mode 2 – Browser-Action Simulation": 2,
        "Mode 3 – Visual Input Simulation": 3,
    }
    mode = mode_map.get(mode_label, 1)

    group_id = _resolve_group_id(group_choice)
    if not group_id:
        return "❌ Please select a valid focus group.", ""

    from simulation_backend.core.models import (
        SimulationParameters,
        SimulationMode,
        StartSimulationRequest,
    )

    params = SimulationParameters(
        mode=SimulationMode(mode),
        target_url=target_url.strip() or None,
        image_urls=[image_url.strip()] if image_url.strip() else None,
        use_browserbase_fallback=use_browserbase,
    )
    request = StartSimulationRequest(
        focus_group_id=group_id,
        content_type="text" if mode != 2 else "url",
        content_payload=content_payload.strip() or target_url.strip(),
        parameters=params,
    )

    try:
        orch = _get_orch()
        job_id = orch.enqueue(request)
        status_msg = f"✅ Job enqueued: `{job_id}`\n\nPolling for results…"

        # Poll until done (max 120 s)
        js = _get_js()
        import time
        for _ in range(120):
            time.sleep(1)
            job = js.get(job_id)
            if job and job.status in ("completed", "failed"):
                break

        job = js.get(job_id)
        if job is None:
            return status_msg, "Job not found after polling."
        if job.status == "failed":
            return f"❌ Simulation failed: {job.message}", ""

        return (
            f"✅ Simulation completed!\n- Mode: {mode_label}\n"
            f"- Progress: {job.progress_percentage:.0f}%\n"
            f"- Message: {job.message}",
            _format_json(job.results),
        )
    except Exception as exc:
        return f"❌ Error: {exc}", ""


# ── Tab 3: Results Viewer ──────────────────────────────────────────────────────

def list_jobs_ui() -> str:
    try:
        js = _get_js()
        jobs = js.list_all()
        if not jobs:
            return "No simulation jobs found."
        lines = []
        for j in sorted(jobs, key=lambda x: x.created_at, reverse=True):
            lines.append(
                f"- `{j.job_id[:12]}…` | Mode {j.mode} | "
                f"**{j.status}** | {j.progress_percentage:.0f}% | "
                f"{j.message[:60]}"
            )
        return "\n".join(lines)
    except Exception as exc:
        return f"Error: {exc}"


def get_job_results_ui(job_id: str) -> str:
    try:
        js = _get_js()
        job = js.get(job_id.strip())
        if job is None:
            return "Job not found."
        return _format_json(job.model_dump())
    except Exception as exc:
        return f"Error: {exc}"


# ── Build the Gradio Blocks app ────────────────────────────────────────────────

def build_gradio_app() -> gr.Blocks:
    with gr.Blocks(
        title="OASIS Simulation Backend",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
# 🌐 OASIS Multi-Mode Simulation Backend
Test persona generation and all three simulation modes from this interface.
            """
        )

        # ── Tab 1 ──────────────────────────────────────────────────────────
        with gr.Tab("🧑‍🤝‍🧑 Persona Generation"):
            gr.Markdown("### Generate a Focus Group via LLM")
            with gr.Row():
                with gr.Column():
                    biz_desc = gr.Textbox(
                        label="Business Description",
                        placeholder="e.g. A sustainable fashion brand targeting Gen-Z consumers…",
                        lines=3,
                    )
                    cust_profile = gr.Textbox(
                        label="Target Customer Profile",
                        placeholder="e.g. 18-28 year olds interested in eco-friendly clothing…",
                        lines=2,
                    )
                    num_p = gr.Slider(
                        label="Number of Personas",
                        minimum=1,
                        maximum=20,
                        step=1,
                        value=5,
                    )
                    api_key_box = gr.Textbox(
                        label="API Key Override (optional)",
                        placeholder="Leave blank to use server default",
                        type="password",
                    )
                    gen_btn = gr.Button("Generate Personas", variant="primary")
                with gr.Column():
                    gen_summary = gr.Markdown(label="Summary")
                    gen_detail = gr.Code(
                        label="Full JSON Output",
                        language="json",
                        lines=20,
                    )
            gen_btn.click(
                fn=generate_personas_ui,
                inputs=[biz_desc, cust_profile, num_p, api_key_box],
                outputs=[gen_summary, gen_detail],
            )

        # ── Tab 2 ──────────────────────────────────────────────────────────
        with gr.Tab("🚀 Simulation Runner"):
            gr.Markdown("### Run a Simulation")
            with gr.Row():
                with gr.Column():
                    group_dd = gr.Dropdown(
                        label="Focus Group",
                        choices=_get_focus_group_choices(),
                        allow_custom_value=False,
                    )
                    refresh_btn = gr.Button("🔄 Refresh Focus Groups")
                    refresh_btn.click(
                        fn=lambda: gr.update(choices=_get_focus_group_choices()),
                        outputs=group_dd,
                    )
                    mode_radio = gr.Radio(
                        label="Simulation Mode",
                        choices=[
                            "Mode 1 – Content Simulation",
                            "Mode 2 – Browser-Action Simulation",
                            "Mode 3 – Visual Input Simulation",
                        ],
                        value="Mode 1 – Content Simulation",
                    )
                    content_box = gr.Textbox(
                        label="Content Payload (text, URL, or image URL)",
                        placeholder="Paste social media copy, a URL, or an image URL…",
                        lines=4,
                    )
                    target_url_box = gr.Textbox(
                        label="Target URL (Mode 2 only)",
                        placeholder="https://example.com",
                    )
                    image_url_box = gr.Textbox(
                        label="Image URL (Mode 3 only)",
                        placeholder="https://example.com/image.jpg",
                    )
                    use_bb_chk = gr.Checkbox(
                        label="Use Browserbase MCP as fallback (Mode 2)",
                        value=True,
                    )
                    run_btn = gr.Button("▶ Run Simulation", variant="primary")
                with gr.Column():
                    run_status = gr.Markdown(label="Status")
                    run_results = gr.Code(
                        label="Results JSON",
                        language="json",
                        lines=30,
                    )
            run_btn.click(
                fn=run_simulation_ui,
                inputs=[
                    group_dd,
                    mode_radio,
                    content_box,
                    target_url_box,
                    image_url_box,
                    use_bb_chk,
                ],
                outputs=[run_status, run_results],
            )

        # ── Tab 3 ──────────────────────────────────────────────────────────
        with gr.Tab("📋 Results Viewer"):
            gr.Markdown("### Past Simulation Jobs")
            list_btn = gr.Button("🔄 Refresh Job List")
            jobs_md = gr.Markdown()
            list_btn.click(fn=list_jobs_ui, outputs=jobs_md)

            gr.Markdown("### Inspect a Job")
            job_id_box = gr.Textbox(label="Job ID", placeholder="Paste full job ID…")
            inspect_btn = gr.Button("Inspect")
            job_detail = gr.Code(label="Job Detail JSON", language="json", lines=30)
            inspect_btn.click(
                fn=get_job_results_ui,
                inputs=job_id_box,
                outputs=job_detail,
            )

    return demo
