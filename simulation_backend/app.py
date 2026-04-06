"""
OASIS Multi-Mode Simulation Backend
=====================================
FastAPI application with Gradio mounted at /gradio.

Start with:
    uvicorn simulation_backend.app:app --host 0.0.0.0 --port 7860 --reload

Or via the helper script:
    python -m simulation_backend
"""
from __future__ import annotations

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from simulation_backend.api.routers.personas import router as personas_router
from simulation_backend.api.routers.simulations import router as simulations_router
from simulation_backend.gradio_ui.interface import build_gradio_app

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
log = logging.getLogger("simulation_backend.app")

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="OASIS Simulation Backend",
    description=(
        "Multi-mode persona simulation server. "
        "Supports content feedback (Mode 1), browser-action simulation (Mode 2), "
        "and visual-input analysis (Mode 3)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow the UserSyncInterface frontend (and local dev) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── REST routers ───────────────────────────────────────────────────────────────
app.include_router(personas_router)
app.include_router(simulations_router)


@app.get("/health", tags=["Health"])
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/v1/modes", tags=["Modes"])
def list_modes() -> dict:
    """Return the available simulation modes and their descriptions."""
    return {
        "modes": [
            {
                "id": 1,
                "name": "Content Simulation",
                "description": (
                    "Personas react to social media content (text, links) "
                    "using standard OASIS social actions."
                ),
            },
            {
                "id": 2,
                "name": "Browser-Action Simulation",
                "description": (
                    "Personas navigate a live website using Playwright "
                    "(primary) or Browserbase MCP (fallback)."
                ),
            },
            {
                "id": 3,
                "name": "Visual Input Simulation",
                "description": (
                    "Personas analyse images (ads, mockups, brand assets) "
                    "using a Vision-Language Model."
                ),
            },
        ]
    }


# ── Gradio interface ───────────────────────────────────────────────────────────
try:
    import gradio as gr

    gradio_app = build_gradio_app()
    app = gr.mount_gradio_app(app, gradio_app, path="/gradio")
    log.info("Gradio interface mounted at /gradio")
except ImportError:
    log.warning("Gradio not installed – /gradio endpoint unavailable.")
