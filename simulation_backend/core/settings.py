"""
Central configuration for the simulation backend.
All values can be overridden via environment variables or a .env file.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── LLM ───────────────────────────────────────────────────────────────
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv(
        "OPENAI_BASE_URL", "https://api.openai.com/v1"
    )
    default_model: str = "gpt-4o-mini"
    vision_model: str = "gpt-4o"          # used for Mode 3 visual input

    # ── Blablador (optional, used by /api/craft proxy) ────────────────────
    blablador_api_key: Optional[str] = None

    # ── Browserbase MCP (Mode 2 fallback) ─────────────────────────────────
    browserbase_api_key: Optional[str] = None

    # ── Parallelism ────────────────────────────────────────────────────────
    # Maximum concurrent Playwright browser sessions (Mode 2 primary)
    max_browser_sessions: int = 8
    # Semaphore for OASIS async agent loop
    oasis_semaphore: int = 64

    # ── Storage ────────────────────────────────────────────────────────────
    data_dir: str = os.path.join(os.path.dirname(__file__), "..", "data")
    focus_groups_dir: str = os.path.join(
        os.path.dirname(__file__), "..", "data", "focus_groups"
    )

    # ── Server ─────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 7860
    reload: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()
