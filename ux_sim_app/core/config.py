"""Central configuration for the UX Simulation App.

Provider model:
  PROVIDER = "openai"      → uses OPENAI_API_KEY + https://api.openai.com/v1
  PROVIDER = "openrouter"  → uses OPENROUTER_API_KEY + https://openrouter.ai/api/v1
  PROVIDER = "custom"      → uses OPENAI_API_KEY (or CUSTOM_API_KEY) + CUSTOM_BASE_URL

Separate model slots:
  TEXT_MODEL   – used for persona generation, content simulation, UX analysis
  VISION_MODEL – used for Mode 3 visual simulation and redesign screenshot analysis
"""
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the app directory.
# override=True ensures .env values always win over any pre-existing system env vars.
_env = Path(__file__).parent.parent / ".env"
if _env.exists():
    load_dotenv(_env, override=True)

# ── Provider ───────────────────────────────────────────────────────────────────
# One of: "openai" | "openrouter" | "custom"
PROVIDER: str = os.environ.get("PROVIDER", "openai").lower()

# ── OpenAI ─────────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")

# ── OpenRouter ─────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY: str = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

# ── Custom provider ────────────────────────────────────────────────────────────
CUSTOM_API_KEY: str = os.environ.get("CUSTOM_API_KEY", "")
CUSTOM_BASE_URL: str = os.environ.get("CUSTOM_BASE_URL", "")

# ── Resolved API key and base URL (used by llm.py) ────────────────────────────
def _resolve_api_key() -> str:
    if PROVIDER == "openrouter":
        return OPENROUTER_API_KEY
    if PROVIDER == "custom":
        return CUSTOM_API_KEY or OPENAI_API_KEY
    return OPENAI_API_KEY  # default: openai

def _resolve_base_url() -> str:
    if PROVIDER == "openrouter":
        return OPENROUTER_BASE_URL
    if PROVIDER == "custom":
        return CUSTOM_BASE_URL or "https://api.openai.com/v1"
    return "https://api.openai.com/v1"  # default: openai

EFFECTIVE_API_KEY: str = _resolve_api_key()
EFFECTIVE_BASE_URL: str = _resolve_base_url()

# ── Model slots ────────────────────────────────────────────────────────────────
# TEXT_MODEL: used for persona generation, content simulation, UX critique, browser analysis
TEXT_MODEL: str = os.environ.get("TEXT_MODEL", os.environ.get("DEFAULT_MODEL", "gpt-4o-mini"))

# VISION_MODEL: used for Mode 3 visual simulation and redesign screenshot analysis
VISION_MODEL: str = os.environ.get("VISION_MODEL", "gpt-4o")

# ── OpenRouter recommended models (shown in Settings dropdown) ─────────────────
OPENROUTER_TEXT_MODELS: list[str] = [
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "openai/gpt-4.1-mini",
    "openai/gpt-4.1",
    "anthropic/claude-3.5-haiku",
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3-opus",
    "google/gemini-2.0-flash-001",
    "google/gemini-2.5-flash-preview",
    "google/gemini-2.5-pro-preview",
    "meta-llama/llama-3.3-70b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
    "mistralai/mistral-small-3.1-24b-instruct",
    "mistralai/mistral-medium-3",
    "deepseek/deepseek-chat-v3-0324",
    "deepseek/deepseek-r1",
    "qwen/qwen-2.5-72b-instruct",
    "x-ai/grok-3-mini-beta",
]

OPENROUTER_VISION_MODELS: list[str] = [
    "openai/gpt-4o",
    "openai/gpt-4.1",
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3-opus",
    "google/gemini-2.0-flash-001",
    "google/gemini-2.5-flash-preview",
    "google/gemini-2.5-pro-preview",
    "meta-llama/llama-3.2-90b-vision-instruct",
    "qwen/qwen-2.5-vl-72b-instruct",
    "mistralai/pixtral-large-2411",
    "x-ai/grok-2-vision-1212",
]

OPENAI_TEXT_MODELS: list[str] = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-4-turbo",
    "o1-mini",
    "o3-mini",
]

OPENAI_VISION_MODELS: list[str] = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4-turbo",
]

# ── Browserbase (optional Mode 2 fallback) ────────────────────────────────────
BROWSERBASE_API_KEY: str = os.environ.get("BROWSERBASE_API_KEY", "")

# ── Email ──────────────────────────────────────────────────────────────────────
SMTP_HOST: str = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT: int = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER: str = os.environ.get("SMTP_USER", "")
SMTP_PASS: str = os.environ.get("SMTP_PASS", "")
SMTP_FROM: str = os.environ.get("SMTP_FROM", SMTP_USER)

# ── Paths ──────────────────────────────────────────────────────────────────────
APP_DIR = Path(__file__).parent.parent
DATA_DIR = APP_DIR / "data"
REPORTS_DIR = DATA_DIR / "reports"
SCREENSHOTS_DIR = DATA_DIR / "screenshots"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Parallelism ────────────────────────────────────────────────────────────────
MAX_BROWSER_SESSIONS: int = int(os.environ.get("MAX_BROWSER_SESSIONS", "3"))
LLM_SEMAPHORE: int = int(os.environ.get("OASIS_SEMAPHORE", "10"))
