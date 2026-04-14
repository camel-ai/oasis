"""Central configuration for the UX Simulation App.

Per-model provider model:
  TEXT_PROVIDER  = "openai" | "openrouter" | "custom"
  VISION_PROVIDER = "openai" | "openrouter" | "custom"

Each provider has its own API key and base URL.
Model slots are also independently configurable:
  TEXT_MODEL   – persona generation, content simulation, UX analysis, browser analysis
  VISION_MODEL – Mode 3 visual simulation and redesign screenshot analysis
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

# ── API Keys ───────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
OPENROUTER_API_KEY: str = os.environ.get("OPENROUTER_API_KEY", "")
CUSTOM_API_KEY: str = os.environ.get("CUSTOM_API_KEY", "")
CUSTOM_BASE_URL: str = os.environ.get("CUSTOM_BASE_URL", "")
CUSTOM_VISION_API_KEY: str = os.environ.get("CUSTOM_VISION_API_KEY", "")
CUSTOM_VISION_BASE_URL: str = os.environ.get("CUSTOM_VISION_BASE_URL", "")

# ── Per-model provider selection ───────────────────────────────────────────────
# TEXT_PROVIDER: provider used for all text/chat tasks
TEXT_PROVIDER: str = os.environ.get("TEXT_PROVIDER", os.environ.get("PROVIDER", "openai")).lower()
# VISION_PROVIDER: provider used for vision/image tasks (Mode 3, redesign)
VISION_PROVIDER: str = os.environ.get("VISION_PROVIDER", TEXT_PROVIDER).lower()

_OPENROUTER_BASE = "https://openrouter.ai/api/v1"
_OPENAI_BASE = "https://api.openai.com/v1"

def _resolve(provider: str, custom_key: str = "", custom_url: str = "") -> tuple[str, str]:
    """Return (api_key, base_url) for the given provider string."""
    if provider == "openrouter":
        return OPENROUTER_API_KEY, _OPENROUTER_BASE
    if provider == "custom":
        return (custom_key or OPENAI_API_KEY), (custom_url or _OPENAI_BASE)
    return OPENAI_API_KEY, _OPENAI_BASE  # default: openai

# Resolved credentials for text tasks
EFFECTIVE_TEXT_API_KEY: str
EFFECTIVE_TEXT_BASE_URL: str
EFFECTIVE_TEXT_API_KEY, EFFECTIVE_TEXT_BASE_URL = _resolve(
    TEXT_PROVIDER, CUSTOM_API_KEY, CUSTOM_BASE_URL
)

# Resolved credentials for vision tasks
EFFECTIVE_VISION_API_KEY: str
EFFECTIVE_VISION_BASE_URL: str
EFFECTIVE_VISION_API_KEY, EFFECTIVE_VISION_BASE_URL = _resolve(
    VISION_PROVIDER, CUSTOM_VISION_API_KEY, CUSTOM_VISION_BASE_URL
)

# Legacy aliases (used by older imports)
EFFECTIVE_API_KEY: str = EFFECTIVE_TEXT_API_KEY
EFFECTIVE_BASE_URL: str = EFFECTIVE_TEXT_BASE_URL

# ── Model slots ────────────────────────────────────────────────────────────────
TEXT_MODEL: str = os.environ.get("TEXT_MODEL", os.environ.get("DEFAULT_MODEL", "gpt-4o-mini"))
VISION_MODEL: str = os.environ.get("VISION_MODEL", "gpt-4o")

# ── Model lists for Settings dropdowns ────────────────────────────────────────
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
