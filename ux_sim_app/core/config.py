"""Central configuration for the UX Simulation App."""
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the app directory or parent
_env = Path(__file__).parent.parent / ".env"
if _env.exists():
    load_dotenv(_env)

# ── API ────────────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL: str = "https://api.openai.com/v1"   # always direct
TEXT_MODEL: str = os.environ.get("DEFAULT_MODEL", "gpt-4o-mini")
VISION_MODEL: str = os.environ.get("VISION_MODEL", "gpt-4o")
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
