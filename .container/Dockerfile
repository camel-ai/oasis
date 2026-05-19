FROM python:3.10-bookworm

# ── System packages ────────────────────────────────────────────────────────────
# Font packages are required for CloakBrowser to pass Kasada/Akamai canvas
# fingerprint checks on headless Linux. Without them, canvas pixel hashes
# don't match any real browser and anti-bot systems flag the session.
RUN apt-get update && apt-get install -y \
    curl \
    git \
    # Anti-bot canvas fingerprint fonts (required for CloakBrowser on Linux)
    fonts-noto-color-emoji \
    fonts-freefont-ttf \
    fonts-unifont \
    fonts-ipafont-gothic \
    fonts-wqy-zenhei \
    fonts-tlwg-loma-otf \
    # Chromium runtime dependencies
    libnss3 \
    libatk-bridge2.0-0 \
    libdrm2 \
    libxkbcommon0 \
    libgbm1 \
    libasound2 \
    && pip install uv \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app/oasis

# Copy code
COPY . .

# Set up venv + install dev deps
RUN uv venv .venv --python=3.10 && \
    . .venv/bin/activate && \
    uv pip install -e ".[dev]" && \
    pip install pre-commit mypy && \
    pre-commit install

# ── UX Sim App dependencies ────────────────────────────────────────────────────
RUN . .venv/bin/activate && \
    pip install -r ux_sim_app/requirements.txt

# ── Browser binaries ───────────────────────────────────────────────────────────
# Pre-download both the standard Playwright Chromium (used by Marpit PDF export
# and the redesign screenshot capture) and the CloakBrowser stealth Chromium
# binary (~200MB). Pre-downloading here avoids a slow download at first runtime.
RUN . .venv/bin/activate && \
    playwright install chromium && \
    python -m cloakbrowser install

# ── Keep container alive for development ──────────────────────────────────────
CMD ["tail", "-f", "/dev/null"]
