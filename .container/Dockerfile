FROM python:3.10-bookworm

# Install uv and common tools
RUN apt-get update && apt-get install -y curl git && \
    pip install uv && \
    rm -rf /var/lib/apt/lists/*

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

# Keep container alive for development
CMD ["tail", "-f", "/dev/null"]
