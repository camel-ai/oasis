FROM ghcr.io/astral-sh/uv:0.10.2 AS uv

FROM python:3.14-bookworm

# Install common system tools and uv.
RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /uvx /usr/local/bin/

WORKDIR /app/oasis

# Copy dependency metadata first so Docker can cache the environment layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project

# Copy the application and install it into the synced virtual environment.
COPY . .
RUN uv sync --locked

ENV PATH="/app/oasis/.venv/bin:$PATH"

# Keep container alive for development.
CMD ["tail", "-f", "/dev/null"]
