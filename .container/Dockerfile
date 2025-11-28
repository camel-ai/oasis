FROM python:3.10-bookworm

# Install system dependencies and Poetry
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/* && \
    pip install poetry

# Set workdir
WORKDIR /app/oasis

# Copy Poetry files and README first for better caching
COPY pyproject.toml README.md ./

# Configure Poetry
RUN poetry config virtualenvs.create false

# Generate new lock file and install dependencies without the root package
RUN poetry lock && poetry install --with dev --no-root --no-interaction --no-ansi

# Copy remaining source code
COPY oasis/ ./oasis/

# Install the root package now that source is available
RUN poetry install --only-root --no-interaction --no-ansi

# Install pre-commit hooks (development)
RUN poetry run pre-commit install || true

# Set environment variables
ENV PYTHONPATH=/app/oasis
ENV PYTHONUNBUFFERED=1

# Keep container alive for development
CMD ["tail", "-f", "/dev/null"]
