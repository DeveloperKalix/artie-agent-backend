# syntax=docker/dockerfile:1
FROM python:3.13-slim

WORKDIR /app

# System dependencies — cached unless this layer changes.
# BuildKit cache mount keeps apt lists across rebuilds on the same host.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libpq-dev \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (cached unless pyproject.toml changes).
COPY pyproject.toml .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install .

# Copy application source last (most frequently changed layer).
COPY app/ ./app/

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
