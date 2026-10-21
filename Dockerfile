# ==============================================================================
# Multi-stage Dockerfile for Steel Surface Defect Inspection Service
# Powered natively by Astral uv for ultra-fast, deterministic dependency resolution
# ==============================================================================

FROM python:3.11-slim as runtime

# Install system dependencies required for OpenCV and image operations
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy uv binary from official Astral image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Create non-root system user for container security
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

WORKDIR /app

# Copy pyproject.toml and uv.lock for deterministic caching
COPY pyproject.toml uv.lock ./

# Install dependencies using uv sync (frozen and no cache)
RUN uv sync --frozen --no-cache --no-dev

# Copy application source code, models, and baseline statistics
COPY src/ /app/src/
COPY models/ /app/models/
COPY reports/ /app/reports/

# Grant permissions to non-root user
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER 10001

# Place uv virtualenv on PATH
ENV PATH="/app/.venv/bin:$PATH"

# Expose HTTP port
EXPOSE 8000

# Container Healthcheck targeting FastAPI /health probe
HEALTHCHECK --interval=20s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch production ASGI server via uv run
CMD ["uv", "run", "uvicorn", "src.serve.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
