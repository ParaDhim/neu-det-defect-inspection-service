# ==============================================================================
# Multi-stage Dockerfile for Steel Surface Defect Inspection Service
# Optimized for minimal footprint, non-root security, and ONNX Runtime CPU execution
# ==============================================================================

FROM python:3.11-slim as runtime

# Install system dependencies required for OpenCV and image operations
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root system user for container security
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

WORKDIR /app

# Copy dependency specifications
COPY requirements.txt .

# Install dependencies into system python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code, models, and reference baseline
COPY src/ /app/src/
COPY models/ /app/models/
COPY reports/ /app/reports/

# Grant permissions to non-root user
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER 10001

# Expose HTTP port
EXPOSE 8000

# Container Healthcheck targeting FastAPI /health probe
HEALTHCHECK --interval=20s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch production ASGI server
CMD ["uvicorn", "src.serve.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
