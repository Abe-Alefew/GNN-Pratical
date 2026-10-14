# --- Build stage ---
FROM python:3.11-slim AS base

# Non-root user for security
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

WORKDIR /app

# Install dependencies first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY configs/ configs/

# Copy model package (must be built before docker build)
# The package dir is set via environment variable
COPY model_packages/ model_packages/

# Environment
ENV MODEL_PACKAGE_DIR=/app/model_packages/latest
ENV PYTHONPATH=/app

# Switch to non-root
USER appuser

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start the API server
CMD ["uvicorn", "src.a4_mlops.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
