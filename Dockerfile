# ============================================================
# Pilot Readiness Monitoring Framework — Dockerfile
# ============================================================
# Multi-stage build for minimal image size.
# Default: launches the real-time streaming demo on port 5000.
#
# Build:
#   docker build -t pilot-readiness .
#
# Run (streaming demo):
#   docker run --rm -p 5000:5000 pilot-readiness
#
# Run (full pipeline with data volume):
#   docker run --rm -v /path/to/Data:/app/Data pilot-readiness python main.py --dataset swell
#
# Run tests:
#   docker run --rm pilot-readiness python -m pytest tests/ -v
# ============================================================

FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements-docker.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-docker.txt

 ============================================================
# Production stage
# ============================================================
FROM python:3.11-slim

LABEL maintainer="Team 104"
LABEL description="Pilot Readiness Monitoring Framework — AI-powered stress detection and performance monitoring"
LABEL org.opencontainers.image.source="https://github.com/mohithbutta/Pilot-Readiness-AI-Framework"

WORKDIR /app

# Install runtime dependencies (libgomp needed by LightGBM)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY src/ ./src/
COPY tests/ ./tests/
COPY templates/ ./templates/
COPY config.py .
COPY main.py .
COPY streaming_demo.py .
COPY requirements.txt .
COPY requirements-docker.txt .
COPY README.md .

# Copy pre-trained models and cached features (if available)
COPY output/models/ ./output/models/
COPY output/features/ ./output/features/

# Create output directories
RUN mkdir -p output/plots output/edge output/experiments

# Expose streaming demo port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" || exit 1

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MPLBACKEND=Agg

# Default: launch streaming demo
CMD ["python", "streaming_demo.py", "--host", "0.0.0.0", "--port", "5000"]
