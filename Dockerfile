# Dockerfile — Generic script runner for Tools
# Purpose: Run any Tools-based Python script in a reproducible environment.
# This image installs the Tools library and its dependencies, then runs
# whatever script or command you pass in via CMD or docker run <image> <cmd>.
#
# Usage:
#   # Build
#   docker build -t tools:latest .
#
#   # Run a script
#   docker run --rm -v $(pwd)/my_script.py:/workspace/my_script.py tools:latest python3 my_script.py
#
#   # Interactive shell
#   docker run --rm -it tools:latest python3
#
# See docs/deployment.md for full containerization documentation.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifests first to leverage Docker layer cache
COPY requirements.txt pyproject.toml setup.py ./
RUN pip install --upgrade pip setuptools wheel && \
    pip install -e ".[all]"

# Copy library source
COPY src/ ./src/
RUN pip install -e .

# Run as non-root
RUN useradd -m -s /bin/bash appuser && chown -R appuser:appuser /workspace
USER appuser

# Default: interactive Python so the image is immediately useful
CMD ["python3"]
