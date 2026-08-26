FROM python:3.13-slim

WORKDIR /app

# Runtime dependencies, EXACTLY pinned.
#
# This list mirrors the `p1am` extra in the repository's pyproject.toml, which
# is the single source of truth; the Docker build context is ./backend and so
# cannot see pyproject.toml. Drift between the two fails
# backend/tests/test_deployment_hardening.py, so add packages there first.
#
# Previously this installed six unpinned packages and omitted pydantic-settings
# (imported by settings.py) and python-multipart (required by main.py's
# File(...) upload), so the app raised ModuleNotFoundError at import and
# `restart: always` turned that into a silent crash loop (issue #4014).
RUN pip install --no-cache-dir \
    fastapi==0.136.1 \
    uvicorn[standard]==0.47.0 \
    pydantic==2.13.4 \
    pydantic-settings==2.14.2 \
    sqlmodel==0.0.38 \
    pymodbus==3.13.0 \
    websockets==16.0 \
    python-multipart==0.0.29 \
    numpy==2.3.5

# Copy source code to working directory
COPY . /app

# Expose FastAPI SCADA port
EXPOSE 8000

# Bind all interfaces INSIDE the container; isolate at the publish layer.
#
# 127.0.0.1 inside a container is unreachable from everywhere — including the
# published port and the frontend container's `proxy_pass http://backend:8000`
# — so the previous default (issue #3289) made the whole stack non-functional
# rather than safe (issue #4014). The isolation it was reaching for lives in
# docker-compose.yml, which publishes this port as `127.0.0.1:8000:8000` so the
# host's loopback is the only way in.
ENV P1AM_BIND_HOST=0.0.0.0
ENV P1AM_BIND_PORT=8000

# The historian SQLite file is created relative to the process working
# directory (see database.py's DB_FILE). Run from /data — the mounted volume —
# so the database survives rebuilds AND the volume is not mounted over the
# source tree this image just copied into /app (issue #4030).
ENV PYTHONPATH=/app
RUN mkdir -p /data

# Execute server
CMD ["sh", "-c", "cd /data && exec uvicorn main:app --host \"$P1AM_BIND_HOST\" --port \"$P1AM_BIND_PORT\""]
