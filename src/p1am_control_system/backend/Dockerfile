FROM python:3.13-slim

WORKDIR /app

# Install standard dependencies directly to keep container fast and lightweight
RUN pip install --no-cache-dir fastapi sqlmodel pymodbus uvicorn websockets

# Copy source code to working directory
COPY . /app

# Expose FastAPI SCADA port
EXPOSE 8000

# Bind to loopback by default (issue #3289). This is an industrial-control
# backend: binding to 0.0.0.0 exposes E-stop/tag-write/DB-wipe endpoints to the
# whole network. Override deliberately with `-e P1AM_BIND_HOST=0.0.0.0` only
# behind proper network isolation, and ensure P1AM_API_KEY is configured.
ENV P1AM_BIND_HOST=127.0.0.1
ENV P1AM_BIND_PORT=8000

# Execute server
CMD ["sh", "-c", "uvicorn main:app --host \"$P1AM_BIND_HOST\" --port \"$P1AM_BIND_PORT\""]
