# Containerization Guide for UD Tools

Complete guide to building, running, and deploying the UD Tools project in Docker.

**Status**: Phase 3.1 — Containerization Foundation (Complete)

## Table of Contents

1. [Quick Start](#quick-start)
2. [Development Environment](#development-environment)
3. [Production Environment](#production-environment)
4. [Health Checks](#health-checks)
5. [Image Specifications](#image-specifications)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Build and Run Development Environment

```bash
# Build development image
docker build -f Dockerfile.dev -t tools:dev .

# Run interactive container
docker run -it -v $(pwd):/workspace tools:dev

# Or use docker-compose (recommended for multi-service setup)
docker-compose up
```

### Build and Run Production Image

```bash
# Build production image (multi-stage, optimized)
docker build -f Dockerfile.prod -t tools:prod .

# Run container
docker run -p 5000:5000 \
  -e FLASK_ENV=production \
  -e SECRET_KEY=your-production-secret-key \
  tools:prod
```

---

## Development Environment

### Using Dockerfile.dev

The development image includes:

- Python 3.11-slim base
- All dependencies (core + optional extras)
- Development tools (pytest, ruff, mypy)
- Non-root user (devuser)
- Live code editing via volume mounts

**Build:**

```bash
docker build -f Dockerfile.dev -t tools:dev .
```

**Run with Live Code Editing:**

```bash
docker run -it \
  -v $(pwd):/workspace \
  -p 5000:5000 \
  tools:dev \
  flask run --host=0.0.0.0
```

**Run Tests Inside Container:**

```bash
docker run --rm \
  -v $(pwd):/workspace \
  tools:dev \
  pytest -n auto --timeout=60
```

### Using docker-compose.yml

The compose file provides a complete development stack:

- **tools** — Main Flask application (with live code editing)
- **postgres** — PostgreSQL database (optional, pre-configured)
- **redis** — Redis cache (optional, pre-configured)
- **tests** — Test runner service (optional, use `--profile tests`)

**Start All Services:**

```bash
docker-compose up
```

**Start Only tools + postgres:**

```bash
docker-compose up tools postgres
```

**Run Tests:**

```bash
docker-compose --profile tests up tests
```

**View Logs:**

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f tools

# Follow specific container
docker-compose logs -f tools | grep "ERROR"
```

**Stop Services:**

```bash
docker-compose down

# With volume cleanup
docker-compose down -v
```

### Volume Mounts

The compose file excludes large/cache directories from live sync:

```yaml
volumes:
  - .:/workspace # Mount entire project
  - /workspace/.git # Exclude .git
  - /workspace/.pytest_cache # Exclude pytest cache
  - /workspace/.ruff_cache # Exclude ruff cache
```

This ensures fast live reload while avoiding sync overhead.

---

## Production Environment

### Using Dockerfile.prod

The production image is a multi-stage build optimized for:

- **Minimal size**: Multi-stage build, wheels cached, dev deps excluded
- **Security**: Non-root user, minimal attack surface
- **Reproducibility**: Exact dependency pinning, no build tools in runtime
- **Health checks**: Built-in readiness/liveness probes

**Build:**

```bash
docker build -f Dockerfile.prod -t tools:prod .
```

**Build with Custom Tag:**

```bash
docker build -f Dockerfile.prod -t docker.io/myorg/tools:1.0.0 .
```

**Stage 1 — Builder:**

- Python 3.11-slim
- Build tools (gcc, build-essential)
- Compiles all dependencies into wheels

**Stage 2 — Runtime:**

- Python 3.11-slim
- Only runtime libraries (no build tools)
- Pre-built wheels from Stage 1
- Non-root user `appuser`

**Image Size Target:**

- Development: ~2GB (includes dev tools)
- Production: < 500MB (minimal runtime only)

### Running Production Container

**Basic:**

```bash
docker run -p 5000:5000 tools:prod
```

**With Environment Variables:**

```bash
docker run -p 5000:5000 \
  -e FLASK_ENV=production \
  -e SECRET_KEY=$(openssl rand -hex 32) \
  -e DATABASE_URL=postgresql://user:pass@db:5432/tools \
  tools:prod
```

**With .env File:**

```bash
docker run -p 5000:5000 \
  --env-file .env.prod \
  tools:prod
```

**With Health Check Verification:**

```bash
docker run -p 5000:5000 \
  --health-interval=30s \
  --health-timeout=10s \
  --health-retries=3 \
  tools:prod
```

---

## Health Checks

The application provides Kubernetes-compatible health check endpoints:

### /api/health (Liveness Probe)

Indicates if the service is running.

```bash
curl http://localhost:5000/api/health
```

**Response (200 OK):**

```json
{
  "status": "ok",
  "service": "ud-tools",
  "timestamp": "2026-04-30T12:34:56Z",
  "version": "1.0.0"
}
```

**Use Case**: Kubernetes livenessProbe. If this fails, the container will be restarted.

### /api/ready (Readiness Probe)

Indicates if the service is ready to serve traffic.

```bash
curl http://localhost:5000/api/ready
```

**Response (200 OK, Ready):**

```json
{
  "status": "ready",
  "ready": true,
  "checks": {
    "python": {
      "healthy": true,
      "version": "3.11"
    },
    "packages": {
      "healthy": true,
      "flask": "3.0.0",
      "numpy": "2.0.1"
    },
    "disk": {
      "healthy": true,
      "free_mb": 5000,
      "free_pct": 45.2
    },
    "memory": {
      "healthy": true,
      "usage_mb": 123.45
    }
  },
  "timestamp": "2026-04-30T12:34:56Z"
}
```

**Response (503 Service Unavailable, Not Ready):**

```json
{
  "status": "not_ready",
  "ready": false,
  "checks": {
    "packages": {
      "healthy": false,
      "error": "No module named 'required_package'"
    }
  },
  "timestamp": "2026-04-30T12:34:56Z"
}
```

**Use Case**: Kubernetes readinessProbe. If this fails, traffic is routed away from the pod.

### Health Check Dependency Validation

The `/api/ready` endpoint checks:

- **Python**: Interpreter available and correct version
- **Packages**: Core dependencies (flask, numpy) importable
- **Disk Space**: > 100MB free
- **Memory Usage**: < 1GB per process

---

## Image Specifications

### Dockerfile.dev

| Component  | Details                                     |
| ---------- | ------------------------------------------- |
| Base Image | `python:3.11-slim`                          |
| User       | `devuser` (non-root)                        |
| Entrypoint | `python3`                                   |
| Size       | ~2GB                                        |
| Build Time | ~3-5 minutes (first build)                  |
| Volumes    | `/workspace` (project root)                 |
| Ports      | 5000 (Flask), 5432 (Postgres), 6379 (Redis) |
| Use Cases  | Development, testing, debugging             |

### Dockerfile.prod

| Component    | Details                            |
| ------------ | ---------------------------------- |
| Base Image   | `python:3.11-slim` (multi-stage)   |
| User         | `appuser` (non-root)               |
| Entrypoint   | `flask run --host=0.0.0.0`         |
| Size         | < 500MB                            |
| Build Time   | ~2-3 minutes                       |
| Volumes      | Read-only (no mount necessary)     |
| Ports        | 5000 (Flask)                       |
| Health Check | Built-in (30s interval, 3 retries) |
| Use Cases    | Production, staging, CI/CD         |

### docker-compose.yml

| Service    | Image              | Purpose                                  |
| ---------- | ------------------ | ---------------------------------------- |
| `tools`    | Dockerfile.dev     | Main application with live reload        |
| `postgres` | postgres:16-alpine | Database (optional, pre-configured)      |
| `redis`    | redis:7-alpine     | Cache (optional, pre-configured)         |
| `tests`    | Dockerfile.dev     | Run test suite (optional, profile=tests) |

---

## Best Practices

### 1. Build Optimization

**Use .dockerignore:**

Exclude large files from build context:

```bash
# Already provided: .dockerignore
# Reduces build context size by ~90%
```

**Layer Caching:**

- Copy dependency files first (rarely change)
- Copy source code last (changes frequently)
- This allows Docker to cache dependency installation

### 2. Security

**Non-Root Users:**

Both images run as non-root:

- Dockerfile.dev: `devuser`
- Dockerfile.prod: `appuser`

**Environment Variables:**

Never commit secrets in Dockerfiles. Use:

```bash
# Pass at runtime
docker run -e SECRET_KEY=xyz ...

# Or use .env file
docker run --env-file .env.prod ...
```

**Minimal Attack Surface:**

Production image excludes:

- Build tools (gcc, git, curl)
- Development packages (pytest, ruff, mypy)
- Source files outside src/

### 3. Logging

**Use stdout/stderr (12-factor app):**

```python
# Good: Logs appear in docker logs
import logging
logger = logging.getLogger(__name__)
logger.info("Application started")

# Avoid: File-based logging in containers
with open("/var/log/app.log", "w") as f:
    f.write("...")  # Don't do this
```

**View Logs:**

```bash
docker logs <container-id>
docker-compose logs -f tools
```

### 4. Environment Variables

**Development (.env):**

```bash
# Copy from .env.example
cp .env.example .env

# Edit as needed (safe test values only)
FLASK_ENV=development
SECRET_KEY=OWASP-TEST-SECRET-KEY-SAFE
```

**Production (.env.prod):**

```bash
FLASK_ENV=production
SECRET_KEY=$(openssl rand -hex 32)
DATABASE_URL=postgresql://user:pass@db:5432/tools
```

### 5. Resource Limits

**Development (no strict limits needed):**

```bash
docker run -m 4g --cpus 2 tools:dev
```

**Production (enforce limits):**

```bash
docker run -m 1g --cpus 1 \
  -e FLASK_ENV=production \
  tools:prod
```

**In docker-compose:**

```yaml
services:
  tools:
    deploy:
      resources:
        limits:
          cpus: "1"
          memory: 1G
        reservations:
          cpus: "0.5"
          memory: 512M
```

---

## Troubleshooting

### Build Failures

**`No such file or directory` during COPY:**

```bash
# Check Docker build context
docker build --debug ... 2>&1 | grep COPY

# Solution: Ensure files exist in project root
ls -la requirements.txt pyproject.toml setup.py
```

**`Permission denied` in docker-compose:**

```bash
# Fix: Use explicit ownership
docker-compose down -v
sudo chown -R $USER:$USER .
docker-compose up
```

### Runtime Issues

**Flask app doesn't start:**

```bash
docker run -it tools:prod flask shell
# Verify imports work interactively
```

**Health checks failing:**

```bash
# Check logs
docker logs <container-id>

# Test health endpoint manually
docker exec <container-id> curl http://localhost:5000/api/health

# Verify dependencies
docker exec <container-id> python3 -c "import flask; print(flask.__version__)"
```

**Volume mount issues (docker-compose):**

```bash
# Verify volume is mounted
docker exec tools-dev mount | grep workspace

# Clear caches and rebuild
docker-compose down -v
rm -rf .pytest_cache .ruff_cache .mypy_cache
docker-compose up --build
```

### Performance

**Slow builds:**

```bash
# Check build context size
du -sh .

# Ensure .dockerignore is comprehensive
wc -l .dockerignore  # Should be ~60+ lines

# Use BuildKit for faster builds
DOCKER_BUILDKIT=1 docker build -f Dockerfile.dev -t tools:dev .
```

**Slow runtime (development):**

```bash
# Check volume sync performance
time docker run -v $(pwd):/workspace tools:dev ls -lR /workspace >/dev/null

# For better performance on macOS/Windows, use rsync
docker run -v $(pwd):/workspace -v rsync_cache:/workspace/.rsync tools:dev
```

---

## Next Steps (Phase 3.2+)

- [ ] Docker Hub push scripts (`push-to-hub.sh`)
- [ ] Kubernetes manifests (pods, services, deployments)
- [ ] CI/CD integration (GitHub Actions, multi-platform builds)
- [ ] Security scanning (Trivy, Grype)
- [ ] Performance benchmarking (image size, startup time)
- [ ] ARM64 support (for Apple Silicon, AWS Graviton)
