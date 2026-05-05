# Docker Quick Reference Card

## Build & Run

### Development
```bash
# Build
docker build -f Dockerfile.dev -t tools:dev .

# Run interactive
docker run -it -v $(pwd):/workspace tools:dev

# Run with Flask
docker run -it -v $(pwd):/workspace -p 5000:5000 \
  tools:dev flask run --host=0.0.0.0
```

### Production
```bash
# Build
docker build -f Dockerfile.prod -t tools:prod .

# Run
docker run -p 5000:5000 \
  -e FLASK_ENV=production \
  -e SECRET_KEY=$(openssl rand -hex 32) \
  tools:prod
```

### docker-compose
```bash
# Start all services
docker-compose up

# Start specific services
docker-compose up tools postgres

# Start with test runner
docker-compose --profile tests up tests

# Stop and cleanup
docker-compose down -v

# View logs
docker-compose logs -f tools
```

---

## Health Checks

```bash
# Liveness probe (is service running?)
curl http://localhost:5000/api/health

# Readiness probe (is service ready?)
curl http://localhost:5000/api/ready

# Pretty-print response
curl -s http://localhost:5000/api/ready | python3 -m json.tool
```

---

## Common Tasks

### Run Tests
```bash
# Inside container
docker-compose exec tools pytest tests/test_health_checks.py -v

# Or standalone
docker run --rm -v $(pwd):/workspace tools:dev \
  pytest tests/test_health_checks.py -v
```

### Format/Lint Code
```bash
# Inside container
docker-compose exec tools python3 -m ruff format src/
docker-compose exec tools python3 -m ruff check src/

# Type checking
docker-compose exec tools python3 -m mypy src/
```

### Execute Python
```bash
# Interactive shell
docker run -it tools:prod python3

# One-liner
docker run --rm tools:prod python3 -c "import flask; print(flask.__version__)"
```

### View Logs
```bash
# Follow logs
docker logs -f <container-id>

# Last 100 lines
docker logs --tail 100 <container-id>

# With timestamps
docker logs -t <container-id>
```

---

## Image Info

### Size Check
```bash
# Development image
docker image inspect tools:dev --format='{{.Size}}' | awk '{print $1/1024/1024/1024 " GB"}'

# Production image
docker image inspect tools:prod --format='{{.Size}}' | awk '{print $1/1024/1024 " MB"}'

# List all
docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}" | grep tools
```

### Layers
```bash
docker history tools:dev
docker history tools:prod
```

### Inspect
```bash
docker inspect tools:dev
docker inspect <container-id>
```

---

## Cleanup

```bash
# Remove container
docker rm <container-id>

# Remove image
docker rmi tools:dev

# Stop all containers
docker stop $(docker ps -q)

# Remove unused images/volumes/networks
docker system prune -a --volumes
```

---

## Troubleshooting

### Build fails
```bash
# Check build output
docker build --no-cache -f Dockerfile.dev -t tools:dev .

# Verbose output
DOCKER_BUILDKIT=1 docker build --progress=plain -f Dockerfile.dev -t tools:dev .
```

### Container exits
```bash
# Check logs
docker logs <container-id>

# Run with /bin/bash to debug
docker run -it tools:prod /bin/bash
```

### Port in use
```bash
# Find process
lsof -i :5000

# Use different port
docker run -p 5001:5000 tools:prod
```

### Volume mount issues
```bash
# Check mount
docker inspect <container-id> | grep -A 10 "Mounts"

# Verify path exists
ls -la /home/user/Tools
```

---

## Resources

- **DOCKER.md** — Full documentation
- **DOCKER_TESTING_GUIDE.md** — Testing procedures
- **CONTAINERIZATION_PHASE_3.1_SUMMARY.md** — Complete phase summary
- **GitHub Issue #2415** — Containerization and deployment readiness

---

## Environment Variables

### Development
```
FLASK_ENV=development
FLASK_DEBUG=1
SECRET_KEY=OWASP-TEST-SECRET-KEY-SAFE-FOR-TESTING-ONLY
FLASK_APP=web_applications.calculator.webapp
```

### Production
```
FLASK_ENV=production
SECRET_KEY=<openssl rand -hex 32>
DATABASE_URL=postgresql://user:pass@db:5432/tools
REDIS_URL=redis://redis:6379/0
```

---

## Health Check Endpoints

### /api/health (200 OK)
```json
{
  "status": "ok",
  "service": "ud-tools",
  "timestamp": "2026-04-30T12:34:56Z",
  "version": "1.0.0"
}
```

### /api/ready (200 OK or 503)
```json
{
  "status": "ready",
  "ready": true,
  "checks": {
    "python": {"healthy": true, "version": "3.11"},
    "packages": {"healthy": true, "flask": "3.0.0"},
    "disk": {"healthy": true, "free_mb": 5000},
    "memory": {"healthy": true, "usage_mb": 123}
  }
}
```

---

**Created**: 2026-04-30 | **Phase**: 3.1 | **Status**: Complete
