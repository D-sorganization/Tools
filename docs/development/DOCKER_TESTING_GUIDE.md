# Docker Testing & Validation Guide

Complete guide to testing the containerization setup. Run these commands in a Docker-enabled environment.

## Pre-Flight Checks

Verify Docker is installed and running:

```bash
docker --version
docker ps  # Should work without errors
```

---

## 1. Build Development Image

### Test Development Build

```bash
cd /home/user/Tools

# Build development image
docker build -f Dockerfile.dev -t tools:dev .

# Expected output:
# Successfully built <SHA>
# Successfully tagged tools:dev:latest

# Verify image was created
docker images | grep tools:dev
```

### Build Size Check

```bash
docker image inspect tools:dev --format='{{.Size}}' | awk '{print $1/1024/1024/1024 " GB"}'
# Expected: ~2GB (includes dev tools)
```

### Test Development Container

```bash
# Run interactive shell
docker run -it tools:dev python3

# Inside container, test imports:
>>> import flask
>>> import numpy
>>> import pytest
>>> print("All imports OK!")
>>> exit()
```

---

## 2. Build Production Image

### Test Production Build

```bash
# Build production image (multi-stage)
docker build -f Dockerfile.prod -t tools:prod .

# Expected output:
# [Stage 1] Building wheels...
# [Stage 2] Creating runtime image...
# Successfully built <SHA>
# Successfully tagged tools:prod:latest
```

### Production Image Size Check

```bash
docker image inspect tools:prod --format='{{.Size}}' | awk '{print $1/1024/1024 " MB"}'
# Expected: < 500 MB (typically 400-450 MB)

# Compare with development image
docker image inspect tools:dev --format='{{.Size}}' | awk '{print $1/1024/1024 " MB"}'
# Expected: ~2000 MB

# Show size difference
echo "Size difference:"
docker images --format "table {{.Repository}}\t{{.Size}}" | grep tools
```

### Test Production Container

```bash
# Run production container
docker run -p 5000:5000 tools:prod

# In another terminal, test health endpoints:
curl http://localhost:5000/api/health
curl http://localhost:5000/api/ready

# Expected responses:
# {
#   "status": "ok",
#   "service": "ud-tools",
#   "timestamp": "2026-04-30T12:34:56Z",
#   "version": "1.0.0"
# }
```

---

## 3. Test docker-compose Stack

### Start All Services

```bash
cd /home/user/Tools

# Start services (all three: tools, postgres, redis)
docker-compose up

# Expected output:
# tools-dev created
# tools-postgres created
# tools-redis created
# [All services starting with health checks...]
```

### Check Service Health

```bash
# In another terminal:
docker-compose ps
# Expected: all services in "running" state

# Check logs
docker-compose logs -f tools

# Test health endpoints
curl http://localhost:5000/api/health
curl http://localhost:5000/api/ready

# Test database connectivity (from container)
docker-compose exec tools python3 -c "import psycopg2; print('PostgreSQL available')"

# Test redis connectivity (from container)
docker-compose exec tools python3 -c "import redis; r = redis.Redis(host='redis'); print('Redis available')"
```

### Stop Services

```bash
# Stop gracefully
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

### Test Individual Services

```bash
# Start only tools and postgres (no redis)
docker-compose up tools postgres

# Start with test runner (requires --profile tests)
docker-compose --profile tests up tests
```

---

## 4. Test Live Code Editing (Development)

### Verify Volume Mounts

```bash
# Start development service
docker-compose up -d tools

# Check mounted volumes
docker inspect tools-dev | grep -A 10 "Mounts"

# Expected: /workspace mounted from current directory
```

### Test Live Reload

```bash
# 1. Start services
docker-compose up

# 2. Make a change to a file in your project
echo "# test comment" >> src/web_applications/__init__.py

# 3. Container should detect the change via volume mount
# 4. Flask auto-reload should restart the server

# 5. Check logs for "Restarting with reloader"
docker-compose logs tools | tail -20
```

### Test Code Execution in Container

```bash
# Execute Python in running container
docker-compose exec tools python3 -c "import sys; print(sys.path)"

# Run tests in container
docker-compose exec tools pytest tests/test_health_checks.py -v

# Run linting
docker-compose exec tools python3 -m ruff check src/
docker-compose exec tools python3 -m ruff format --check src/
```

---

## 5. Test Health Check Endpoints

### Manual Health Checks

```bash
# Start production container
docker run -d -p 5000:5000 --name tools-test tools:prod
sleep 5

# Test liveness probe
curl -v http://localhost:5000/api/health
# Expected: 200 OK

# Test readiness probe
curl -v http://localhost:5000/api/ready
# Expected: 200 OK (if ready) or 503 Service Unavailable

# View full response
curl -s http://localhost:5000/api/ready | python3 -m json.tool

# Cleanup
docker stop tools-test
docker rm tools-test
```

### Automated Health Check Testing

```bash
# Run test suite
docker run --rm \
  -v $(pwd):/workspace \
  tools:dev \
  pytest tests/test_health_checks.py -v
```

---

## 6. Network Testing

### Test Inter-Service Communication

```bash
# Start services
docker-compose up -d

# From tools container, ping other services
docker-compose exec tools ping postgres
docker-compose exec tools ping redis

# Test DNS resolution
docker-compose exec tools getent hosts postgres
docker-compose exec tools getent hosts redis

# Test port connectivity
docker-compose exec tools python3 -c \
  "import socket; s = socket.socket(); s.connect(('postgres', 5432)); print('PostgreSQL: OK')"
```

### Network Inspection

```bash
# View network
docker network ls | grep tools

# Inspect network details
docker network inspect $(docker network ls -q -f name=tools)

# Expected: all services connected on tools-network
```

---

## 7. Stress Testing

### Memory and CPU Limits

```bash
# Run with resource limits
docker run -m 1g --cpus 1 -p 5000:5000 tools:prod

# Monitor resource usage
docker stats
```

### Concurrent Requests

```bash
# Start container
docker run -d -p 5000:5000 --name tools-stress tools:prod
sleep 5

# Send concurrent health check requests
for i in {1..100}; do
  curl -s http://localhost:5000/api/health &
done
wait

# Check container health
docker stats tools-stress --no-stream

docker stop tools-stress
docker rm tools-stress
```

---

## 8. Log Collection

### View Logs

```bash
# Development (via docker-compose)
docker-compose logs tools

# Production (specific container)
docker logs <container-id>

# Follow logs in real-time
docker-compose logs -f tools

# Show last 100 lines
docker logs --tail 100 <container-id>
```

### Log Integration Testing

```bash
# Verify logging works correctly
docker run -it tools:prod flask shell
>>> import logging
>>> logger = logging.getLogger("test")
>>> logger.info("Test message")
>>> exit()

# Logs should appear in docker logs output
```

---

## 9. Security Scanning

### Container Layer Analysis

```bash
# Inspect image layers
docker history tools:prod

# Expected: production image has fewer, smaller layers than dev

# Check non-root user
docker run --rm tools:prod id
# Expected output: uid=1000(appuser) gid=1000(appuser)

docker run --rm tools:dev id
# Expected output: uid=1000(devuser) gid=1000(devuser)
```

### Vulnerability Scanning (if Trivy installed)

```bash
# Install Trivy: https://github.com/aquasecurity/trivy

# Scan images
trivy image tools:dev
trivy image tools:prod

# Expected: minimal vulnerabilities in production image
```

---

## 10. Performance Benchmarks

### Build Time

```bash
# Time development build
time docker build -f Dockerfile.dev -t tools:dev .

# Time production build
time docker build -f Dockerfile.prod -t tools:prod .

# Expected:
# - Dev: 3-5 minutes
# - Prod: 2-3 minutes
```

### Startup Time

```bash
# Measure container startup
docker run --rm -p 5000:5000 tools:prod &
PID=$!
time (sleep 2 && curl http://localhost:5000/api/health)
kill $PID

# Expected: < 2 seconds for health check to respond
```

### Request Performance

```bash
# Start container
docker run -d -p 5000:5000 --name tools-perf tools:prod
sleep 2

# Benchmark health checks
ab -n 1000 -c 10 http://localhost:5000/api/health/

# Expected: > 100 requests/second

docker stop tools-perf
docker rm tools-perf
```

---

## Troubleshooting

### Common Issues

**Build fails with "pip: command not found":**

```bash
# Ensure base image has pip installed
# Check if using python:3.11-slim (should have pip)
# May need to install: RUN apt-get install -y python3-pip
```

**Container exits immediately:**

```bash
# Check logs
docker logs <container-id>

# Run with entrypoint override
docker run -it tools:prod /bin/bash
# Now debug interactively
```

**Port 5000 already in use:**

```bash
# Find process using port 5000
lsof -i :5000

# Run on different port
docker run -p 5001:5000 tools:prod
```

**Volume mount issues (docker-compose):**

```bash
# Verify mount
docker-compose exec tools mount | grep workspace

# Rebuild without cache
docker-compose down -v
docker-compose build --no-cache
docker-compose up
```

---

## Test Checklist

- [ ] Development image builds successfully
- [ ] Development image < 2.5GB
- [ ] Production image builds successfully
- [ ] Production image < 500MB
- [ ] Production image 50-60% smaller than dev
- [ ] Both images run without errors
- [ ] Health endpoint returns 200
- [ ] Ready endpoint returns 200 or 503
- [ ] docker-compose stack starts all services
- [ ] Services communicate via network
- [ ] Live code editing works (development)
- [ ] Tests run inside container
- [ ] Non-root users configured
- [ ] Health checks have required fields
- [ ] Logs appear in docker logs/docker-compose logs

---

## Next Steps

Once all tests pass:

1. Push images to Docker Hub (Phase 3.2)
2. Create Kubernetes manifests (Phase 3.3)
3. Set up CI/CD with multi-platform builds (Phase 3.4)
4. Implement security scanning (Phase 3.5)
