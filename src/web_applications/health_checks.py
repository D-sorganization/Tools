"""health_checks.py — Health and readiness check endpoints for containerization.

Provides /api/health (liveness) and /api/ready (readiness) endpoints for
Kubernetes and container orchestration platforms.

Status Codes:
- 200 OK: Service is healthy and ready
- 503 Service Unavailable: Service is unhealthy or not ready
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


def get_health_status() -> dict[str, Any]:
    """Get current health status of the application.

    Returns liveness probe data:
    - status: "ok" if service is running
    - timestamp: ISO 8601 timestamp
    - uptime: Seconds since process start (simplified)
    """
    return {
        "status": "ok",
        "service": "ud-tools",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "version": os.getenv("APP_VERSION", "1.0.0"),
    }


def get_readiness_status() -> dict[str, Any]:
    """Get readiness status with dependency checks.

    Returns readiness probe data with dependency health:
    - status: "ready" if all critical dependencies are available
    - checks: Dictionary of dependency health checks
    - ready: Boolean indicating if service is ready to serve traffic

    Checks include:
    - python: Python interpreter available
    - packages: Required packages importable
    - disk_space: Disk space available (> 100MB)
    - memory: Process memory usage reasonable (< 1GB)
    """
    checks = {}

    # Python interpreter check
    checks["python"] = {
        "healthy": True,
        "version": f"{sys.version_info.major}.{sys.version_info.minor}",
    }

    # Package imports check
    try:
        import flask
        import numpy

        checks["packages"] = {
            "healthy": True,
            "flask": flask.__version__,
            "numpy": numpy.__version__,
        }
    except ImportError as e:
        checks["packages"] = {"healthy": False, "error": str(e)}

    # Disk space check (simplified)
    try:
        import shutil

        total, used, free = shutil.disk_usage("/")
        checks["disk"] = {
            "healthy": free > 100 * 1024 * 1024,  # > 100MB
            "free_mb": free // (1024 * 1024),
            "free_pct": (free / total * 100) if total > 0 else 0,
        }
    except Exception as e:
        checks["disk"] = {"healthy": False, "error": str(e)}

    # Memory usage check (simplified)
    try:
        import psutil

        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)
        checks["memory"] = {
            "healthy": memory_mb < 1024,  # < 1GB
            "usage_mb": round(memory_mb, 2),
        }
    except ImportError:
        # psutil not available, skip check
        checks["memory"] = {"healthy": True, "skipped": "psutil not installed"}
    except Exception as e:
        checks["memory"] = {"healthy": False, "error": str(e)}

    # Determine overall readiness
    ready = all(check.get("healthy", True) for check in checks.values())

    return {
        "status": "ready" if ready else "not_ready",
        "ready": ready,
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


def register_health_endpoints(app: Any) -> None:
    """Register health check endpoints on Flask app.

    Args:
        app: Flask application instance

    Endpoints:
        GET /api/health - Liveness probe (is service running?)
        GET /api/ready - Readiness probe (is service ready to serve traffic?)
    """
    from flask import jsonify

    @app.get("/api/health")
    def health() -> tuple[dict[str, Any], int]:
        """Liveness probe endpoint.

        Returns 200 if the service is running.
        Use for container restart decisions (Kubernetes livenessProbe).
        """
        try:
            status = get_health_status()
            return jsonify(status), 200
        except Exception as e:
            logger.exception("Health check failed")
            return jsonify({"status": "error", "error": str(e)}), 503

    @app.get("/api/ready")
    def ready() -> tuple[dict[str, Any], int]:
        """Readiness probe endpoint.

        Returns 200 if the service is ready to serve traffic.
        Returns 503 if dependencies are unavailable.
        Use for traffic routing decisions (Kubernetes readinessProbe).
        """
        try:
            status = get_readiness_status()
            status_code = 200 if status["ready"] else 503
            return jsonify(status), status_code
        except Exception as e:
            logger.exception("Readiness check failed")
            return jsonify({"status": "error", "error": str(e)}), 503
