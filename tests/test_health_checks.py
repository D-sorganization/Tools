"""test_health_checks.py — Tests for health check endpoints.

Tests containerization health check endpoints:
- /api/health — Liveness probe
- /api/ready — Readiness probe
"""

from __future__ import annotations

import json

import pytest

from web_applications.calculator.webapp import create_app
from web_applications.health_checks import get_health_status, get_readiness_status


class TestHealthCheckFunctions:
    """Test health check status functions."""

    def test_get_health_status_returns_dict(self) -> None:
        """Health status should return dict with required fields."""
        status = get_health_status()
        assert isinstance(status, dict)
        assert status["status"] == "ok"
        assert "timestamp" in status
        assert "service" in status
        assert status["service"] == "ud-tools"

    def test_get_health_status_has_valid_timestamp(self) -> None:
        """Health status timestamp should be ISO 8601."""
        status = get_health_status()
        timestamp = status["timestamp"]
        # Should end with Z and contain T
        assert timestamp.endswith("Z")
        assert "T" in timestamp

    def test_get_readiness_status_returns_dict(self) -> None:
        """Readiness status should return dict with checks."""
        status = get_readiness_status()
        assert isinstance(status, dict)
        assert "checks" in status
        assert isinstance(status["checks"], dict)
        assert "ready" in status
        assert isinstance(status["ready"], bool)

    def test_get_readiness_status_includes_python_check(self) -> None:
        """Readiness status should include Python version check."""
        status = get_readiness_status()
        assert "python" in status["checks"]
        python_check = status["checks"]["python"]
        assert python_check["healthy"] is True
        assert "version" in python_check

    def test_get_readiness_status_includes_packages_check(self) -> None:
        """Readiness status should include package import checks."""
        status = get_readiness_status()
        assert "packages" in status["checks"]
        packages_check = status["checks"]["packages"]
        assert packages_check["healthy"] is True
        assert "flask" in packages_check
        assert "numpy" in packages_check

    def test_get_readiness_status_includes_disk_check(self) -> None:
        """Readiness status should include disk space check."""
        status = get_readiness_status()
        assert "disk" in status["checks"]
        disk_check = status["checks"]["disk"]
        assert "healthy" in disk_check
        assert "free_mb" in disk_check
        assert "free_pct" in disk_check

    def test_get_readiness_status_includes_memory_check(self) -> None:
        """Readiness status should include memory usage check."""
        status = get_readiness_status()
        assert "memory" in status["checks"]
        memory_check = status["checks"]["memory"]
        assert "healthy" in memory_check


class TestHealthCheckEndpoints:
    """Test health check Flask endpoints."""

    @pytest.fixture
    def app(self):
        """Create test Flask app."""
        app = create_app()
        app.config["TESTING"] = True
        return app

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return app.test_client()

    def test_health_endpoint_returns_200(self, client) -> None:
        """Health endpoint should return 200 OK."""
        response = client.get("/api/health")
        assert response.status_code == 200

    def test_health_endpoint_returns_json(self, client) -> None:
        """Health endpoint should return JSON."""
        response = client.get("/api/health")
        assert response.content_type == "application/json"
        data = json.loads(response.data)
        assert data["status"] == "ok"

    def test_health_endpoint_has_required_fields(self, client) -> None:
        """Health endpoint response should have required fields."""
        response = client.get("/api/health")
        data = json.loads(response.data)
        assert "status" in data
        assert "service" in data
        assert "timestamp" in data

    def test_ready_endpoint_returns_200_or_503(self, client) -> None:
        """Ready endpoint should return 200 or 503."""
        response = client.get("/api/ready")
        assert response.status_code in (200, 503)

    def test_ready_endpoint_returns_json(self, client) -> None:
        """Ready endpoint should return JSON."""
        response = client.get("/api/ready")
        assert response.content_type == "application/json"
        data = json.loads(response.data)
        assert "status" in data
        assert "ready" in data

    def test_ready_endpoint_has_checks(self, client) -> None:
        """Ready endpoint should include dependency checks."""
        response = client.get("/api/ready")
        data = json.loads(response.data)
        assert "checks" in data
        assert isinstance(data["checks"], dict)
        assert len(data["checks"]) > 0

    def test_ready_endpoint_check_keys_are_consistent(self, client) -> None:
        """Ready endpoint checks should have consistent structure."""
        response = client.get("/api/ready")
        data = json.loads(response.data)
        for check_name, check_value in data["checks"].items():
            assert isinstance(check_value, dict)
            assert "healthy" in check_value

    def test_health_endpoint_accessible(self, client) -> None:
        """Health endpoint should be accessible from root."""
        response = client.get("/api/health")
        assert response.status_code == 200

    def test_ready_endpoint_accessible(self, client) -> None:
        """Ready endpoint should be accessible from root."""
        response = client.get("/api/ready")
        assert response.status_code in (200, 503)


@pytest.mark.unit
class TestHealthCheckIntegration:
    """Integration tests for health checks in Flask app."""

    @pytest.fixture
    def app(self):
        """Create test Flask app."""
        app = create_app()
        app.config["TESTING"] = True
        return app

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return app.test_client()

    def test_both_endpoints_accessible_from_app(self, client) -> None:
        """Both health endpoints should be registered in app."""
        health = client.get("/api/health")
        ready = client.get("/api/ready")
        assert health.status_code == 200
        assert ready.status_code in (200, 503)

    def test_health_ready_response_format_matches_spec(self, client) -> None:
        """Responses should match containerization spec format."""
        health = client.get("/api/health")
        health_data = json.loads(health.data)

        ready = client.get("/api/ready")
        ready_data = json.loads(ready.data)

        # Health response format
        assert health_data["status"] == "ok"
        assert "timestamp" in health_data

        # Ready response format
        assert ready_data["status"] in ("ready", "not_ready")
        assert "checks" in ready_data
        assert "ready" in ready_data
        assert isinstance(ready_data["ready"], bool)
