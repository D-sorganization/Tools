# ruff: noqa: E501
"""Tests for URDF Viewer web application API.

Uses FastAPI TestClient (via httpx) for integration testing of all endpoints.
Skips gracefully when run from CI root (where cors/shared deps may be missing).
"""

from __future__ import annotations

import pytest

# Path setup is handled by conftest.py — no sys.path manipulation needed here.

# Skip entire module if FastAPI app can't be imported/initialized
# (CI may lack python-multipart, cors deps, etc.)
try:
    from app import app
except (
    Exception
) as _exc:  # noqa: BLE001 — CI may lack optional deps; skip entire module
    pytest.skip(
        f"Skipping urdf_viewer tests — app import failed: {_exc}",
        allow_module_level=True,
    )

from fastapi.testclient import TestClient


@pytest.fixture()
def client():  # type: ignore[no-untyped-def]
    """Create a FastAPI TestClient for testing."""
    return TestClient(app)


class TestStaticRoutes:
    """Test static file serving."""

    def test_root_returns_html(self, client) -> None:  # type: ignore[no-untyped-def]
        """GET / should return the viewer page."""
        response = client.get("/")
        assert response.status_code == 200


class TestModelsAPI:
    """Test model CRUD endpoints."""

    def test_list_models_returns_list(self, client) -> None:  # type: ignore[no-untyped-def]
        """GET /api/models should return a list."""
        response = client.get("/api/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], list)

    def test_get_nonexistent_model_returns_404(self, client) -> None:  # type: ignore[no-untyped-def]
        """GET /api/models/nonexistent.urdf should return 404."""
        response = client.get("/api/models/nonexistent.urdf")
        assert response.status_code == 404


class TestGenerateAPI:
    """Test URDF generation endpoint."""

    def test_generate_default_returns_xml(self, client) -> None:  # type: ignore[no-untyped-def]
        """POST /api/generate with defaults should return valid XML."""
        response = client.post("/api/generate", json={})
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/xml"
        assert response.text.startswith("<?xml")
        assert "<robot" in response.text

    def test_generate_custom_params(self, client) -> None:  # type: ignore[no-untyped-def]
        """POST /api/generate with custom params should succeed."""
        response = client.post(
            "/api/generate",
            json={
                "robot_name": "test_robot",
                "height_m": 1.80,
                "mass_kg": 80.0,
                "template": "Upper Body Only",
                "damping": 2.0,
            },
        )
        assert response.status_code == 200
        assert "test_robot" in response.text

    def test_generate_all_templates(self, client) -> None:  # type: ignore[no-untyped-def]
        """All templates should generate valid URDF."""
        templates_response = client.get("/api/templates")
        templates = templates_response.json()["templates"]

        for template in templates:
            response = client.post(
                "/api/generate",
                json={"template": template},
            )
            assert response.status_code == 200, f"Template '{template}' failed"
            assert "<robot" in response.text, f"Template '{template}' has no <robot>"

    def test_generate_invalid_height_rejected(self, client) -> None:  # type: ignore[no-untyped-def]
        """Negative height should be rejected by Pydantic validation."""
        response = client.post("/api/generate", json={"height_m": -1.0})
        assert response.status_code == 422


class TestPreviewAPI:
    """Test preview generation endpoint."""

    def test_preview_default(self, client) -> None:  # type: ignore[no-untyped-def]
        """POST /api/preview with defaults should return preview text."""
        response = client.post("/api/preview", json={})
        assert response.status_code == 200
        data = response.json()
        assert "preview" in data
        assert "Model Structure Preview" in data["preview"]

    def test_preview_contains_params(self, client) -> None:  # type: ignore[no-untyped-def]
        """Preview should reflect the requested parameters."""
        response = client.post(
            "/api/preview",
            json={"robot_name": "my_bot", "height_m": 2.0},
        )
        data = response.json()
        assert "my_bot" in data["preview"]
        assert "2.00" in data["preview"]


class TestTemplatesAPI:
    """Test templates listing endpoint."""

    def test_list_templates(self, client) -> None:  # type: ignore[no-untyped-def]
        """GET /api/templates should return template names."""
        response = client.get("/api/templates")
        assert response.status_code == 200
        data = response.json()
        assert "templates" in data
        assert "Full Humanoid" in data["templates"]
        assert len(data["templates"]) >= 5


class TestPathSafety:
    """Test path traversal protection."""

    def test_traversal_blocked(self, client) -> None:  # type: ignore[no-untyped-def]
        """Path traversal attempts should be blocked."""
        response = client.get("/api/models/../../etc/passwd")
        # Should either 404 or 400, but NOT serve the file
        assert response.status_code in (400, 403, 404)

    def test_upload_rejects_traversal_filename(self, client) -> None:  # type: ignore[no-untyped-def]
        """Upload should reject traversal filenames."""
        response = client.post(
            "/api/upload",
            files={"file": ("../../etc/passwd", b"<robot />")},
        )
        assert response.status_code == 400

    def test_upload_rejects_path_separators(self, client) -> None:  # type: ignore[no-untyped-def]
        """Upload should reject filename paths with separators."""
        response = client.post(
            "/api/upload",
            files={"file": ("nested/robot.urdf", b"<robot />")},
        )
        assert response.status_code == 400
