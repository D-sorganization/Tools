"""Integration tests for the split model_generation REST API adapters."""

from __future__ import annotations

from pathlib import Path

import pytest

Flask = pytest.importorskip("flask").Flask
FastAPI = pytest.importorskip("fastapi").FastAPI
TestClient = pytest.importorskip("fastapi.testclient").TestClient

from model_generation.api import FastAPIAdapter, FlaskAdapter, ModelGenerationAPI

SIMPLE_URDF = """<?xml version="1.0"?>
<robot name="test_robot">
    <link name="base_link">
        <inertial>
            <mass value="1.0"/>
            <inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/>
        </inertial>
    </link>
</robot>
"""


def _nonblank_line_count(path: Path) -> int:
    return sum(
        1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    )


def test_rest_api_is_a_small_compatibility_shim(repo_root: Path) -> None:
    rest_api = (
        repo_root
        / "src"
        / "shared"
        / "python"
        / "model_generation"
        / "api"
        / "rest_api.py"
    )
    assert _nonblank_line_count(rest_api) <= 30
    content = rest_api.read_text(encoding="utf-8")
    assert "rest_api_routes" in content
    assert "rest_api_flask" in content
    assert "rest_api_fastapi" in content


def test_flask_adapter_registers_and_serves_health() -> None:
    app = Flask(__name__)
    FlaskAdapter(ModelGenerationAPI()).register(app)

    client = app.test_client()
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.get_json()["status"] == "healthy"


def test_fastapi_adapter_registers_and_serves_validation() -> None:
    app = FastAPI()
    FastAPIAdapter(ModelGenerationAPI()).register(app)

    client = TestClient(app)
    response = client.post("/api/v1/validate", json={"content": SIMPLE_URDF})

    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is True
