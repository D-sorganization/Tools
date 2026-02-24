"""Tests for the Rotation Converter API endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient
from shared.python.calc_backend.app import app

client = TestClient(app)


def test_rotation_converter_quaternion_post() -> None:
    """Test standard quaternion to euler conversion endpoint."""
    response = client.post(
        "/api/calc/rotation-converter",
        json={
            "type": "quaternion",
            "value": [1.0, 0.0, 0.0, 0.0],
            "euler_convention": "xyz",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "representations" in data
    reps = data["representations"]

    assert reps["quaternion"] == [1.0, 0.0, 0.0, 0.0]
    assert reps["euler"] == [0.0, 0.0, 0.0]
    assert reps["euler_convention"] == "xyz"
    assert reps["axis_angle"]["axis"] == [1.0, 0.0, 0.0]
    assert reps["axis_angle"]["angle"] == 0.0


def test_rotation_converter_euler_post() -> None:
    """Test euler input conversion."""
    response = client.post(
        "/api/calc/rotation-converter",
        json={
            "type": "euler",
            "value": [1.5707963267948966, 0.0, 0.0],
            "euler_convention": "xyz",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "representations" in data

    # 90 degrees around X translates to quat [0.707, 0.707, 0, 0]
    quat = data["representations"]["quaternion"]
    assert abs(quat[0] - 0.7071) < 1e-3
    assert abs(quat[1] - 0.7071) < 1e-3


def test_rotation_converter_invalid_type() -> None:
    """Test arbitrary bad input throws validation exception."""
    response = client.post(
        "/api/calc/rotation-converter",
        json={
            "type": "invalid_magic_type",
            "value": [1.0, 0.0, 0.0, 0.0],
        },
    )
    # Pydantic Literal validation should fail before the endpoint logic
    assert response.status_code == 422
