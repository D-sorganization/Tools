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


def test_reference_frame_twist_conversion_identity() -> None:
    """Identity transform should preserve the input twist."""
    response = client.post(
        "/api/calc/rotation-converter/reference-frame",
        json={
            "operation": "twist_frame_conversion",
            "transform": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "twist": [0.1, 0.2, 0.3, 1.0, 2.0, 3.0],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["operation"] == "twist_frame_conversion"
    assert data["results"]["output_twist"] == [0.1, 0.2, 0.3, 1.0, 2.0, 3.0]


def test_reference_frame_homogeneous_transform() -> None:
    """Homogeneous construction endpoint should emit a valid 4x4 transform."""
    response = client.post(
        "/api/calc/rotation-converter/reference-frame",
        json={
            "operation": "homogeneous_transform",
            "rotation_matrix": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            "translation": [1.0, 2.0, 3.0],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["operation"] == "homogeneous_transform"
    assert data["results"]["homogeneous_transform"][0][3] == 1.0
    assert data["results"]["homogeneous_transform"][1][3] == 2.0
    assert data["results"]["homogeneous_transform"][2][3] == 3.0
    assert data["results"]["homogeneous_transform"][3] == [0.0, 0.0, 0.0, 1.0]


def test_reference_frame_so3_so3_maps() -> None:
    """so(3) mapping endpoint should produce exp/log compatible outputs."""
    response = client.post(
        "/api/calc/rotation-converter/reference-frame",
        json={
            "operation": "so3_so3_maps",
            "so3_vector": [0.0, 0.0, 0.5],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["operation"] == "so3_so3_maps"
    assert len(data["results"]["so3_hat_matrix"]) == 3
    assert len(data["results"]["so3_exponential_SO3"]) == 3


def test_reference_frame_homogeneous_transform_rejects_twist_payload() -> None:
    """Homogeneous mode should reject fields belonging to other operations."""
    response = client.post(
        "/api/calc/rotation-converter/reference-frame",
        json={
            "operation": "homogeneous_transform",
            "rotation_matrix": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            "translation": [0.0, 0.0, 0.0],
            "twist": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
    )
    assert response.status_code == 422


def test_reference_frame_so3_map_requires_single_input_variant() -> None:
    """so(3)<->SO(3) mode should accept exactly one input source."""
    response = client.post(
        "/api/calc/rotation-converter/reference-frame",
        json={
            "operation": "so3_so3_maps",
            "so3_vector": [0.0, 0.0, 0.5],
            "so3_matrix": [
                [0.0, -0.5, 0.0],
                [0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
        },
    )
    assert response.status_code == 422
