from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
from model_generation.api.rest_api_routes import (
    ALLOWED_MESH_SUFFIXES,
    MAX_MESH_UPLOAD_BYTES,
    ModelGenerationAPI,
)
from model_generation.api.rest_api_types import APIRequest, HTTPMethod

_ASCII_STL = b"solid mesh\nendsolid mesh\n"


class _FakeInertiaMesh:
    volume = 0.25
    center_mass = np.array([0.1, 0.2, 0.3])

    def __init__(self) -> None:
        self.density = 1.0

    @property
    def mass(self) -> float:
        return self.volume * self.density

    @property
    def moment_inertia(self) -> np.ndarray:
        return np.eye(3) * self.mass


def _mesh_request(mesh: bytes, **body: object) -> APIRequest:
    return APIRequest(
        method=HTTPMethod.POST,
        path="/api/v1/inertia/from-mesh",
        body={"mass": 1.0, "filename": "mesh.stl", **body},
        files={"mesh": mesh},
    )


def test_max_upload_bytes_is_50_mib() -> None:
    """Constant must equal exactly 50 MiB (issue #2479 requirement)."""
    assert MAX_MESH_UPLOAD_BYTES == 50 * 1024 * 1024


def test_mesh_upload_rejects_oversized_payload() -> None:
    """Payloads exceeding MAX_MESH_UPLOAD_BYTES must return HTTP 413."""
    api = ModelGenerationAPI()

    response = api.inertia_from_mesh(_mesh_request(b"0" * (MAX_MESH_UPLOAD_BYTES + 1)))

    assert response.status_code == 413
    assert "exceeds" in response.body["error"]


def test_mesh_upload_accepts_payload_at_size_limit() -> None:
    """Payload exactly at the limit must NOT be rejected by the size check."""
    api = ModelGenerationAPI()

    response = api.inertia_from_mesh(_mesh_request(b"0" * MAX_MESH_UPLOAD_BYTES))

    assert response.status_code != 413


def test_mesh_upload_rejects_unsupported_filename() -> None:
    api = ModelGenerationAPI()

    response = api.inertia_from_mesh(_mesh_request(_ASCII_STL, filename="mesh.exe"))

    assert response.status_code == 413
    assert "Unsupported mesh file type" in response.body["error"]


@pytest.mark.parametrize("suffix", sorted(ALLOWED_MESH_SUFFIXES))
def test_mesh_upload_allows_all_supported_extensions(suffix: str) -> None:
    """Every extension in ALLOWED_MESH_SUFFIXES must pass the filename check."""
    api = ModelGenerationAPI()

    response = api.inertia_from_mesh(
        _mesh_request(_ASCII_STL, filename=f"mesh{suffix}")
    )

    assert response.status_code != 413


def test_mesh_upload_cleans_temp_file_on_parser_failure(monkeypatch) -> None:
    """Temp file must be deleted even when the parser raises ValueError."""
    captured_path: dict[str, Path] = {}

    def load(path: str | Path) -> object:
        captured_path["path"] = Path(path)
        raise ValueError("bad mesh")

    monkeypatch.setitem(sys.modules, "trimesh", types.SimpleNamespace(load=load))
    api = ModelGenerationAPI()

    response = api.inertia_from_mesh(_mesh_request(_ASCII_STL))

    assert response.status_code == 400
    assert "Mesh processing failed" in response.body["error"]
    assert captured_path["path"].exists() is False


def test_mesh_upload_cleans_temp_file_on_unexpected_exception(monkeypatch) -> None:
    """Temp file must be deleted when trimesh raises an unexpected RuntimeError."""
    captured_path: dict[str, Path] = {}

    def load(path: str | Path) -> object:
        captured_path["path"] = Path(path)
        raise RuntimeError("unexpected crash")

    monkeypatch.setitem(sys.modules, "trimesh", types.SimpleNamespace(load=load))
    api = ModelGenerationAPI()

    response = api.inertia_from_mesh(_mesh_request(_ASCII_STL))

    assert response.status_code == 400
    assert captured_path["path"].exists() is False


def test_mesh_upload_uses_correct_suffix_for_ply_file(monkeypatch) -> None:
    """Temp file suffix must match the uploaded filename extension (.ply)."""
    captured_path: dict[str, Path] = {}

    def load(path: str | Path) -> object:
        captured_path["path"] = Path(path)
        raise ValueError("stop early")

    monkeypatch.setitem(sys.modules, "trimesh", types.SimpleNamespace(load=load))
    api = ModelGenerationAPI()

    api.inertia_from_mesh(_mesh_request(b"ply\n", filename="model.ply"))

    assert captured_path["path"].suffix == ".ply"


def test_mesh_upload_density_path_returns_volume(monkeypatch) -> None:
    """Density-based inertia must return the mesh volume instead of crashing."""

    fake_mesh = _FakeInertiaMesh()

    def load(path: str | Path) -> _FakeInertiaMesh:
        assert Path(path).suffix == ".stl"
        return fake_mesh

    monkeypatch.setitem(sys.modules, "trimesh", types.SimpleNamespace(load=load))
    api = ModelGenerationAPI()
    request = APIRequest(
        method=HTTPMethod.POST,
        path="/api/v1/inertia/from-mesh",
        body={"density": 8.0, "filename": "mesh.stl"},
        files={"mesh": _ASCII_STL},
    )

    response = api.inertia_from_mesh(request)

    assert response.status_code == 200
    assert fake_mesh.density == 8.0
    assert response.body["mass"] == pytest.approx(2.0)
    assert response.body["volume"] == pytest.approx(0.25)
    assert response.body["inertia"]["ixx"] == pytest.approx(2.0)


def test_mesh_upload_mass_path_still_returns_scaled_inertia(monkeypatch) -> None:
    """Mass-based inertia must still scale the mesh inertia tensor."""
    fake_mesh = _FakeInertiaMesh()

    def load(path: str | Path) -> _FakeInertiaMesh:
        assert Path(path).suffix == ".stl"
        return fake_mesh

    monkeypatch.setitem(sys.modules, "trimesh", types.SimpleNamespace(load=load))
    api = ModelGenerationAPI()

    response = api.inertia_from_mesh(_mesh_request(_ASCII_STL, mass=4.0))

    assert response.status_code == 200
    assert response.body["mass"] == pytest.approx(4.0)
    assert response.body["volume"] == pytest.approx(0.25)
    assert response.body["inertia"]["ixx"] == pytest.approx(4.0)


def test_mesh_upload_no_trimesh_returns_501() -> None:
    """When trimesh is not installed the route must return HTTP 501."""
    api = ModelGenerationAPI()

    original = sys.modules.get("trimesh")
    sys.modules["trimesh"] = None  # type: ignore[assignment]
    try:
        response = api.inertia_from_mesh(_mesh_request(_ASCII_STL))
    finally:
        if original is not None:
            sys.modules["trimesh"] = original
        else:
            sys.modules.pop("trimesh", None)

    assert response.status_code == 501
    assert "trimesh" in response.body["error"].lower()


def test_mesh_upload_missing_file_returns_error() -> None:
    """Request without a mesh file must return an error response."""
    api = ModelGenerationAPI()

    request = APIRequest(
        method=HTTPMethod.POST,
        path="/api/v1/inertia/from-mesh",
        body={"mass": 1.0},
        files={},
    )
    response = api.inertia_from_mesh(request)

    assert response.status_code == 400
    assert "Missing mesh file" in response.body["error"]


def test_mesh_upload_requires_mass_or_density() -> None:
    """Request without mass or density must return an error response."""
    api = ModelGenerationAPI()

    request = APIRequest(
        method=HTTPMethod.POST,
        path="/api/v1/inertia/from-mesh",
        body={"filename": "mesh.stl"},
        files={"mesh": _ASCII_STL},
    )
    response = api.inertia_from_mesh(request)

    assert response.status_code == 400
    assert (
        "mass" in response.body["error"].lower()
        or "density" in response.body["error"].lower()
    )
