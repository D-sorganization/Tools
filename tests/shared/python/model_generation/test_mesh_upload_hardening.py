from __future__ import annotations

import sys
import types
from pathlib import Path

from model_generation.api.rest_api_routes import ModelGenerationAPI
from model_generation.api.rest_api_types import APIRequest, HTTPMethod


def _mesh_request(mesh: bytes, **body: object) -> APIRequest:
    return APIRequest(
        method=HTTPMethod.POST,
        path="/api/v1/inertia/from-mesh",
        body={"mass": 1.0, "filename": "mesh.stl", **body},
        files={"mesh": mesh},
    )


def test_mesh_upload_rejects_oversized_payload() -> None:
    api = ModelGenerationAPI()

    response = api.inertia_from_mesh(_mesh_request(b"0" * (10 * 1024 * 1024 + 1)))

    assert response.status_code == 413
    assert "exceeds" in response.body["error"]


def test_mesh_upload_rejects_unsupported_filename() -> None:
    api = ModelGenerationAPI()

    response = api.inertia_from_mesh(
        _mesh_request(b"solid mesh\nendsolid mesh\n", filename="mesh.exe")
    )

    assert response.status_code == 413
    assert "Unsupported mesh file type" in response.body["error"]


def test_mesh_upload_cleans_temp_file_on_parser_failure(
    monkeypatch,
) -> None:
    captured_path: dict[str, Path] = {}

    def load(path: str | Path) -> object:
        captured_path["path"] = Path(path)
        raise ValueError("bad mesh")

    monkeypatch.setitem(sys.modules, "trimesh", types.SimpleNamespace(load=load))
    api = ModelGenerationAPI()

    response = api.inertia_from_mesh(_mesh_request(b"solid mesh\nendsolid mesh\n"))

    assert response.status_code == 400
    assert "Mesh processing failed" in response.body["error"]
    assert captured_path["path"].exists() is False
