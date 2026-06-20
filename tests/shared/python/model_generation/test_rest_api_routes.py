"""Tests for model_generation.api.rest_api_routes.ModelGenerationAPI handler methods.

Covers: route registration count; health/info shape; every handler's
valid→200 and missing-field→4xx paths; inertia/convert/validate/parse
success + error branches; security headers on all responses.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
from model_generation.api.rest_api_routes import ModelGenerationAPI
from model_generation.api.rest_api_types import APIRequest, APIResponse, HTTPMethod

SIMPLE_URDF = """<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base_link">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
</robot>"""

MINIMAL_MJCF = """<mujoco model="test">
  <worldbody>
    <body name="base" pos="0 0 0">
      <geom type="sphere" size="0.1"/>
    </body>
  </worldbody>
</mujoco>"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get(api: ModelGenerationAPI, path: str, **kwargs) -> APIResponse:
    return api.handle_request(APIRequest(method=HTTPMethod.GET, path=path, **kwargs))


def _post(api: ModelGenerationAPI, path: str, body=None, files=None) -> APIResponse:
    kw: dict = {}
    if body is not None:
        kw["body"] = body
    if files is not None:
        kw["files"] = files
    return api.handle_request(APIRequest(method=HTTPMethod.POST, path=path, **kw))


def _delete(api: ModelGenerationAPI, path: str, **kwargs) -> APIResponse:
    return api.handle_request(APIRequest(method=HTTPMethod.DELETE, path=path, **kwargs))


@pytest.fixture()
def api() -> ModelGenerationAPI:
    return ModelGenerationAPI()


# ---------------------------------------------------------------------------
# Constructor / prefix
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_init_default_prefix() -> None:
    a = ModelGenerationAPI()
    assert a.prefix == "/api/v1"


@pytest.mark.unit
def test_init_custom_prefix() -> None:
    a = ModelGenerationAPI(prefix="/v2")
    assert a.prefix == "/v2"
    paths = [r.path for r in a.get_routes()]
    assert all(p.startswith("/v2") for p in paths)


@pytest.mark.unit
def test_init_none_prefix_raises() -> None:
    with pytest.raises(ValueError, match="prefix"):
        ModelGenerationAPI(prefix=None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------


EXPECTED_ROUTE_COUNT = 18


@pytest.mark.unit
def test_route_count(api: ModelGenerationAPI) -> None:
    assert len(api.get_routes()) == EXPECTED_ROUTE_COUNT


@pytest.mark.unit
def test_route_paths_include_all_groups(api: ModelGenerationAPI) -> None:
    paths = {r.path for r in api.get_routes()}
    for expected in (
        "/api/v1/health",
        "/api/v1/info",
        "/api/v1/generate/humanoid",
        "/api/v1/generate/from-params",
        "/api/v1/convert/simscape-to-urdf",
        "/api/v1/convert/mjcf-to-urdf",
        "/api/v1/convert/urdf-to-mjcf",
        "/api/v1/validate",
        "/api/v1/parse",
        "/api/v1/inertia/calculate",
        "/api/v1/inertia/from-mesh",
        "/api/v1/library/models",
        "/api/v1/library/models/{model_id}",
        "/api/v1/editor/compose",
        "/api/v1/editor/diff",
    ):
        assert expected in paths, f"Missing route: {expected}"


# ---------------------------------------------------------------------------
# Health / Info
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_health_check_status_200(api: ModelGenerationAPI) -> None:
    resp = _get(api, "/api/v1/health")
    assert resp.status_code == 200


@pytest.mark.unit
def test_health_check_body_shape(api: ModelGenerationAPI) -> None:
    resp = _get(api, "/api/v1/health")
    assert resp.body["status"] == "healthy"
    assert resp.body["service"] == "model_generation"


@pytest.mark.unit
def test_get_api_info_status_200(api: ModelGenerationAPI) -> None:
    resp = _get(api, "/api/v1/info")
    assert resp.status_code == 200


@pytest.mark.unit
def test_get_api_info_body_shape(api: ModelGenerationAPI) -> None:
    resp = _get(api, "/api/v1/info")
    body = resp.body
    assert "name" in body
    assert "version" in body
    assert "description" in body
    assert "endpoints" in body
    assert isinstance(body["endpoints"], list)
    assert len(body["endpoints"]) == EXPECTED_ROUTE_COUNT


@pytest.mark.unit
def test_get_api_info_endpoint_fields(api: ModelGenerationAPI) -> None:
    resp = _get(api, "/api/v1/info")
    ep = resp.body["endpoints"][0]
    assert "method" in ep
    assert "path" in ep
    assert "description" in ep
    assert "tags" in ep


# ---------------------------------------------------------------------------
# handle_request dispatch
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_unknown_route_returns_404(api: ModelGenerationAPI) -> None:
    resp = _get(api, "/api/v1/nonexistent")
    assert resp.status_code == 404
    assert "error" in resp.body


@pytest.mark.unit
def test_security_headers_on_health_response(api: ModelGenerationAPI) -> None:
    resp = _get(api, "/api/v1/health")
    assert "Content-Security-Policy" in resp.headers
    assert "X-Content-Type-Options" in resp.headers
    assert "X-Frame-Options" in resp.headers
    assert "Strict-Transport-Security" in resp.headers


@pytest.mark.unit
def test_security_headers_on_error_response(api: ModelGenerationAPI) -> None:
    resp = _get(api, "/api/v1/nonexistent")
    assert "Content-Security-Policy" in resp.headers


@pytest.mark.unit
def test_handle_request_does_not_leak_raw_exception_text(
    api: ModelGenerationAPI,
) -> None:
    def _raise_secret(_request: APIRequest) -> APIResponse:
        raise RuntimeError("secret filesystem path C:/tmp/private-model.urdf")

    api.add_route(HTTPMethod.GET, "/boom", _raise_secret)

    resp = _get(api, "/api/v1/boom")

    assert resp.status_code == 500
    assert "secret filesystem path" not in resp.body["error"]
    assert "internal server error" in resp.body["error"].lower()


# ---------------------------------------------------------------------------
# generate_from_params — missing-field guard (no heavy imports needed)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_generate_from_params_missing_links_returns_400(
    api: ModelGenerationAPI,
) -> None:
    resp = _post(api, "/api/v1/generate/from-params", body={"name": "bot"})
    assert resp.status_code == 400
    assert "error" in resp.body


# ---------------------------------------------------------------------------
# Conversion handlers — missing-content guard
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_convert_simscape_missing_content_returns_400(api: ModelGenerationAPI) -> None:
    resp = _post(api, "/api/v1/convert/simscape-to-urdf", body={})
    assert resp.status_code == 400
    assert "error" in resp.body


@pytest.mark.unit
def test_convert_mjcf_to_urdf_missing_content_returns_400(
    api: ModelGenerationAPI,
) -> None:
    resp = _post(api, "/api/v1/convert/mjcf-to-urdf", body={})
    assert resp.status_code == 400
    assert "error" in resp.body


@pytest.mark.unit
def test_convert_mjcf_to_urdf_invalid_upload_encoding_returns_422(
    api: ModelGenerationAPI,
) -> None:
    resp = _post(api, "/api/v1/convert/mjcf-to-urdf", files={"file": b"\xff"})
    assert resp.status_code == 422
    assert "utf-8" in resp.body["error"].lower()


@pytest.mark.unit
def test_convert_urdf_to_mjcf_missing_content_returns_400(
    api: ModelGenerationAPI,
) -> None:
    resp = _post(api, "/api/v1/convert/urdf-to-mjcf", body={})
    assert resp.status_code == 400
    assert "error" in resp.body


# ---------------------------------------------------------------------------
# Validate handler
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_validate_missing_content_returns_400(api: ModelGenerationAPI) -> None:
    resp = _post(api, "/api/v1/validate", body={})
    assert resp.status_code == 400
    assert "error" in resp.body


@pytest.mark.unit
def test_validate_valid_urdf_returns_200_valid_true(api: ModelGenerationAPI) -> None:
    resp = _post(api, "/api/v1/validate", body={"content": SIMPLE_URDF})
    assert resp.status_code == 200
    assert resp.body["valid"] is True
    assert resp.body["error_count"] == 0


@pytest.mark.unit
def test_validate_invalid_urdf_returns_200_valid_false(api: ModelGenerationAPI) -> None:
    resp = _post(api, "/api/v1/validate", body={"content": "<robot><bad></robot>"})
    assert resp.status_code == 200
    assert resp.body["valid"] is False
    assert resp.body["error_count"] > 0


@pytest.mark.unit
def test_validate_response_has_messages_list(api: ModelGenerationAPI) -> None:
    resp = _post(api, "/api/v1/validate", body={"content": SIMPLE_URDF})
    assert "messages" in resp.body
    assert isinstance(resp.body["messages"], list)


# ---------------------------------------------------------------------------
# Parse handler
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_parse_missing_content_returns_400(api: ModelGenerationAPI) -> None:
    resp = _post(api, "/api/v1/parse", body={})
    assert resp.status_code == 400
    assert "error" in resp.body


@pytest.mark.unit
def test_parse_valid_urdf_returns_200(api: ModelGenerationAPI) -> None:
    resp = _post(api, "/api/v1/parse", body={"content": SIMPLE_URDF})
    assert resp.status_code == 200


@pytest.mark.unit
def test_parse_response_body_shape(api: ModelGenerationAPI) -> None:
    resp = _post(api, "/api/v1/parse", body={"content": SIMPLE_URDF})
    body = resp.body
    assert body["name"] == "test_robot"
    assert "links" in body
    assert "joints" in body
    assert "materials" in body
    assert "warnings" in body


# ---------------------------------------------------------------------------
# Inertia — calculate_inertia
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_inertia_missing_shape_returns_400(api: ModelGenerationAPI) -> None:
    resp = _post(
        api, "/api/v1/inertia/calculate", body={"mass": 1.0, "dimensions": [0.1]}
    )
    assert resp.status_code == 400
    assert "error" in resp.body


@pytest.mark.unit
def test_inertia_unknown_shape_returns_400(api: ModelGenerationAPI) -> None:
    resp = _post(
        api,
        "/api/v1/inertia/calculate",
        body={"shape": "torus", "mass": 1.0, "dimensions": [0.1]},
    )
    assert resp.status_code == 400
    assert "error" in resp.body


@pytest.mark.unit
def test_inertia_box_success(api: ModelGenerationAPI) -> None:
    resp = _post(
        api,
        "/api/v1/inertia/calculate",
        body={"shape": "box", "mass": 1.0, "dimensions": [0.1, 0.2, 0.3]},
    )
    assert resp.status_code == 200
    inertia = resp.body["inertia"]
    for key in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz"):
        assert key in inertia
    assert resp.body["is_positive_definite"] is True


@pytest.mark.unit
def test_inertia_box_wrong_dimension_count_returns_400(api: ModelGenerationAPI) -> None:
    resp = _post(
        api,
        "/api/v1/inertia/calculate",
        body={"shape": "box", "mass": 1.0, "dimensions": [0.1, 0.2]},
    )
    assert resp.status_code == 400


@pytest.mark.unit
def test_inertia_sphere_success_symmetric(api: ModelGenerationAPI) -> None:
    resp = _post(
        api,
        "/api/v1/inertia/calculate",
        body={"shape": "sphere", "mass": 1.0, "dimensions": [0.1]},
    )
    assert resp.status_code == 200
    inertia = resp.body["inertia"]
    assert abs(inertia["ixx"] - inertia["iyy"]) < 1e-10
    assert abs(inertia["iyy"] - inertia["izz"]) < 1e-10


@pytest.mark.unit
def test_inertia_sphere_wrong_dimension_count_returns_400(
    api: ModelGenerationAPI,
) -> None:
    resp = _post(
        api,
        "/api/v1/inertia/calculate",
        body={"shape": "sphere", "mass": 1.0, "dimensions": [0.1, 0.2]},
    )
    assert resp.status_code == 400


@pytest.mark.unit
def test_inertia_cylinder_success(api: ModelGenerationAPI) -> None:
    resp = _post(
        api,
        "/api/v1/inertia/calculate",
        body={"shape": "cylinder", "mass": 1.0, "dimensions": [0.05, 0.2]},
    )
    assert resp.status_code == 200
    assert "inertia" in resp.body


@pytest.mark.unit
def test_inertia_cylinder_wrong_dimension_count_returns_400(
    api: ModelGenerationAPI,
) -> None:
    resp = _post(
        api,
        "/api/v1/inertia/calculate",
        body={"shape": "cylinder", "mass": 1.0, "dimensions": [0.05]},
    )
    assert resp.status_code == 400


@pytest.mark.unit
def test_inertia_capsule_success(api: ModelGenerationAPI) -> None:
    resp = _post(
        api,
        "/api/v1/inertia/calculate",
        body={"shape": "capsule", "mass": 1.0, "dimensions": [0.05, 0.3]},
    )
    assert resp.status_code == 200
    assert "inertia" in resp.body


@pytest.mark.unit
def test_inertia_capsule_wrong_dimension_count_returns_400(
    api: ModelGenerationAPI,
) -> None:
    resp = _post(
        api,
        "/api/v1/inertia/calculate",
        body={"shape": "capsule", "mass": 1.0, "dimensions": [0.05]},
    )
    assert resp.status_code == 400


@pytest.mark.unit
def test_inertia_response_includes_shape_and_mass_echo(api: ModelGenerationAPI) -> None:
    resp = _post(
        api,
        "/api/v1/inertia/calculate",
        body={"shape": "box", "mass": 2.5, "dimensions": [0.1, 0.2, 0.3]},
    )
    assert resp.body["shape"] == "box"
    assert resp.body["mass"] == 2.5


# ---------------------------------------------------------------------------
# Inertia — inertia_from_mesh
# ---------------------------------------------------------------------------


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


@pytest.mark.unit
def test_inertia_from_mesh_missing_file_returns_400(api: ModelGenerationAPI) -> None:
    resp = _post(api, "/api/v1/inertia/from-mesh", body={"mass": 1.0})
    assert resp.status_code == 400
    assert "error" in resp.body


@pytest.mark.unit
def test_inertia_from_mesh_no_mass_or_density_returns_400(
    api: ModelGenerationAPI,
) -> None:
    resp = _post(
        api,
        "/api/v1/inertia/from-mesh",
        body={},
        files={"mesh": b"\x00" * 100},
    )
    assert resp.status_code == 400
    assert "error" in resp.body


@pytest.mark.unit
def test_inertia_from_mesh_mass_branch_returns_scaled_inertia(
    api: ModelGenerationAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mesh = _FakeInertiaMesh()

    def load(path: str | Path) -> _FakeInertiaMesh:
        assert Path(path).suffix == ".stl"
        return fake_mesh

    monkeypatch.setitem(sys.modules, "trimesh", types.SimpleNamespace(load=load))

    resp = _post(
        api,
        "/api/v1/inertia/from-mesh",
        body={"mass": 4.0, "filename": "mesh.stl"},
        files={"mesh": _ASCII_STL},
    )

    assert resp.status_code == 200
    assert resp.body["mass"] == pytest.approx(4.0)
    assert resp.body["volume"] == pytest.approx(0.25)
    assert resp.body["center_of_mass"] == pytest.approx([0.1, 0.2, 0.3])
    assert resp.body["inertia"]["ixx"] == pytest.approx(4.0)


@pytest.mark.unit
def test_inertia_from_mesh_density_branch_returns_volume_and_inertia(
    api: ModelGenerationAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mesh = _FakeInertiaMesh()

    def load(path: str | Path) -> _FakeInertiaMesh:
        assert Path(path).suffix == ".stl"
        return fake_mesh

    monkeypatch.setitem(sys.modules, "trimesh", types.SimpleNamespace(load=load))

    resp = _post(
        api,
        "/api/v1/inertia/from-mesh",
        body={"density": 8.0, "filename": "mesh.stl"},
        files={"mesh": _ASCII_STL},
    )

    assert resp.status_code == 200
    assert fake_mesh.density == 8.0
    assert resp.body["mass"] == pytest.approx(2.0)
    assert resp.body["volume"] == pytest.approx(0.25)
    assert resp.body["inertia"]["ixx"] == pytest.approx(2.0)


@pytest.mark.unit
@pytest.mark.parametrize("field", ["mass", "density"])
def test_inertia_from_mesh_rejects_non_positive_mass_inputs(
    api: ModelGenerationAPI,
    field: str,
) -> None:
    resp = _post(
        api,
        "/api/v1/inertia/from-mesh",
        body={field: -1.0, "filename": "mesh.stl"},
        files={"mesh": b"solid mesh\nendsolid mesh\n"},
    )
    assert resp.status_code == 400
    assert field in resp.body["error"]


# ---------------------------------------------------------------------------
# Library handlers
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_library_list_models_returns_200(api: ModelGenerationAPI) -> None:
    resp = _get(api, "/api/v1/library/models")
    assert resp.status_code == 200
    assert "models" in resp.body
    assert "count" in resp.body


@pytest.mark.unit
def test_library_get_model_missing_id_guard(api: ModelGenerationAPI) -> None:
    # Call the handler directly with an empty query_params so model_id is absent.
    req = APIRequest(
        method=HTTPMethod.GET, path="/api/v1/library/models/", query_params={}
    )
    resp = api.library_get_model(req)
    assert resp.status_code == 400
    assert "error" in resp.body


@pytest.mark.unit
def test_library_get_model_nonexistent_returns_404(api: ModelGenerationAPI) -> None:
    resp = _get(api, "/api/v1/library/models/nonexistent_xyz_999")
    assert resp.status_code == 404
    assert "error" in resp.body


@pytest.mark.unit
def test_library_add_model_missing_content_returns_400(api: ModelGenerationAPI) -> None:
    resp = _post(api, "/api/v1/library/models", body={"name": "my_model"})
    assert resp.status_code == 400
    assert "error" in resp.body


@pytest.mark.unit
def test_library_remove_model_missing_id_returns_400(api: ModelGenerationAPI) -> None:
    req = APIRequest(
        method=HTTPMethod.DELETE,
        path="/api/v1/library/models/",
        query_params={},
    )
    resp = api.library_remove_model(req)
    assert resp.status_code == 400
    assert "error" in resp.body


@pytest.mark.unit
def test_library_remove_model_nonexistent_returns_404(api: ModelGenerationAPI) -> None:
    # ModelLibrary.remove_model returns False for an unknown id -> 404 (#3327).
    resp = _delete(
        api,
        "/api/v1/library/models/nonexistent_xyz_999",
        query_params={"model_id": "nonexistent_xyz_999"},
    )
    assert resp.status_code == 404
    assert "error" in resp.body


@pytest.mark.unit
def test_library_remove_model_success_returns_200(
    api: ModelGenerationAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A successful removal returns 200 and echoes the removed id (#3327)."""
    captured: dict[str, object] = {}

    class _FakeLibrary:
        def remove_model(self, model_id: str, delete_files: bool = False) -> bool:
            captured["model_id"] = model_id
            captured["delete_files"] = delete_files
            return True

    monkeypatch.setattr(
        "model_generation.library.ModelLibrary", _FakeLibrary, raising=False
    )

    resp = _delete(
        api,
        "/api/v1/library/models/abc123",
        query_params={"model_id": "abc123", "delete_files": "true"},
    )
    assert resp.status_code == 200
    assert resp.body == {"removed": True, "id": "abc123"}
    assert captured == {"model_id": "abc123", "delete_files": True}


@pytest.mark.unit
def test_library_download_model_missing_id_guard(api: ModelGenerationAPI) -> None:
    # Call the handler directly with no model_id in query_params.
    req = APIRequest(
        method=HTTPMethod.GET, path="/api/v1/library/models//download", query_params={}
    )
    resp = api.library_download_model(req)
    assert resp.status_code == 400
    assert "error" in resp.body


@pytest.mark.unit
def test_library_download_model_nonexistent_returns_404(
    api: ModelGenerationAPI,
) -> None:
    resp = _get(api, "/api/v1/library/models/nonexistent_xyz_999/download")
    assert resp.status_code == 404
    assert "error" in resp.body


# ---------------------------------------------------------------------------
# Editor handlers
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_compose_models_missing_sources_returns_400(api: ModelGenerationAPI) -> None:
    resp = _post(api, "/api/v1/editor/compose", body={"name": "bot"})
    assert resp.status_code == 400
    assert "error" in resp.body


@pytest.mark.unit
@pytest.mark.parametrize(
    ("operation", "missing_key"),
    [
        ({"type": "copy_subtree", "source": "base"}, "link"),
        ({"type": "delete_subtree"}, "link"),
        ({"type": "rename", "old_name": "base"}, "new_name"),
    ],
)
def test_compose_models_validates_required_operation_fields(
    api: ModelGenerationAPI,
    operation: dict[str, str],
    missing_key: str,
) -> None:
    resp = _post(
        api,
        "/api/v1/editor/compose",
        body={"sources": {"base": SIMPLE_URDF}, "operations": [operation]},
    )
    assert resp.status_code == 400
    assert missing_key in resp.body["error"]


@pytest.mark.unit
def test_compose_models_rejects_unknown_operation_type(
    api: ModelGenerationAPI,
) -> None:
    resp = _post(
        api,
        "/api/v1/editor/compose",
        body={
            "sources": {"base": SIMPLE_URDF},
            "operations": [{"type": "cop_subtree", "source": "base", "link": "root"}],
        },
    )
    assert resp.status_code == 400
    assert "unknown operation" in resp.body["error"].lower()


@pytest.mark.unit
def test_diff_missing_content_a_returns_400(api: ModelGenerationAPI) -> None:
    resp = _post(api, "/api/v1/editor/diff", body={"content_b": SIMPLE_URDF})
    assert resp.status_code == 400
    assert "error" in resp.body


@pytest.mark.unit
def test_diff_missing_content_b_returns_400(api: ModelGenerationAPI) -> None:
    resp = _post(api, "/api/v1/editor/diff", body={"content_a": SIMPLE_URDF})
    assert resp.status_code == 400
    assert "error" in resp.body


@pytest.mark.unit
def test_diff_identical_urdfs_has_no_changes(api: ModelGenerationAPI) -> None:
    resp = _post(
        api,
        "/api/v1/editor/diff",
        body={"content_a": SIMPLE_URDF, "content_b": SIMPLE_URDF},
    )
    assert resp.status_code == 200
    assert resp.body["has_changes"] is False


@pytest.mark.unit
def test_diff_changed_urdfs_has_changes(api: ModelGenerationAPI) -> None:
    modified = SIMPLE_URDF.replace("test_robot", "changed_robot")
    resp = _post(
        api,
        "/api/v1/editor/diff",
        body={"content_a": SIMPLE_URDF, "content_b": modified},
    )
    assert resp.status_code == 200
    assert resp.body["has_changes"] is True
    assert "unified_diff" in resp.body
    assert resp.body["additions"] > 0
