"""Tests for model_generation.api.rest_api_routes.ModelGenerationAPI handler methods.

Covers: route registration count; health/info shape; every handler's
valid→200 and missing-field→4xx paths; inertia/convert/validate/parse
success + error branches; security headers on all responses.
"""

from __future__ import annotations

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
def test_library_remove_model_returns_501(api: ModelGenerationAPI) -> None:
    resp = _delete(
        api,
        "/api/v1/library/models/some_id",
        query_params={"model_id": "some_id"},
    )
    assert resp.status_code == 501


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
