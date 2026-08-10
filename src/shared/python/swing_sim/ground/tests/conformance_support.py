"""Reusable scientific-conformance fixture loader and assertion helpers."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

FIXTURE_DIR = (
    Path(__file__).parents[5]
    / "rate_of_closure"
    / "web"
    / "src"
    / "model"
    / "__fixtures__"
)
CORPUS_PATH = FIXTURE_DIR / "ground_reference_conformance_v1.json"
TEMPLATE_FIXTURE = "ground_reference_pipeline_golden_v1.json"


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AssertionError(f"duplicate conformance key: {key}")
        result[key] = value
    return result


def _decode_pointer_token(raw_token: str) -> str:
    decoded: list[str] = []
    index = 0
    while index < len(raw_token):
        if raw_token[index] != "~":
            decoded.append(raw_token[index])
            index += 1
            continue
        if index + 1 >= len(raw_token) or raw_token[index + 1] not in {"0", "1"}:
            raise AssertionError(f"invalid JSON pointer escape: {raw_token}")
        decoded.append("~" if raw_token[index + 1] == "0" else "/")
        index += 2
    return "".join(decoded)


def _array_index(token: str, length: int) -> int:
    if (
        not token
        or any(character < "0" or character > "9" for character in token)
        or (len(token) > 1 and token.startswith("0"))
    ):
        raise AssertionError(f"noncanonical JSON pointer array index: {token}")
    index = int(token)
    if index >= length:
        raise AssertionError(f"JSON pointer array index is out of range: {token}")
    return index


def _pointer(document: object, pointer: str) -> Any:
    if pointer == "":
        return document
    if not pointer.startswith("/"):
        raise AssertionError(f"invalid JSON pointer: {pointer}")
    current = document
    for raw_token in pointer[1:].split("/"):
        token = _decode_pointer_token(raw_token)
        if isinstance(current, list):
            current = current[_array_index(token, len(current))]
        elif isinstance(current, dict):
            current = current[token]
        else:
            raise AssertionError(f"pointer traverses a scalar: {pointer}")
    return current


def apply_overrides(document: dict[str, Any], overrides: dict[str, Any]) -> None:
    """Replace existing leaves using strict canonical RFC 6901 pointers."""
    for pointer, replacement in overrides.items():
        if not pointer.startswith("/"):
            raise AssertionError(f"invalid JSON pointer: {pointer}")
        parent_path, _, leaf = pointer.rpartition("/")
        parent = _pointer(document, parent_path)
        leaf = _decode_pointer_token(leaf)
        if isinstance(parent, list):
            parent[_array_index(leaf, len(parent))] = deepcopy(replacement)
        elif isinstance(parent, dict) and leaf in parent:
            parent[leaf] = deepcopy(replacement)
        else:
            raise AssertionError(
                f"override does not replace an existing leaf: {pointer}"
            )


def _dot(left: list[float], right: list[float]) -> float:
    return sum(left[index] * right[index] for index in range(3))


def _cross(left: list[float], right: list[float]) -> list[float]:
    return [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]


def _assert_close(actual: float, expected: float, check: dict[str, Any]) -> None:
    assert math.isclose(
        actual,
        expected,
        rel_tol=check["relative_tolerance"],
        abs_tol=check["absolute_tolerance"],
    ), check["description"]


def _assert_rolling_constraint(
    result: dict[str, Any], request: dict[str, Any], check: dict[str, Any]
) -> None:
    event = result["events"][check["event_index"]]
    normal = request["surface"]["normal_unit"]
    arm = [-request["ball_radius_m"] * value for value in normal]
    velocity = event["velocity_after_m_s"]
    spin_velocity = _cross(event["angular_velocity_after_rad_s"], arm)
    surface_velocity = request["surface"]["surface_velocity_m_s"]
    contact = [
        velocity[index] + spin_velocity[index] - surface_velocity[index]
        for index in range(3)
    ]
    normal_speed = _dot(contact, normal)
    tangent = [contact[index] - normal_speed * normal[index] for index in range(3)]
    assert math.sqrt(_dot(tangent, tangent)) <= check["absolute_tolerance"]


def _assert_contact_plane_constraint(
    result: dict[str, Any], request: dict[str, Any], check: dict[str, Any]
) -> None:
    surface = request["surface"]
    normal = surface["normal_unit"]
    origin = [0.0, float(surface["height_m"]), 0.0]
    radius = float(request["ball_radius_m"])
    tolerance = float(check["absolute_tolerance_m"])
    for point in result["trajectory"]:
        if point["phase"] == "bounce":
            continue
        offset = [
            float(point["position_m"][index]) - origin[index] for index in range(3)
        ]
        error = abs(_dot(offset, normal) - radius)
        assert error <= tolerance, (
            f"contact point leaves the declared plane: error={error}"
        )


def _impact_energy(
    event: dict[str, Any], request: dict[str, Any], suffix: str
) -> float:
    velocity = [float(value) for value in event[f"velocity_{suffix}_m_s"]]
    spin = [float(value) for value in event[f"angular_velocity_{suffix}_rad_s"]]
    mass = float(request["ball_mass_kg"])
    radius = float(request["ball_radius_m"])
    inertia = float(request["rotational_inertia_factor"]) * mass * radius**2
    return float(
        0.5 * mass * _dot(velocity, velocity) + 0.5 * inertia * _dot(spin, spin)
    )


def _assert_check(
    result: dict[str, Any], request: dict[str, Any], check: dict[str, Any]
) -> None:
    kind = check["kind"]
    if kind == "value_equal":
        assert _pointer(result, check["path"]) == check["expected"]
    elif kind == "scalar_close":
        _assert_close(float(_pointer(result, check["path"])), check["expected"], check)
    elif kind == "vector_close":
        actual = _pointer(result, check["path"])
        assert len(actual) == len(check["expected"])
        for actual_value, expected_value in zip(actual, check["expected"], strict=True):
            _assert_close(float(actual_value), expected_value, check)
    elif kind == "terminal_vector_close":
        actual = result["trajectory"][-1][check["field"]]
        assert len(actual) == len(check["expected"])
        for actual_value, expected_value in zip(actual, check["expected"], strict=True):
            _assert_close(float(actual_value), expected_value, check)
    elif kind == "event_types_equal":
        assert [event["event_type"] for event in result["events"]] == check["expected"]
    elif kind == "restitution_ratio":
        event = result["events"][check["event_index"]]
        normal = request["surface"]["normal_unit"]
        before = _dot(event["velocity_before_m_s"], normal)
        after = _dot(event["velocity_after_m_s"], normal)
        _assert_close(after / -before, check["expected"], check)
    elif kind == "rolling_constraint":
        _assert_rolling_constraint(result, request, check)
    elif kind == "contact_plane_constraint":
        _assert_contact_plane_constraint(result, request, check)
    elif kind == "impact_energy_nonincrease":
        event = result["events"][check["event_index"]]
        assert _impact_energy(event, request, "after") <= (
            _impact_energy(event, request, "before") + check["absolute_tolerance_j"]
        )
    else:
        raise AssertionError(f"unsupported conformance check: {kind}")


def _load_unique_json(path: Path) -> dict[str, Any]:
    document = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object
    )
    assert isinstance(document, dict)
    return document


def load_conformance_cases(
    corpus_path: Path = CORPUS_PATH,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load and minimally qualify the versioned corpus and template."""
    corpus = _load_unique_json(corpus_path)
    assert set(corpus) == {
        "schema_version",
        "template_fixture",
        "authority",
        "coordinate_frame",
        "cases",
    }
    assert corpus["schema_version"] == "ground-reference-conformance/v1"
    assert corpus["template_fixture"] == TEMPLATE_FIXTURE
    template = _load_unique_json(corpus_path.parent / TEMPLATE_FIXTURE)
    case_ids = [case["case_id"] for case in corpus["cases"]]
    assert len(case_ids) == len(set(case_ids))
    return template, corpus["cases"]


def materialize_case(
    template: dict[str, Any], case: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Materialize canonical request and execution objects for one case."""
    request = deepcopy(template["request"])
    apply_overrides(request, case["request_overrides"])
    execution = deepcopy(template["execution"])
    execution["schema_version"] = template["execution_schema_version"]
    return request, execution


def assert_conformance_case(
    result: dict[str, Any], request: dict[str, Any], case: dict[str, Any]
) -> None:
    """Apply every whitelisted analytic assertion for one runtime result."""
    for check in case["checks"]:
        _assert_check(result, request, check)


__all__ = [
    "apply_overrides",
    "assert_conformance_case",
    "load_conformance_cases",
    "materialize_case",
]
