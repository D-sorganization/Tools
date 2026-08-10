"""Python-authority checks for the shared scientific conformance corpus."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from shared.python.swing_sim.ground import (
    GroundSimulationRequest,
    run_ground_reference,
)
from shared.python.swing_sim.ground.tests.conformance_support import (
    apply_overrides,
    assert_conformance_case,
    load_conformance_cases,
    materialize_case,
)

TEMPLATE, CASES = load_conformance_cases()
SUPPORTED_CHECKS = {
    "event_types_equal",
    "impact_energy_nonincrease",
    "restitution_ratio",
    "rolling_constraint",
    "scalar_close",
    "terminal_vector_close",
    "value_equal",
    "vector_close",
}
MAX_ABSOLUTE_TOLERANCE = 1e-6
MAX_RELATIVE_TOLERANCE = 1e-6
MAX_ENERGY_TOLERANCE_J = 1e-8
CHECK_KEYS = {
    "event_types_equal": {"kind", "expected", "description"},
    "impact_energy_nonincrease": {
        "kind",
        "event_index",
        "absolute_tolerance_j",
        "unit",
        "description",
    },
    "restitution_ratio": {
        "kind",
        "event_index",
        "expected",
        "absolute_tolerance",
        "relative_tolerance",
        "unit",
        "description",
    },
    "rolling_constraint": {
        "kind",
        "event_index",
        "absolute_tolerance",
        "unit",
        "description",
    },
    "scalar_close": {
        "kind",
        "path",
        "expected",
        "absolute_tolerance",
        "relative_tolerance",
        "unit",
        "description",
    },
    "terminal_vector_close": {
        "kind",
        "field",
        "expected",
        "absolute_tolerance",
        "relative_tolerance",
        "unit",
        "description",
    },
    "value_equal": {"kind", "path", "expected", "description"},
    "vector_close": {
        "kind",
        "path",
        "expected",
        "absolute_tolerance",
        "relative_tolerance",
        "unit",
        "description",
    },
}


def _assert_finite_numeric(value: object) -> None:
    if isinstance(value, list):
        assert value and all(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(item)
            for item in value
        )
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        assert math.isfinite(value)
    else:
        raise AssertionError(f"expected finite numeric evidence, got {type(value)}")


def _assert_check_contract(check: dict[str, Any]) -> None:
    assert check["kind"] in SUPPORTED_CHECKS
    assert set(check) == CHECK_KEYS[check["kind"]]
    assert check["description"].strip()
    numeric_expected = {
        "restitution_ratio",
        "scalar_close",
        "terminal_vector_close",
        "vector_close",
    }
    if check["kind"] in numeric_expected:
        _assert_finite_numeric(check["expected"])
    for key in ("absolute_tolerance", "relative_tolerance", "absolute_tolerance_j"):
        if key in check:
            _assert_finite_numeric(check[key])
    for key in ("absolute_tolerance", "relative_tolerance", "absolute_tolerance_j"):
        if key in check:
            assert check[key] >= 0.0
    if "absolute_tolerance" in check:
        assert check["absolute_tolerance"] <= MAX_ABSOLUTE_TOLERANCE
    if "relative_tolerance" in check:
        assert check["relative_tolerance"] <= MAX_RELATIVE_TOLERANCE
    if "absolute_tolerance_j" in check:
        assert check["absolute_tolerance_j"] <= MAX_ENERGY_TOLERANCE_J
    if "path" in check:
        assert check["path"].startswith("/")
    if "event_index" in check:
        assert type(check["event_index"]) is int and check["event_index"] >= 0
    if check["kind"] == "terminal_vector_close":
        assert check["field"] in {"velocity_m_s", "angular_velocity_rad_s"}
    if check["kind"] == "event_types_equal":
        assert check["expected"] and all(
            isinstance(value, str) and value for value in check["expected"]
        )
    if check["kind"] in numeric_expected:
        assert {"absolute_tolerance", "relative_tolerance", "unit"} <= set(check)
    if check["kind"] == "rolling_constraint":
        assert {"absolute_tolerance", "unit"} <= set(check)
    if check["kind"] == "impact_energy_nonincrease":
        assert check["unit"] == "J"
        assert "absolute_tolerance_j" in check


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["case_id"])
def test_python_reference_satisfies_shared_scientific_corpus(
    case: dict[str, Any],
) -> None:
    request, _ = materialize_case(TEMPLATE, case)

    actual = run_ground_reference(GroundSimulationRequest.from_dict(request)).to_dict()

    assert case["platforms"] == ["python", "native", "pyo3", "wasm"]
    assert_conformance_case(actual, request, case)


def test_scientific_corpus_declares_nonempty_basis_units_and_tolerances() -> None:
    tolerance_kinds = {
        "scalar_close",
        "terminal_vector_close",
        "vector_close",
        "restitution_ratio",
    }
    for case in CASES:
        assert set(case) == {
            "case_id",
            "scientific_basis",
            "platforms",
            "request_overrides",
            "checks",
        }
        assert case["case_id"].strip()
        assert case["scientific_basis"].strip()
        assert case["platforms"] == ["python", "native", "pyo3", "wasm"]
        assert case["request_overrides"]
        assert all(pointer.startswith("/") for pointer in case["request_overrides"])
        assert case["checks"]
        for check in case["checks"]:
            _assert_check_contract(check)
            if check["kind"] in tolerance_kinds:
                assert check["unit"].strip()
                assert check["absolute_tolerance"] >= 0.0
                assert check["relative_tolerance"] >= 0.0


def test_override_json_pointers_decode_escapes_and_reject_ambiguous_indices() -> None:
    document = {"a/b": 1, "a~b": 2, "items": [3, 4]}
    apply_overrides(document, {"/a~1b": 10, "/a~0b": 20, "/items/1": 40})
    assert document == {"a/b": 10, "a~b": 20, "items": [3, 40]}

    for pointer in ("/items/-1", "/items/01", "/items/2", "/items/~2"):
        with pytest.raises(AssertionError):
            apply_overrides({"items": [1, 2]}, {pointer: 9})


def test_loader_pins_template_and_rejects_duplicate_template_keys(
    tmp_path: Path,
) -> None:
    corpus = {
        "schema_version": "ground-reference-conformance/v1",
        "template_fixture": "../outside.json",
        "authority": {},
        "coordinate_frame": {},
        "cases": [],
    }
    corpus_path = tmp_path / "ground_reference_conformance_v1.json"
    corpus_path.write_text(json.dumps(corpus), encoding="utf-8")
    with pytest.raises(AssertionError):
        load_conformance_cases(corpus_path)

    corpus["template_fixture"] = "ground_reference_pipeline_golden_v1.json"
    corpus_path.write_text(json.dumps(corpus), encoding="utf-8")
    (tmp_path / "ground_reference_pipeline_golden_v1.json").write_text(
        '{"request":{"value":1,"value":2}}', encoding="utf-8"
    )
    with pytest.raises(AssertionError):
        load_conformance_cases(corpus_path)
