"""Tests for the advisory-only MPC result record (SCADA F16, issue #5533).

The record wraps MPC output with provenance and is advisory by construction:
``authoritative_write_available`` is ``Literal[False]`` and there is no import
path from ``mpc.py`` / ``advisory_result.py`` to any PLC write seam.
"""

from __future__ import annotations

import ast
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import advisory_result
import mpc
import pytest
from advisory_result import AdvisoryResult, build_advisory_result
from pydantic import ValidationError

NOW = datetime(2026, 10, 11, 12, 0, tzinfo=UTC)


def _kwargs(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "model_id": "dmc-fopdt",
        "model_version": "1",
        "model_parameters": {"process_gain": 1.2, "rho": 0.1},
        "inputs": {"setpoint": 50.0},
        "input_window": {"dt_s": 0.5, "steps": 50},
        "constraint_set": {"cv_min": 0.0, "cv_max": 100.0},
        "solver_status": "success",
        "created_at": NOW,
    }
    base.update(over)
    return base


@pytest.mark.unit
def test_record_is_advisory_and_not_writable() -> None:
    rec = build_advisory_result(**_kwargs())
    assert rec.authoritative_write_available is False
    assert rec.finding_type == "advisory"


@pytest.mark.unit
@pytest.mark.parametrize("bad", [True, 1, "false", None])
def test_authoritative_write_cannot_be_true(bad: Any) -> None:
    data = build_advisory_result(**_kwargs()).model_dump()
    data["authoritative_write_available"] = bad
    with pytest.raises(ValidationError):
        AdvisoryResult(**data)


@pytest.mark.unit
def test_record_is_frozen() -> None:
    rec = build_advisory_result(**_kwargs())
    with pytest.raises(ValidationError):
        rec.authoritative_write_available = True  # type: ignore[assignment,misc]


@pytest.mark.unit
def test_hashes_are_stable_for_identical_inputs() -> None:
    a = build_advisory_result(**_kwargs())
    b = build_advisory_result(**_kwargs(created_at=datetime(2030, 1, 1, tzinfo=UTC)))
    assert a.model_hash == b.model_hash
    assert a.inputs_hash == b.inputs_hash
    assert len(a.model_hash) == 64


@pytest.mark.unit
def test_hashes_change_with_inputs_and_model() -> None:
    base = build_advisory_result(**_kwargs())
    other_in = build_advisory_result(**_kwargs(inputs={"setpoint": 51.0}))
    other_model = build_advisory_result(**_kwargs(model_version="2"))
    assert base.inputs_hash != other_in.inputs_hash
    assert base.model_hash == other_in.model_hash
    assert base.model_hash != other_model.model_hash


@pytest.mark.unit
@pytest.mark.parametrize(
    "field", ["model_id", "model_version", "solver_status", "created_at"]
)
def test_missing_provenance_fails_closed(field: str) -> None:
    kwargs = _kwargs()
    del kwargs[field]
    with pytest.raises(TypeError):
        build_advisory_result(**kwargs)


@pytest.mark.unit
@pytest.mark.parametrize("field", ["model_id", "model_version", "solver_status"])
@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_provenance_fails_closed(field: str, blank: str) -> None:
    with pytest.raises(ValidationError):
        build_advisory_result(**_kwargs(**{field: blank}))


@pytest.mark.unit
def test_naive_timestamp_rejected() -> None:
    with pytest.raises(ValidationError):
        build_advisory_result(**_kwargs(created_at=datetime(2026, 1, 1)))


@pytest.mark.unit
def test_malformed_hash_rejected() -> None:
    data = build_advisory_result(**_kwargs()).model_dump()
    data["model_hash"] = "nothex"
    with pytest.raises(ValidationError):
        AdvisoryResult(**data)


@pytest.mark.unit
def test_non_finite_parameters_rejected() -> None:
    with pytest.raises(ValueError):
        build_advisory_result(**_kwargs(inputs={"setpoint": float("nan")}))


@pytest.mark.unit
def test_simulate_with_advisory_wraps_result() -> None:
    class _P:
        prediction_horizon = 10
        control_horizon = 3
        setpoint = 50.0
        rho = 0.1
        process_gain = 1.2
        process_tau = 5.0
        process_delay = 1.0

    first = mpc.simulate_with_advisory(_P())
    second = mpc.simulate_with_advisory(_P())
    assert first["status"] == "success"
    adv = first["advisory"]
    assert adv["authoritative_write_available"] is False
    assert adv["solver_status"] == "success"
    assert adv["model_hash"] == second["advisory"]["model_hash"]
    assert adv["inputs_hash"] == second["advisory"]["inputs_hash"]
    assert adv["constraint_set"]["cv_max"] == mpc.CV_MAX


# --- Static guard: no path from the advisory modules to a PLC write seam ----

_FORBIDDEN_FRAGMENTS = (
    "plc",
    "modbus",
    "hardware",
    "simulator_client",
    "shutdown_safety",
    "power_supply",
    "fastapi",
    "state",
)
_FORBIDDEN_NAMES = {"write_pid_setpoint", "write_output", "write_coil"}


def _imports_and_names(path: str) -> tuple[list[str], set[str]]:
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    imported: list[str] = []
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
            imported += [a.name for a in node.names]
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return imported, names


@pytest.mark.unit
@pytest.mark.parametrize("module", [advisory_result, mpc])
def test_no_import_path_to_plc_write_seam(module: Any) -> None:
    imported, names = _imports_and_names(module.__file__)
    bad = [
        m
        for m in imported
        for f in _FORBIDDEN_FRAGMENTS
        if f in m.lower().split(".")[0]
    ]
    assert bad == [], f"{module.__name__} imports a write-capable module: {bad}"
    assert not (names & _FORBIDDEN_NAMES), names & _FORBIDDEN_NAMES


@pytest.mark.unit
def test_guard_detects_a_forbidden_import(tmp_path: Path) -> None:
    f = tmp_path / "evil.py"
    f.write_text("from plc_interface import write_pid_setpoint\n", encoding="utf-8")
    imported, names = _imports_and_names(str(f))
    assert "plc_interface" in imported
    assert "write_pid_setpoint" in imported or "write_pid_setpoint" in names
