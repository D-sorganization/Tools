"""Standalone fallback coverage for vessel_drafter contracts."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_fallback_contracts(
    monkeypatch: pytest.MonkeyPatch, dbc_level: str
) -> ModuleType:
    repo_root = Path(__file__).resolve().parents[2]
    module_path = (
        repo_root
        / "src"
        / "vessel_drafter"
        / "python"
        / "vessel_drafter"
        / "contracts.py"
    )
    module_name = f"_vessel_contracts_fallback_{dbc_level}"

    monkeypatch.setenv("DBC_LEVEL", dbc_level)
    monkeypatch.setitem(sys.modules, "contracts", None)
    monkeypatch.setitem(sys.modules, "shared", None)
    monkeypatch.setitem(sys.modules, "shared.python", None)
    monkeypatch.setitem(sys.modules, "shared.python.contracts", None)

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fallback_ensure_raises_postcondition_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contracts = _load_fallback_contracts(monkeypatch, "enforce")

    with pytest.raises(contracts.PostconditionError, match="valid output"):
        contracts.ensure(False, "valid output", -1)


def test_fallback_dbc_off_disables_require_ensure_and_wrappers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contracts = _load_fallback_contracts(monkeypatch, "off")

    contracts.require(False, "valid input", -1)
    contracts.ensure(False, "valid output", -1)
    contracts.require_positive("diameter", -1.0)
    contracts.require_finite("wall_temp", float("nan"))
