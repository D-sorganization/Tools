"""Regression tests for scripts/verify_coverage.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "verify_coverage.py"


def _load_verify_coverage_module():
    spec = importlib.util.spec_from_file_location("verify_coverage_script", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_main_returns_nonzero_when_pytest_fails(monkeypatch) -> None:
    """Coverage success must not mask failing pytest execution."""
    module = _load_verify_coverage_module()

    monkeypatch.setattr(module, "parse_args", lambda: SimpleNamespace(html=False))
    monkeypatch.setattr(
        module,
        "run_pytest_with_coverage",
        lambda html: (2, "unused.json"),
    )
    monkeypatch.setattr(
        module,
        "load_coverage_json",
        lambda path: {"totals": {}, "files": {}},
    )
    monkeypatch.setattr(module, "check_thresholds", lambda coverage_data: True)

    assert module.main() == 1


def test_main_returns_zero_when_pytest_and_thresholds_pass(monkeypatch) -> None:
    """The verifier should succeed only when both checks succeed."""
    module = _load_verify_coverage_module()

    monkeypatch.setattr(module, "parse_args", lambda: SimpleNamespace(html=False))
    monkeypatch.setattr(
        module,
        "run_pytest_with_coverage",
        lambda html: (0, "unused.json"),
    )
    monkeypatch.setattr(
        module,
        "load_coverage_json",
        lambda path: {"totals": {}, "files": {}},
    )
    monkeypatch.setattr(module, "check_thresholds", lambda coverage_data: True)

    assert module.main() == 0
