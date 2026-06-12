from pathlib import Path

from scripts import check_blocking_quality_gates
from scripts.check_blocking_quality_gates import validate_ci_standard


def test_ci_standard_quality_gate_steps_are_blocking() -> None:
    assert validate_ci_standard(Path(".github/workflows/ci-standard.yml")) == []


def test_ci_standard_quality_gate_text_fallback(
    monkeypatch,
) -> None:
    monkeypatch.setattr(check_blocking_quality_gates, "yaml", None)

    assert validate_ci_standard(Path(".github/workflows/ci-standard.yml")) == []
