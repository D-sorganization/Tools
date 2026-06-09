from pathlib import Path

from scripts.check_blocking_quality_gates import validate_ci_standard


def test_ci_standard_quality_gate_steps_are_blocking() -> None:
    assert validate_ci_standard(Path(".github/workflows/ci-standard.yml")) == []
