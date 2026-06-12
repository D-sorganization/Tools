"""Regression tests for vessel drafter contract helpers."""

from __future__ import annotations

from pathlib import Path

import pytest


def _enable_vessel_drafter_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(repo_root / "src" / "vessel_drafter" / "python"))


def test_require_positive_accepts_shared_argument_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The helper should accept the fleet-standard ``(value, name)`` order."""
    _enable_vessel_drafter_imports(monkeypatch)

    from vessel_drafter.contracts import require_positive

    require_positive(1.0, "diameter_in")


def test_require_positive_preserves_legacy_argument_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Existing vessel drafter call sites should keep working during migration."""
    _enable_vessel_drafter_imports(monkeypatch)

    from vessel_drafter.contracts import require_positive

    require_positive("diameter_in", 1.0)


def test_require_positive_rejects_non_numeric_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bad argument pairs should fail explicitly instead of comparing strings."""
    _enable_vessel_drafter_imports(monkeypatch)

    from vessel_drafter.contracts import require_positive

    with pytest.raises(TypeError, match="expects"):
        require_positive("diameter_in", "not-a-number")
