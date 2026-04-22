"""Import contract for the verification package marker."""

from __future__ import annotations

from importlib import import_module


def test_verification_package_imports() -> None:
    assert import_module("verification").__name__ == "verification"
