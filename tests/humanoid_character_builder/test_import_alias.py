"""Compatibility tests for the humanoid-character-builder import shim."""

from __future__ import annotations

import importlib


def test_humanoid_character_builder_uses_shared_python_package() -> None:
    """The legacy top-level package resolves to the canonical shared package."""
    legacy = importlib.import_module("humanoid_character_builder")
    canonical = importlib.import_module("shared.python.humanoid_character_builder")

    assert legacy is canonical
