"""Tests for python.src.utils.compatibility module.

Covers:
- check_python_version function
- UTC constant export
- StrEnum backport
"""

from __future__ import annotations

from datetime import datetime, timedelta

from utils.compatibility import UTC, StrEnum, check_python_version


class TestCheckPythonVersion:
    """Tests for check_python_version function."""

    def test_current_version_passes(self) -> None:
        """Should not raise since we're running on 3.10+."""
        check_python_version()  # No exception


class TestUTC:
    """Tests for UTC constant."""

    def test_utc_is_timezone(self) -> None:
        dt = datetime.now(UTC)
        assert dt.tzinfo is not None

    def test_utc_offset_zero(self) -> None:
        dt = datetime.now(UTC)
        assert dt.utcoffset() == timedelta(0)


class TestStrEnum:
    """Tests for StrEnum backport."""

    def test_basic_enum(self) -> None:
        class Color(StrEnum):
            RED = "red"
            GREEN = "green"
            BLUE = "blue"

        assert Color.RED == "red"
        assert Color.RED.value == "red"

    def test_str_returns_value(self) -> None:
        class Animal(StrEnum):
            DOG = "dog"

        assert str(Animal.DOG) == "dog"

    def test_repr_includes_class_name(self) -> None:
        class Animal(StrEnum):
            CAT = "cat"

        r = repr(Animal.CAT)
        assert "Animal" in r
        assert "CAT" in r

    def test_is_string(self) -> None:
        class Status(StrEnum):
            ACTIVE = "active"

        assert isinstance(Status.ACTIVE, str)

    def test_comparison_with_string(self) -> None:
        class Level(StrEnum):
            HIGH = "high"
            LOW = "low"

        assert Level.HIGH == "high"
        assert Level.LOW != "high"
