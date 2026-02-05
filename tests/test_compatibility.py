"""Tests for compatibility - Python version compatibility shims.

These tests verify the compatibility utility functions using
Design by Contract principles.
"""

import sys
from datetime import datetime, timezone

import pytest


class TestCheckPythonVersionContract:
    """Design by Contract tests for check_python_version function."""

    def test_does_not_exit_on_valid_version(self):
        """Postcondition: Does not exit when version is valid."""
        # If we get here, the module was imported successfully
        # which means check_python_version passed
        from utils.compatibility import check_python_version

        # Re-running should not raise
        check_python_version()


class TestUTCConstant:
    """Tests for UTC constant."""

    def test_utc_is_timezone(self):
        """Test that UTC is a timezone object."""
        from utils.compatibility import UTC

        assert UTC is not None
        # Should be usable with datetime
        now = datetime.now(UTC)
        assert now.tzinfo is not None

    def test_utc_matches_standard_utc(self):
        """Test that UTC matches timezone.utc."""
        from utils.compatibility import UTC

        assert UTC == timezone.utc


class TestStrEnumContract:
    """Design by Contract tests for StrEnum class."""

    def test_is_enum_subclass(self):
        """Postcondition: StrEnum is an Enum subclass."""
        from enum import Enum

        from utils.compatibility import StrEnum

        assert issubclass(StrEnum, Enum)

    def test_members_are_strings(self):
        """Postcondition: StrEnum members are strings."""
        from utils.compatibility import StrEnum

        class Color(StrEnum):
            RED = "red"
            GREEN = "green"

        assert isinstance(Color.RED, str)
        assert isinstance(Color.GREEN, str)


class TestStrEnum:
    """Functional tests for StrEnum."""

    def test_str_returns_value(self):
        """Test that str() returns the value."""
        from utils.compatibility import StrEnum

        class Status(StrEnum):
            ACTIVE = "active"
            INACTIVE = "inactive"

        assert str(Status.ACTIVE) == "active"
        assert str(Status.INACTIVE) == "inactive"

    def test_repr_format(self):
        """Test repr format."""
        from utils.compatibility import StrEnum

        class Level(StrEnum):
            HIGH = "high"
            LOW = "low"

        assert "Level.HIGH" in repr(Level.HIGH)

    def test_value_comparison(self):
        """Test string value comparison."""
        from utils.compatibility import StrEnum

        class Mode(StrEnum):
            DEBUG = "debug"
            RELEASE = "release"

        assert Mode.DEBUG == "debug"
        assert Mode.RELEASE == "release"

    def test_can_use_in_string_operations(self):
        """Test using StrEnum in string operations."""
        from utils.compatibility import StrEnum

        class Prefix(StrEnum):
            USER = "user_"
            ADMIN = "admin_"

        # Can concatenate like strings
        result = Prefix.USER + "123"
        assert result == "user_123"

    def test_membership_in_list(self):
        """Test membership checking in lists."""
        from utils.compatibility import StrEnum

        class Animal(StrEnum):
            DOG = "dog"
            CAT = "cat"

        animals = ["dog", "bird", "fish"]
        assert Animal.DOG in animals
        assert Animal.CAT not in animals

    def test_as_dict_key(self):
        """Test using as dictionary key."""
        from utils.compatibility import StrEnum

        class Key(StrEnum):
            NAME = "name"
            VALUE = "value"

        data = {Key.NAME: "test", Key.VALUE: 42}
        assert data["name"] == "test"
        assert data[Key.NAME] == "test"

    def test_iteration(self):
        """Test iteration over StrEnum."""
        from utils.compatibility import StrEnum

        class Direction(StrEnum):
            NORTH = "north"
            SOUTH = "south"
            EAST = "east"
            WEST = "west"

        values = list(Direction)
        assert len(values) == 4
        assert Direction.NORTH in values

    def test_comparison_with_enum(self):
        """Test comparison between StrEnum members."""
        from utils.compatibility import StrEnum

        class Priority(StrEnum):
            HIGH = "high"
            LOW = "low"

        assert Priority.HIGH != Priority.LOW
        assert Priority.HIGH == Priority.HIGH
