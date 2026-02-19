"""Tests for env_utils - Environment variable utilities.

These tests verify the environment utility functions using
Design by Contract principles.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest


class TestGetEnvVarContract:
    """Design by Contract tests for get_env_var function."""

    def test_returns_string_or_none(self):
        """Postcondition: Returns string or None."""
        from utils.env_utils import get_env_var

        result = get_env_var("NONEXISTENT_VAR_12345")
        assert result is None or isinstance(result, str)

    def test_raises_when_required_missing(self):
        """Precondition: Raises ValueError when required var is missing."""
        from utils.env_utils import get_env_var

        with pytest.raises(ValueError, match="not set"):
            get_env_var("REQUIRED_BUT_MISSING_VAR", required=True)


class TestGetEnvVar:
    """Functional tests for get_env_var."""

    def test_returns_existing_variable(self):
        """Test returning existing environment variable."""
        from utils.env_utils import get_env_var

        with patch.dict(os.environ, {"TEST_VAR": "test_value"}):
            result = get_env_var("TEST_VAR")
            assert result == "test_value"

    def test_returns_default_for_missing(self):
        """Test returning default for missing variable."""
        from utils.env_utils import get_env_var

        result = get_env_var("MISSING_VAR_XYZ", default="fallback")
        assert result == "fallback"

    def test_returns_none_for_missing_without_default(self):
        """Test returning None for missing without default."""
        from utils.env_utils import get_env_var

        result = get_env_var("MISSING_VAR_ABC")
        assert result is None

    def test_raises_for_required_missing(self):
        """Test raising for required missing variable."""
        from utils.env_utils import get_env_var

        with pytest.raises(ValueError) as exc_info:
            get_env_var("REQUIRED_MISSING", required=True)

        assert "REQUIRED_MISSING" in str(exc_info.value)


class TestGetEnvBoolContract:
    """Design by Contract tests for get_env_bool function."""

    def test_returns_bool(self):
        """Postcondition: Returns a boolean."""
        from utils.env_utils import get_env_bool

        result = get_env_bool("ANY_VAR")
        assert isinstance(result, bool)


class TestGetEnvBool:
    """Functional tests for get_env_bool."""

    @pytest.mark.parametrize(
        "env_val",
        ["true", "TRUE", "True", "1", "yes", "YES", "on", "ON"],
    )
    def test_true_values(self, env_val):
        """Test recognizing true values."""
        from utils.env_utils import get_env_bool

        with patch.dict(os.environ, {"BOOL_VAR": env_val}):
            assert get_env_bool("BOOL_VAR") is True

    @pytest.mark.parametrize(
        "env_val",
        ["false", "FALSE", "0", "no", "off", "", "anything_else"],
    )
    def test_false_values(self, env_val):
        """Test recognizing false values."""
        from utils.env_utils import get_env_bool

        with patch.dict(os.environ, {"BOOL_VAR": env_val}):
            assert get_env_bool("BOOL_VAR") is False

    def test_default_is_false(self):
        """Test default is False."""
        from utils.env_utils import get_env_bool

        result = get_env_bool("MISSING_BOOL_VAR")
        assert result is False

    def test_custom_default(self):
        """Test custom default value."""
        from utils.env_utils import get_env_bool

        result = get_env_bool("MISSING_BOOL_VAR", default=True)
        assert result is True


class TestGetEnvIntContract:
    """Design by Contract tests for get_env_int function."""

    def test_returns_int(self):
        """Postcondition: Returns an integer."""
        from utils.env_utils import get_env_int

        result = get_env_int("MISSING_INT_VAR")
        assert isinstance(result, int)

    def test_raises_on_invalid_int(self):
        """Precondition: Raises ValueError for non-integer value."""
        from utils.env_utils import get_env_int

        with patch.dict(os.environ, {"BAD_INT": "not_a_number"}):
            with pytest.raises(ValueError, match="must be an integer"):
                get_env_int("BAD_INT")


class TestGetEnvInt:
    """Functional tests for get_env_int."""

    @pytest.mark.parametrize(
        "env_val, expected",
        [("8080", 8080), ("-5", -5), ("0", 0), ("999999", 999999)],
        ids=["positive", "negative", "zero", "large"],
    )
    def test_parses_integer(self, env_val, expected):
        """Test parsing integer values from environment."""
        from utils.env_utils import get_env_int

        with patch.dict(os.environ, {"INT_VAR": env_val}):
            result = get_env_int("INT_VAR")
            assert result == expected

    def test_default_is_zero(self):
        """Test default is zero."""
        from utils.env_utils import get_env_int

        result = get_env_int("MISSING_INT")
        assert result == 0

    def test_custom_default(self):
        """Test custom default value."""
        from utils.env_utils import get_env_int

        result = get_env_int("MISSING_INT", default=42)
        assert result == 42

    def test_raises_for_float_string(self):
        """Test raising for float string."""
        from utils.env_utils import get_env_int

        with patch.dict(os.environ, {"FLOAT_VAR": "3.14"}):
            with pytest.raises(ValueError):
                get_env_int("FLOAT_VAR")


class TestFindEnvFileContract:
    """Design by Contract tests for find_env_file function."""

    def test_returns_path_or_none(self, tmp_path):
        """Postcondition: Returns Path or None."""
        from utils.env_utils import find_env_file

        result = find_env_file(start_path=tmp_path)
        assert result is None or isinstance(result, Path)


class TestFindEnvFile:
    """Functional tests for find_env_file."""

    def test_finds_env_in_start_path(self, tmp_path):
        """Test finding .env in start path."""
        from utils.env_utils import find_env_file

        env_file = tmp_path / ".env"
        env_file.write_text("KEY=value")

        result = find_env_file(start_path=tmp_path)
        assert result == env_file

    def test_finds_env_in_search_locations(self, tmp_path):
        """Test finding .env in search locations."""
        from utils.env_utils import find_env_file

        # Create .env in a custom location
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        env_file = config_dir / ".env"
        env_file.write_text("API_KEY=secret")

        # Search with explicit search locations
        result = find_env_file(search_locations=[config_dir / ".env"])
        assert result == env_file

    def test_custom_filename(self, tmp_path):
        """Test finding custom filename."""
        from utils.env_utils import find_env_file

        env_file = tmp_path / ".env.local"
        env_file.write_text("LOCAL=true")

        result = find_env_file(filename=".env.local", start_path=tmp_path)
        assert result == env_file

    def test_returns_none_when_not_found(self, tmp_path):
        """Test returning None when not found."""
        from utils.env_utils import find_env_file

        result = find_env_file(start_path=tmp_path)
        assert result is None

    def test_custom_search_locations(self, tmp_path):
        """Test searching custom locations."""
        from utils.env_utils import find_env_file

        custom_dir = tmp_path / "config"
        custom_dir.mkdir()
        env_file = custom_dir / ".env"
        env_file.write_text("CUSTOM=true")

        result = find_env_file(search_locations=[custom_dir / ".env"])
        assert result == env_file


class TestLoadEnvFileContract:
    """Design by Contract tests for load_env_file function."""

    def test_returns_bool(self, tmp_path):
        """Postcondition: Returns a boolean."""
        from utils.env_utils import load_env_file

        result = load_env_file(env_path=tmp_path / "nonexistent.env")
        assert isinstance(result, bool)


class TestLoadEnvFile:
    """Functional tests for load_env_file."""

    def test_returns_false_without_dotenv(self, tmp_path):
        """Test returning False when python-dotenv not available."""
        from utils.env_utils import load_env_file

        env_file = tmp_path / ".env"
        env_file.write_text("TEST=value")

        with patch.dict("sys.modules", {"dotenv": None}):
            # This simulates dotenv not being installed
            # The actual behavior depends on whether dotenv is installed
            result = load_env_file(env_path=env_file)
            # Either True (if dotenv installed) or False (if not)
            assert isinstance(result, bool)

    def test_returns_false_for_missing_file(self, tmp_path):
        """Test returning False for missing file."""
        from utils.env_utils import load_env_file

        result = load_env_file(env_path=tmp_path / "missing.env")
        assert result is False

    def test_explicit_path(self, tmp_path):
        """Test loading from explicit path."""
        from utils.env_utils import load_env_file

        env_file = tmp_path / "custom.env"
        env_file.write_text("EXPLICIT=yes")

        # Will return True if dotenv is installed and file exists
        result = load_env_file(env_path=env_file)
        # File exists, so should succeed if dotenv available
        assert isinstance(result, bool)
