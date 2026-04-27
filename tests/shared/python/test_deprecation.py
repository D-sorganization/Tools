"""Tests for the shared deprecation utility.

Covers:
- @deprecated emits DeprecationWarning on call
- Warning message includes function name
- Warning message includes reason when provided
- Warning message includes removal_version when provided
- Decorated function still returns correct value
- functools.wraps metadata is preserved
- TypeError raised for invalid reason type
- ValueError raised for empty removal_version string
- Works on methods as well as module-level functions
- Package version bump: programmatic_pid.__version__ == "1.0.0"
- pyproject.toml version == "1.0.0"
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
from deprecation import deprecated

# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def legacy_func():
    """A simple deprecated function with reason and removal_version."""

    @deprecated(reason="Use new_func() instead.", removal_version="2.0.0")
    def old_func(x: float) -> float:
        """Original docstring."""
        return x * 2.0

    return old_func


# ---------------------------------------------------------------------------
# Core decorator behaviour
# ---------------------------------------------------------------------------


class TestDeprecatedDecorator:
    """Unit tests for the @deprecated decorator."""

    @pytest.mark.unit
    def test_emits_deprecation_warning(self, legacy_func):
        """Calling a deprecated function must emit a DeprecationWarning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            legacy_func(1.0)

        assert len(caught) == 1
        assert issubclass(caught[0].category, DeprecationWarning)

    @pytest.mark.unit
    def test_warning_message_contains_function_name(self, legacy_func):
        """The warning message must include the deprecated function's name."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            legacy_func(1.0)

        assert "old_func" in str(caught[0].message)

    @pytest.mark.unit
    def test_warning_message_contains_reason(self, legacy_func):
        """The warning message must include the supplied reason."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            legacy_func(1.0)

        assert "Use new_func() instead" in str(caught[0].message)

    @pytest.mark.unit
    def test_warning_message_contains_removal_version(self, legacy_func):
        """The warning message must mention the removal version."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            legacy_func(1.0)

        assert "2.0.0" in str(caught[0].message)

    @pytest.mark.unit
    def test_return_value_preserved(self, legacy_func):
        """The deprecated wrapper must return the original return value."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = legacy_func(3.0)

        assert result == pytest.approx(6.0)

    @pytest.mark.unit
    def test_docstring_preserved(self, legacy_func):
        """functools.wraps must copy the original docstring."""
        assert legacy_func.__doc__ == "Original docstring."

    @pytest.mark.unit
    def test_function_name_preserved(self, legacy_func):
        """functools.wraps must preserve __name__."""
        assert legacy_func.__name__ == "old_func"

    @pytest.mark.unit
    def test_no_reason_no_removal_version(self):
        """Decorator with default args still emits a DeprecationWarning."""

        @deprecated()
        def bare_func() -> str:
            return "bare"

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = bare_func()

        assert issubclass(caught[0].category, DeprecationWarning)
        assert result == "bare"

    @pytest.mark.unit
    def test_only_reason_provided(self):
        """Decorator with reason but no removal_version omits the version text."""

        @deprecated(reason="no longer needed")
        def reason_only() -> None:
            pass

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            reason_only()

        message = str(caught[0].message)
        assert "no longer needed" in message
        assert "will be removed" not in message

    @pytest.mark.unit
    def test_works_on_methods(self):
        """@deprecated must work on instance methods."""

        class Klass:
            @deprecated(reason="use new_method()")
            def old_method(self, val: int) -> int:
                return val + 1

        obj = Klass()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = obj.old_method(5)

        assert issubclass(caught[0].category, DeprecationWarning)
        assert result == 6


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestDeprecatedInputValidation:
    """@deprecated enforces its own preconditions."""

    @pytest.mark.unit
    def test_reason_must_be_string(self):
        """Non-string reason must raise TypeError immediately (not on call)."""
        with pytest.raises(TypeError, match="reason must be a string"):
            deprecated(reason=42)  # type: ignore[arg-type]

    @pytest.mark.unit
    def test_removal_version_empty_string_raises(self):
        """An empty removal_version string must raise ValueError immediately."""
        with pytest.raises(
            ValueError, match="removal_version must not be an empty string"
        ):
            deprecated(removal_version="   ")


# ---------------------------------------------------------------------------
# Semantic versioning — package metadata
# ---------------------------------------------------------------------------


class TestSemanticVersioning:
    """Verify that the package version has been bumped to 1.0.0."""

    @pytest.mark.unit
    def test_pyproject_version_is_1_0_0(self):
        """pyproject.toml must declare version 1.0.0."""
        repo_root = Path(__file__).resolve().parents[3]
        pyproject = repo_root / "pyproject.toml"
        assert pyproject.exists(), "pyproject.toml not found"
        content = pyproject.read_text()
        # Naive but reliable: look for the project version declaration
        assert 'version = "1.0.0"' in content, (
            "Expected 'version = \"1.0.0\"' in pyproject.toml"
        )

    @pytest.mark.unit
    def test_programmatic_pid_version_is_1_0_0(self):
        """programmatic_pid __init__.py must declare __version__ == '1.0.0'.

        Reads the source file directly to avoid importing the package, which
        would fail in environments without its optional dependency ``ezdxf``.
        """
        repo_root = Path(__file__).resolve().parents[3]
        init_py = (
            repo_root / "src" / "shared" / "python" / "programmatic_pid" / "__init__.py"
        )
        assert init_py.exists(), f"programmatic_pid __init__.py not found at {init_py}"
        content = init_py.read_text()
        assert '__version__ = "1.0.0"' in content, (
            "Expected '__version__ = \"1.0.0\"' in programmatic_pid/__init__.py"
        )
