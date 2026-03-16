"""Tests for urdf_builder_gui/contracts.py — lightweight DbC support."""

from __future__ import annotations

import pytest

from urdf_builder_gui.contracts import (
    PostconditionError,
    PreconditionError,
    ensure,
    require,
)


class TestRequire:
    """Tests for the require() precondition function."""

    def test_passes_when_condition_true(self) -> None:
        require(True, "should not raise")  # no exception

    def test_raises_precondition_error_when_false(self) -> None:
        with pytest.raises(PreconditionError, match="value must be positive"):
            require(False, "value must be positive")

    def test_precondition_error_is_value_error(self) -> None:
        with pytest.raises(ValueError):
            require(False, "must fail")

    def test_message_included_in_error(self) -> None:
        msg = "my custom error message"
        with pytest.raises(PreconditionError, match=msg):
            require(1 > 2, msg)

    def test_args_appended_to_message(self) -> None:
        with pytest.raises(PreconditionError, match="got: 42"):
            require(False, "bad value", 42)

    def test_multiple_args_in_message(self) -> None:
        with pytest.raises(PreconditionError, match="got: foo, bar"):
            require(False, "bad inputs", "foo", "bar")

    def test_disabled_mode_skips_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DBC_LEVEL", "disabled")
        # Re-import to pick up env change — instead test by reaching into module
        import urdf_builder_gui.contracts as c

        original = c._DBC_LEVEL
        c._DBC_LEVEL = "disabled"
        try:
            require(False, "should not raise when disabled")
        finally:
            c._DBC_LEVEL = original

    def test_zero_is_falsy(self) -> None:
        with pytest.raises(PreconditionError):
            require(0, "zero is falsy")  # type: ignore[arg-type]

    def test_none_is_falsy(self) -> None:
        with pytest.raises(PreconditionError):
            require(None, "none is falsy")  # type: ignore[arg-type]


class TestEnsure:
    """Tests for the ensure() postcondition function."""

    def test_passes_when_condition_true(self) -> None:
        ensure(True, "should pass")  # no exception

    def test_raises_postcondition_error_when_false(self) -> None:
        with pytest.raises(PostconditionError, match="result must be valid"):
            ensure(False, "result must be valid")

    def test_postcondition_error_is_value_error(self) -> None:
        with pytest.raises(ValueError):
            ensure(False, "postcondition failed")

    def test_message_included_in_error(self) -> None:
        msg = "post condition violated"
        with pytest.raises(PostconditionError, match=msg):
            ensure(False, msg)

    def test_args_appended_to_message(self) -> None:
        with pytest.raises(PostconditionError, match="got: -1"):
            ensure(False, "bad result", -1)

    def test_disabled_skips_check(self) -> None:
        import urdf_builder_gui.contracts as c

        original = c._DBC_LEVEL
        c._DBC_LEVEL = "disabled"
        try:
            ensure(False, "should not raise when disabled")
        finally:
            c._DBC_LEVEL = original

    def test_non_empty_string_is_truthy(self) -> None:
        ensure("result", "has a value")  # type: ignore[arg-type]


class TestContractsPublicApi:
    """Smoke-test the __all__ exports."""

    def test_all_exports_available(self) -> None:
        from urdf_builder_gui.contracts import __all__

        assert "require" in __all__
        assert "ensure" in __all__
        assert "PreconditionError" in __all__
        assert "PostconditionError" in __all__

    def test_precondition_error_class(self) -> None:
        err = PreconditionError("test")
        assert isinstance(err, ValueError)
        assert str(err) == "test"

    def test_postcondition_error_class(self) -> None:
        err = PostconditionError("post test")
        assert isinstance(err, ValueError)
        assert str(err) == "post test"
