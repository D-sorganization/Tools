"""Tests for ODE Solver PyQt6 main window — DbC and helper methods.

Covers:
- _parse_dict_input: type guard, valid input, edge cases
- _on_preset_changed: DbC type/value guards (precondition only, no Qt init)

These tests deliberately avoid instantiating ODESolverWindow (which requires
a live QApplication and display) by invoking the target methods via
types.MethodType or by calling the unbound function with a mock self.
All precondition branches complete before any Qt interaction occurs.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

# Import the functions under test by extracting them from the module — avoids
# importing at module level which triggers Qt enum evaluation at module-load time.
# That is fine in the test environment as long as PyQt6 is installed.
from ode_solver.ui.pyqt6.main_window import ODESolverWindow


class TestParseDictInput:
    """Unit tests for ODESolverWindow._parse_dict_input."""

    def _call(self, text: str) -> dict[str, str]:
        """Call _parse_dict_input on a mock instance (no Qt init needed)."""
        mock_self = MagicMock(spec=ODESolverWindow)
        return ODESolverWindow._parse_dict_input(mock_self, text)

    def test_valid_single_pair(self) -> None:
        """Single key: value pair is parsed correctly."""
        result = self._call("k: 0.1")
        assert result == {"k": "0.1"}

    def test_valid_multiple_pairs(self) -> None:
        """Multiple key: value pairs across lines are all parsed."""
        result = self._call("k: 0.1\nT_env: 350")
        assert result == {"k": "0.1", "T_env": "350"}

    def test_empty_string_returns_empty_dict(self) -> None:
        """Empty input returns an empty dict (not an error)."""
        result = self._call("")
        assert result == {}

    def test_lines_without_colon_are_skipped(self) -> None:
        """Lines without a colon separator are silently ignored."""
        result = self._call("no_colon_here\nk: 0.5")
        assert result == {"k": "0.5"}

    def test_value_with_colon_preserved(self) -> None:
        """Value may itself contain colons — only first colon splits key/value."""
        result = self._call("url: http://example.com")
        assert result == {"url": "http://example.com"}

    def test_whitespace_stripped(self) -> None:
        """Leading/trailing whitespace on keys and values is stripped."""
        result = self._call("  alpha :  beta  ")
        assert result == {"alpha": "beta"}

    def test_blank_lines_ignored(self) -> None:
        """Blank lines between pairs are skipped."""
        result = self._call("a: 1\n\nb: 2")
        assert result == {"a": "1", "b": "2"}

    def test_type_error_for_non_str(self) -> None:
        """Passing a non-str raises TypeError (DbC)."""
        mock_self = MagicMock(spec=ODESolverWindow)
        with pytest.raises(TypeError, match="text must be str"):
            ODESolverWindow._parse_dict_input(mock_self, 42)

    def test_type_error_for_none(self) -> None:
        """Passing None raises TypeError (DbC) — replaces old assert None check."""
        mock_self = MagicMock(spec=ODESolverWindow)
        with pytest.raises(TypeError, match="text must be str"):
            ODESolverWindow._parse_dict_input(mock_self, None)

    def test_type_error_for_list(self) -> None:
        """Passing a list raises TypeError (DbC)."""
        mock_self = MagicMock(spec=ODESolverWindow)
        with pytest.raises(TypeError, match="text must be str"):
            ODESolverWindow._parse_dict_input(mock_self, ["k: 1"])


class TestRunSolveValidationHandling:
    """_run_solve reports invalid input instead of raising (issue #3321).

    Exercised via an unbound call with a mock ``self`` so no live QApplication
    is required. The key guarantee is that bad user input never escapes the slot
    (which would abort the whole app under PyQt6).
    """

    def _mock_window(self) -> MagicMock:
        # No spec= here: _run_solve reads instance attributes (the Qt edit
        # widgets, results_text) that are not part of the class namespace, so a
        # spec-restricted mock would reject them. The widgets' return values are
        # irrelevant because _parse_dict_input is stubbed below.
        mock_self = MagicMock()
        mock_self._SOLVER_TIMEOUT_S = 30.0
        return mock_self

    def test_bad_parameter_value_is_reported_not_raised(self) -> None:
        mock_self = self._mock_window()
        # derivatives valid; a parameter value is non-numeric -> float() raises
        # ValueError inside _run_solve, which must be caught.
        mock_self._parse_dict_input.side_effect = [
            {"x": "-k*x"},  # derivatives
            {"k": "abc"},  # parameters (non-numeric)
            {"x": "1.0"},  # initial conditions
        ]
        ODESolverWindow._run_solve(mock_self)  # must NOT raise

        mock_self.results_text.setPlainText.assert_called_once()
        message = mock_self.results_text.setPlainText.call_args[0][0]
        assert "Invalid input" in message

    def test_missing_initial_condition_is_reported_not_raised(self) -> None:
        mock_self = self._mock_window()
        mock_self._parse_dict_input.side_effect = [
            {"x": "-k*x"},  # derivatives
            {"k": "0.5"},  # parameters
            {},  # initial conditions — missing x
        ]
        ODESolverWindow._run_solve(mock_self)  # must NOT raise

        message = mock_self.results_text.setPlainText.call_args[0][0]
        assert "Invalid input" in message


class TestThemeIntegration:
    """ODE Solver should delegate colors to the shared theme manager."""

    def test_launch_uses_saved_theme_stylesheet(
        self, qapp, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        from theme.theme_manager import ThemeManager, get_theme_manager

        settings_app = f"ODESolverWindowThemeTest-{tmp_path.name}"
        monkeypatch.setattr(ODESolverWindow, "_settings_app", settings_app)
        ThemeManager.reset_instance()
        bootstrap_manager = get_theme_manager(settings_app=settings_app)
        previous_theme = bootstrap_manager.get_theme_preference()
        bootstrap_manager.change_theme("Light")
        ThemeManager.reset_instance()

        window = ODESolverWindow()
        manager = window.get_theme_manager()
        try:
            assert manager.get_current_theme_name() == "Light"
            assert window.styleSheet() == manager.get_current_stylesheet()
        finally:
            if previous_theme in manager.get_available_themes():
                manager.change_theme(previous_theme)
            window.close()
            ThemeManager.reset_instance()

    def test_theme_change_reapplies_result_color_tokens(
        self, qapp, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        from theme.theme_manager import ThemeManager

        settings_app = f"ODESolverWindowThemeTest-{tmp_path.name}"
        monkeypatch.setattr(ODESolverWindow, "_settings_app", settings_app)
        ThemeManager.reset_instance()
        window = ODESolverWindow()
        manager = window.get_theme_manager()
        previous_theme = manager.get_theme_preference()
        try:
            manager.change_theme("Dark")
            dark_stylesheet = window.styleSheet()
            window._set_results_status("success")
            dark_results_style = window.results_text.styleSheet()

            manager.change_theme("Light")
            qapp.processEvents()

            light_success = manager.get_current_colors()["success"]
            assert window.styleSheet() != dark_stylesheet
            assert light_success in window.results_text.styleSheet()
            assert window.results_text.styleSheet() != dark_results_style
            assert "#a6e3a1" not in window.results_text.styleSheet().lower()
        finally:
            if previous_theme in manager.get_available_themes():
                manager.change_theme(previous_theme)
            window.close()
            ThemeManager.reset_instance()


class TestOnPresetChangedPreconditions:
    """Unit tests for ODESolverWindow._on_preset_changed DbC guards.

    Only the precondition branches are tested here (they complete before any
    Qt widget interaction). Full integration testing of preset loading requires
    a live QApplication and is omitted as headless-unsafe.
    """

    def test_type_error_for_non_str(self) -> None:
        """Passing a non-str raises TypeError (DbC)."""
        mock_self = MagicMock(spec=ODESolverWindow)
        with pytest.raises(TypeError, match="preset_name must be str"):
            ODESolverWindow._on_preset_changed(mock_self, 123)

    def test_type_error_for_none(self) -> None:
        """Passing None raises TypeError (DbC) — replaces old assert None check."""
        mock_self = MagicMock(spec=ODESolverWindow)
        with pytest.raises(TypeError, match="preset_name must be str"):
            ODESolverWindow._on_preset_changed(mock_self, None)

    def test_value_error_for_empty_string(self) -> None:
        """Passing an empty string raises ValueError (DbC)."""
        mock_self = MagicMock(spec=ODESolverWindow)
        with pytest.raises(ValueError, match="preset_name must not be empty"):
            ODESolverWindow._on_preset_changed(mock_self, "")
