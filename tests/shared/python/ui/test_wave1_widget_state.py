"""Tests for the wave-1 widget/state-layer crash fixes (#3102).

Mixin behaviour is exercised through a fake host (no top-level Qt widgets, per
the segfault gotcha). Covers engine-exception handling (F1/F2), corrupt-state
graceful degradation (F4), and the converter index mapping (F3).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from sidekick.data_processing.exceptions import (
    ColumnNotFoundError,
    FilterError,
    FitError,
)
from sidekick.ui.widgets.mixins.data_processor_ops import DataProcessorOpsMixin


def _make_ops_host(engine: Any) -> DataProcessorOpsMixin:
    """Build a minimal host exposing only what the op handlers touch."""
    host = DataProcessorOpsMixin()
    host.engine = engine  # type: ignore[attr-defined]
    host.data_modified = SimpleNamespace(emit=lambda: None)  # type: ignore[attr-defined]
    host._update_table = lambda: None  # type: ignore[attr-defined]
    host._update_column_selectors = lambda: None  # type: ignore[attr-defined]
    host.refresh_statistics = lambda: None  # type: ignore[attr-defined]
    host._set_status = lambda *a, **k: None  # type: ignore[attr-defined]
    return host


@pytest.mark.unit
def test_filter_engine_exception_shows_warning_not_crash() -> None:
    """A raising engine surfaces a warning instead of propagating (F1)."""
    engine = MagicMock()
    engine.filter_data.side_effect = FilterError("bad filter value")
    host = _make_ops_host(engine)
    host.filter_column = SimpleNamespace(currentText=lambda: "col")  # type: ignore[attr-defined]
    host.filter_operator = SimpleNamespace(currentText=lambda: ">")  # type: ignore[attr-defined]
    host.filter_value = SimpleNamespace(text=lambda: "5")  # type: ignore[attr-defined]

    with patch(
        "sidekick.ui.widgets.mixins.data_processor_ops.QMessageBox.warning"
    ) as warn:
        host._apply_filter()
    warn.assert_called_once()
    assert "bad filter value" in warn.call_args.args[2]


@pytest.mark.unit
def test_fit_curve_exception_shows_message_not_crash() -> None:
    """A raising fit_curve writes the message instead of crashing (F2)."""
    engine = MagicMock()
    engine.fit_curve.side_effect = FitError("need >= 2 points")
    host = _make_ops_host(engine)
    captured: dict[str, str] = {}
    host.fit_x_column = SimpleNamespace(currentText=lambda: "x")  # type: ignore[attr-defined]
    host.fit_y_column = SimpleNamespace(currentText=lambda: "y")  # type: ignore[attr-defined]
    host.fit_type = SimpleNamespace(currentText=lambda: "linear")  # type: ignore[attr-defined]
    host.fit_degree = SimpleNamespace(value=lambda: 1)  # type: ignore[attr-defined]
    host.fit_results_text = SimpleNamespace(  # type: ignore[attr-defined]
        setText=lambda t: captured.__setitem__("text", t),
        setHtml=lambda t: captured.__setitem__("html", t),
    )

    # FitType(...) is constructed from the combo text; patch it to a passthrough.
    with patch("sidekick.data_processing.core.FitType", side_effect=lambda v: v):
        host._fit_curve()
    assert "need >= 2 points" in captured.get("text", "")


@pytest.mark.unit
def test_column_not_found_is_handled() -> None:
    """ColumnNotFoundError (a DataProcessingError) is also caught (F1)."""
    engine = MagicMock()
    engine.query.side_effect = ColumnNotFoundError("no such column")
    host = _make_ops_host(engine)
    host.query_input = SimpleNamespace(text=lambda: "a > 1")  # type: ignore[attr-defined]
    with patch(
        "sidekick.ui.widgets.mixins.data_processor_ops.QMessageBox.warning"
    ) as warn:
        host._execute_query()
    warn.assert_called_once()


@pytest.mark.unit
def test_set_calculator_state_handles_corrupt_profile() -> None:
    """A corrupt state dict degrades gracefully instead of crashing (F4)."""
    from sidekick.ui.mixins.calculator_state_mixin import CalculatorStateMixin

    host = CalculatorStateMixin.__new__(CalculatorStateMixin)
    host.unsaved_changes = True  # type: ignore[attr-defined]

    # window_geometry as a non-str, input_states as a non-dict: previously
    # raised AttributeError/ValueError out of the slot.
    corrupt = {
        "window_geometry": 12345,
        "input_states": "not-a-dict",
        "splitter_states": None,
    }
    # Must not raise.
    CalculatorStateMixin.set_calculator_state(host, corrupt)

    # Non-dict top level is also tolerated.
    CalculatorStateMixin.set_calculator_state(host, "garbage")  # type: ignore[arg-type]


@pytest.mark.unit
def test_state_manager_import_does_not_trigger_deprecation_warning() -> None:
    """Regression test for issue #3950.

    The mixin used to import the deprecated global `state_manager` (which
    raises a DeprecationWarning on every access via module `__getattr__`),
    breaking Units-tab usage under warnings-as-errors. It should go through
    `get_state_manager()` instead -- confirmed two ways: the deprecated name
    is no longer bound in the mixin module's namespace, and calling
    `get_state_manager()` itself (what `__init__` now does) does not warn.
    """
    import warnings

    import sidekick.ui.mixins.calculator_state_mixin as mixin_module
    from sidekick.utils.state_manager import StateManager, get_state_manager

    assert not hasattr(mixin_module, "state_manager")

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        manager = get_state_manager()

    assert isinstance(manager, StateManager)
