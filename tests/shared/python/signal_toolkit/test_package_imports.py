"""Package import contract tests for Signal Toolkit."""

from __future__ import annotations

import importlib
import sys


def test_signal_toolkit_package_import_does_not_load_optional_widget_stack(
    monkeypatch,
) -> None:
    """Importing the package must not import optional GUI or SciPy calculus modules."""
    for module_name in list(sys.modules):
        if module_name == "signal_toolkit" or module_name.startswith("signal_toolkit."):
            monkeypatch.delitem(sys.modules, module_name, raising=False)

    importlib.import_module("signal_toolkit")

    assert "signal_toolkit.widget" not in sys.modules
    assert "signal_toolkit.widget_plotting" not in sys.modules
    assert "signal_toolkit.calculus" not in sys.modules
