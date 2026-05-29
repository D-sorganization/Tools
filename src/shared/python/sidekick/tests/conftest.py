"""Pytest configuration for upstream_drift_tools tests.

Uses shared path setup from utils.path_helpers.
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import patch

import pytest
from PyQt6.QtWidgets import QMessageBox
from utils.path_helpers import ensure_utils_in_path

# Ensure utils is available for test imports
ensure_utils_in_path()


@pytest.fixture(autouse=True)
def _no_blocking_message_boxes() -> Iterator[None]:
    """Prevent informational modal dialogs from blocking headless test runs.

    ``QMessageBox.warning/critical/information/about`` are modal and block on
    user input. Under full-suite ordering, widget-state pollution can drive a
    widget down an error path that pops one of these dialogs in a test that did
    not patch it locally, hanging the entire run (see issue #3096). These
    methods' return values are never meaningfully consumed by production code,
    so stubbing them to a non-blocking ``Ok`` is safe.

    ``QMessageBox.question`` is intentionally **not** stubbed: its return value
    drives control flow, and the tests that exercise it patch it explicitly.
    Local ``patch`` calls inside individual tests still take precedence over
    this fixture within their context.
    """
    ok = QMessageBox.StandardButton.Ok
    with (
        patch.object(QMessageBox, "warning", return_value=ok),
        patch.object(QMessageBox, "critical", return_value=ok),
        patch.object(QMessageBox, "information", return_value=ok),
        patch.object(QMessageBox, "about", return_value=None),
    ):
        yield
