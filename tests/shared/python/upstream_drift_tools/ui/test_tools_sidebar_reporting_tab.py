"""Tests for tools_sidebar reporting_tab.

Covers:
- _ReportWorker happy path (finished_with_report signal emitted)
- _ReportWorker failure path (failed signal emitted)
- Fallback path when no generator is configured
- _gather_session_context error is logged as warning, not silently swallowed
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from upstream_drift_tools.ui.tools_sidebar.reporting_tab import (
    SidekickReportingWidget,
    _format_local_report,
    _gather_session_context,
    _ReportWorker,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sidebar(
    *,
    variables: list[str] | None = None,
    project_root: str = "/fake/root",
    registry_raises: bool = False,
) -> MagicMock:
    """Return a minimal sidebar mock.

    Args:
        variables: Names to expose via ``sidebar.registry.variables()``.
        project_root: String returned for ``sidebar.project_root``.
        registry_raises: When True the registry raises RuntimeError.
    """
    sidebar = MagicMock()
    sidebar.project_root = project_root
    if registry_raises:
        sidebar.registry.variables.side_effect = RuntimeError("registry not ready")
    else:
        names = variables or []
        mock_vars = []
        for n in names:
            var = MagicMock()
            var.name = n
            mock_vars.append(var)
        sidebar.registry.variables.return_value = mock_vars
    return sidebar


def _make_generator(
    *, raises: Exception | None = None, result: str = "insights"
) -> Any:
    """Return an async mock generator.

    Args:
        raises: If provided, ``generate_agentic_insights`` will raise this.
        result: The string to return on success.
    """
    gen = MagicMock()

    async def _async_gen(context: dict[str, Any]) -> str:
        if raises is not None:
            raise raises
        return result

    gen.generate_agentic_insights = _async_gen
    return gen


# ---------------------------------------------------------------------------
# _gather_session_context
# ---------------------------------------------------------------------------


def test_gather_session_context_returns_expected_keys() -> None:
    sidebar = _make_sidebar(variables=["x", "y"], project_root="/proj")
    ctx = _gather_session_context(sidebar)

    assert ctx["workspace_variables"] == ["x", "y"]
    assert ctx["project_root"] == "/proj"


def test_gather_session_context_logs_warning_on_registry_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sidebar = _make_sidebar(registry_raises=True)

    _logger = "upstream_drift_tools.ui.tools_sidebar.reporting_tab"
    with caplog.at_level(logging.WARNING, logger=_logger):
        ctx = _gather_session_context(sidebar)

    assert ctx["workspace_variables"] == []
    assert any("workspace variables" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# _format_local_report
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "variables,expected_fragment",
    [
        (["alpha", "beta"], "- `alpha`"),
        ([], "_(none)_"),
    ],
)
def test_format_local_report_renders_variable_list(
    variables: list[str], expected_fragment: str
) -> None:
    ctx: dict[str, Any] = {"workspace_variables": variables, "project_root": "/p"}
    report = _format_local_report(ctx)

    assert "# Session Report" in report
    assert expected_fragment in report


# ---------------------------------------------------------------------------
# _ReportWorker
# ---------------------------------------------------------------------------


def test_report_worker_emits_finished_on_success() -> None:
    """Worker should emit ``finished_with_report`` when generator succeeds."""
    generator = _make_generator(result="detailed insights")
    ctx: dict[str, Any] = {"workspace_variables": [], "project_root": "."}

    received: list[Any] = []
    worker = _ReportWorker(generator=generator, context=ctx)
    worker.finished_with_report.connect(received.append)

    worker.run()  # run synchronously in test — no Qt app needed for run()

    assert received == ["detailed insights"]


def test_report_worker_emits_failed_on_exception() -> None:
    """Worker should emit ``failed`` with the error message on exception."""
    generator = _make_generator(raises=ValueError("backend offline"))
    ctx: dict[str, Any] = {"workspace_variables": [], "project_root": "."}

    errors: list[str] = []
    worker = _ReportWorker(generator=generator, context=ctx)
    worker.failed.connect(errors.append)

    worker.run()

    assert errors == ["backend offline"]


def test_report_worker_requires_generator() -> None:
    """Constructing a worker with generator=None raises ValueError."""
    with pytest.raises(ValueError, match="generator must be provided"):
        _ReportWorker(generator=None, context={})


@pytest.mark.parametrize(
    "exc_type,message",
    [
        (RuntimeError, "loop error"),
        (OSError, "network unreachable"),
        (TimeoutError, "timed out"),
    ],
)
def test_report_worker_captures_various_exception_types(
    exc_type: type[Exception], message: str
) -> None:
    """Worker failure signal is emitted for any exception type."""
    generator = _make_generator(raises=exc_type(message))
    ctx: dict[str, Any] = {}

    errors: list[str] = []
    worker = _ReportWorker(generator=generator, context=ctx)
    worker.failed.connect(errors.append)
    worker.run()

    assert errors == [message]


# ---------------------------------------------------------------------------
# SidekickReportingWidget — fallback path (no generator)
# ---------------------------------------------------------------------------


def test_widget_falls_back_to_local_report_when_no_generator(
    qtbot: Any,
) -> None:
    """When no generator is available, clicking Generate produces a local report."""
    sidebar = _make_sidebar(variables=["flow_rate"], project_root="/project")
    _mod = (
        "upstream_drift_tools.ui.tools_sidebar"
        ".reporting_tab._try_import_report_generator"
    )
    with patch(_mod, return_value=None):
        widget = SidekickReportingWidget(sidebar=sidebar)
    qtbot.addWidget(widget)

    assert widget._report_generator is None

    widget._on_generate_clicked()

    preview_text = widget._report_preview.toPlainText()
    assert "# Session Report" in preview_text
    assert "/project" in preview_text


def test_widget_requires_sidebar() -> None:
    """SidekickReportingWidget raises ValueError when sidebar is None."""
    with pytest.raises(ValueError, match="sidebar must be provided"):
        SidekickReportingWidget(sidebar=None)


# ---------------------------------------------------------------------------
# SidekickReportingWidget — worker integration (mocked QThread)
# ---------------------------------------------------------------------------


def test_widget_creates_worker_and_connects_signals_when_generator_present(
    qtbot: Any,
) -> None:
    """When a generator is configured, _run_async_report creates a worker."""
    sidebar = _make_sidebar()
    widget = SidekickReportingWidget(sidebar=sidebar)
    qtbot.addWidget(widget)

    generator = _make_generator(result="ai insights")
    widget._report_generator = generator

    with patch.object(_ReportWorker, "start") as mock_start:
        widget._run_async_report({"workspace_variables": [], "project_root": "."})

    assert widget._worker is not None
    mock_start.assert_called_once()


def test_widget_on_worker_report_includes_insights(qtbot: Any) -> None:
    """_on_worker_report appends insights to the local report."""
    sidebar = _make_sidebar(variables=["v1"], project_root="/x")
    widget = SidekickReportingWidget(sidebar=sidebar)
    qtbot.addWidget(widget)

    widget._on_worker_report("my insights text")

    preview = widget._report_preview.toPlainText()
    assert "## Agentic Insights" in preview
    assert "my insights text" in preview


def test_widget_on_report_failed_logs_error_and_shows_fallback(
    qtbot: Any,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """_on_report_failed logs the error and shows a local report with error section."""
    sidebar = _make_sidebar(project_root="/y")
    widget = SidekickReportingWidget(sidebar=sidebar)
    qtbot.addWidget(widget)

    _logger = "upstream_drift_tools.ui.tools_sidebar.reporting_tab"
    with caplog.at_level(logging.ERROR, logger=_logger):
        widget._on_report_failed("something went wrong")

    preview = widget._report_preview.toPlainText()
    assert "## Insights Error" in preview
    assert "something went wrong" in preview
    assert any("something went wrong" in r.message for r in caplog.records)
