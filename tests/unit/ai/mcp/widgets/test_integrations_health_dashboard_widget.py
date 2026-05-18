# ruff: noqa: E501
"""Tests for the shared IntegrationsHealthDashboardWidget.

This widget surfaces live status for every configured integration
(Linear / Notion / Affine / Obsidian / MCP pool / providers / GitHub).
It pulls data from the pure-data health_query_api module so that the
same data is available to non-GUI consumers.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.shared.python.ai.mcp.widgets.health_query_api import (  # noqa: E402
    IntegrationStatus,
    IntegrationStatusLevel,
)
from src.shared.python.ai.mcp.widgets.integrations_health_dashboard_widget import (  # noqa: E402
    IntegrationsHealthDashboardWidget,
)


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _make_status(
    integration_id: str = "linear",
    *,
    level: IntegrationStatusLevel = IntegrationStatusLevel.OK,
    tools: int = 3,
) -> IntegrationStatus:
    return IntegrationStatus(
        integration_id=integration_id,
        display_name=integration_id.title(),
        level=level,
        message="seeded",
        tools_exposed=tools,
        latency_ms=10.0,
    )


@pytest.mark.unit
class TestIntegrationsHealthDashboardWidgetConstruction:
    def test_construct_empty(self, qapp: QApplication) -> None:
        widget = IntegrationsHealthDashboardWidget(status_provider=lambda: [])
        assert widget.row_count == 0
        # An empty state label should be visible.
        assert widget.has_empty_state_visible

    def test_construct_with_seed(self, qapp: QApplication) -> None:
        widget = IntegrationsHealthDashboardWidget(
            status_provider=lambda: [_make_status("linear")]
        )
        widget.refresh()
        assert widget.row_count == 1
        assert not widget.has_empty_state_visible


@pytest.mark.unit
class TestIntegrationsHealthDashboardWidgetRefresh:
    def test_refresh_replaces_rows(self, qapp: QApplication) -> None:
        sequence = iter(
            [
                [_make_status("linear")],
                [_make_status("linear"), _make_status("notion")],
            ]
        )
        widget = IntegrationsHealthDashboardWidget(
            status_provider=lambda: next(sequence)
        )
        widget.refresh()
        assert widget.row_count == 1
        widget.refresh()
        assert widget.row_count == 2

    def test_refresh_button_triggers_provider(self, qapp: QApplication) -> None:
        provider = MagicMock(return_value=[_make_status("linear")])
        widget = IntegrationsHealthDashboardWidget(status_provider=provider)
        widget.refresh()
        widget.refresh()
        assert provider.call_count == 2


@pytest.mark.unit
class TestIntegrationsHealthDashboardWidgetRowFormat:
    def test_row_shows_display_name_and_status(self, qapp: QApplication) -> None:
        widget = IntegrationsHealthDashboardWidget(
            status_provider=lambda: [
                _make_status("linear", level=IntegrationStatusLevel.OK, tools=4),
                _make_status("notion", level=IntegrationStatusLevel.ERROR, tools=0),
            ]
        )
        widget.refresh()
        rows = widget.row_data
        assert rows[0]["display_name"] == "Linear"
        assert rows[0]["level"] == "ok"
        assert rows[0]["tools_exposed"] == 4
        assert rows[1]["level"] == "error"

    def test_row_handles_unconfigured(self, qapp: QApplication) -> None:
        widget = IntegrationsHealthDashboardWidget(
            status_provider=lambda: [
                _make_status("affine", level=IntegrationStatusLevel.UNCONFIGURED),
            ]
        )
        widget.refresh()
        rows = widget.row_data
        assert rows[0]["level"] == "unconfigured"


@pytest.mark.unit
class TestIntegrationsHealthDashboardWidgetSafety:
    def test_provider_exception_is_surfaced_as_error_row(
        self, qapp: QApplication
    ) -> None:
        def boom() -> list[IntegrationStatus]:
            raise RuntimeError("backend offline")

        widget = IntegrationsHealthDashboardWidget(status_provider=boom)
        widget.refresh()
        # We expect a single error row reporting the failure rather than a
        # raised exception (the dashboard must never crash the host app).
        assert widget.row_count == 1
        assert widget.row_data[0]["level"] == "error"
        assert "backend offline" in widget.row_data[0]["message"]
