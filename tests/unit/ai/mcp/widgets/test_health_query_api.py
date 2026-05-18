"""Tests for the integrations health query API (pure-data layer).

This API is consumed by the IntegrationsHealthDashboardWidget but is
intentionally Qt-free so that any consumer (CLI, server, alt-GUI) can
ask the same question: "which integrations are healthy right now?"
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.shared.python.ai.mcp.widgets.health_query_api import (
    IntegrationStatus,
    IntegrationStatusLevel,
    list_all_integrations,
    query_integration_status,
)


@pytest.mark.unit
class TestIntegrationStatus:
    def test_round_trip(self) -> None:
        status = IntegrationStatus(
            integration_id="linear",
            display_name="Linear",
            level=IntegrationStatusLevel.OK,
            message="connected",
            tools_exposed=4,
            latency_ms=23.5,
        )
        assert status.integration_id == "linear"
        assert status.level is IntegrationStatusLevel.OK
        assert status.tools_exposed == 4

    def test_level_enum_values(self) -> None:
        assert IntegrationStatusLevel.OK.value == "ok"
        assert IntegrationStatusLevel.DEGRADED.value == "degraded"
        assert IntegrationStatusLevel.ERROR.value == "error"
        assert IntegrationStatusLevel.UNCONFIGURED.value == "unconfigured"


@pytest.mark.unit
class TestQueryIntegrationStatus:
    def test_unknown_integration_returns_unconfigured(self) -> None:
        status = query_integration_status("does-not-exist")
        assert status.level is IntegrationStatusLevel.UNCONFIGURED
        assert status.integration_id == "does-not-exist"

    def test_linear_uses_token_probe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LINEAR_API_KEY", "lin_test_token")
        status = query_integration_status("linear")
        assert status.integration_id == "linear"
        # With a token set we are at least "configured" (not UNCONFIGURED).
        assert status.level is not IntegrationStatusLevel.UNCONFIGURED

    def test_linear_without_token_unconfigured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("LINEAR_API_KEY", raising=False)
        # Also clear any module-level token cached.
        from src.shared.python.ai.integrations import linear as linear_mod

        linear_mod._LINEAR_API_TOKEN = None  # noqa: SLF001 — test cleanup
        status = query_integration_status("linear")
        assert status.level is IntegrationStatusLevel.UNCONFIGURED


@pytest.mark.unit
class TestListAllIntegrations:
    def test_lists_known_integrations(self) -> None:
        statuses = list_all_integrations()
        ids = {s.integration_id for s in statuses}
        # Core set must be present.
        assert "linear" in ids
        assert "notion" in ids
        assert "affine" in ids
        assert "obsidian" in ids
        assert "mcp-pool" in ids

    def test_mcp_pool_status_with_injected_pool(self) -> None:
        # The pool is injected so tests don't need a real one.
        pool = MagicMock()
        pool.connected_count.return_value = 2
        pool.total_count.return_value = 3
        statuses = list_all_integrations(mcp_pool=pool)
        mcp_status = next(s for s in statuses if s.integration_id == "mcp-pool")
        assert mcp_status.tools_exposed >= 0
        # 2 of 3 connected → DEGRADED (partial connectivity).
        assert mcp_status.level is IntegrationStatusLevel.DEGRADED

    def test_mcp_pool_all_connected_is_ok(self) -> None:
        pool = MagicMock()
        pool.connected_count.return_value = 3
        pool.total_count.return_value = 3
        statuses = list_all_integrations(mcp_pool=pool)
        mcp_status = next(s for s in statuses if s.integration_id == "mcp-pool")
        assert mcp_status.level is IntegrationStatusLevel.OK

    def test_mcp_pool_zero_total_is_unconfigured(self) -> None:
        pool = MagicMock()
        pool.connected_count.return_value = 0
        pool.total_count.return_value = 0
        statuses = list_all_integrations(mcp_pool=pool)
        mcp_status = next(s for s in statuses if s.integration_id == "mcp-pool")
        assert mcp_status.level is IntegrationStatusLevel.UNCONFIGURED
