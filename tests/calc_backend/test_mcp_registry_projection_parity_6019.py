"""Epic #6019: REST + MCP projections must share one registry (#6020, #6021)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("calc_backend")

from shared.python.calc_backend import mcp_server
from shared.python.calc_backend.agent_api import CONTRACT_VERSION
from shared.python.calc_backend.registry.store import (
    CALCULATOR_REGISTRY,
    describe_calculator,
    registered_calculators,
)


async def _mcp_describe_via_fastmcp(calculator_id: str) -> dict[str, Any]:
    """Invoke the real MCP describe_calculator tool closure (_build_fastmcp)."""
    pytest.importorskip("mcp.server.fastmcp")
    server = mcp_server._build_fastmcp()
    assert server is not None
    _content, payload = await server.call_tool(
        "describe_calculator", {"id": calculator_id}
    )
    return payload


@pytest.mark.unit
def test_list_calculators_projection_matches_registry_ids() -> None:
    mcp_server._ensure_registry()
    page = mcp_server._list_page(category=None, query=None, limit=500, cursor=None)
    listed_ids = {item["id"] for item in page["calculators"]}
    registry_ids = {spec.id for spec in registered_calculators()}
    assert listed_ids == registry_ids


@pytest.mark.unit
def test_list_calculators_projection_carries_registry_metadata() -> None:
    mcp_server._ensure_registry()
    by_id = {spec.id: spec for spec in registered_calculators()}
    page = mcp_server._list_page(category=None, query=None, limit=500, cursor=None)
    for item in page["calculators"]:
        spec = by_id[item["id"]]
        assert item["title"] == spec.title
        assert item["category"] == spec.category
        assert item["maturity"] == spec.maturity


@pytest.mark.unit
def test_mcp_describe_unknown_id_uses_tool_error_envelope() -> None:
    """Guards against comparing registry_describe only (KeyError, not FAILED envelope)."""
    mcp_server._ensure_registry()
    payload = asyncio.run(_mcp_describe_via_fastmcp("not-a-registered-calculator-id"))
    assert payload.get("status") == "FAILED"
    assert payload["errors"][0]["field"] == "id"


@pytest.mark.unit
def test_mcp_describe_projection_matches_registry_for_every_calculator() -> None:
    mcp_server._ensure_registry()
    for calculator_id in CALCULATOR_REGISTRY:
        projected = asyncio.run(_mcp_describe_via_fastmcp(calculator_id))
        expected = dict(describe_calculator(calculator_id))
        expected["contract_version"] = CONTRACT_VERSION
        assert projected == expected, calculator_id