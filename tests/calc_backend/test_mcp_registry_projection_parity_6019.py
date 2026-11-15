"""Epic #6019: REST + MCP projections must share one registry (#6020, #6021)."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("calc_backend")

from shared.python.calc_backend import mcp_server
from shared.python.calc_backend.registry.store import (
    CALCULATOR_REGISTRY,
    describe_calculator,
    registered_calculators,
)


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
def test_mcp_describe_projection_matches_registry_for_every_calculator() -> None:
    from shared.python.calc_backend.agent_api import CONTRACT_VERSION

    mcp_server._ensure_registry()
    for calculator_id in CALCULATOR_REGISTRY:
        registry_payload = describe_calculator(calculator_id)
        projected = dict(mcp_server.registry_describe(calculator_id))
        projected["contract_version"] = CONTRACT_VERSION
        expected = dict(registry_payload)
        expected["contract_version"] = CONTRACT_VERSION
        assert projected == expected, calculator_id
