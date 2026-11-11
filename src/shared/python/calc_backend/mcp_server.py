"""Stdio MCP server for engineering calculators (issues #6021, #6022)."""

from __future__ import annotations

import logging
import sys
from typing import Any

from .agent_api import (
    CONTRACT_VERSION,
    build_result,
    parse_inputs,
    serialize_outputs,
    validation_problems,
)
from .registry.store import CALCULATOR_REGISTRY, registered_calculators
from .registry.store import describe_calculator as registry_describe
from .registry.wire import wire_calculator_registry

logger = logging.getLogger(__name__)

_LIST_DEFAULT_LIMIT = 50
_CONVERT_UNITS_ID = "flow-rate"


def _ensure_registry() -> None:
    if CALCULATOR_REGISTRY:
        return
    wire_calculator_registry()


def _list_page(
    *,
    category: str | None,
    query: str | None,
    limit: int,
    cursor: str | None,
) -> dict[str, Any]:
    _ensure_registry()
    limit = max(1, min(limit, _LIST_DEFAULT_LIMIT))
    items = [
        {
            "id": spec.id,
            "title": spec.title,
            "category": spec.category,
            "maturity": spec.maturity,
        }
        for spec in registered_calculators()
    ]
    if category:
        items = [item for item in items if item["category"] == category]
    if query:
        needle = query.lower()
        items = [
            item
            for item in items
            if needle in item["id"].lower() or needle in item["title"].lower()
        ]
    start = int(cursor or 0)
    page = items[start : start + limit]
    next_cursor = str(start + limit) if start + limit < len(items) else None
    return {
        "calculators": page,
        "next_cursor": next_cursor,
        "contract_version": CONTRACT_VERSION,
    }


def _run_calculator(calculator_id: str, inputs: dict[str, Any]) -> dict[str, Any]:
    _ensure_registry()
    entry = CALCULATOR_REGISTRY.get(calculator_id)
    if entry is None:
        return {
            "status": "FAILED",
            "errors": [
                {
                    "field": "id",
                    "problem": f"unknown calculator {calculator_id!r}",
                    "allowed_range": sorted(CALCULATOR_REGISTRY),
                    "example": "flow-rate",
                }
            ],
            "provenance": {"contract_version": CONTRACT_VERSION},
        }
    spec = entry.spec
    try:
        request = parse_inputs(spec.request_model, inputs)
    except Exception as exc:
        from pydantic import ValidationError

        if isinstance(exc, ValidationError):
            return {
                "status": "FAILED",
                "errors": validation_problems(exc),
                "provenance": {"contract_version": CONTRACT_VERSION},
            }
        raise
    try:
        raw = spec.fn(request)
    except Exception as exc:
        logger.exception("calculator %s failed", calculator_id)
        return {
            "status": "FAILED",
            "errors": [
                {
                    "field": "inputs",
                    "problem": str(exc),
                    "allowed_range": None,
                    "example": registry_describe(calculator_id)["worked_example"][
                        "request"
                    ],
                }
            ],
            "provenance": {"contract_version": CONTRACT_VERSION},
        }
    outputs, warnings = serialize_outputs(raw)
    return build_result(
        calculator_id=calculator_id,
        formula_ids=spec.formula_ids,
        outputs=outputs,
        warnings=warnings,
        units=dict(spec.units),
        status="SUCCEEDED",
    )


def _build_fastmcp():
    try:
        from mcp.server.fastmcp import FastMCP  # type: ignore[import-not-found]
    except ImportError:
        return None

    mcp = FastMCP("ud-tools-calculators")

    @mcp.tool(annotations={"readOnlyHint": True, "idempotentHint": True})
    def list_calculators(
        category: str | None = None,
        query: str | None = None,
        limit: int = _LIST_DEFAULT_LIMIT,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """List registered calculators with optional filter and cursor paging."""
        return _list_page(category=category, query=query, limit=limit, cursor=cursor)

    @mcp.tool(annotations={"readOnlyHint": True, "idempotentHint": True})
    def describe_calculator(id: str) -> dict[str, Any]:
        """Return JSON Schema, units, ranges, and worked example for one calculator."""
        _ensure_registry()
        if id not in CALCULATOR_REGISTRY:
            return {
                "status": "FAILED",
                "errors": [
                    {
                        "field": "id",
                        "problem": f"unknown calculator {id!r}",
                        "allowed_range": sorted(CALCULATOR_REGISTRY),
                        "example": "flow-rate",
                    }
                ],
            }
        payload = registry_describe(id)
        payload["contract_version"] = CONTRACT_VERSION
        return payload

    @mcp.tool(annotations={"idempotentHint": True})
    def run_calculation(
        id: str, inputs: dict[str, Any], options: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Execute one registered calculator with structured validation errors."""
        _ = options
        return _run_calculator(id, inputs)

    @mcp.tool(annotations={"idempotentHint": True})
    def convert_units(
        value: float,
        from_unit: str,
        to_unit: str,
        category: str = "mass",
    ) -> dict[str, Any]:
        """Flow-rate unit conversion via the registered flow-rate calculator."""
        return _run_calculator(
            _CONVERT_UNITS_ID,
            {
                "value": value,
                "from_unit": from_unit,
                "to_unit": to_unit,
                "category": category,
            },
        )

    return mcp


def main(argv: list[str] | None = None) -> int:
    server = _build_fastmcp()
    if server is None:
        sys.stderr.write(
            "ud-tools-mcp: install the MCP SDK: pip install 'ud-tools[calc-mcp]'\n"
        )
        return 2
    _ensure_registry()
    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
