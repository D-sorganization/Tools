"""Fail-closed canonical migration gateways for ground contract payloads."""

from __future__ import annotations

from typing import Any, cast

from .contract_records import GroundSimulationRequest, GroundSimulationResult


def migrate_request_to_current(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize a request; v1 has no implicit predecessor."""
    request = cast(GroundSimulationRequest, GroundSimulationRequest.from_dict(payload))
    return request.to_dict()


def migrate_result_to_current(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize a result; v1 has no implicit predecessor."""
    result = cast(GroundSimulationResult, GroundSimulationResult.from_dict(payload))
    return result.to_dict()


__all__ = ["migrate_request_to_current", "migrate_result_to_current"]
