"""Strict authority-response validation before public gateway publication."""

from __future__ import annotations

from typing import Final

from rate_of_closure.application._workspace_validation import exact_mapping
from rate_of_closure.application.regional_ground_authority_status import (
    regional_ground_authority_job_status_from_json,
)
from rate_of_closure.application.regional_ground_execution_job import (
    regional_ground_execution_job_from_json,
    regional_ground_execution_job_to_json,
)
from rate_of_closure.application.regional_ground_execution_result import (
    regional_ground_execution_result_from_json,
    regional_ground_execution_result_to_json,
)
from rate_of_closure.web_authority.capability import AuthorityCapability
from shared.python.swing_sim.canonical_numeric_json import canonical_numeric_json
from shared.python.swing_sim.ground.strict_json import strict_json_object

from .contracts import CompanionApiOperation, CompanionRoute

_ERROR_FIELDS: Final = frozenset({"code", "detail"})
_MAX_ERROR_DETAIL: Final = 240
_ERROR_CODES: Final = {
    400: frozenset({"invalid_job", "invalid_preparation"}),
    404: frozenset({"job_not_found"}),
    409: frozenset({"job_conflict", "result_unavailable"}),
    413: frozenset({"body_too_large"}),
    415: frozenset({"unsupported_media_type"}),
    422: frozenset({"preparation_failed"}),
    503: frozenset({"execution_unavailable", "preparation_unavailable"}),
}


def _validated_error(text: str, status: int) -> bytes:
    payload = exact_mapping(strict_json_object(text), _ERROR_FIELDS, "authority error")
    code = payload["code"]
    detail = payload["detail"]
    if type(code) is not str or code not in _ERROR_CODES.get(status, frozenset()):
        raise ValueError("authority error code does not match its status")
    if (
        type(detail) is not str
        or not detail
        or detail != detail.strip()
        or len(detail) > _MAX_ERROR_DETAIL
    ):
        raise ValueError("authority error detail is invalid")
    return str(canonical_numeric_json(payload)).encode("utf-8")


def _validated_success(text: str, operation: CompanionApiOperation) -> bytes:
    if operation is CompanionApiOperation.CAPABILITY:
        payload = AuthorityCapability.from_json(text).to_wire()
        return str(canonical_numeric_json(payload)).encode("utf-8")
    if operation is CompanionApiOperation.PREPARE:
        canonical = regional_ground_execution_job_to_json(
            regional_ground_execution_job_from_json(text)
        )
        return canonical.encode("utf-8")
    if operation in {
        CompanionApiOperation.SUBMIT,
        CompanionApiOperation.STATUS,
        CompanionApiOperation.CANCEL,
    }:
        snapshot = regional_ground_authority_job_status_from_json(text)
        return str(canonical_numeric_json(snapshot.to_wire())).encode("utf-8")
    if operation is CompanionApiOperation.RESULT:
        canonical = regional_ground_execution_result_to_json(
            regional_ground_execution_result_from_json(text)
        )
        return canonical.encode("utf-8")
    raise AssertionError("unreachable companion operation")


def validate_authority_response(
    route: CompanionRoute, *, status: int, media_type: str, body: bytes
) -> bytes:
    """Return canonical validated bytes or reject the authority response."""
    if (
        route.operation is None
        or status not in route.response_statuses
        or media_type != "application/json"
        or type(body) is not bytes
        or len(body) > route.response_limit
    ):
        raise ValueError("authority response metadata is invalid")
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("authority response must be UTF-8") from exc
    if status in {200, 202}:
        return _validated_success(text, route.operation)
    return _validated_error(text, status)


__all__ = ["validate_authority_response"]
