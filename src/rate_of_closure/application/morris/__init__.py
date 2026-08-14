"""Lazy Morris authority exports for dependency-neutral UI contracts."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_CONTRACT_EXPORTS = {
    "MORRIS_AUTHORITY_SCHEMA_VERSION",
    "MORRIS_JOB_SCHEMA_ID",
    "MORRIS_REQUEST_SCHEMA_ID",
    "MorrisAuthorityRequest",
    "MorrisJobEnvelope",
    "parse_morris_request",
}


def __getattr__(name: str) -> Any:
    if name in _CONTRACT_EXPORTS:
        return getattr(import_module(".contracts", __name__), name)
    if name in {"MorrisExecutionService", "RateMorrisService"}:
        return getattr(import_module(".service", __name__), name)
    raise AttributeError(name)


__all__ = [
    "MORRIS_AUTHORITY_SCHEMA_VERSION",
    "MORRIS_JOB_SCHEMA_ID",
    "MORRIS_REQUEST_SCHEMA_ID",
    "MorrisAuthorityRequest",
    "MorrisExecutionService",
    "MorrisJobEnvelope",
    "RateMorrisService",
    "parse_morris_request",
]
