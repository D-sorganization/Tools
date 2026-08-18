"""Canonical identity binding for internal ground-model phase results."""

from __future__ import annotations

import hashlib

from .contract_records import GroundSimulationRequest


def ground_request_fingerprint(request: GroundSimulationRequest) -> str:
    """Return the SHA-256 of the strict canonical request document."""
    if type(request) is not GroundSimulationRequest:
        raise ValueError("request fingerprint requires an exact ground request")
    return hashlib.sha256(request.to_json().encode("utf-8")).hexdigest()


def validate_request_fingerprint(value: str) -> str:
    """Return a canonical SHA-256 digest or fail closed."""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("request fingerprint must be 64 lowercase hex characters")
    return value


__all__ = ["ground_request_fingerprint", "validate_request_fingerprint"]
