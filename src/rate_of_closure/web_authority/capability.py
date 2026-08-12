"""Versioned fail-closed capability contract for the local Python authority."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, TypedDict

AUTHORITY_CAPABILITY_SCHEMA_VERSION: Final = (
    "rate-of-closure/regional-ground-authority-capability/v1"
)
AUTHORITY_ID: Final = "rate-of-closure-python-authority"
AUTHORITY_VERSION: Final = "1"
MAX_CAPABILITY_DETAIL_LENGTH: Final = 240

AuthorityReasonCode = Literal[
    "execution_profile_unqualified",
    "runner_not_started",
]


class AuthorityCapabilityWire(TypedDict):
    """Exact JSON-compatible regional-ground capability shape."""

    schema_version: str
    authority_id: str
    authority_version: str
    available: bool
    regional_ground_execution: bool
    reason_code: AuthorityReasonCode
    detail: str


@dataclass(frozen=True, slots=True)
class AuthorityCapability:
    """Fail-closed execution availability advertised by the Python authority."""

    reason_code: AuthorityReasonCode
    detail: str

    def __post_init__(self) -> None:
        """Reject ambiguous or unbounded diagnostic text at construction."""
        if not self.detail or self.detail != self.detail.strip():
            raise ValueError("capability detail must be nonempty and trimmed")
        if len(self.detail) > MAX_CAPABILITY_DETAIL_LENGTH:
            raise ValueError("capability detail exceeds the v1 length bound")

    @classmethod
    def unavailable(
        cls,
        *,
        reason_code: AuthorityReasonCode,
        detail: str,
    ) -> AuthorityCapability:
        """Build an explicitly non-executable capability document."""
        return cls(reason_code=reason_code, detail=detail)

    def to_wire(self) -> AuthorityCapabilityWire:
        """Return the exact v1 JSON-compatible wire record."""
        return {
            "schema_version": AUTHORITY_CAPABILITY_SCHEMA_VERSION,
            "authority_id": AUTHORITY_ID,
            "authority_version": AUTHORITY_VERSION,
            "available": False,
            "regional_ground_execution": False,
            "reason_code": self.reason_code,
            "detail": self.detail,
        }


DEFAULT_UNAVAILABLE_CAPABILITY: Final = AuthorityCapability.unavailable(
    reason_code="execution_profile_unqualified",
    detail="Exact flight and ground execution profile is not qualified.",
)
