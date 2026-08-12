"""Versioned fail-closed capability contract for the local Python authority."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, TypedDict, cast

from shared.python.swing_sim.ground.strict_json import strict_json_object

AUTHORITY_CAPABILITY_SCHEMA_VERSION: Final = (
    "rate-of-closure/regional-ground-authority-capability/v1"
)
AUTHORITY_ID: Final = "rate-of-closure-python-authority"
AUTHORITY_VERSION: Final = "1"
MAX_CAPABILITY_DETAIL_LENGTH: Final = 240
MAX_CAPABILITY_BYTES: Final = 4_096
_CAPABILITY_FIELDS: Final = frozenset(
    {
        "schema_version",
        "authority_id",
        "authority_version",
        "available",
        "regional_ground_execution",
        "reason_code",
        "detail",
    }
)
_REASON_CODES: Final = frozenset(
    {
        "qualified_execution_profile",
        "execution_profile_unqualified",
        "runner_not_started",
    }
)

AuthorityReasonCode = Literal[
    "qualified_execution_profile",
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

    available: bool
    regional_ground_execution: bool
    reason_code: AuthorityReasonCode
    detail: str

    def __post_init__(self) -> None:
        """Reject ambiguous or unbounded diagnostic text at construction."""
        if type(self.reason_code) is not str:
            raise TypeError("capability reason code must be text")
        if self.reason_code not in _REASON_CODES:
            raise ValueError("capability reason code is unsupported")
        if type(self.detail) is not str:
            raise TypeError("capability detail must be text")
        if not self.detail or self.detail != self.detail.strip():
            raise ValueError("capability detail must be nonempty and trimmed")
        if len(self.detail) > MAX_CAPABILITY_DETAIL_LENGTH:
            raise ValueError("capability detail exceeds the v1 length bound")
        if (
            type(self.available) is not bool
            or type(self.regional_ground_execution) is not bool
        ):
            raise TypeError("capability flags must be exact bool values")
        if self.available is not self.regional_ground_execution:
            raise ValueError("capability flags must be consistent")
        qualified = self.reason_code == "qualified_execution_profile"
        if self.available is not qualified:
            raise ValueError("qualified capability reason and flags must agree")

    @classmethod
    def unavailable(
        cls,
        *,
        reason_code: AuthorityReasonCode,
        detail: str,
    ) -> AuthorityCapability:
        """Build an explicitly non-executable capability document."""
        return cls(
            available=False,
            regional_ground_execution=False,
            reason_code=reason_code,
            detail=detail,
        )

    @classmethod
    def qualified(cls) -> AuthorityCapability:
        """Build the exact service-level qualified execution capability."""
        return cls(
            available=True,
            regional_ground_execution=True,
            reason_code="qualified_execution_profile",
            detail="Qualified Python regional-ground execution is available.",
        )

    def to_wire(self) -> AuthorityCapabilityWire:
        """Return the exact v1 JSON-compatible wire record."""
        return {
            "schema_version": AUTHORITY_CAPABILITY_SCHEMA_VERSION,
            "authority_id": AUTHORITY_ID,
            "authority_version": AUTHORITY_VERSION,
            "available": self.available,
            "regional_ground_execution": self.regional_ground_execution,
            "reason_code": self.reason_code,
            "detail": self.detail,
        }

    @classmethod
    def from_json(cls, text: str) -> AuthorityCapability:
        """Parse one bounded duplicate-safe exact v1 capability document."""
        if type(text) is not str:
            raise TypeError("capability JSON must be text")
        if len(text.encode("utf-8")) > MAX_CAPABILITY_BYTES:
            raise ValueError("capability JSON exceeds the v1 byte bound")
        payload = strict_json_object(text)
        if set(payload) != _CAPABILITY_FIELDS:
            raise ValueError("capability JSON fields must match v1 exactly")
        if payload["schema_version"] != AUTHORITY_CAPABILITY_SCHEMA_VERSION:
            raise ValueError("unsupported capability schema")
        if payload["authority_id"] != AUTHORITY_ID:
            raise ValueError("unsupported capability authority")
        if payload["authority_version"] != AUTHORITY_VERSION:
            raise ValueError("unsupported capability authority version")
        reason = payload["reason_code"]
        if type(reason) is not str or reason not in _REASON_CODES:
            raise ValueError("unsupported capability reason")
        detail = payload["detail"]
        if type(detail) is not str:
            raise TypeError("capability detail must be text")
        return cls(
            available=payload["available"],
            regional_ground_execution=payload["regional_ground_execution"],
            reason_code=cast(AuthorityReasonCode, reason),
            detail=detail,
        )


DEFAULT_UNAVAILABLE_CAPABILITY: Final = AuthorityCapability.unavailable(
    reason_code="execution_profile_unqualified",
    detail="Exact flight and ground execution profile is not qualified.",
)
QUALIFIED_EXECUTION_CAPABILITY: Final = AuthorityCapability.qualified()
