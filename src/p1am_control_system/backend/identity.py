"""Named principals, role contracts, and short-lived opaque SCADA sessions."""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import threading
from collections.abc import Callable, Sequence
from dataclasses import InitVar, dataclass, field
from datetime import datetime, timedelta, timezone

from shared.python.compatibility import StrEnum

try:
    from datetime import UTC
except ImportError:  # Python 3.10 support
    UTC = timezone.utc  # noqa: UP017

MINIMUM_CREDENTIAL_LENGTH = 16
MAXIMUM_SESSION_TTL = timedelta(days=1)
DEFAULT_SESSION_TTL = timedelta(hours=8)
SESSION_TOKEN_BYTES = 32


class Role(StrEnum):
    """Ordered SCADA authorization roles."""

    VIEWER = "viewer"
    OPERATOR = "operator"
    ENGINEER = "engineer"
    ADMIN = "admin"


_ROLE_RANK = {
    Role.VIEWER: 0,
    Role.OPERATOR: 1,
    Role.ENGINEER: 2,
    Role.ADMIN: 3,
}


def _required_text(value: object, field_name: str) -> str:
    """Return a stripped non-empty string or raise a contract error."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


@dataclass(frozen=True)
class Principal:
    """Authenticated named identity and its effective role."""

    subject: str
    display_name: str
    role: Role

    def __post_init__(self) -> None:
        object.__setattr__(self, "subject", _required_text(self.subject, "subject"))
        object.__setattr__(
            self,
            "display_name",
            _required_text(self.display_name, "display_name"),
        )
        if not isinstance(self.role, Role):
            raise TypeError("role must be a Role")

    def allows(self, required_role: Role) -> bool:
        """Return whether this principal meets ``required_role``."""
        if not isinstance(required_role, Role):
            raise TypeError("required_role must be a Role")
        return _ROLE_RANK[self.role] >= _ROLE_RANK[required_role]


@dataclass(frozen=True)
class CredentialRecord:
    """Principal paired with an API credential that is always redacted."""

    principal: Principal
    api_key: str = field(repr=False)
    minimum_length: InitVar[int] = MINIMUM_CREDENTIAL_LENGTH

    def __post_init__(self, minimum_length: int) -> None:
        if not isinstance(self.principal, Principal):
            raise TypeError("principal must be a Principal")
        if not isinstance(minimum_length, int) or minimum_length < 1:
            raise ValueError("minimum_length must be a positive integer")
        secret = _required_text(self.api_key, "api_key")
        if len(secret) < minimum_length:
            raise ValueError(
                f"api_key must contain at least {minimum_length} characters"
            )
        object.__setattr__(self, "api_key", secret)


def _parse_record(raw: object) -> CredentialRecord:
    """Validate one JSON principal record."""
    if not isinstance(raw, dict):
        raise TypeError("each principal configuration entry must be an object")
    try:
        role = Role(raw.get("role"))
    except (TypeError, ValueError) as exc:
        raise ValueError("role must be viewer, operator, engineer, or admin") from exc
    return CredentialRecord(
        principal=Principal(
            subject=_required_text(raw.get("subject"), "subject"),
            display_name=_required_text(raw.get("display_name"), "display_name"),
            role=role,
        ),
        api_key=_required_text(raw.get("api_key"), "api_key"),
    )


def parse_principal_config(raw_json: str) -> tuple[CredentialRecord, ...]:
    """Parse the named-principal JSON contract without logging credentials."""
    text = _required_text(raw_json, "principal configuration")
    try:
        raw_records = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("principal configuration must be valid JSON") from exc
    if not isinstance(raw_records, list):
        raise TypeError("principal configuration must be a list")
    if not raw_records:
        raise ValueError("principal configuration must contain at least one entry")
    records = tuple(_parse_record(raw_record) for raw_record in raw_records)
    subjects = [record.principal.subject for record in records]
    if len(set(subjects)) != len(subjects):
        raise ValueError("principal configuration contains a duplicate subject")
    return records


class CredentialRegistry:
    """Authenticate credentials against a validated immutable principal set."""

    def __init__(self, records: Sequence[CredentialRecord]) -> None:
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            raise TypeError("records must be a sequence of CredentialRecord")
        normalized = tuple(records)
        if not normalized:
            raise ValueError("records must contain at least one credential")
        if not all(isinstance(record, CredentialRecord) for record in normalized):
            raise TypeError("records must contain only CredentialRecord values")
        self._reject_duplicate_credentials(normalized)
        self._records = normalized

    @staticmethod
    def _reject_duplicate_credentials(records: Sequence[CredentialRecord]) -> None:
        for index, record in enumerate(records):
            for candidate in records[index + 1 :]:
                if hmac.compare_digest(record.api_key, candidate.api_key):
                    raise ValueError(
                        "principal configuration contains a duplicate credential"
                    )

    def authenticate(self, api_key: str | None) -> Principal | None:
        """Return the matching named principal without exposing credential data."""
        if not api_key or not isinstance(api_key, str):
            return None
        matched: Principal | None = None
        for record in self._records:
            if hmac.compare_digest(api_key, record.api_key):
                matched = record.principal
        return matched


@dataclass(frozen=True)
class IssuedSession:
    """One newly issued opaque session token and its public metadata."""

    token: str = field(repr=False)
    principal: Principal
    expires_at: datetime


@dataclass(frozen=True)
class _StoredSession:
    principal: Principal
    expires_at: datetime


class SessionStore:
    """Thread-safe in-memory store that retains token digests, never raw tokens."""

    def __init__(
        self,
        ttl: timedelta = DEFAULT_SESSION_TTL,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(ttl, timedelta):
            raise TypeError("ttl must be a timedelta")
        if ttl <= timedelta(0) or ttl > MAXIMUM_SESSION_TTL:
            raise ValueError("ttl must be greater than zero and at most one day")
        self._ttl = ttl
        self._clock = clock or (lambda: datetime.now(UTC))
        self._sessions: dict[str, _StoredSession] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _digest(token: str) -> str:
        if not isinstance(token, str):
            raise TypeError("token must be a string")
        if not token:
            raise ValueError("token must be non-empty")
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def _now(self) -> datetime:
        now = self._clock()
        if not isinstance(now, datetime) or now.tzinfo is None:
            raise ValueError("clock must return a timezone-aware datetime")
        return now

    def create(self, principal: Principal) -> IssuedSession:
        """Issue one opaque session for ``principal`` within the TTL contract."""
        if not isinstance(principal, Principal):
            raise TypeError("principal must be a Principal")
        now = self._now()
        token = secrets.token_urlsafe(SESSION_TOKEN_BYTES)
        expires_at = now + self._ttl
        with self._lock:
            self._sessions[self._digest(token)] = _StoredSession(
                principal=principal,
                expires_at=expires_at,
            )
        return IssuedSession(token=token, principal=principal, expires_at=expires_at)

    def resolve(self, token: str) -> Principal | None:
        """Resolve a valid session and remove it if it has expired."""
        digest = self._digest(token)
        with self._lock:
            stored = self._sessions.get(digest)
            if stored is None:
                return None
            if stored.expires_at <= self._now():
                del self._sessions[digest]
                return None
            return stored.principal

    def revoke(self, token: str) -> bool:
        """Revoke ``token`` and report whether an active record existed."""
        digest = self._digest(token)
        with self._lock:
            return self._sessions.pop(digest, None) is not None
