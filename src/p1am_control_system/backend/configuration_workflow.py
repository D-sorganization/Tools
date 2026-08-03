"""Canonical protected workflow for immutable SCADA configuration revisions."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime, timezone
from typing import Protocol

from alarm_service import manager_from_routing
from identity import Principal, Role
from models import RoutingConfig
from pydantic import BaseModel, ConfigDict, Field

from shared.python.compatibility import StrEnum

try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc  # noqa: UP017


class ConfigurationState(StrEnum):
    DRAFT = "draft"
    VALIDATED = "validated"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    ACTIVE = "active"
    SUPERSEDED = "superseded"


class ConfigurationDiff(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str = Field(min_length=1)
    before: object | None
    after: object | None


class ConfigurationRevision(BaseModel):
    """One immutable payload and its explicit workflow metadata."""

    model_config = ConfigDict(frozen=True)

    revision_id: str
    version: int = Field(gt=0)
    state: ConfigurationState
    payload: RoutingConfig
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reason: str
    created_by: str
    created_at: datetime
    validated_by: str | None = None
    reviewed_by: str | None = None
    approved_by: str | None = None
    activated_by: str | None = None
    activated_at: datetime | None = None
    activation_identity: str | None = None
    source_revision_id: str | None = None


class RevisionRepository(Protocol):
    def next_version(self) -> int: ...
    def save(self, revision: ConfigurationRevision) -> None: ...
    def get(self, revision_id: str) -> ConfigurationRevision: ...
    def list(self) -> list[ConfigurationRevision]: ...
    def activate(self, revision: ConfigurationRevision) -> ConfigurationRevision: ...


class InMemoryRevisionRepository:
    """Deterministic repository used by tests and isolated demonstrations."""

    def __init__(self) -> None:
        self._revisions: dict[str, ConfigurationRevision] = {}
        self._lock = threading.RLock()

    def next_version(self) -> int:
        with self._lock:
            return (
                max((item.version for item in self._revisions.values()), default=0) + 1
            )

    def save(self, revision: ConfigurationRevision) -> None:
        if not isinstance(revision, ConfigurationRevision):
            raise TypeError("revision must be a ConfigurationRevision")
        with self._lock:
            self._revisions[revision.revision_id] = revision

    def get(self, revision_id: str) -> ConfigurationRevision:
        with self._lock:
            try:
                return self._revisions[revision_id]
            except KeyError as exc:
                raise KeyError(
                    f"unknown configuration revision {revision_id!r}"
                ) from exc

    def list(self) -> list[ConfigurationRevision]:
        with self._lock:
            return sorted(self._revisions.values(), key=lambda item: item.version)

    def activate(self, revision: ConfigurationRevision) -> ConfigurationRevision:
        if revision.state is not ConfigurationState.ACTIVE:
            raise ValueError("activated revision must have active state")
        with self._lock:
            for revision_id, current in tuple(self._revisions.items()):
                if current.state is ConfigurationState.ACTIVE:
                    self._revisions[revision_id] = current.model_copy(
                        update={"state": ConfigurationState.SUPERSEDED}
                    )
            self._revisions[revision.revision_id] = revision
            return revision


def _required_reason(reason: object) -> str:
    if not isinstance(reason, str):
        raise TypeError("reason must be a string")
    normalized = reason.strip()
    if not normalized:
        raise ValueError("reason must be non-empty")
    if len(normalized) > 500:
        raise ValueError("reason must contain at most 500 characters")
    return normalized


def _require_role(principal: Principal, role: Role) -> None:
    if not isinstance(principal, Principal):
        raise TypeError("principal must be a Principal")
    if not principal.allows(role):
        raise PermissionError(f"{role.value} role required")


def _payload_hash(payload: RoutingConfig) -> str:
    canonical = json.dumps(
        payload.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _flatten(value: object, prefix: str = "") -> dict[str, object]:
    if isinstance(value, Mapping):
        flattened: dict[str, object] = {}
        for key in sorted(value):
            path = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten(value[key], path))
        return flattened
    if isinstance(value, list):
        flattened = {}
        for index, item in enumerate(value):
            path = f"{prefix}.{index}" if prefix else str(index)
            flattened.update(_flatten(item, path))
        return flattened
    return {prefix: value}


class ConfigurationWorkflow:
    """Application service enforcing every protected configuration transition."""

    def __init__(
        self,
        repository: RevisionRepository,
        deploy: Callable[[RoutingConfig], Awaitable[None]],
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not callable(deploy):
            raise TypeError("deploy must be callable")
        self._repository = repository
        self._deploy = deploy
        self._clock = clock or (lambda: datetime.now(UTC))
        self._mutation_lock = threading.RLock()
        self._activation_lock = asyncio.Lock()

    def _now(self) -> datetime:
        now = self._clock()
        if not isinstance(now, datetime) or now.tzinfo is None:
            raise ValueError("clock must return an aware datetime")
        return now

    def get(self, revision_id: str) -> ConfigurationRevision:
        return self._repository.get(revision_id)

    def list(self) -> list[ConfigurationRevision]:
        return self._repository.list()

    def active(self) -> ConfigurationRevision | None:
        return next(
            (
                item
                for item in reversed(self.list())
                if item.state is ConfigurationState.ACTIVE
            ),
            None,
        )

    def create_draft(
        self, payload: RoutingConfig, principal: Principal, reason: str
    ) -> ConfigurationRevision:
        _require_role(principal, Role.ENGINEER)
        if not isinstance(payload, RoutingConfig):
            raise TypeError("payload must be a RoutingConfig")
        with self._mutation_lock:
            version = self._repository.next_version()
            digest = _payload_hash(payload)
            revision = ConfigurationRevision(
                revision_id=f"cfg-{version:06d}-{digest[:12]}",
                version=version,
                state=ConfigurationState.DRAFT,
                payload=payload.model_copy(deep=True),
                payload_sha256=digest,
                reason=_required_reason(reason),
                created_by=principal.subject,
                created_at=self._now(),
            )
            self._repository.save(revision)
        return revision

    def _transition(
        self,
        revision_id: str,
        expected: ConfigurationState,
        target: ConfigurationState,
        **updates: object,
    ) -> ConfigurationRevision:
        with self._mutation_lock:
            revision = self.get(revision_id)
            if revision.state is not expected:
                raise ValueError(f"revision must be {expected.value}")
            changed = revision.model_copy(update={"state": target, **updates})
            self._repository.save(changed)
        return changed

    def validate(self, revision_id: str, principal: Principal) -> ConfigurationRevision:
        _require_role(principal, Role.ENGINEER)
        revision = self.get(revision_id)
        manager_from_routing(revision.payload)
        return self._transition(
            revision_id,
            ConfigurationState.DRAFT,
            ConfigurationState.VALIDATED,
            validated_by=principal.subject,
        )

    def submit_for_review(
        self, revision_id: str, principal: Principal
    ) -> ConfigurationRevision:
        _require_role(principal, Role.ENGINEER)
        return self._transition(
            revision_id,
            ConfigurationState.VALIDATED,
            ConfigurationState.IN_REVIEW,
            reviewed_by=principal.subject,
        )

    def approve(
        self, revision_id: str, principal: Principal, reason: str
    ) -> ConfigurationRevision:
        _require_role(principal, Role.ENGINEER)
        _required_reason(reason)
        return self._transition(
            revision_id,
            ConfigurationState.IN_REVIEW,
            ConfigurationState.APPROVED,
            approved_by=principal.subject,
        )

    def diff(
        self, revision_id: str, base_revision_id: str | None = None
    ) -> list[ConfigurationDiff]:
        revision = self.get(revision_id)
        base = self.get(base_revision_id) if base_revision_id else self.active()
        before = _flatten(base.payload.model_dump(mode="json")) if base else {}
        after = _flatten(revision.payload.model_dump(mode="json"))
        return [
            ConfigurationDiff(path=path, before=before.get(path), after=after.get(path))
            for path in sorted(before.keys() | after.keys())
            if before.get(path) != after.get(path)
        ]

    async def activate(
        self, revision_id: str, principal: Principal
    ) -> ConfigurationRevision:
        _require_role(principal, Role.ADMIN)
        async with self._activation_lock:
            revision = self.get(revision_id)
            if revision.state is not ConfigurationState.APPROVED:
                raise ValueError("revision must be approved")
            await self._deploy(revision.payload.model_copy(deep=True))
            active = revision.model_copy(
                update={
                    "state": ConfigurationState.ACTIVE,
                    "activated_by": principal.subject,
                    "activated_at": self._now(),
                    "activation_identity": revision.revision_id,
                }
            )
            return self._repository.activate(active)

    async def rollback(
        self,
        source_revision_id: str,
        principal: Principal,
        reason: str,
    ) -> ConfigurationRevision:
        _require_role(principal, Role.ADMIN)
        source = self.get(source_revision_id)
        if source.state not in {
            ConfigurationState.ACTIVE,
            ConfigurationState.SUPERSEDED,
        }:
            raise ValueError("rollback source must be active or superseded")
        with self._mutation_lock:
            version = self._repository.next_version()
            clone = ConfigurationRevision(
                revision_id=f"cfg-{version:06d}-{source.payload_sha256[:12]}",
                version=version,
                state=ConfigurationState.APPROVED,
                payload=source.payload.model_copy(deep=True),
                payload_sha256=source.payload_sha256,
                reason=_required_reason(reason),
                created_by=principal.subject,
                created_at=self._now(),
                validated_by=principal.subject,
                reviewed_by=principal.subject,
                approved_by=principal.subject,
                source_revision_id=source.revision_id,
            )
            self._repository.save(clone)
        return await self.activate(clone.revision_id, principal)
