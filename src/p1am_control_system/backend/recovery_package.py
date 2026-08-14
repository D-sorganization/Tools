"""Checksum-verified configuration recovery packages with no energized state."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

from alarm_service import manager_from_routing
from configuration_workflow import ConfigurationRevision, ConfigurationWorkflow
from identity import Principal
from models import RoutingConfig
from pydantic import BaseModel, ConfigDict, Field

try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc  # noqa: UP017

PACKAGE_SCHEMA = "p1am.configuration-recovery/v1"
EXPECTED_ENTRIES = frozenset({"manifest.json", "configuration.json"})
MAX_PACKAGE_BYTES = 5_000_000
MAX_ENTRY_BYTES = 2_000_000


class RecoveryManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_id: str = PACKAGE_SCHEMA
    created_at: datetime
    software_revision: str = Field(min_length=1, max_length=200)
    configuration_revision: str = Field(min_length=1, max_length=200)
    configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    entries: dict[str, str]
    data_classification: str = "configuration_backup"
    not_for_live_control: bool = True
    energized_state_included: bool = False
    limitations: tuple[str, ...] = (
        "Restores configuration into a draft only.",
        "Does not contain credentials, runtime commands, or energized state.",
        "Requires validation, review, approval, and activation after restore.",
    )


@dataclass(frozen=True)
class RecoveryArtifact:
    payload: bytes = field(repr=False)
    sha256: str
    manifest: RecoveryManifest


@dataclass(frozen=True)
class VerifiedRecovery:
    manifest: RecoveryManifest
    configuration: RoutingConfig
    package_sha256: str


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _required_revision(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("software_revision must be a non-empty string")
    return value.strip()


class RecoveryPackageService:
    """Create and restore narrowly scoped, de-energized recovery artifacts."""

    def __init__(
        self,
        workflow: ConfigurationWorkflow,
        software_revision: str,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(workflow, ConfigurationWorkflow):
            raise TypeError("workflow must be a ConfigurationWorkflow")
        self._workflow = workflow
        self._software_revision = _required_revision(software_revision)
        self._clock = clock or (lambda: datetime.now(UTC))
        self._last_verified_at: datetime | None = None

    @property
    def last_verified_at(self) -> datetime | None:
        return self._last_verified_at

    def _now(self) -> datetime:
        now = self._clock()
        if not isinstance(now, datetime) or now.tzinfo is None:
            raise ValueError("clock must return an aware datetime")
        return now

    def create(self) -> RecoveryArtifact:
        active = self._workflow.active()
        if active is None or not active.activation_identity:
            raise ValueError("an identified active configuration is required")
        configuration = json.dumps(
            active.payload.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        manifest = RecoveryManifest(
            created_at=self._now(),
            software_revision=self._software_revision,
            configuration_revision=active.activation_identity,
            configuration_sha256=active.payload_sha256,
            entries={"configuration.json": _sha256(configuration)},
        )
        manifest_payload = manifest.model_dump_json(indent=2).encode("utf-8")
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", manifest_payload)
            archive.writestr("configuration.json", configuration)
        payload = output.getvalue()
        return RecoveryArtifact(
            payload=payload,
            sha256=_sha256(payload),
            manifest=manifest,
        )

    def verify(
        self,
        payload: bytes,
        expected_sha256: str | None = None,
    ) -> VerifiedRecovery:
        if not isinstance(payload, bytes):
            raise TypeError("payload must be bytes")
        if not payload or len(payload) > MAX_PACKAGE_BYTES:
            raise ValueError("recovery package size is outside the allowed boundary")
        package_sha256 = _sha256(payload)
        if expected_sha256 is not None and package_sha256 != expected_sha256.lower():
            raise ValueError("recovery package checksum does not match")
        try:
            with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
                names = frozenset(archive.namelist())
                if names != EXPECTED_ENTRIES:
                    raise ValueError("recovery package entries are not allowed")
                for info in archive.infolist():
                    if info.file_size > MAX_ENTRY_BYTES:
                        raise ValueError("recovery package entry is too large")
                manifest_payload = archive.read("manifest.json")
                configuration_payload = archive.read("configuration.json")
        except (zipfile.BadZipFile, RuntimeError) as exc:
            raise ValueError("recovery package is not a valid archive") from exc
        manifest = RecoveryManifest.model_validate_json(manifest_payload)
        if manifest.schema_id != PACKAGE_SCHEMA:
            raise ValueError("recovery package schema is unsupported")
        if not manifest.not_for_live_control or manifest.energized_state_included:
            raise ValueError("recovery package violates the de-energized contract")
        expected_entry = manifest.entries.get("configuration.json")
        if expected_entry != _sha256(configuration_payload):
            raise ValueError("configuration entry checksum does not match")
        configuration = RoutingConfig.model_validate_json(configuration_payload)
        manager_from_routing(configuration)
        if (
            _sha256(
                json.dumps(
                    configuration.model_dump(mode="json"),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            != manifest.configuration_sha256
        ):
            raise ValueError("configuration identity checksum does not match")
        self._last_verified_at = self._now()
        return VerifiedRecovery(manifest, configuration, package_sha256)

    def restore_as_draft(
        self,
        payload: bytes,
        principal: Principal,
        reason: str,
        expected_sha256: str | None = None,
    ) -> ConfigurationRevision:
        verified = self.verify(payload, expected_sha256)
        return self._workflow.create_draft(
            verified.configuration,
            principal,
            reason,
        )
