"""Explicit producer and source identity for persisted variation plans."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from shared.python.contracts import require

PRODUCER_PROVENANCE_SCHEMA_ID = "rate-of-closure/variation-plan-provenance"
PRODUCER_PROVENANCE_SCHEMA_VERSION = 1
SOURCE_REVISION_STATUSES = ("exact", "unavailable")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_STABLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
_FIELDS = frozenset(
    {
        "schema_id",
        "schema_version",
        "producer_id",
        "producer_version",
        "source_repository",
        "source_revision",
        "source_revision_status",
        "source_revision_reason",
    }
)


@dataclass(frozen=True)
class PlanProducerProvenance:
    """Producer identity with an exact revision or explicit absence."""

    producer_id: str
    producer_version: int
    source_repository: str
    source_revision: str | None
    source_revision_status: str
    source_revision_reason: str | None
    schema_id: str = PRODUCER_PROVENANCE_SCHEMA_ID
    schema_version: int = PRODUCER_PROVENANCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name, value in (
            ("producer_id", self.producer_id),
            ("source_repository", self.source_repository),
        ):
            require(
                isinstance(value, str) and bool(_STABLE_ID.fullmatch(value)),
                f"{name} must be a stable identifier",
                value,
            )
        require(
            isinstance(self.producer_version, int)
            and not isinstance(self.producer_version, bool)
            and self.producer_version > 0,
            "producer_version must be a positive integer",
            self.producer_version,
        )
        require(
            self.source_revision_status in SOURCE_REVISION_STATUSES,
            "source_revision_status is unsupported",
            self.source_revision_status,
        )
        if self.source_revision_status == "exact":
            require(
                isinstance(self.source_revision, str)
                and bool(_COMMIT.fullmatch(self.source_revision)),
                "exact source revision must be a lowercase Git commit",
                self.source_revision,
            )
            require(
                self.source_revision_reason is None,
                "exact source revision must not have an unavailability reason",
            )
        else:
            require(
                self.source_revision is None,
                "unavailable source revision must be null",
                self.source_revision,
            )
            require(
                isinstance(self.source_revision_reason, str)
                and len(self.source_revision_reason.strip()) >= 16,
                "unavailable source revision requires an unavailability reason",
                self.source_revision_reason,
            )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "producer_id": self.producer_id,
            "producer_version": self.producer_version,
            "source_repository": self.source_repository,
            "source_revision": self.source_revision,
            "source_revision_status": self.source_revision_status,
            "source_revision_reason": self.source_revision_reason,
        }

    @property
    def sha256(self) -> str:
        canonical = json.dumps(
            self.to_json_dict(), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


PYTHON_DEFAULT_PROVENANCE = PlanProducerProvenance(
    producer_id="rate-of-closure/python",
    producer_version=1,
    source_repository="D-sorganization/Tools",
    source_revision=None,
    source_revision_status="unavailable",
    source_revision_reason=(
        "This runtime did not receive an exact Tools source revision at build time."
    ),
)


def provenance_from_json_dict(value: object) -> PlanProducerProvenance:
    require(isinstance(value, Mapping), "provenance must be an object", value)
    item = cast(Mapping[str, object], value)
    require(set(item) == _FIELDS, "provenance fields mismatch", tuple(item))
    require(
        item["schema_id"] == PRODUCER_PROVENANCE_SCHEMA_ID,
        "provenance schema_id mismatch",
    )
    require(
        item["schema_version"] == PRODUCER_PROVENANCE_SCHEMA_VERSION,
        "provenance schema_version mismatch",
    )
    return PlanProducerProvenance(
        producer_id=cast(str, item["producer_id"]),
        producer_version=cast(int, item["producer_version"]),
        source_repository=cast(str, item["source_repository"]),
        source_revision=cast(str | None, item["source_revision"]),
        source_revision_status=cast(str, item["source_revision_status"]),
        source_revision_reason=cast(str | None, item["source_revision_reason"]),
    )


__all__ = [
    "PRODUCER_PROVENANCE_SCHEMA_ID",
    "PRODUCER_PROVENANCE_SCHEMA_VERSION",
    "PYTHON_DEFAULT_PROVENANCE",
    "PlanProducerProvenance",
    "provenance_from_json_dict",
]
