"""Strict public-client types generated from pinned Upstream v2 schemas."""

from __future__ import annotations

from dataclasses import dataclass
from re import fullmatch
from typing import Any

DATASET_JOB_CONTRACT_VERSION = "launch-monitor-dataset-job/1.0.0"
PLAYER_COVARIATION_CONTRACT_VERSION = "launch-monitor-player-covariation/1.0.0"
MAX_CANONICAL_INLINE_RECORDS = 20_000
MAX_DATASET_JOB_PAGE_SIZE = 200
CANONICAL_DATASET_METRICS = frozenset(
    {
        "club_speed",
        "ball_speed",
        "smash_factor",
        "launch_angle",
        "launch_direction",
        "spin_rate",
        "back_spin",
        "side_spin",
        "spin_axis",
        "attack_angle",
        "club_path",
        "face_angle",
        "carry_distance",
        "total_distance",
        "descent_angle",
        "lateral_carry",
        "flight_time",
    }
)
_SHA256_PATTERN = r"[0-9a-f]{64}"
_COMMIT_PATTERN = r"[0-9a-f]{40}"
_ROOT_ID_PATTERN = r"[a-z][a-z0-9-]{0,62}"
_REPOSITORY_PATTERN = r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+"
_PRIVATE_ROW_KEYS = frozenset({"shot_id", "source_row", "row_index"})
_STATUS_KEYS = frozenset(
    {
        "contract_version",
        "job_id",
        "status",
        "submitted_at_utc",
        "completed_at_utc",
        "input_row_count",
        "result_item_count",
        "unavailable",
    }
)
_PAGE_KEYS = frozenset(
    {
        "contract_version",
        "job_id",
        "offset",
        "limit",
        "total_items",
        "next_offset",
        "items",
    }
)
_AGGREGATE_KEYSETS = frozenset(
    {
        frozenset(
            {
                "source_id",
                "row_count",
                "vendor_key",
                "redistribution_status",
                "license_spdx",
                "backing_repository",
                "backing_commit",
                "backing_object_digests",
            }
        ),
        frozenset(
            {
                "group_by",
                "group",
                "metric",
                "n",
                "mean",
                "standard_deviation",
                "minimum",
                "maximum",
            }
        ),
        frozenset(
            {
                "group_by",
                "group",
                "left_metric",
                "right_metric",
                "n",
                "correlation",
            }
        ),
    }
)


@dataclass(frozen=True)
class CanonicalDatasetReference:
    """Immutable server-authorized dataset reference without a filesystem path."""

    root_id: str
    repository: str
    commit: str
    manifest_sha256: str
    content_sha256: str
    expected_row_count: int

    def to_wire(self) -> dict[str, object]:
        """Return the exact Upstream dataset-reference wire shape."""

        return {
            "root_id": self.root_id,
            "repository": self.repository,
            "commit": self.commit,
            "manifest_sha256": self.manifest_sha256,
            "content_sha256": self.content_sha256,
            "expected_row_count": self.expected_row_count,
        }


def _strict_object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _matched_text(value: object, pattern: str, label: str) -> str:
    if not isinstance(value, str) or fullmatch(pattern, value) is None:
        raise ValueError(f"{label} is invalid")
    return value


def validate_job_id(value: object) -> str:
    """Return one canonical opaque dataset job identifier."""

    return _matched_text(value, r"[0-9a-f]{32}", "job_id")


def load_canonical_dataset_reference(value: object) -> CanonicalDatasetReference:
    """Validate a user-authorized opaque reference; paths are never accepted."""

    item = _strict_object(value, "dataset reference")
    required = {
        "root_id",
        "repository",
        "commit",
        "manifest_sha256",
        "content_sha256",
        "expected_row_count",
    }
    if set(item) != required:
        raise ValueError("dataset reference has missing or unknown fields")
    row_count = item["expected_row_count"]
    if (
        not isinstance(row_count, int)
        or isinstance(row_count, bool)
        or not (1 <= row_count <= 10_000_000)
    ):
        raise ValueError("expected_row_count is outside the canonical bounds")
    return CanonicalDatasetReference(
        root_id=_matched_text(item["root_id"], _ROOT_ID_PATTERN, "root_id"),
        repository=_matched_text(item["repository"], _REPOSITORY_PATTERN, "repository"),
        commit=_matched_text(item["commit"], _COMMIT_PATTERN, "commit"),
        manifest_sha256=_matched_text(
            item["manifest_sha256"], _SHA256_PATTERN, "manifest_sha256"
        ),
        content_sha256=_matched_text(
            item["content_sha256"], _SHA256_PATTERN, "content_sha256"
        ),
        expected_row_count=row_count,
    )


def build_dataset_job_request(
    reference: CanonicalDatasetReference,
    kind: str,
    *,
    metrics: tuple[str, ...] = (),
    group_by: str | None = None,
    minimum_group_rows: int = 10,
) -> dict[str, object]:
    """Build one allow-listed aggregate job; arbitrary query text is forbidden."""

    if kind not in {"source_summary", "metric_summary", "correlation"}:
        raise ValueError("dataset operation kind is unsupported")
    if group_by not in {None, "source_id", "monitor", "club"}:
        raise ValueError("dataset operation group_by is unsupported")
    if minimum_group_rows < 10:
        raise ValueError("minimum_group_rows must be at least 10")
    if (
        len(metrics) > 12
        or len(set(metrics)) != len(metrics)
        or not set(metrics).issubset(CANONICAL_DATASET_METRICS)
    ):
        raise ValueError("metrics must be unique canonical dataset metrics")
    if kind == "source_summary" and (metrics or group_by is not None):
        raise ValueError("source_summary does not accept metrics or group_by")
    if kind == "metric_summary" and not metrics:
        raise ValueError("metric_summary requires metrics")
    if kind == "correlation" and len(metrics) < 2:
        raise ValueError("correlation requires at least two metrics")
    return {
        "contract_version": DATASET_JOB_CONTRACT_VERSION,
        "dataset": reference.to_wire(),
        "operation": {
            "kind": kind,
            "metrics": list(metrics),
            "group_by": group_by,
            "minimum_group_rows": minimum_group_rows,
        },
    }


def validate_dataset_job_status(value: object) -> dict[str, Any]:
    """Validate data-free asynchronous job state from the canonical authority."""

    item = _strict_object(value, "dataset job status")
    if set(item) != _STATUS_KEYS:
        raise ValueError("dataset job status has missing or unknown fields")
    if item.get("contract_version") != DATASET_JOB_CONTRACT_VERSION:
        raise ValueError("dataset job status has an unsupported contract version")
    validate_job_id(item.get("job_id"))
    if item.get("status") not in {
        "queued",
        "running",
        "completed",
        "unavailable",
        "failed",
    }:
        raise ValueError("dataset job status is invalid")
    for key in ("input_row_count", "result_item_count"):
        if not isinstance(item.get(key), int) or item[key] < 0:
            raise ValueError(f"dataset job {key} is invalid")
    return item


def validate_dataset_job_page(value: object) -> dict[str, Any]:
    """Validate a bounded aggregate page and reject observation-shaped items."""

    item = _strict_object(value, "dataset job result page")
    if set(item) != _PAGE_KEYS:
        raise ValueError("dataset job page has missing or unknown fields")
    if item.get("contract_version") != DATASET_JOB_CONTRACT_VERSION:
        raise ValueError("dataset job page has an unsupported contract version")
    validate_job_id(item.get("job_id"))
    for key in ("offset", "total_items"):
        if (
            not isinstance(item.get(key), int)
            or isinstance(item[key], bool)
            or item[key] < 0
        ):
            raise ValueError(f"dataset job page {key} is invalid")
    next_offset = item.get("next_offset")
    if next_offset is not None and (
        not isinstance(next_offset, int)
        or isinstance(next_offset, bool)
        or next_offset < 0
    ):
        raise ValueError("dataset job page next_offset is invalid")
    limit = item.get("limit")
    if (
        not isinstance(limit, int)
        or isinstance(limit, bool)
        or not 1 <= limit <= MAX_DATASET_JOB_PAGE_SIZE
    ):
        raise ValueError("dataset job page limit is invalid")
    items = item.get("items")
    if not isinstance(items, list) or len(items) > limit:
        raise ValueError("dataset job page items are invalid")
    for entry in items:
        if not isinstance(entry, dict) or _PRIVATE_ROW_KEYS.intersection(entry):
            raise ValueError("dataset job pages cannot expose private rows")
        if frozenset(entry) not in _AGGREGATE_KEYSETS:
            raise ValueError("dataset job page item does not match an aggregate schema")
    return item


def build_player_covariation_payload(
    records: list[dict[str, object]],
    *,
    player_column: str,
    x_column: str,
    y_column: str,
    min_samples: int,
    confidence_level: float,
) -> dict[str, object]:
    """Build the canonical inline request and fail closed above its hard limit."""

    if not 1 <= len(records) <= MAX_CANONICAL_INLINE_RECORDS:
        raise ValueError("canonical player covariation accepts at most 20,000 rows")
    if len({player_column, x_column, y_column}) != 3 or not all(
        value.strip() for value in (player_column, x_column, y_column)
    ):
        raise ValueError("player, x, and y columns must be distinct and non-empty")
    if min_samples < 4 or not 0.5 < confidence_level < 1:
        raise ValueError("canonical covariation options are invalid")
    return {
        "records": records,
        "request": {
            "x_column": x_column,
            "y_column": y_column,
            "player_column": player_column,
            "min_samples": min_samples,
            "confidence_level": confidence_level,
        },
        "context": {
            "player_identity": {
                "trust_level": "explicit_user_attested",
                "identifier_column": player_column,
                "evidence": (
                    f"Dataset owner attested {player_column} in this client session."
                ),
            }
        },
    }


def validate_player_covariation_response(value: object) -> dict[str, Any]:
    """Validate the canonical evidence envelope before either UI consumes it."""

    item = _strict_object(value, "player covariation response")
    if item.get("contract_version") != PLAYER_COVARIATION_CONTRACT_VERSION:
        raise ValueError("player covariation response has an unsupported contract")
    common = {
        "analysis_kind",
        "contract_version",
        "status",
        "request",
        "lineage",
        "player_identity",
        "vendor_provenance",
        "claims",
        "warnings",
    }
    selected = common | {
        "pooled",
        "within_player",
        "between_player",
        "per_player",
        "meta_analysis",
        "missingness",
        "units",
        "availability",
        "uncertainty",
        "definitions",
    }
    scan = common | {
        "pair_count",
        "available_pair_count",
        "unavailable_pair_count",
        "ranking",
        "method_description",
    }
    expected = selected if item.get("analysis_kind") == "selected_pair" else scan
    if set(item) != expected:
        raise ValueError("player covariation response is missing required fields")
    if item["analysis_kind"] not in {"selected_pair", "pair_scan"}:
        raise ValueError("player covariation analysis_kind is invalid")
    lineage = _strict_object(item["lineage"], "covariation lineage")
    if not isinstance(lineage.get("backing_records"), list):
        raise ValueError("player covariation backing lineage is invalid")
    identity = _strict_object(item["player_identity"], "player identity")
    if identity.get("trust_level") not in {
        "explicit_user_attested",
        "pseudonymous_stable",
        "verified_external",
    }:
        raise ValueError("player covariation requires trusted identity evidence")
    claims = _strict_object(item["claims"], "covariation claims")
    if any(
        claims.get(name) is not False
        for name in ("device_emulation", "device_certification", "causal_inference")
    ):
        raise ValueError("player covariation response makes an unsupported claim")
    return item


__all__ = [
    "DATASET_JOB_CONTRACT_VERSION",
    "CANONICAL_DATASET_METRICS",
    "MAX_CANONICAL_INLINE_RECORDS",
    "MAX_DATASET_JOB_PAGE_SIZE",
    "PLAYER_COVARIATION_CONTRACT_VERSION",
    "CanonicalDatasetReference",
    "build_dataset_job_request",
    "build_player_covariation_payload",
    "load_canonical_dataset_reference",
    "validate_dataset_job_page",
    "validate_dataset_job_status",
    "validate_job_id",
    "validate_player_covariation_response",
]
