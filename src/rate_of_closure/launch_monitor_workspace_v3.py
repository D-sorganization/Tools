"""Portable, row-free launch-monitor workspace v3 persistence and export."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_ID = "launch-monitor-workspace/v3"
_CLASSIFICATIONS = {"public", "internal", "restricted"}
_ROW_KEYS = {"rows", "records", "backing_data", "source_rows"}


@dataclass(frozen=True)
class WorkspaceExportAuthorization:
    """Explicit authority for backing-row export; projects remain row-free."""

    platform: str = "desktop"
    include_backing_rows: bool = False
    restricted_data_approved: bool = False


class WorkspaceProject(dict[str, Any]):
    """Validated JSON mapping with ergonomic read-only-style attribute access."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error


def _attribute_mapping(value: Any) -> Any:
    if isinstance(value, Mapping):
        return WorkspaceProject(
            {key: _attribute_mapping(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return [_attribute_mapping(item) for item in value]
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(char in "0123456789abcdef" for char in value.lower())
    )


def _contains_rows(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            key in _ROW_KEYS
            or (key == "backing_rows" and not isinstance(item, str))
            or _contains_rows(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_rows(item) for item in value)
    return False


def _validate_identity(identity: Mapping[str, Any], name: str) -> None:
    _require(isinstance(identity, Mapping), f"{name} identity evidence is required")
    _require(bool(identity.get("column")), f"{name} column is required")
    _require(
        identity.get("user_attested") is True, f"{name} identity must be user-attested"
    )
    _require(bool(identity.get("evidence")), f"{name} evidence is required")


def _validate_analysis(
    analysis: Mapping[str, Any], identities: Mapping[str, Any]
) -> None:
    _require(bool(analysis.get("analysis_id")), "analysis_id is required")
    operation = analysis.get("operation")
    _require(
        operation in {"player_covariation", "longitudinal", "performance_summary"},
        "unsupported operation",
    )
    if operation in {"player_covariation", "longitudinal"}:
        _validate_identity(identities.get("player", {}), "player")
    if operation == "longitudinal":
        _validate_identity(identities.get("session", {}), "session")
        _validate_identity(identities.get("order", {}), "order")
    result = analysis.get("result")
    if not isinstance(result, Mapping):
        raise ValueError("analysis result is required")
    _require(
        not _contains_rows(result.get("payload")),
        "result payload must not contain rows",
    )
    status = result.get("status")
    _require(status in {"available", "unavailable"}, "invalid result status")
    if status == "available":
        _require(
            result.get("payload") is not None, "available result payload is required"
        )
        _require(
            _is_hex(result.get("response_sha256"), 64), "response SHA-256 is required"
        )
    else:
        _require(
            result.get("payload") is None, "unavailable result payload must be null"
        )
        _require(
            result.get("response_sha256") is None,
            "unavailable result hash must be null",
        )
        _require(
            bool(result.get("exclusions")), "unavailable result exclusions are required"
        )


def _validate_dataset(dataset: object) -> None:
    if not isinstance(dataset, Mapping):
        raise ValueError("dataset metadata is required")
    _require(
        dataset.get("classification") in _CLASSIFICATIONS, "invalid data classification"
    )
    _require(_is_hex(dataset.get("content_sha256"), 64), "dataset SHA-256 is required")
    if dataset.get("authority_commit") is not None:
        _require(
            _is_hex(dataset["authority_commit"], 40),
            "authority commit must be a full SHA",
        )
    if dataset.get("manifest_sha256") is not None:
        _require(
            _is_hex(dataset["manifest_sha256"], 64), "manifest SHA-256 is required"
        )


def _validate_document(document: Mapping[str, Any]) -> None:
    allowed = {
        "schema_id",
        "schema_version",
        "name",
        "dataset",
        "identity_evidence",
        "analyses",
        "export_policy",
    }
    _require(document.get("schema_id") == SCHEMA_ID, "unsupported workspace schema")
    _require(document.get("schema_version") == 3, "unsupported workspace version")
    _require(
        not _contains_rows(document),
        "workspace projects must not contain rows or row-bearing records",
    )
    _require(set(document) <= allowed, "unknown workspace fields")
    _validate_dataset(document.get("dataset"))
    identities = document.get("identity_evidence")
    if not isinstance(identities, Mapping):
        raise ValueError("identity evidence is required")
    analyses = document.get("analyses")
    if not isinstance(analyses, list) or not analyses:
        raise ValueError("at least one analysis is required")
    for analysis in analyses:
        _require(isinstance(analysis, Mapping), "analysis entries must be objects")
        _validate_analysis(analysis, identities)
    policy = document.get("export_policy")
    if not isinstance(policy, Mapping):
        raise ValueError("export policy is required")
    _require(policy.get("persist_rows") is False, "saved projects must be row-free")


def parse_workspace_project(value: str | bytes | Mapping[str, Any]) -> WorkspaceProject:
    """Parse and validate a v3 project without accepting embedded source rows."""

    document = json.loads(value) if isinstance(value, (str, bytes)) else dict(value)
    _validate_document(document)
    converted = _attribute_mapping(document)
    if not isinstance(converted, WorkspaceProject):
        raise TypeError("workspace conversion failed")
    return converted


def serialize_workspace_project(project: Mapping[str, Any]) -> str:
    """Return deterministic JSON for a validated, row-free workspace."""

    validated = parse_workspace_project(project)
    return json.dumps(
        validated, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def _normalize_json_number(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _normalize_json_number(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_json_number(item) for item in value]
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _canonical_row(row: Mapping[str, Any]) -> str:
    normalized = _normalize_json_number(row)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _csv_bytes(rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=fieldnames, lineterminator="\n", extrasaction="ignore"
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _backing_permission(
    project: Mapping[str, Any], auth: WorkspaceExportAuthorization
) -> tuple[bool, str | None]:
    if not auth.include_backing_rows:
        return False, "backing rows were not requested"
    classification = project["dataset"]["classification"]
    if auth.platform == "browser" and classification == "restricted":
        return False, "browser export of restricted backing rows is unavailable"
    if classification == "restricted" and not auth.restricted_data_approved:
        return (
            False,
            "restricted backing-row export requires explicit restricted approval",
        )
    return True, None


def _backing_files(rows: list[Mapping[str, Any]], allowed: bool) -> dict[str, bytes]:
    if not allowed or not rows:
        return {}
    canonical = [_canonical_row(row) for row in rows]
    hashes = [hashlib.sha256(row.encode("utf-8")).hexdigest() for row in canonical]
    joins = [
        {"result_row_index": index, "row_sha256": digest}
        for index, digest in enumerate(hashes)
    ]
    fields = list(dict.fromkeys(key for row in rows for key in row))
    return {
        "backing_join.csv": _csv_bytes(joins, ("result_row_index", "row_sha256")),
        "backing_rows.csv": _csv_bytes(rows, fields),
    }


def _write_bundle_files(
    destination: Path, files: Mapping[str, bytes], backing_status: dict[str, Any]
) -> None:
    manifest: dict[str, Any] = {
        "schema_id": "launch-monitor-workspace-export/v3",
        "backing_data": backing_status,
        "files": {},
    }
    for name, content in files.items():
        (destination / name).write_bytes(content)
        manifest["files"][name] = {
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    manifest_bytes = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    (destination / "manifest.json").write_bytes(manifest_bytes)


def create_workspace_bundle(
    output_directory: str | Path,
    project: Mapping[str, Any],
    backing_rows: Any = (),
    authorization: WorkspaceExportAuthorization | None = None,
) -> Path:
    """Write a portable bundle and only export rows under explicit authority."""

    validated = parse_workspace_project(project)
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    project_bytes = (serialize_workspace_project(validated) + "\n").encode("utf-8")
    result_bytes = (
        json.dumps(validated["analyses"], sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    files = {"project.json": project_bytes, "results.json": result_bytes}
    auth = authorization or WorkspaceExportAuthorization()
    allowed, reason = _backing_permission(validated, auth)
    rows: list[Mapping[str, Any]] = (
        backing_rows.to_dict(orient="records")
        if hasattr(backing_rows, "to_dict")
        else list(backing_rows)
    )
    files.update(_backing_files(rows, allowed))
    if allowed and not rows:
        reason = "no backing rows were supplied"
    _write_bundle_files(
        destination,
        files,
        {
            "status": "available" if allowed and rows else "unavailable",
            "reason": reason,
        },
    )
    return destination
