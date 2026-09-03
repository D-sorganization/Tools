"""Reference-only project and export seam for launch-monitor analytics.

The canonical inferential layer now lives in Tools' `src/shared/python/
launch_monitor/` per UpstreamDrift ADR-0046 (Stage 1 complete 2026-09-02).
This module remains part of `rate_of_closure`'s web-twinned application
layer: it validates the portable client state exchanged with that canonical
layer and deliberately keeps private corpus rows out of persistent project
documents.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd

from rate_of_closure.application.atomic_text_files import write_utf8_text_atomic
from rate_of_closure.launch_monitor_v2_client import (
    CanonicalDatasetReference,
    load_canonical_dataset_reference,
)
from rate_of_closure.launch_monitor_workspace_v3 import (
    WorkspaceExportAuthorization,
    create_workspace_bundle,
    parse_workspace_project,
    serialize_workspace_project,
)

CONTRACT_VERSION = "2.0.0"


def _non_empty(value: str, label: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{label} must be non-empty")
    return cleaned


@dataclass(frozen=True)
class DatasetReference:
    """Immutable provenance reference to a dataset; never the dataset itself."""

    source_name: str
    repository: str
    revision: str
    relative_path: str
    sha256: str
    row_count: int

    def __post_init__(self) -> None:
        for value, label in (
            (self.source_name, "source name"),
            (self.repository, "repository"),
            (self.revision, "revision"),
            (self.relative_path, "relative path"),
        ):
            _non_empty(value, label)
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256.lower()
        ):
            raise ValueError("dataset SHA-256 must contain 64 hexadecimal characters")
        if self.row_count < 0:
            raise ValueError("row count cannot be negative")


@dataclass(frozen=True)
class PlayerIdentityBinding:
    """User assertion that one source column is a real player identifier."""

    column: str
    user_attested: bool

    def __post_init__(self) -> None:
        _non_empty(self.column, "player identity column")
        if not self.user_attested:
            raise ValueError("player identity must be explicitly user-attested")


@dataclass(frozen=True)
class AnalysisSelection:
    """Variables and uncertainty settings for player covariation."""

    x: str
    y: str
    min_samples: int = 10
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        _non_empty(self.x, "x variable")
        _non_empty(self.y, "y variable")
        if self.x == self.y:
            raise ValueError("x and y variables must be different")
        if self.min_samples < 3:
            raise ValueError("minimum samples must be at least three")
        if not 0.5 < self.confidence_level < 1.0:
            raise ValueError("confidence level must be between 0.5 and 1")


@dataclass(frozen=True)
class LaunchMonitorProject:
    """Validated, reference-only state shared by the PyQt and React clients."""

    name: str
    dataset: DatasetReference
    identity: PlayerIdentityBinding
    selection: AnalysisSelection
    canonical_dataset: CanonicalDatasetReference | None = None
    contract_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _non_empty(self.name, "project name")
        if self.contract_version != CONTRACT_VERSION:
            raise ValueError("unsupported launch-monitor project contract")
        if self.identity.column in {self.selection.x, self.selection.y}:
            raise ValueError("identity and analysis variables must be different")

    def to_wire(self) -> dict[str, Any]:
        """Return the snake-case backend and persistence representation."""

        payload = asdict(self)
        if self.canonical_dataset is None:
            payload.pop("canonical_dataset")
        return payload


def build_player_covariation_request(project: LaunchMonitorProject) -> dict[str, Any]:
    """Build a reference-only request for the authoritative analytics backend."""

    return {
        "contract_version": CONTRACT_VERSION,
        "operation": "player_covariation",
        "dataset": asdict(project.dataset),
        "player_identity": {
            "column": project.identity.column,
            "user_attested": project.identity.user_attested,
        },
        "variables": {"x": project.selection.x, "y": project.selection.y},
        "options": {
            "min_samples": project.selection.min_samples,
            "confidence_level": project.selection.confidence_level,
        },
    }


def dataset_reference_for_frame(
    frame: pd.DataFrame, source_name: str
) -> DatasetReference:
    """Build a deterministic local-data reference without embedding records."""

    serialized = frame.to_json(orient="records", date_format="iso")
    return DatasetReference(
        source_name=_non_empty(source_name, "source name"),
        repository="local-user-data",
        revision="unversioned",
        relative_path=_non_empty(source_name, "source name"),
        sha256=sha256(serialized.encode("utf-8")).hexdigest(),
        row_count=len(frame),
    )


def _project_json(project: LaunchMonitorProject) -> str:
    return str(serialize_workspace_project(_workspace_v3(project))) + "\n"


def _row_free_result(value: Any) -> Any:
    """Retain scalar/aggregate evidence while excluding row-shaped collections."""

    if isinstance(value, dict):
        forbidden = {"rows", "records", "backing_data", "backing_rows", "per_player"}
        return {
            key: _row_free_result(item)
            for key, item in value.items()
            if key not in forbidden
            and not (isinstance(item, list) and item and isinstance(item[0], dict))
        }
    if isinstance(value, list):
        return [_row_free_result(item) for item in value]
    return value


def _dataset_v3(project: LaunchMonitorProject) -> dict[str, Any]:
    canonical = project.canonical_dataset
    dataset: dict[str, Any] = {
        "source_name": project.dataset.source_name,
        "repository": project.dataset.repository,
        "revision": project.dataset.revision,
        "relative_path": project.dataset.relative_path,
        "content_sha256": project.dataset.sha256,
        "row_count": project.dataset.row_count,
        "classification": "restricted",
        "authority_commit": None,
        "manifest_sha256": None,
    }
    if canonical is not None:
        dataset.update(
            authority_root_id=canonical.root_id,
            authority_repository=canonical.repository,
            authority_commit=canonical.commit,
            manifest_sha256=canonical.manifest_sha256,
            authority_content_sha256=canonical.content_sha256,
            authority_row_count=canonical.expected_row_count,
        )
    return dataset


def _result_v3(
    project: LaunchMonitorProject, result: dict[str, Any] | None
) -> dict[str, Any]:
    canonical = project.canonical_dataset
    payload = None if result is None else _row_free_result(result)
    response_hash = (
        None
        if payload is None
        else sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    )
    return {
        "status": "available" if payload is not None else "unavailable",
        "authority": "upstream-v2" if canonical else "offline-compatibility-v1",
        "authority_commit": canonical.commit if canonical else None,
        "response_sha256": response_hash,
        "payload": payload,
        "units": {
            project.selection.x: "source-unit-unavailable",
            project.selection.y: "source-unit-unavailable",
        },
        "formulas": ["pairwise-complete player covariation"],
        "exclusions": ["Row-aligned records are retained outside the saved project."],
    }


def _analysis_v3(
    project: LaunchMonitorProject, result: dict[str, Any] | None
) -> dict[str, Any]:
    return {
        "analysis_id": "player-covariation",
        "operation": "player_covariation",
        "settings": {
            "x_column": project.selection.x,
            "y_column": project.selection.y,
            "method": "pearson",
            "minimum_samples": project.selection.min_samples,
            "confidence_level": project.selection.confidence_level,
        },
        "result": _result_v3(project, result),
        "backing_join": {
            "algorithm": "sha256-canonical-json-v1",
            "row_count": project.dataset.row_count,
            "sha256": None,
            "status": "available-on-authorized-export",
            "reason": None,
        },
    }


def _workspace_v3(
    project: LaunchMonitorProject, result: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "schema_id": "launch-monitor-workspace/v3",
        "schema_version": 3,
        "name": project.name,
        "dataset": _dataset_v3(project),
        "identity_evidence": {
            "player": {
                "column": project.identity.column,
                "user_attested": project.identity.user_attested,
                "evidence": "Dataset owner explicitly attested this player identifier.",
            }
        },
        "analyses": [_analysis_v3(project, result)],
        "export_policy": {
            "persist_rows": False,
            "backing_rows": "explicit-restricted-approval",
            "reason": (
                "Restricted rows remain outside saved projects and browser persistence."
            ),
        },
    }


def save_project(destination: str | Path, project: LaunchMonitorProject) -> Path:
    """Atomically persist a reference-only project document."""

    path = Path(destination)
    write_utf8_text_atomic(
        _project_json(project), path, document_name="launch-monitor project"
    )
    return path


def load_project(source: str | Path) -> LaunchMonitorProject:
    """Load v3 or a labelled compatibility v2 project document."""

    payload = json.loads(Path(source).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("launch-monitor project must be a JSON object")
    if payload.get("schema_id") == "launch-monitor-workspace/v3":
        return _project_from_v3(payload)
    required = {"name", "dataset", "identity", "selection", "contract_version"}
    allowed = required | {"canonical_dataset"}
    if not required.issubset(payload) or not set(payload).issubset(allowed):
        raise ValueError("launch-monitor project has missing or unknown fields")
    canonical = payload.get("canonical_dataset")
    return LaunchMonitorProject(
        name=str(payload["name"]),
        dataset=DatasetReference(**payload["dataset"]),
        identity=PlayerIdentityBinding(**payload["identity"]),
        selection=AnalysisSelection(**payload["selection"]),
        canonical_dataset=(
            None if canonical is None else load_canonical_dataset_reference(canonical)
        ),
        contract_version=str(payload["contract_version"]),
    )


def load_project_versioned(
    source: str | Path,
) -> tuple[LaunchMonitorProject, str]:
    """Load a project and expose whether a compatibility adapter was used."""

    payload = json.loads(Path(source).read_text(encoding="utf-8"))
    imported_from = (
        "v3"
        if isinstance(payload, dict)
        and payload.get("schema_id") == "launch-monitor-workspace/v3"
        else "v2-compatibility"
    )
    return load_project(source), imported_from


def _project_from_v3(payload: dict[str, Any]) -> LaunchMonitorProject:
    workspace = parse_workspace_project(payload)
    analysis = next(
        item for item in workspace.analyses if item.operation == "player_covariation"
    )
    dataset = workspace.dataset
    canonical = None
    if dataset.get("authority_root_id"):
        canonical = CanonicalDatasetReference(
            root_id=dataset.authority_root_id,
            repository=dataset.authority_repository,
            commit=dataset.authority_commit,
            manifest_sha256=dataset.manifest_sha256,
            content_sha256=dataset.authority_content_sha256,
            expected_row_count=dataset.authority_row_count,
        )
    return LaunchMonitorProject(
        name=workspace.name,
        dataset=DatasetReference(
            dataset.source_name,
            dataset.repository,
            dataset.revision,
            dataset.relative_path,
            dataset.content_sha256,
            dataset.row_count,
        ),
        identity=PlayerIdentityBinding(
            workspace.identity_evidence.player.column,
            workspace.identity_evidence.player.user_attested,
        ),
        selection=AnalysisSelection(
            analysis.settings.x_column,
            analysis.settings.y_column,
            analysis.settings.minimum_samples,
            analysis.settings.confidence_level,
        ),
        canonical_dataset=canonical,
    )


def export_analysis_bundle(
    destination: str | Path,
    project: LaunchMonitorProject,
    result: dict[str, Any],
    backing_rows: pd.DataFrame,
    authorization: WorkspaceExportAuthorization | None = None,
) -> Path:
    """Export v3 evidence, failing closed for unapproved restricted rows."""

    approved = authorization or WorkspaceExportAuthorization(include_backing_rows=True)
    return Path(
        create_workspace_bundle(
            destination, _workspace_v3(project, result), backing_rows, approved
        )
    )


__all__ = [
    "CONTRACT_VERSION",
    "AnalysisSelection",
    "DatasetReference",
    "LaunchMonitorProject",
    "PlayerIdentityBinding",
    "build_player_covariation_request",
    "dataset_reference_for_frame",
    "export_analysis_bundle",
    "load_project",
    "load_project_versioned",
    "save_project",
]
