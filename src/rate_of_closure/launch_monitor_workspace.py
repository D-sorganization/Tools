"""Reference-only project and export seam for launch-monitor analytics.

Statistics remain owned by the UpstreamDrift backend.  This module validates
the portable client state sent to that backend and deliberately keeps private
corpus rows out of persistent project documents.
"""

from __future__ import annotations

import csv
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
    return json.dumps(project.to_wire(), indent=2, sort_keys=True) + "\n"


def save_project(destination: str | Path, project: LaunchMonitorProject) -> Path:
    """Atomically persist a reference-only project document."""

    path = Path(destination)
    write_utf8_text_atomic(
        _project_json(project), path, document_name="launch-monitor project"
    )
    return path


def load_project(source: str | Path) -> LaunchMonitorProject:
    """Load and fully validate a version-2 project document."""

    payload = json.loads(Path(source).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("launch-monitor project must be a JSON object")
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


def _write_bundle_file(directory: Path, name: str, content: str) -> dict[str, Any]:
    path = directory / name
    write_utf8_text_atomic(content, path, document_name="analysis export")
    encoded = content.encode("utf-8")
    return {"sha256": sha256(encoded).hexdigest(), "bytes": len(encoded)}


def export_analysis_bundle(
    destination: str | Path,
    project: LaunchMonitorProject,
    result: dict[str, Any],
    backing_rows: pd.DataFrame,
) -> Path:
    """Export project, result, and explicit backing rows with file hashes."""

    directory = Path(destination)
    directory.mkdir(parents=True, exist_ok=False)
    project_text = _project_json(project)
    result_text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    csv_text = backing_rows.to_csv(
        None, index=False, quoting=csv.QUOTE_MINIMAL, lineterminator="\n"
    )
    if not isinstance(csv_text, str):
        raise TypeError("pandas did not return a CSV string")
    backing_metadata = _write_bundle_file(directory, "backing_rows.csv", csv_text)
    files = {
        "project.json": _write_bundle_file(directory, "project.json", project_text),
        "result.json": _write_bundle_file(directory, "result.json", result_text),
        "backing_rows.csv": backing_metadata,
    }
    manifest = {
        "contract_version": CONTRACT_VERSION,
        "purpose": "explicit full analysis export including backing rows",
        "files": files,
    }
    _write_bundle_file(
        directory,
        "manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    return directory


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
    "save_project",
]
