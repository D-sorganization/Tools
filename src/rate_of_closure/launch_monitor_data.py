"""Private campaign discovery, dataset catalog, units, and project persistence."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_SCHEMA_VERSION = "2.0.0"
CAMPAIGN_ENVIRONMENT_VARIABLE = "LAUNCH_MONITOR_CAMPAIGN_REPO"


@dataclass(frozen=True)
class DatasetDescriptor:
    """One manifested, locally available campaign table."""

    dataset_id: str
    label: str
    path: Path
    row_count: int
    column_count: int
    sha256: str


@dataclass(frozen=True)
class CampaignDatasetCatalog:
    """Traceable datasets that remain owned by the private repository."""

    root: Path
    source_sha256: str
    datasets: tuple[DatasetDescriptor, ...]


@dataclass(frozen=True)
class AnalysisProject:
    """Reloadable UI state bound to a campaign source identity."""

    campaign_root: str
    dataset_id: str
    source_sha256: str
    selections: dict[str, object]
    dataset_sha256: str = ""
    data_path: str = ""
    schema_version: str = PROJECT_SCHEMA_VERSION


@dataclass(frozen=True)
class ResolvedProjectData:
    """Verified data resolved without mutating presentation state."""

    frame: pd.DataFrame
    catalog: CampaignDatasetCatalog | None
    descriptor: DatasetDescriptor | None
    path: Path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def infer_unit(column: str) -> str:
    """Infer a display unit from canonical and common launch-monitor names."""

    lowered = column.lower()
    suffixes = {
        "_mph": "mph",
        "_rpm": "rpm",
        "_deg": "deg",
        "_ft": "ft",
        "_yd": "yd",
        "_m": "m",
        "_s": "s",
    }
    for suffix, unit in suffixes.items():
        if lowered.endswith(suffix):
            return unit
    if "speed" in lowered:
        return "mph"
    if "spin" in lowered:
        return "rpm"
    if any(token in lowered for token in ("angle", "direction", "axis")):
        return "deg"
    if any(token in lowered for token in ("carry", "distance", "lateral")):
        return "yd"
    if "strokes_gained" in lowered:
        return "strokes"
    if any(token in lowered for token in ("coefficient", "correlation", "r2")):
        return "unitless"
    return "unitless"


def axis_label(column: str) -> str:
    """Create a readable, unit-bearing chart label."""

    label = column.replace("_", " ").strip().title()
    unit = infer_unit(column)
    return label if unit == "unitless" else f"{label} ({unit})"


def discover_campaign_repository(explicit: Path | None = None) -> Path | None:
    """Find the private campaign without copying data into this repository."""

    candidates: list[Path] = []
    if explicit is not None:
        resolved = explicit.expanduser().resolve()
        if (resolved / "campaign.toml").is_file() and (
            resolved / "results" / "run_manifest.json"
        ).is_file():
            return resolved
        return None
    configured = os.environ.get(CAMPAIGN_ENVIRONMENT_VARIABLE)
    if configured:
        resolved = Path(configured).expanduser().resolve()
        if (resolved / "campaign.toml").is_file() and (
            resolved / "results" / "run_manifest.json"
        ).is_file():
            return resolved
    candidates.extend(
        [
            Path.home()
            / "Repositories"
            / "Launch-Monitor-Flight-Model-Campaign-worktrees"
            / "model-explainability",
            Path.home() / "Repositories" / "Launch-Monitor-Flight-Model-Campaign",
            Path.cwd() / "Launch-Monitor-Flight-Model-Campaign",
            Path.cwd().parent / "Launch-Monitor-Flight-Model-Campaign",
        ]
    )
    valid: list[Path] = []
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if (resolved / "campaign.toml").is_file() and (
            resolved / "results" / "run_manifest.json"
        ).is_file():
            valid.append(resolved)
    if not valid:
        return None
    return max(
        valid,
        key=lambda root: (root / "results" / "run_manifest.json").stat().st_mtime,
    )


def _csv_shape(path: Path, *, has_unit_row: bool = False) -> tuple[int, int]:
    columns = len(pd.read_csv(path, nrows=0).columns)
    with path.open("rb") as source:
        rows = max(0, sum(1 for _ in source) - 1 - int(has_unit_row))
    return rows, columns


def _verified_sha256(path: Path, expected: object = "") -> str:
    actual = _sha256_file(path)
    expected_text = str(expected or "")
    if expected_text and not hmac.compare_digest(actual, expected_text):
        raise ValueError(f"SHA-256 mismatch for manifested dataset: {path}")
    return actual


def campaign_dataset_catalog(root: Path) -> CampaignDatasetCatalog:
    """Read all CSV outputs declared by a private campaign configuration."""

    resolved = root.resolve()
    config_path = resolved / "campaign.toml"
    manifest_path = resolved / "results" / "run_manifest.json"
    if not config_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            "campaign.toml and results/run_manifest.json are required"
        )
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_hashes = manifest.get("output_sha256", {})
    datasets: list[DatasetDescriptor] = []
    source_config = config.get("source", {})
    source_relative = source_config.get("csv")
    if source_relative:
        source_path = (resolved / str(source_relative)).resolve()
        if source_path.suffix.lower() == ".csv" and source_path.is_file():
            rows, columns = _csv_shape(source_path, has_unit_row=True)
            source_hash = _verified_sha256(
                source_path,
                source_config.get("expected_sha256") or manifest.get("source_sha256"),
            )
            datasets.append(
                DatasetDescriptor(
                    dataset_id="source",
                    label="Exact Source Data",
                    path=source_path,
                    row_count=rows,
                    column_count=columns,
                    sha256=source_hash,
                )
            )
    for dataset_id, relative in config.get("outputs", {}).items():
        path = (resolved / str(relative)).resolve()
        if path.suffix.lower() != ".csv" or not path.is_file():
            continue
        rows, columns = _csv_shape(path)
        actual_hash = _verified_sha256(path, output_hashes.get(dataset_id))
        datasets.append(
            DatasetDescriptor(
                dataset_id=str(dataset_id),
                label=str(dataset_id).replace("_", " ").title(),
                path=path,
                row_count=rows,
                column_count=columns,
                sha256=actual_hash,
            )
        )
    if not datasets:
        raise ValueError("campaign has no available CSV outputs")
    return CampaignDatasetCatalog(
        root=resolved,
        source_sha256=str(manifest.get("source_sha256", "")),
        datasets=tuple(datasets),
    )


def load_campaign_dataset(descriptor: DatasetDescriptor) -> pd.DataFrame:
    """Load every retained row and column for the selected descriptor."""

    actual_hash = _verified_sha256(descriptor.path, descriptor.sha256)
    if actual_hash != descriptor.sha256:
        raise ValueError("dataset hash changed after catalog discovery")
    if descriptor.dataset_id == "source":
        frame = pd.read_csv(descriptor.path, low_memory=False, skiprows=[1])
    else:
        frame = pd.read_csv(descriptor.path, low_memory=False)
    if len(frame) != descriptor.row_count:
        raise ValueError("dataset row count changed after catalog discovery")
    return frame


def load_imported_dataset(path: Path, expected_sha256: str = "") -> pd.DataFrame:
    """Load a user-selected table after verifying its persisted identity."""

    resolved = path.expanduser().resolve()
    _verified_sha256(resolved, expected_sha256)
    if resolved.suffix.lower() == ".csv":
        return pd.read_csv(resolved, low_memory=False)
    if resolved.suffix.lower() == ".json":
        payload: Any = json.loads(resolved.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or any(
            not isinstance(row, dict) for row in payload
        ):
            raise ValueError("JSON data must be an array of record objects")
        return pd.DataFrame.from_records(payload)
    raise ValueError("Launch-monitor import supports CSV and JSON")


def file_sha256(path: Path) -> str:
    """Return the actual SHA-256 used to bind an imported project."""

    return _sha256_file(path.expanduser().resolve())


def resolve_project_data(project: AnalysisProject) -> ResolvedProjectData:
    """Resolve and verify project data before a UI applies any saved state."""

    if project.campaign_root:
        catalog = campaign_dataset_catalog(Path(project.campaign_root))
        descriptor = next(
            (
                item
                for item in catalog.datasets
                if item.dataset_id == project.dataset_id
            ),
            None,
        )
        if descriptor is None:
            raise ValueError("saved campaign dataset is unavailable")
        if project.source_sha256 != catalog.source_sha256:
            raise ValueError("saved project source hash does not match the campaign")
        if project.dataset_sha256 != descriptor.sha256:
            raise ValueError("saved project dataset hash does not match the campaign")
        return ResolvedProjectData(
            load_campaign_dataset(descriptor), catalog, descriptor, descriptor.path
        )
    if project.data_path:
        path = Path(project.data_path).expanduser().resolve()
        return ResolvedProjectData(
            load_imported_dataset(path, project.dataset_sha256), None, None, path
        )
    raise ValueError("saved project does not identify reloadable data")


def save_analysis_project(path: Path, project: AnalysisProject) -> None:
    """Persist analysis state as deterministic, inspectable JSON."""

    payload = asdict(project)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def load_analysis_project(path: Path) -> AnalysisProject:
    """Load and validate a persisted analysis project."""

    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("analysis project must be a JSON object")
    if payload.get("schema_version") != PROJECT_SCHEMA_VERSION:
        raise ValueError("unsupported analysis project schema version")
    selections = payload.get("selections")
    if not isinstance(selections, dict):
        raise ValueError("analysis project selections must be an object")
    return AnalysisProject(
        campaign_root=str(payload.get("campaign_root", "")),
        dataset_id=str(payload.get("dataset_id", "")),
        source_sha256=str(payload.get("source_sha256", "")),
        selections=selections,
        dataset_sha256=str(payload.get("dataset_sha256", "")),
        data_path=str(payload.get("data_path", "")),
        schema_version=str(payload["schema_version"]),
    )


__all__ = [
    "AnalysisProject",
    "CampaignDatasetCatalog",
    "DatasetDescriptor",
    "ResolvedProjectData",
    "axis_label",
    "campaign_dataset_catalog",
    "discover_campaign_repository",
    "file_sha256",
    "infer_unit",
    "load_analysis_project",
    "load_campaign_dataset",
    "load_imported_dataset",
    "resolve_project_data",
    "save_analysis_project",
]
