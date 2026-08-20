"""Safe contracts for private neural training requests and portable inference."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CAPABILITY_SCHEMA = "launch-monitor-capability-manifest/v1"
BUNDLE_SCHEMA = "launch-monitor-neural-bundle/v2"
TRAINING_SCHEMA = "launch-monitor-neural-training/v2"
FORBIDDEN_SPLIT_GROUPS = frozenset({"shot_id", "source_row_number", "row_index"})


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value.strip()


def _digest(value: object, label: str) -> str:
    text = _text(value, label).lower()
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{label} must be a 64-character SHA-256")
    return text


def _finite(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{label} must be finite")
    return float(value)


@dataclass(frozen=True)
class VendorCapability:
    vendor: str
    state: str
    row_count: int
    strict_row_count: int
    artifact_state: str
    blockers: tuple[str, ...]


@dataclass(frozen=True)
class CapabilityManifest:
    schema: str
    policy_sha256: str
    vendors: tuple[VendorCapability, ...]


def load_capability_manifest(path: Path | None = None) -> CapabilityManifest:
    source = path or Path(
        str(
            files("rate_of_closure").joinpath("data/neural_vendor_capabilities.v2.json")
        )
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema") != CAPABILITY_SCHEMA or not isinstance(
        payload.get("vendors"), list
    ):
        raise ValueError("unsupported neural capability manifest")
    labels = {
        "trackman": "TrackMan",
        "foresight": "Foresight",
        "flightscope": "FlightScope",
        "unconfirmed": "Unconfirmed",
    }
    vendors = tuple(
        VendorCapability(
            vendor=labels.get(
                _text(item.get("vendor_key"), "vendor key"),
                _text(item.get("vendor_key"), "vendor key"),
            ),
            state="available"
            if isinstance(item.get("allowed_operations"), dict)
            and item["allowed_operations"].get("vendor_training") is True
            else "unavailable",
            row_count=int(item.get("rows", -1)),
            strict_row_count=int(item.get("strict_model_input_rows", -1)),
            artifact_state=_text(
                item.get("current_surrogate_artifact_status"), "artifact state"
            ),
            blockers=tuple(
                f"{str(reason).replace('_', ' ')}: {count} "
                "source-metric policy decisions"
                for reason, count in item.get("training_blockers", {}).items()
            ),
        )
        for item in payload["vendors"]
    )
    if any(
        item.state not in {"available", "unavailable"}
        or item.row_count < 0
        or item.strict_row_count < 0
        for item in vendors
    ):
        raise ValueError("invalid neural capability values")
    return CapabilityManifest(
        CAPABILITY_SCHEMA,
        _digest(payload.get("policy_sha256"), "capability policy SHA-256"),
        vendors,
    )


@dataclass(frozen=True)
class DatasetAuthority:
    dataset_id: str
    repository: str
    commit: str
    dataset_path: str
    sha256: str
    row_count: int

    def __post_init__(self) -> None:
        _text(self.dataset_id, "dataset id")
        _text(self.repository, "repository")
        if len(self.commit) != 40 or any(
            c not in "0123456789abcdef" for c in self.commit.lower()
        ):
            raise ValueError("dataset commit must be a 40-character git commit")
        _text(self.dataset_path, "dataset path")
        _digest(self.sha256, "dataset SHA-256")
        if self.row_count < 1:
            raise ValueError("dataset row count must be positive")


@dataclass(frozen=True)
class TrainingSelection:
    vendor: str
    features: tuple[str, ...]
    targets: tuple[str, ...]
    split_group: str
    split_group_policy_approved: bool


@dataclass(frozen=True)
class GroupSummary:
    column: str
    distinct_groups: int
    repeated_groups: int


@dataclass(frozen=True)
class TrainingManifest:
    dataset: DatasetAuthority
    selection: TrainingSelection
    groups: GroupSummary
    schema: str = TRAINING_SCHEMA

    def to_wire(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "dataset": asdict(self.dataset),
            "vendor": self.selection.vendor,
            "features": list(self.selection.features),
            "targets": list(self.selection.targets),
            "split": {**asdict(self.groups), "policy_approved": True},
        }

    @property
    def sha256(self) -> str:
        value = json.dumps(self.to_wire(), sort_keys=True, separators=(",", ":"))
        return sha256(value.encode()).hexdigest()


def validate_training_groups(
    frame: pd.DataFrame, column: str, *, policy_approved: bool
) -> GroupSummary:
    normalized = _text(column, "split group").lower()
    if normalized in FORBIDDEN_SPLIT_GROUPS:
        raise ValueError(
            f"split group {column!r} is forbidden because it is row-like or unique"
        )
    if not policy_approved:
        raise ValueError("split group must be explicitly policy-approved")
    if column not in frame:
        raise ValueError("split group column is absent from the dataset")
    counts = frame[column].dropna().astype(str).value_counts()
    summary = GroupSummary(column, len(counts), int((counts >= 2).sum()))
    if summary.distinct_groups < 3:
        raise ValueError("split group requires at least three distinct groups")
    if summary.repeated_groups < 1:
        raise ValueError("split group requires at least one repeated group")
    return summary


def build_training_manifest(
    authority: DatasetAuthority, frame: pd.DataFrame, selection: TrainingSelection
) -> TrainingManifest:
    features = tuple(_text(item, "feature") for item in selection.features)
    targets = tuple(_text(item, "target") for item in selection.targets)
    if not features or not targets or set(features) & set(targets):
        raise ValueError("features and targets must be non-empty and disjoint")
    missing = (set(features) | set(targets)) - set(frame.columns)
    if missing:
        raise ValueError(
            f"dataset is missing selected columns: {', '.join(sorted(missing))}"
        )
    if authority.row_count != len(frame):
        raise ValueError("dataset authority row count does not match loaded dataset")
    groups = validate_training_groups(
        frame,
        selection.split_group,
        policy_approved=selection.split_group_policy_approved,
    )
    return TrainingManifest(authority, selection, groups)


@dataclass(frozen=True)
class Variable:
    name: str
    unit: str
    mean: float
    scale: float
    minimum: float | None = None
    maximum: float | None = None


@dataclass(frozen=True)
class Layer:
    activation: str
    weights: np.ndarray[Any, np.dtype[np.float64]]
    bias: np.ndarray[Any, np.dtype[np.float64]]


@dataclass(frozen=True)
class PortableModel:
    model_id: str
    vendor: str
    training_manifest_sha256: str
    dataset_sha256: str
    training_manifest: Mapping[str, object]
    features: tuple[Variable, ...]
    targets: tuple[Variable, ...]
    layers: tuple[Layer, ...]
    model_card: Mapping[str, object]
    metrics: tuple[Mapping[str, object], ...]
    residuals: Mapping[str, object]


@dataclass(frozen=True)
class Prediction:
    values: dict[str, float]
    warnings: tuple[str, ...]


def _variable(payload: Mapping[str, object], bounded: bool) -> Variable:
    name = _text(payload.get("name"), "variable name")
    scale = _finite(payload.get("scale"), f"{name} scale")
    if scale <= 0:
        raise ValueError(f"{name} scale must be positive")
    minimum = _finite(payload.get("min"), f"{name} min") if bounded else None
    maximum = _finite(payload.get("max"), f"{name} max") if bounded else None
    if bounded and minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError(f"{name} bounds are reversed")
    return Variable(
        name,
        str(payload.get("unit", "unitless")),
        _finite(payload.get("mean"), f"{name} mean"),
        scale,
        minimum,
        maximum,
    )


def load_portable_model(source: Path | Mapping[str, object]) -> PortableModel:
    payload = (
        json.loads(source.read_text(encoding="utf-8"))
        if isinstance(source, Path)
        else dict(source)
    )
    if payload.get("schema") != BUNDLE_SCHEMA:
        raise ValueError(f"unsupported model schema; expected {BUNDLE_SCHEMA}")
    features = tuple(_variable(item, True) for item in payload.get("features", []))
    targets = tuple(_variable(item, False) for item in payload.get("targets", []))
    if not 1 <= len(features) <= 64 or not 1 <= len(targets) <= 32:
        raise ValueError("portable model feature or target count is unsafe")
    layers: list[Layer] = []
    width = len(features)
    for raw in payload.get("layers", []):
        weights = np.asarray(raw.get("weights"), dtype=float)
        bias = np.asarray(raw.get("bias"), dtype=float)
        activation = raw.get("activation")
        if (
            activation not in {"linear", "relu", "tanh"}
            or weights.shape != (len(bias), width)
            or weights.size > 5_000_000
        ):
            raise ValueError("portable model layer is invalid or unsafe")
        if not np.isfinite(weights).all() or not np.isfinite(bias).all():
            raise ValueError("portable model layer values must be finite")
        layers.append(Layer(str(activation), weights, bias))
        width = len(bias)
    if not layers or width != len(targets):
        raise ValueError("portable model final layer does not match targets")
    residuals = payload.get(
        "residuals", {"state": "unavailable", "reason": "row-aligned residuals absent"}
    )
    if not isinstance(residuals, dict) or residuals.get("state") not in {
        "available",
        "unavailable",
    }:
        raise ValueError("residual availability must be explicit")
    training_manifest = payload.get("training_manifest")
    if not isinstance(training_manifest, dict):
        raise ValueError("portable model must embed its reference-only manifest")
    manifest_text = json.dumps(training_manifest, sort_keys=True, separators=(",", ":"))
    manifest_digest = sha256(manifest_text.encode()).hexdigest()
    declared_manifest_digest = _digest(
        payload.get("training_manifest_sha256"), "training manifest SHA-256"
    )
    if manifest_digest != declared_manifest_digest:
        raise ValueError("embedded training manifest SHA-256 does not match")
    declared_dataset_digest = _digest(payload.get("dataset_sha256"), "dataset SHA-256")
    manifest_dataset = training_manifest.get("dataset")
    if (
        not isinstance(manifest_dataset, dict)
        or manifest_dataset.get("sha256") != declared_dataset_digest
    ):
        raise ValueError("training manifest dataset SHA-256 does not match model")
    return PortableModel(
        _text(payload.get("model_id"), "model id"),
        _text(payload.get("vendor"), "vendor"),
        declared_manifest_digest,
        declared_dataset_digest,
        training_manifest,
        features,
        targets,
        tuple(layers),
        payload.get("model_card")
        if isinstance(payload.get("model_card"), dict)
        else {},
        tuple(item for item in payload.get("metrics", []) if isinstance(item, dict)),
        residuals,
    )


def predict_one(model: PortableModel, inputs: Mapping[str, object]) -> Prediction:
    raw = np.array(
        [_finite(inputs.get(item.name), item.name) for item in model.features],
        dtype=float,
    )
    warnings = tuple(
        f"{item.name} is outside training range "
        f"[{item.minimum:g}, {item.maximum:g}] {item.unit}."
        for index, item in enumerate(model.features)
        if (item.minimum is not None and raw[index] < item.minimum)
        or (item.maximum is not None and raw[index] > item.maximum)
    )
    values = (raw - [item.mean for item in model.features]) / [
        item.scale for item in model.features
    ]
    for layer in model.layers:
        values = layer.weights @ values + layer.bias
        if layer.activation == "relu":
            values = np.maximum(values, 0)
        elif layer.activation == "tanh":
            values = np.tanh(values)
    values = values * [item.scale for item in model.targets] + [
        item.mean for item in model.targets
    ]
    return Prediction(
        dict(zip((item.name for item in model.targets), values.tolist(), strict=True)),
        warnings,
    )
