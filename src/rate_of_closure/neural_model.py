"""Safe, framework-neutral neural-surrogate bundle loading and inference."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

BUNDLE_SCHEMA = "launch-monitor-neural-bundle/v1"
MAX_FEATURES = 128
MAX_LAYERS = 32
MAX_LAYER_VALUES = 5_000_000
SUPPORTED_ACTIVATIONS = frozenset({"linear", "relu", "tanh"})


@dataclass(frozen=True)
class VariableSpec:
    """One normalized model input or output variable."""

    name: str
    unit: str
    mean: float
    scale: float
    minimum: float | None = None
    maximum: float | None = None


@dataclass(frozen=True)
class DenseLayer:
    """One validated dense layer using output-by-input weight storage."""

    weights: NDArray[np.float64]
    bias: NDArray[np.float64]
    activation: str


@dataclass(frozen=True)
class NeuralModelBundle:
    """Portable, non-executable launch-monitor surrogate model."""

    model_id: str
    vendor: str
    created_at: str
    features: tuple[VariableSpec, ...]
    outputs: tuple[VariableSpec, ...]
    layers: tuple[DenseLayer, ...]
    metrics: object
    learning_curve: tuple[Mapping[str, object], ...]
    provenance: Mapping[str, object]


@dataclass(frozen=True)
class PredictionResult:
    """Numerical predictions and any out-of-domain warnings."""

    values: NDArray[np.float64]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class FramePrediction:
    """Batch output retaining source columns and warning metadata."""

    frame: pd.DataFrame
    warnings: tuple[str, ...]


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _variable(payload: object, *, bounded: bool) -> VariableSpec:
    if not isinstance(payload, dict):
        raise ValueError("variable specifications must be objects")
    name = payload.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("variable name must be non-empty")
    scale = _finite_number(payload.get("scale"), f"{name} scale")
    if scale <= 0.0:
        raise ValueError(f"{name} scale must be positive")
    minimum = _finite_number(payload.get("min"), f"{name} min") if bounded else None
    maximum = _finite_number(payload.get("max"), f"{name} max") if bounded else None
    if bounded and minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError(f"{name} applicability bounds are reversed")
    return VariableSpec(
        name=name,
        unit=str(payload.get("unit", "unitless")),
        mean=_finite_number(payload.get("mean"), f"{name} mean"),
        scale=scale,
        minimum=minimum,
        maximum=maximum,
    )


def _layer(payload: object, expected_inputs: int) -> DenseLayer:
    if not isinstance(payload, dict):
        raise ValueError("layers must be objects")
    activation = payload.get("activation")
    if activation not in SUPPORTED_ACTIVATIONS:
        raise ValueError(f"unsupported layer activation: {activation}")
    weights = np.asarray(payload.get("weights"), dtype=float)
    bias = np.asarray(payload.get("bias"), dtype=float)
    if weights.ndim != 2 or bias.ndim != 1:
        raise ValueError("layer weights and bias have invalid dimensions")
    if weights.shape != (len(bias), expected_inputs):
        raise ValueError("layer input/output dimension mismatch")
    if weights.size > MAX_LAYER_VALUES:
        raise ValueError("layer exceeds safe bundle size limit")
    if not np.isfinite(weights).all() or not np.isfinite(bias).all():
        raise ValueError("layer values must be finite")
    return DenseLayer(weights=weights, bias=bias, activation=str(activation))


def _read_payload(source: Path | Mapping[str, object]) -> Mapping[str, object]:
    if isinstance(source, Path):
        if not source.name.lower().endswith(".json"):
            raise ValueError("only JSON neural bundles are accepted")
        payload: Any = json.loads(source.read_text(encoding="utf-8"))
    else:
        payload = source
    if not isinstance(payload, dict):
        raise ValueError("neural bundle must be a JSON object")
    return payload


def load_neural_bundle(source: Path | Mapping[str, object]) -> NeuralModelBundle:
    """Load a bounded JSON bundle without deserializing executable objects."""

    payload = _read_payload(source)
    if payload.get("schema") != BUNDLE_SCHEMA:
        raise ValueError(f"unsupported neural bundle schema; expected {BUNDLE_SCHEMA}")
    raw_features = payload.get("features")
    raw_outputs = payload.get("outputs")
    raw_layers = payload.get("layers")
    if not all(
        isinstance(value, list) for value in (raw_features, raw_outputs, raw_layers)
    ):
        raise ValueError("features, outputs, and layers must be arrays")
    assert isinstance(raw_features, list)
    assert isinstance(raw_outputs, list)
    assert isinstance(raw_layers, list)
    if not 1 <= len(raw_features) <= MAX_FEATURES or not raw_outputs:
        raise ValueError("bundle feature/output count exceeds supported limits")
    if not 1 <= len(raw_layers) <= MAX_LAYERS:
        raise ValueError("bundle layer count exceeds supported limits")
    features = tuple(_variable(item, bounded=True) for item in raw_features)
    outputs = tuple(_variable(item, bounded=False) for item in raw_outputs)
    layers: list[DenseLayer] = []
    width = len(features)
    for item in raw_layers:
        layer = _layer(item, width)
        layers.append(layer)
        width = len(layer.bias)
    if width != len(outputs):
        raise ValueError("final layer output dimension does not match outputs")
    return NeuralModelBundle(
        model_id=str(payload.get("modelId", "unnamed-model")),
        vendor=str(payload.get("vendor", "unspecified")),
        created_at=str(payload.get("createdAt", "")),
        features=features,
        outputs=outputs,
        layers=tuple(layers),
        metrics=_metadata(payload.get("metrics")),
        learning_curve=_curve(payload.get("learningCurve")),
        provenance=_mapping(payload.get("provenance")),
    )


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, dict) else {}


def _metadata(value: object) -> object:
    """Retain inspectable JSON metadata while rejecting scalar surprises."""

    return value if isinstance(value, (dict, list)) else {}


def _curve(value: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, dict))


def _activate(values: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    if name == "relu":
        return np.maximum(values, 0.0)
    if name == "tanh":
        return np.tanh(values)
    return values


def _warnings(model: NeuralModelBundle, matrix: NDArray[np.float64]) -> tuple[str, ...]:
    warnings: list[str] = []
    for index, feature in enumerate(model.features):
        below = feature.minimum is not None and np.any(
            matrix[:, index] < feature.minimum
        )
        above = feature.maximum is not None and np.any(
            matrix[:, index] > feature.maximum
        )
        if below or above:
            warnings.append(
                f"{feature.name} is outside training range "
                f"[{feature.minimum:g}, {feature.maximum:g}] {feature.unit}."
            )
    return tuple(warnings)


def predict_records(
    model: NeuralModelBundle, records: Sequence[Mapping[str, object]]
) -> PredictionResult:
    """Run a forward pass for records containing every named feature."""

    if not records:
        raise ValueError("at least one query row is required")
    try:
        raw_rows = [
            [_numeric_value(row[feature.name]) for feature in model.features]
            for row in records
        ]
        matrix: NDArray[np.float64] = np.array(raw_rows, dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "query rows must contain finite numeric model features"
        ) from exc
    if not np.isfinite(matrix).all():
        raise ValueError("query rows must contain finite numeric model features")
    values = (matrix - [item.mean for item in model.features]) / [
        item.scale for item in model.features
    ]
    for layer in model.layers:
        values = _activate(values @ layer.weights.T + layer.bias, layer.activation)
    values = values * [item.scale for item in model.outputs] + [
        item.mean for item in model.outputs
    ]
    return PredictionResult(values=values, warnings=_warnings(model, matrix))


def _numeric_value(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError("feature value is not numeric")
    return float(value)


def predict_frame(model: NeuralModelBundle, frame: pd.DataFrame) -> FramePrediction:
    """Predict a whole dataframe while preserving all original columns."""

    missing = [feature.name for feature in model.features if feature.name not in frame]
    if missing:
        raise ValueError(f"dataset is missing model features: {', '.join(missing)}")
    records: list[dict[str, object]] = [
        {str(key): value for key, value in row.items()}
        for row in frame[[item.name for item in model.features]].to_dict("records")
    ]
    result = predict_records(model, records)
    output = frame.copy()
    for index, target in enumerate(model.outputs):
        output[f"predicted_{target.name}"] = result.values[:, index]
    return FramePrediction(frame=output, warnings=result.warnings)


__all__ = [
    "BUNDLE_SCHEMA",
    "FramePrediction",
    "NeuralModelBundle",
    "PredictionResult",
    "VariableSpec",
    "load_neural_bundle",
    "predict_frame",
    "predict_records",
]
