"""Validated pipeline configuration models for the Data Processor CLI."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from data_processor.constants import SUPPORTED_FORMATS

_ALLOWED_FILTERS = {
    "Moving Average",
    "Butterworth Low-pass",
    "Butterworth High-pass",
    "Median Filter",
    "Hampel Filter",
    "Z-Score Filter",
    "Savitzky-Golay",
    "Gaussian Filter",
    "FFT Low-pass",
    "FFT High-pass",
    "FFT Band-pass",
    "FFT Band-stop",
}

_INT_PARAMETERS: dict[str, tuple[int, int]] = {
    "ma_window": (3, 10000),
    "bw_order": (1, 10),
    "median_kernel": (3, 10001),
    "hampel_window": (3, 10001),
    "savgol_window": (3, 10001),
    "savgol_polyorder": (1, 9),
}

_FLOAT_PARAMETERS: dict[str, tuple[float, float]] = {
    "bw_cutoff": (0.0001, 0.9999),
    "hampel_threshold": (0.0, 1_000.0),
    "zscore_threshold": (0.0, 1_000.0),
    "gaussian_sigma": (0.0, 10_000.0),
    "fft_freq_low": (0.0, 1_000_000.0),
    "fft_freq_high": (0.0, 1_000_000.0),
    "fft_transition_bw": (0.0, 1_000_000.0),
}

_STRING_PARAMETERS: set[str] = {
    "zscore_method",
    "gaussian_mode",
    "fft_window_shape",
}


@dataclass(frozen=True)
class FilterConfig:
    """Configuration for a single filter operation."""

    filter_type: str
    parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> FilterConfig:
        """Construct a FilterConfig from a mapping with validation."""
        assert mapping is not None, "mapping must be provided"
        if "filter_type" not in mapping:
            msg = "filter.filter_type is required"
            raise ValueError(msg)

        filter_type = str(mapping["filter_type"])
        if filter_type not in _ALLOWED_FILTERS:
            msg = (
                f"Unsupported filter_type '{filter_type}'. Allowed values: "
                f"{', '.join(sorted(_ALLOWED_FILTERS))}"
            )
            raise ValueError(msg)

        parameters = {
            key: value for key, value in mapping.items() if key != "filter_type"
        }
        validated = cls._validate_parameters(parameters)
        return cls(filter_type=filter_type, parameters=validated)

    @staticmethod
    def _validate_parameters(raw: Mapping[str, Any]) -> dict[str, Any]:
        cleaned: dict[str, Any] = {}

        allowed_keys = (
            set(_INT_PARAMETERS) | set(_FLOAT_PARAMETERS) | _STRING_PARAMETERS
        )

        for key, value in raw.items():
            if key in _INT_PARAMETERS:
                cleaned[key] = _coerce_int(key, value, *_INT_PARAMETERS[key])
            elif key in _FLOAT_PARAMETERS:
                cleaned[key] = _coerce_float(key, value, *_FLOAT_PARAMETERS[key])
            elif key in _STRING_PARAMETERS:
                cleaned[key] = str(value)
            else:
                msg = (
                    f"Unknown filter parameter '{key}'. Allowed keys: "
                    f"{', '.join(sorted(allowed_keys))}"
                )
                raise ValueError(msg)
        return cleaned

    def to_engine_parameters(self) -> dict[str, Any]:
        """Return parameters in the shape expected by the filter engine."""
        return dict(self.parameters)


@dataclass(frozen=True)
class IntegrationConfig:
    """Configuration for signal integration."""

    method: str = "trapezoidal"
    signals: list[str] = field(default_factory=list)
    initial_condition: float = 0.0


@dataclass(frozen=True)
class DifferentiationConfig:
    """Configuration for signal differentiation."""

    method: str = "finite_difference"
    signals: list[str] = field(default_factory=list)
    order: int = 1


@dataclass(frozen=True)
class OutputConfig:
    """Validated output configuration."""

    path: Path
    format: str = "csv"

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> OutputConfig:
        assert mapping is not None, "mapping must be provided"
        if "path" not in mapping:
            msg = "output.path is required when an output section is provided"
            raise ValueError(msg)

        fmt = str(mapping.get("format", "csv")).lower()
        if fmt == "excel":
            fmt = "xlsx"
        if fmt not in {ext.lstrip(".") for ext in SUPPORTED_FORMATS}:
            msg = (
                f"Unsupported output format '{fmt}'. Allowed formats: "
                f"{', '.join(sorted({ext.lstrip('.') for ext in SUPPORTED_FORMATS}))}"
            )
            raise ValueError(msg)

        path = Path(mapping["path"]).expanduser()
        return cls(path=path, format=fmt)

    def ensure_directory_for_uncombined(self) -> Path:
        """Return a directory path for uncombined outputs, validating semantics."""
        if self.path.suffix:
            msg = "When combine is disabled, output.path must reference a directory"
            raise ValueError(msg)
        return self.path


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level pipeline configuration."""

    files: list[str]
    combine: bool = True
    selected_signals: list[str] | None = None
    filter: FilterConfig | None = None
    output: OutputConfig | None = None

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> PipelineConfig:
        assert mapping is not None, "mapping must be provided"
        normalized: MutableMapping[str, Any] = dict(mapping)
        files = _normalize_files(normalized.get("files"))
        combine = bool(normalized.get("combine", True))
        selected_signals = _normalize_optional_str_list(
            normalized.get("selected_signals")
        )

        filter_cfg: FilterConfig | None = None
        if "filter" in normalized and normalized["filter"] is not None:
            filter_cfg = FilterConfig.from_mapping(
                _ensure_mapping(normalized["filter"], "filter"),
            )

        output_cfg: OutputConfig | None = None
        if "output" in normalized and normalized["output"] is not None:
            output_cfg = OutputConfig.from_mapping(
                _ensure_mapping(normalized["output"], "output"),
            )

        return cls(
            files=files,
            combine=combine,
            selected_signals=selected_signals,
            filter=filter_cfg,
            output=output_cfg,
        )

    def summary(self) -> dict[str, Any]:
        """Provide a structured summary suitable for logging or debugging."""
        return {
            "files": self.files,
            "combine": self.combine,
            "selected_signals": self.selected_signals,
            "filter": self.filter.parameters if self.filter else None,
            "output": (
                {
                    "path": str(self.output.path),
                    "format": self.output.format,
                }
                if self.output
                else None
            ),
        }


def _normalize_files(value: Any) -> list[str]:
    if value is None:
        msg = "At least one input file must be supplied"
        raise ValueError(msg)

    if isinstance(value, str | Path):
        return [str(value)]

    if isinstance(value, Sequence):
        files = [str(item) for item in value if isinstance(item, str | Path)]

        if not files:
            msg = "files must contain at least one path"
            raise ValueError(msg)
        return files

    msg = "files must be a string path or list of paths"
    raise ValueError(msg)


def _normalize_optional_str_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str | Path):
        return [str(value)]
    if isinstance(value, Sequence):
        result = [str(item) for item in value if isinstance(item, str | Path)]

        return result or None
    msg = "selected_signals must be a list of strings if provided"
    raise ValueError(msg)


def _coerce_int(name: str, value: Any, min_value: int, max_value: int) -> int:
    try:
        coerced = int(value)
    except (TypeError, ValueError) as exc:
        msg = f"{name} must be an integer"
        raise ValueError(msg) from exc
    if not (min_value <= coerced <= max_value):
        msg = f"{name} must be between {min_value} and {max_value}"
        raise ValueError(msg)
    return coerced


def _coerce_float(name: str, value: Any, min_value: float, max_value: float) -> float:
    try:
        coerced = float(value)
    except (TypeError, ValueError) as exc:
        msg = f"{name} must be a number"
        raise ValueError(msg) from exc
    if not (min_value <= coerced <= max_value):
        msg = f"{name} must be between {min_value} and {max_value}"
        raise ValueError(msg)
    return coerced


def _ensure_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        msg = f"{field_name} must be an object"
        raise ValueError(msg)
    return value
