"""Controlled deterministic CAD and mesh export for wedge solids."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

from ._validation import require_finite_float, require_identifier
from .wedge_cad import WedgeMeasuredMetrics, build_wedge_solid
from .wedge_parameters import WedgeHeadParameters

WEDGE_EXPORT_FORMAT = "golf_club.wedge_export/1"
_FIXED_STEP_TIMESTAMP = datetime(1970, 1, 1)
_MIN_LINEAR_TOLERANCE_M = 1.0e-6
_MAX_LINEAR_TOLERANCE_M = 1.0e-3
_MIN_ANGULAR_TOLERANCE_RAD = 1.0e-4
_MAX_ANGULAR_TOLERANCE_RAD = 0.5


class WedgeExportFormat(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    """Supported exact and tessellated interchange artifacts."""

    STEP = "step"
    STL = "stl"
    BREP = "brep"


@dataclass(frozen=True)
class WedgeExportRequest:
    """Validated output location, formats, and tessellation tolerances."""

    output_directory: Path | str
    stem: str = "wedge-head"
    formats: tuple[WedgeExportFormat, ...] = tuple(WedgeExportFormat)
    linear_tolerance_m: float = 5.0e-5
    angular_tolerance_rad: float = 0.05

    def __post_init__(self) -> None:
        if not isinstance(self.output_directory, (Path, str)):
            raise TypeError("output_directory must be a path")
        object.__setattr__(self, "output_directory", Path(self.output_directory))
        safe_stem = require_identifier(self.stem, "stem")
        if Path(safe_stem).name != safe_stem or any(
            separator in safe_stem for separator in ("/", "\\")
        ):
            raise ValueError("stem must be a filename stem without path separators")
        if not isinstance(self.formats, tuple) or not self.formats:
            raise ValueError("formats must be a nonempty tuple")
        if not all(isinstance(item, WedgeExportFormat) for item in self.formats):
            raise TypeError("formats must contain only WedgeExportFormat values")
        if len(set(self.formats)) != len(self.formats):
            raise ValueError("formats must not contain duplicate values")
        linear = require_finite_float(
            self.linear_tolerance_m, "linear_tolerance_m", positive=True
        )
        if not _MIN_LINEAR_TOLERANCE_M <= linear <= _MAX_LINEAR_TOLERANCE_M:
            raise ValueError(
                "linear_tolerance_m must be in "
                f"[{_MIN_LINEAR_TOLERANCE_M}, {_MAX_LINEAR_TOLERANCE_M}]"
            )
        angular = require_finite_float(
            self.angular_tolerance_rad, "angular_tolerance_rad", positive=True
        )
        if not _MIN_ANGULAR_TOLERANCE_RAD <= angular <= _MAX_ANGULAR_TOLERANCE_RAD:
            raise ValueError(
                "angular_tolerance_rad must be in "
                f"[{_MIN_ANGULAR_TOLERANCE_RAD}, {_MAX_ANGULAR_TOLERANCE_RAD}]"
            )
        object.__setattr__(self, "linear_tolerance_m", linear)
        object.__setattr__(self, "angular_tolerance_rad", angular)


@dataclass(frozen=True)
class WedgeExportArtifact:
    """One generated artifact and its declared interchange format."""

    format: WedgeExportFormat
    path: Path


@dataclass(frozen=True)
class WedgeExportResult:
    """Complete artifact set, manifest path, and measured solid metrics."""

    artifacts: tuple[WedgeExportArtifact, ...]
    manifest_path: Path
    measured: WedgeMeasuredMetrics


def export_wedge_artifacts(
    parameters: WedgeHeadParameters,
    request: WedgeExportRequest,
) -> WedgeExportResult:
    """Build once and export deterministic exact/mesh artifacts plus metadata."""
    if not isinstance(parameters, WedgeHeadParameters):
        raise TypeError("parameters must be WedgeHeadParameters")
    if not isinstance(request, WedgeExportRequest):
        raise TypeError("request must be WedgeExportRequest")
    output_directory = request.output_directory
    assert isinstance(output_directory, Path)  # normalized by request contract
    output_directory.mkdir(parents=True, exist_ok=True)
    if not output_directory.is_dir():
        raise ValueError("output_directory must resolve to a directory")
    result = build_wedge_solid(parameters)
    artifacts = tuple(
        _export_one(result.solid, output_directory, request, export_format)
        for export_format in request.formats
    )
    manifest_path = output_directory / f"{request.stem}.json"
    manifest_path.write_text(
        _manifest_json(parameters, request, result.measured, artifacts),
        encoding="utf-8",
    )
    return WedgeExportResult(
        artifacts=artifacts,
        manifest_path=manifest_path,
        measured=result.measured,
    )


def _export_one(
    solid: object,
    output_directory: Path,
    request: WedgeExportRequest,
    export_format: WedgeExportFormat,
) -> WedgeExportArtifact:
    from build123d import export_brep, export_step, export_stl

    path = output_directory / f"{request.stem}.{export_format.value}"
    if export_format is WedgeExportFormat.STEP:
        succeeded = export_step(solid, path, timestamp=_FIXED_STEP_TIMESTAMP)
    elif export_format is WedgeExportFormat.STL:
        succeeded = export_stl(
            solid,
            path,
            tolerance=request.linear_tolerance_m * 1_000.0,
            angular_tolerance=request.angular_tolerance_rad,
            ascii_format=False,
        )
    else:
        succeeded = export_brep(solid, path)
    if not succeeded or not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"failed to export {export_format.value} artifact")
    return WedgeExportArtifact(format=export_format, path=path)


def _manifest_json(
    parameters: WedgeHeadParameters,
    request: WedgeExportRequest,
    measured: WedgeMeasuredMetrics,
    artifacts: tuple[WedgeExportArtifact, ...],
) -> str:
    parameter_values = asdict(parameters)
    parameter_values["handedness"] = parameters.handedness.value
    payload = {
        "format": WEDGE_EXPORT_FORMAT,
        "units": {"angle": "degree", "length": "metre", "mass": "kilogram"},
        "kernel": {"name": "build123d/OpenCascade", "model_unit": "millimetre"},
        "parameters": parameter_values,
        "measured": asdict(measured),
        "tessellation": {
            "linear_tolerance_m": request.linear_tolerance_m,
            "angular_tolerance_rad": request.angular_tolerance_rad,
        },
        "artifacts": [
            {"format": item.format.value, "filename": item.path.name}
            for item in artifacts
        ],
    }
    if not all(
        math.isfinite(value)
        for value in asdict(measured).values()
        if isinstance(value, float)
    ):
        raise RuntimeError("measured export metrics must be finite")
    return json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"


__all__ = [
    "WEDGE_EXPORT_FORMAT",
    "WedgeExportArtifact",
    "WedgeExportFormat",
    "WedgeExportRequest",
    "WedgeExportResult",
    "export_wedge_artifacts",
]
