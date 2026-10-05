"""Controlled deterministic CAD and mesh export for iron solids.

The export formats, request validation, and artifact record are the
family-neutral wedge definitions (one definition each); only the default
stem, the manifest identity, and the measured payload are iron-specific.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from ._head_cad import export_solid_file
from .iron_cad import IronMeasuredMetrics, build_iron_solid
from .iron_parameters import IronHeadParameters
from .iron_serialization import IRON_PARAMETERS_FORMAT, iron_parameters_to_json
from .wedge_export import WedgeExportArtifact, WedgeExportRequest

IRON_EXPORT_FORMAT = "golf_club.iron_export/1"


@dataclass(frozen=True)
class IronExportRequest(WedgeExportRequest):
    """Validated output location, formats, and tessellation tolerances."""

    stem: str = "iron-head"


@dataclass(frozen=True)
class IronExportResult:
    """Complete artifact set, manifest path, and measured solid metrics."""

    artifacts: tuple[WedgeExportArtifact, ...]
    manifest_path: Path
    measured: IronMeasuredMetrics


def export_iron_artifacts(
    parameters: IronHeadParameters,
    request: IronExportRequest,
) -> IronExportResult:
    """Build once and export deterministic exact/mesh artifacts plus manifest.

    Postcondition: every requested artifact exists and is nonempty, and the
    manifest records the versioned parameters, the recovered metrics, and the
    tessellation tolerances that bound the mesh-vs-exact deviation.
    """
    if not isinstance(parameters, IronHeadParameters):
        raise TypeError("parameters must be IronHeadParameters")
    if not isinstance(request, IronExportRequest):
        raise TypeError("request must be IronExportRequest")
    output_directory = request.output_directory
    assert isinstance(output_directory, Path)  # normalized by request contract
    output_directory.mkdir(parents=True, exist_ok=True)
    result = build_iron_solid(parameters)
    tolerances = (request.linear_tolerance_m, request.angular_tolerance_rad)
    artifacts = []
    for export_format in request.formats:
        path = output_directory / f"{request.stem}.{export_format.value}"
        export_solid_file(result.solid, path, export_format.value, tolerances)
        artifacts.append(WedgeExportArtifact(format=export_format, path=path))
    manifest_path = output_directory / f"{request.stem}.json"
    manifest_path.write_text(
        _manifest_json(parameters, request, result.measured, tuple(artifacts)),
        encoding="utf-8",
        newline="\n",
    )
    return IronExportResult(
        artifacts=tuple(artifacts),
        manifest_path=manifest_path,
        measured=result.measured,
    )


def _manifest_json(
    parameters: IronHeadParameters,
    request: IronExportRequest,
    measured: IronMeasuredMetrics,
    artifacts: tuple[WedgeExportArtifact, ...],
) -> str:
    measured_values = asdict(measured)
    scalars = [
        *measured.cg_m,
        *(v for v in measured_values.values() if isinstance(v, float)),
    ]
    if not all(math.isfinite(value) for value in scalars):
        raise RuntimeError("measured export metrics must be finite")
    payload = {
        "format": IRON_EXPORT_FORMAT,
        "parameters_format": IRON_PARAMETERS_FORMAT,
        "units": {"angle": "degree", "length": "metre", "mass": "kilogram"},
        "kernel": {"name": "build123d/OpenCascade", "model_unit": "millimetre"},
        "mass_properties": "uniform density, OpenCascade exact-solid integration",
        "parameters": json.loads(iron_parameters_to_json(parameters))["parameters"],
        "measured": measured_values,
        "tessellation": {
            "linear_tolerance_m": request.linear_tolerance_m,
            "angular_tolerance_rad": request.angular_tolerance_rad,
        },
        "artifacts": [
            {"format": item.format.value, "filename": item.path.name}
            for item in artifacts
        ],
    }
    return json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"


__all__ = [
    "IRON_EXPORT_FORMAT",
    "IronExportRequest",
    "IronExportResult",
    "export_iron_artifacts",
]
