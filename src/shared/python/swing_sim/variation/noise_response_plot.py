"""Row-oriented plotting boundary for governed geometric response fields."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from shared.python.contracts import require

from .noise_response_fingerprint import response_field_fingerprint
from .noise_response_record import PositionNoiseResponseField


@dataclass(frozen=True)
class PositionNoiseResponsePlotRow:
    """One immutable input/time/point row with paired reviewer context."""

    input_id: str
    input_unit: str
    input_declared_scale: float
    input_normalization_scale: float
    source_layout_id: str
    adapter_id: str
    time_s: float
    coordinate_frame: str
    point_id: str
    signed_response: tuple[float, float, float]
    response_magnitude: float
    matched_absolute_rms_scatter_m: float
    all_eligible_absolute_rms_scatter_m: float
    availability_count: int
    all_eligible_count: int
    adequacy: str
    method_id: str
    normalization_id: str
    scientific_boundary: str
    field_sha256: str


def _plot_row(
    field: PositionNoiseResponseField,
    field_sha256: str,
    cell: tuple[int, int, int],
) -> PositionNoiseResponsePlotRow:
    input_index, sample_index, point_index = cell
    signed = field.signed_response_m_per_declared_scale[cell]
    return PositionNoiseResponsePlotRow(
        input_id=field.input_ids[input_index],
        input_unit=field.input_units[input_index],
        input_declared_scale=float(field.input_declared_scales[input_index]),
        input_normalization_scale=float(field.input_normalization_scales[input_index]),
        source_layout_id=field.source_layout_ids[input_index],
        adapter_id=field.adapter_ids[input_index],
        time_s=float(field.sample_times_s[sample_index]),
        coordinate_frame=field.coordinate_frame,
        point_id=field.point_ids[point_index],
        signed_response=(float(signed[0]), float(signed[1]), float(signed[2])),
        response_magnitude=float(field.response_magnitude_m_per_declared_scale[cell]),
        matched_absolute_rms_scatter_m=float(
            field.matched_absolute_rms_scatter_m[cell]
        ),
        all_eligible_absolute_rms_scatter_m=float(
            field.all_eligible_absolute_rms_scatter_m[cell]
        ),
        availability_count=int(field.availability_count[cell]),
        all_eligible_count=int(field.all_eligible_count[cell]),
        adequacy=str(field.adequacy[cell]),
        method_id=field.method_id,
        normalization_id=field.normalization_id,
        scientific_boundary=field.scientific_boundary,
        field_sha256=field_sha256,
    )


def iter_position_noise_response_plot_rows(
    field: PositionNoiseResponseField,
) -> Iterator[PositionNoiseResponsePlotRow]:
    """Yield plot-ready rows without duplicating source trace tensors."""
    require(isinstance(field, PositionNoiseResponseField), "invalid response field")
    field_sha256 = response_field_fingerprint(field)
    shape = field.response_magnitude_m_per_declared_scale.shape
    for cell in np.ndindex(shape):
        yield _plot_row(field, field_sha256, cell)


__all__ = [
    "PositionNoiseResponsePlotRow",
    "iter_position_noise_response_plot_rows",
]
