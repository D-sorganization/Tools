"""Truncation-induced mean shift and skewness analysis (#4253 item d).

Quantifies and highlights the shift in effective parameter mean, variance,
and distribution shape caused by bounding (clipping) input noise specs.
When input distributions are truncated asymmetrically, the realized sample mean
shifts away from the nominal base value, subtly moving the physical operating point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shared.python.contracts import require

from .engine import VariationDataset


@dataclass(frozen=True)
class TruncationShiftNote:
    """Quantifies mean-shift and skewness induced by bounding/truncating an input."""

    variable_key: str
    nominal_base: float
    realized_mean: float
    mean_shift: float
    nominal_scale: float
    realized_std: float
    lower_bound: float | None
    upper_bound: float | None
    truncated_lower_count: int
    truncated_upper_count: int
    has_shift: bool
    note: str


def detect_truncation_mean_shifts(
    dataset: VariationDataset,
    shift_threshold_fraction: float = 0.05,
) -> tuple[TruncationShiftNote, ...]:
    """Inspect input columns of a dataset for truncation-induced mean shifts."""
    require(dataset is not None, "dataset cannot be None")
    base = dataset.plan.resolved_base()
    notes: list[TruncationShiftNote] = []

    for idx, spec in enumerate(dataset.plan.noise):
        if spec.lower is None and spec.upper is None:
            continue

        var_key = spec.variable_key
        nominal_base = float(base[var_key])
        nominal_scale = float(spec.scale)

        inputs = dataset.inputs[dataset.success, idx]
        if inputs.size == 0:
            continue

        realized_mean = float(np.mean(inputs))
        mean_shift = realized_mean - nominal_base
        realized_std = float(np.std(inputs, ddof=1)) if inputs.size >= 2 else 0.0

        lo_count = (
            int(np.count_nonzero(inputs <= spec.lower + 1e-9))
            if spec.lower is not None
            else 0
        )
        hi_count = (
            int(np.count_nonzero(inputs >= spec.upper - 1e-9))
            if spec.upper is not None
            else 0
        )

        # Flag if mean shifted by > threshold fraction of scale, or clamping occurred
        has_shift = (
            abs(mean_shift) > shift_threshold_fraction * nominal_scale
            or lo_count > 0
            or hi_count > 0
        )

        if not has_shift:
            continue

        lo_str = f"{spec.lower}" if spec.lower is not None else "-inf"
        hi_str = f"{spec.upper}" if spec.upper is not None else "+inf"
        bounds_str = f"[{lo_str}, {hi_str}]"
        note_text = (
            f"{var_key}: nominal base {nominal_base:.3g} shifted by "
            f"{mean_shift:+.3g} to realized mean {realized_mean:.3g} "
            f"(nominal scale {nominal_scale:.3g} -> realized std {realized_std:.3g}) "
            f"due to truncation bounds {bounds_str} with {lo_count} lower and "
            f"{hi_count} upper clamp events. Output sensitivities and dispersion "
            "reflect this shifted operating point."
        )

        notes.append(
            TruncationShiftNote(
                variable_key=var_key,
                nominal_base=nominal_base,
                realized_mean=realized_mean,
                mean_shift=mean_shift,
                nominal_scale=nominal_scale,
                realized_std=realized_std,
                lower_bound=spec.lower,
                upper_bound=spec.upper,
                truncated_lower_count=lo_count,
                truncated_upper_count=hi_count,
                has_shift=has_shift,
                note=note_text,
            )
        )

    return tuple(notes)


__all__ = [
    "TruncationShiftNote",
    "detect_truncation_mean_shifts",
]
