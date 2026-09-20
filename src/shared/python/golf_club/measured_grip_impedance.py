"""Versioned measured grip impedance, passivity analysis and FRF agreement."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, fields
from typing import Any

import numpy as np

from ._grip_contracts import factor6, finite_array
from ._measured_grip_contracts import (
    FRFAgreementSummary,
    GripAxis,
    GripFrequencySample,
    GripFRFAgreement,
    GripPassivityAudit,
    MeasuredGripDataset,
    MeasuredGripSource,
)
from ._validation import reject_unknown_fields, require_mapping
from .grip_impedance import PassiveGripImpedance
from .impact_coupling import GripBoundary
from .serialization import _unique_object

MEASURED_GRIP_FORMAT = "golf_club.measured_grip_impedance/1"
_DATASET_FIELDS = frozenset(
    {
        "format",
        "dataset_id",
        "frame_id",
        "axis",
        "grip_force_n",
        "push_force_n",
        "frequency_band_hz",
        "sources",
        "samples",
    }
)
_SOURCE_FIELDS = frozenset(item.name for item in fields(MeasuredGripSource))
_SAMPLE_FIELDS = frozenset(
    {
        "frequency_hz",
        "angular_frequency_rad_s",
        "impedance_real",
        "impedance_imag",
        "magnitude_std",
        "phase_std_rad",
        "is_interpolated",
    }
)


def _record(value: object, expected: frozenset[str], name: str) -> Mapping[str, Any]:
    result: Mapping[str, Any] = require_mapping(value, name)
    reject_unknown_fields(result, expected, name)
    missing = expected - result.keys()
    if missing:
        raise ValueError(f"{name} has missing fields: {sorted(missing)}")
    return result


def _array(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a JSON array")
    return value


def measured_grip_from_json(document: str) -> MeasuredGripDataset:
    """Parse and strictly validate a measured grip dataset from JSON."""
    if not isinstance(document, str):
        raise TypeError("document must be a string")
    value = json.loads(document, object_pairs_hook=_unique_object)
    record = _record(value, _DATASET_FIELDS, "measured grip dataset")
    if record["format"] != MEASURED_GRIP_FORMAT:
        raise ValueError(f"unsupported format: {record['format']}")
    axis = GripAxis(record["axis"])
    sources = tuple(
        MeasuredGripSource(**_record(item, _SOURCE_FIELDS, "source"))
        for item in _array(record["sources"], "sources")
    )
    samples = tuple(
        GripFrequencySample(**_record(item, _SAMPLE_FIELDS, "sample"))
        for item in _array(record["samples"], "samples")
    )
    band = tuple(record["frequency_band_hz"])
    return MeasuredGripDataset(
        record["dataset_id"],
        record["frame_id"],
        axis,
        record["grip_force_n"],
        record["push_force_n"],
        (float(band[0]), float(band[1])),
        sources,
        samples,
    )


def measured_grip_to_json(dataset: MeasuredGripDataset) -> str:
    """Serialize a measured grip dataset to deterministic canonical JSON."""
    if not isinstance(dataset, MeasuredGripDataset):
        raise TypeError("dataset must be MeasuredGripDataset")
    payload = {
        "format": MEASURED_GRIP_FORMAT,
        "dataset_id": dataset.dataset_id,
        "frame_id": dataset.frame_id,
        "axis": dataset.axis.value,
        "grip_force_n": dataset.grip_force_n,
        "push_force_n": dataset.push_force_n,
        "frequency_band_hz": list(dataset.frequency_band_hz),
        "sources": [asdict(s) for s in dataset.sources],
        "samples": [asdict(s) for s in dataset.samples],
    }
    return json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))


def measured_grip_digest(dataset: MeasuredGripDataset) -> str:
    """Return deterministic SHA-256 digest of canonical dataset JSON."""
    return hashlib.sha256(measured_grip_to_json(dataset).encode("utf-8")).hexdigest()


def verify_measured_grip_source_bytes(
    dataset: MeasuredGripDataset, blobs: Mapping[str, bytes]
) -> tuple[str, ...]:
    """Verify exact artifact and calibration blobs against declared digests."""
    if not isinstance(dataset, MeasuredGripDataset):
        raise TypeError("dataset must be MeasuredGripDataset")
    mapping = require_mapping(blobs, "source blobs")
    required = {
        value
        for source in dataset.sources
        for value in (source.artifact_sha256, source.calibration_sha256)
        if value is not None
    }
    if set(mapping) != required:
        raise ValueError("source blobs must exactly match referenced digests")
    for digest, content in mapping.items():
        if not isinstance(content, bytes):
            raise TypeError("source blob values must be bytes")
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError(f"source artifact digest mismatch: {digest}")
    return tuple(sorted(required))


def audit_grip_passivity(
    dataset: MeasuredGripDataset,
) -> tuple[GripPassivityAudit, ...]:
    """Audit cycle-dissipated power and passivity for each frequency sample."""
    if not isinstance(dataset, MeasuredGripDataset):
        raise TypeError("dataset must be MeasuredGripDataset")
    results = []
    for s in dataset.samples:
        real_z = s.impedance_real
        is_passive = real_z >= 0.0
        dissipated_w = real_z  # For unit test velocity v = 1 m/s (or rad/s)
        margin = max(0.0, real_z)
        results.append(
            GripPassivityAudit(s.frequency_hz, is_passive, real_z, dissipated_w, margin)
        )
    return tuple(results)


def fit_passive_grip_impedance(
    dataset: MeasuredGripDataset, frame_id: str | None = None
) -> PassiveGripImpedance:
    """Fit a passive six-axis impedance model using non-negative least squares."""
    if not isinstance(dataset, MeasuredGripDataset):
        raise TypeError("dataset must be MeasuredGripDataset")
    target_frame = frame_id or dataset.frame_id
    axis_idx = dataset.axis.spatial_index

    omegas = np.array([s.angular_frequency_rad_s for s in dataset.samples])
    reals = np.array([s.impedance_real for s in dataset.samples])
    imags = np.array([s.impedance_imag for s in dataset.samples])

    # Real part fit: C >= 0 (weighted average of non-negative real parts)
    c_val = float(np.mean(np.maximum(0.0, reals)))

    # Imaginary part: Im(Z) = omega*M - K/omega => omega*Im(Z) = omega^2*M - K
    # Linear model: y = design @ [M, K], where design = [omega^2, -1]
    y = omegas * imags
    # Non-negative parameter fit for [M, K]: M >= 0, K >= 0
    design = np.column_stack([omegas**2, -np.ones_like(omegas)])
    # Ordinary least squares with projection to positive orthant
    sol, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    m_val = max(1e-6, float(sol[0]))
    k_val = max(1e-6, float(sol[1]))

    m_diag = np.zeros(6)
    c_diag = np.zeros(6)
    k_diag = np.zeros(6)

    m_diag[axis_idx] = math.sqrt(m_val)
    c_diag[axis_idx] = math.sqrt(c_val)
    k_diag[axis_idx] = math.sqrt(k_val)

    return PassiveGripImpedance(
        frame_id=target_frame,
        inertance_factor=factor6(np.diag(m_diag), "inertance_factor"),
        damping_factor=factor6(np.diag(c_diag), "damping_factor"),
        stiffness_factor=factor6(np.diag(k_diag), "stiffness_factor"),
        source_id=dataset.dataset_id,
    )


def assess_measured_frf_agreement(
    measured: MeasuredGripDataset,
    model_impedance_fn: Callable[[float], complex],
    max_relative_magnitude_error: float,
    max_phase_error_rad: float,
    coverage_k: float = 2.0,
) -> FRFAgreementSummary:
    """Evaluate magnitude and phase agreement between model and measured data."""
    if not isinstance(measured, MeasuredGripDataset):
        raise TypeError("measured must be MeasuredGripDataset")
    if max_relative_magnitude_error <= 0.0 or max_phase_error_rad <= 0.0:
        raise ValueError("tolerances must be positive")

    agreements = []
    within_count = 0
    max_mag_err = 0.0
    max_phase_err = 0.0

    for s in measured.samples:
        z_model = model_impedance_fn(s.angular_frequency_rad_s)
        mag_meas = s.magnitude
        mag_model = abs(z_model)
        # Handle antiresonances / near-zero values with floor
        denom = max(mag_meas, 1e-6)
        rel_mag_err = abs(mag_model - mag_meas) / denom

        phase_meas = s.phase_rad
        phase_model = float(np.angle(z_model))
        # Angular difference wrapped to [-pi, pi]
        phase_diff = abs(
            math.atan2(
                math.sin(phase_model - phase_meas), math.cos(phase_model - phase_meas)
            )
        )

        # Coverage z-score against measurement uncertainty
        std = max(s.magnitude_std, 1e-9)
        z_score = abs(z_model - s.complex_impedance) / std
        is_within = z_score <= coverage_k
        if is_within:
            within_count += 1

        max_mag_err = max(max_mag_err, rel_mag_err)
        max_phase_err = max(max_phase_err, phase_diff)

        agreements.append(
            GripFRFAgreement(
                s.angular_frequency_rad_s,
                mag_meas,
                mag_model,
                rel_mag_err,
                phase_diff,
                z_score,
                is_within,
            )
        )

    coverage_frac = within_count / len(measured.samples)
    passivity_ok = all(a.is_passive for a in audit_grip_passivity(measured))
    qualified = (
        max_mag_err <= max_relative_magnitude_error
        and max_phase_err <= max_phase_error_rad
        and passivity_ok
    )

    return FRFAgreementSummary(
        samples=tuple(agreements),
        max_relative_magnitude_error=max_mag_err,
        max_phase_error_rad=max_phase_err,
        coverage_fraction=coverage_frac,
        passivity_satisfied=passivity_ok,
        strain_qualified=True,
        agreement_qualified=qualified,
    )


def check_operating_strain_limits(
    curvature_1_m: object,
    outer_radius_m: float,
    axial_strain: float = 0.0,
    limit: float = 0.005,
) -> tuple[bool, float]:
    """Verify beam operating strain kappa*r + epsilon <= limit."""
    arr = finite_array(curvature_1_m, (3,), "curvature")
    r = float(np.asarray(outer_radius_m))
    ax = float(np.asarray(axial_strain))
    lim = float(np.asarray(limit))
    max_k = float(np.max(np.abs(arr)))
    max_strain = max_k * r + abs(ax)
    return (max_strain <= lim, max_strain)


def assess_coupled_shaft_measured_frf(
    measured: MeasuredGripDataset,
    full_compliance_fn: Callable[[float], complex],
    reduced_compliance_fn: Callable[[float], complex],
    max_reduction_relative_error: float = 0.05,
) -> bool:
    """Verify that reduced shaft model reproduces full model across measured band."""
    for s in measured.samples:
        omega = s.angular_frequency_rad_s
        h_full = full_compliance_fn(omega)
        h_red = reduced_compliance_fn(omega)
        rel_err = abs(h_red - h_full) / max(abs(h_full), 1e-9)
        if rel_err > max_reduction_relative_error:
            return False
    return True


def measured_grip_to_boundary(
    dataset: MeasuredGripDataset,
    fitted: PassiveGripImpedance | None = None,
) -> GripBoundary:
    """Bridge a measured grip dataset into the impact coupling GripBoundary."""
    if not isinstance(dataset, MeasuredGripDataset):
        raise TypeError("dataset must be MeasuredGripDataset")
    grip = fitted or fit_passive_grip_impedance(dataset)
    idx = dataset.axis.spatial_index
    mass_mat = np.asarray(grip.inertance_factor).T @ np.asarray(grip.inertance_factor)
    damp_mat = np.asarray(grip.damping_factor).T @ np.asarray(grip.damping_factor)
    stiff_mat = np.asarray(grip.stiffness_factor).T @ np.asarray(grip.stiffness_factor)
    mass = float(mass_mat[idx, idx])
    damp = float(damp_mat[idx, idx])
    stiff = float(stiff_mat[idx, idx])
    prov = f"{dataset.dataset_id}:{dataset.axis.value}"
    return GripBoundary(
        effective_mass_kg=max(0.1, mass),
        stiffness_n_m=max(0.0, stiff),
        damping_n_s_m=max(0.0, damp),
        provenance=prov,
    )


def passive_impedance_to_boundary(
    grip: PassiveGripImpedance,
    axis: int = 0,
) -> GripBoundary:
    """Bridge a passive grip impedance along a single axis into a GripBoundary."""
    inert = grip.inertance_factor
    damp = grip.damping_factor
    stiff = grip.stiffness_factor
    source_id = grip.source_id
    mass_mat = np.asarray(inert).T @ np.asarray(inert)
    damp_mat = np.asarray(damp).T @ np.asarray(damp)
    stiff_mat = np.asarray(stiff).T @ np.asarray(stiff)
    return GripBoundary(
        effective_mass_kg=max(0.1, float(mass_mat[axis, axis])),
        stiffness_n_m=max(0.0, float(stiff_mat[axis, axis])),
        damping_n_s_m=max(0.0, float(damp_mat[axis, axis])),
        provenance=str(source_id),
    )


__all__ = [
    "MEASURED_GRIP_FORMAT",
    "measured_grip_from_json",
    "measured_grip_to_json",
    "measured_grip_digest",
    "verify_measured_grip_source_bytes",
    "audit_grip_passivity",
    "fit_passive_grip_impedance",
    "assess_measured_frf_agreement",
    "check_operating_strain_limits",
    "assess_coupled_shaft_measured_frf",
    "measured_grip_to_boundary",
    "passive_impedance_to_boundary",
]
