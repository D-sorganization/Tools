"""Tests for measured grip impedance, passivity and FRF agreement."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any

import numpy as np
import pytest

from shared.python.golf_club._measured_grip_contracts import (
    GripAxis,
    GripFrequencySample,
    MeasuredGripDataset,
    MeasuredGripSource,
)
from shared.python.golf_club.grip_impedance import (
    PassiveGripImpedance,
    grip_frequency_impedance,
)
from shared.python.golf_club.impact_coupling import (
    CoupledImpactConfig,
    GripBoundary,
    simulate_coupled_impact,
)
from shared.python.golf_club.measured_grip_impedance import (
    MEASURED_GRIP_FORMAT,
    assess_coupled_shaft_measured_frf,
    assess_measured_frf_agreement,
    audit_grip_passivity,
    check_operating_strain_limits,
    fit_passive_grip_impedance,
    measured_grip_digest,
    measured_grip_from_json,
    measured_grip_to_boundary,
    measured_grip_to_json,
    passive_impedance_to_boundary,
    verify_measured_grip_source_bytes,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract, pytest.mark.scientific]


def _synthetic_source() -> dict[str, Any]:
    raw = b"kit-1000194060-raw-data"
    return {
        "source_id": "synthetic-passive-grip-fixture",
        "kind": "synthetic",
        "artifact_sha256": hashlib.sha256(raw).hexdigest(),
        "calibration_sha256": None,
        "method": "closed_form_mass_damping_stiffness_fixture",
        "uncertainty_note": "deterministic numerical fixture uncertainty",
        "data_license": "CC0-1.0",
    }


def _measurement_source() -> dict[str, Any]:
    source = _synthetic_source()
    source["source_id"] = "measurement-contract-fixture"
    source["kind"] = "measurement-derived"
    source["calibration_sha256"] = hashlib.sha256(
        b"kit-calibration-rig-data"
    ).hexdigest()
    source["method"] = "contract_fixture_with_calibration_identity"
    return source


def _synthetic_dataset_dict() -> dict[str, Any]:
    # Mass = 2.0 kg, Damping = 40.0 N s/m, Stiffness = 20000.0 N/m
    # Z(omega) = C + i*(omega*M - K/omega)
    m_true, c_true, k_true = 2.0, 40.0, 20000.0
    freqs_hz = [10.0, 20.0, 30.0, 50.0, 100.0, 150.0, 200.0]
    samples = []
    for f in freqs_hz:
        omega = 2.0 * math.pi * f
        z_exact = c_true + 1j * (omega * m_true - k_true / omega)
        mag = abs(z_exact)
        samples.append(
            {
                "frequency_hz": f,
                "angular_frequency_rad_s": omega,
                "impedance_real": float(z_exact.real),
                "impedance_imag": float(z_exact.imag),
                "magnitude_std": 0.05 * mag,
                "phase_std_rad": 0.03,
                "is_interpolated": f in (100.0, 200.0),
            }
        )
    return {
        "format": MEASURED_GRIP_FORMAT,
        "dataset_id": "kit-xh-translation-p1",
        "frame_id": "grip",
        "axis": "tx",
        "grip_force_n": 50.0,
        "push_force_n": 20.0,
        "frequency_band_hz": [10.0, 200.0],
        "sources": [_synthetic_source()],
        "samples": samples,
    }


@pytest.fixture
def dataset_payload() -> dict[str, Any]:
    return _synthetic_dataset_dict()


@pytest.fixture
def dataset(dataset_payload: dict[str, Any]) -> MeasuredGripDataset:
    doc = json.dumps(dataset_payload)
    return measured_grip_from_json(doc)


def test_schema_serialization_roundtrip_and_digest(
    dataset: MeasuredGripDataset, dataset_payload: dict[str, Any]
) -> None:
    json_str = measured_grip_to_json(dataset)
    restored = measured_grip_from_json(json_str)
    assert restored.dataset_id == dataset.dataset_id
    assert restored.axis == GripAxis.TRANSLATION_X
    assert restored.frame_id == "grip"
    assert len(restored.samples) == len(dataset.samples)
    assert restored.frequency_band_hz == (10.0, 200.0)
    assert restored.grip_force_n == pytest.approx(50.0)

    # Digest verification
    digest1 = measured_grip_digest(dataset)
    digest2 = measured_grip_digest(restored)
    assert digest1 == digest2
    assert len(digest1) == 64

    # Source byte verification
    blobs = {
        dataset.sources[0].artifact_sha256: b"kit-1000194060-raw-data",
    }
    verified = verify_measured_grip_source_bytes(dataset, blobs)
    assert len(verified) == 1


def test_schema_refusal_cases(dataset_payload: dict[str, Any]) -> None:
    # Invalid format string
    corrupt = copy.deepcopy(dataset_payload)
    corrupt["format"] = "invalid_format/9"
    with pytest.raises(ValueError, match="format"):
        measured_grip_from_json(json.dumps(corrupt))

    # Missing calibration digest for measurement-derived source
    corrupt = copy.deepcopy(dataset_payload)
    corrupt["sources"] = [_measurement_source()]
    corrupt["sources"][0]["calibration_sha256"] = None
    with pytest.raises(ValueError, match="calibration_sha256"):
        measured_grip_from_json(json.dumps(corrupt))

    # Frequency out of order
    corrupt = copy.deepcopy(dataset_payload)
    corrupt["samples"][1]["frequency_hz"] = 5.0  # smaller than sample 0 (10.0)
    corrupt["frequency_band_hz"] = [1.0, 200.0]
    with pytest.raises(ValueError, match="ascending"):
        measured_grip_from_json(json.dumps(corrupt))

    # Inverted frequency band
    corrupt = copy.deepcopy(dataset_payload)
    corrupt["frequency_band_hz"] = [200.0, 10.0]
    with pytest.raises(ValueError, match="band"):
        measured_grip_from_json(json.dumps(corrupt))

    # Negative uncertainty
    corrupt = copy.deepcopy(dataset_payload)
    corrupt["samples"][0]["magnitude_std"] = -1.0
    with pytest.raises(ValueError, match="uncertainty standard deviations"):
        measured_grip_from_json(json.dumps(corrupt))


def test_audit_grip_passivity_clean(dataset: MeasuredGripDataset) -> None:
    audits = audit_grip_passivity(dataset)
    assert len(audits) == len(dataset.samples)
    for audit in audits:
        assert audit.is_passive is True
        assert audit.minimum_real_impedance >= 0.0
        assert audit.dissipated_power_w >= 0.0
        assert audit.passivity_margin >= 0.0


def test_audit_grip_passivity_violation(dataset_payload: dict[str, Any]) -> None:
    # Inject negative real impedance (active behavior) at 50 Hz
    corrupt = copy.deepcopy(dataset_payload)
    for s in corrupt["samples"]:
        if s["frequency_hz"] == 50.0:
            s["impedance_real"] = -5.0
    dataset = measured_grip_from_json(json.dumps(corrupt))
    audits = audit_grip_passivity(dataset)
    f50_audits = [a for a in audits if a.frequency_hz == 50.0]
    assert len(f50_audits) == 1
    assert f50_audits[0].is_passive is False
    assert f50_audits[0].minimum_real_impedance < 0.0
    assert f50_audits[0].dissipated_power_w < 0.0


def test_fit_passive_grip_impedance(dataset: MeasuredGripDataset) -> None:
    # Fitting yields a PassiveGripImpedance with PSD Gram factors
    grip = fit_passive_grip_impedance(dataset)
    assert isinstance(grip, PassiveGripImpedance)
    assert grip.frame_id == dataset.frame_id
    assert grip.source_id == dataset.dataset_id

    # The identified parameters should be close to true 2.0 kg, 40.0 Ns/m, 20000.0 N/m
    mass_matrix = np.asarray(grip.inertance_factor).T @ np.asarray(
        grip.inertance_factor
    )
    damping_matrix = np.asarray(grip.damping_factor).T @ np.asarray(grip.damping_factor)
    stiff_matrix = np.asarray(grip.stiffness_factor).T @ np.asarray(
        grip.stiffness_factor
    )

    # Check x-translation entry (index 0)
    assert mass_matrix[0, 0] == pytest.approx(2.0, rel=0.05)
    assert damping_matrix[0, 0] == pytest.approx(40.0, rel=0.05)
    assert stiff_matrix[0, 0] == pytest.approx(20000.0, rel=0.05)

    # Frequency impedance must be strictly passive for all omega > 0
    for omega in (50.0, 100.0, 200.0, 500.0):
        z_model = grip_frequency_impedance(grip, omega)
        assert z_model[0, 0].real >= 0.0


def test_passive_fit_uses_constrained_optimum_when_mass_is_active(
    dataset_payload: dict[str, Any],
) -> None:
    """A zero-mass boundary must re-fit stiffness instead of clipping OLS."""
    corrupted = copy.deepcopy(dataset_payload)
    target_values = []
    for sample in corrupted["samples"]:
        omega = float(sample["angular_frequency_rad_s"])
        # y = omega * Im(Z) = -2 * omega**2 - 10 gives an unconstrained
        # fit M=-2, K=10.  With M constrained to zero, least squares instead
        # requires K=-mean(y), which differs substantially from clipping K=10.
        target = -2.0 * omega**2 - 10.0
        sample["impedance_imag"] = target / omega
        target_values.append(target)
    constrained = fit_passive_grip_impedance(
        measured_grip_from_json(json.dumps(corrupted))
    )
    stiffness = np.asarray(constrained.stiffness_factor).T @ np.asarray(
        constrained.stiffness_factor
    )
    mass = np.asarray(constrained.inertance_factor).T @ np.asarray(
        constrained.inertance_factor
    )

    assert mass[0, 0] == pytest.approx(0.0, abs=1e-12)
    assert stiffness[0, 0] == pytest.approx(-float(np.mean(target_values)))


def test_passive_fit_uses_constrained_optimum_when_damping_is_active(
    dataset_payload: dict[str, Any],
) -> None:
    """The scalar damping fit is the non-negative mean, not mean of positives."""
    corrupted = copy.deepcopy(dataset_payload)
    observed_damping = [-5.0, 1.0, 1.0, -2.0, 1.0, 1.0, 1.0]
    for sample, damping in zip(corrupted["samples"], observed_damping, strict=True):
        sample["impedance_real"] = damping
    constrained = fit_passive_grip_impedance(
        measured_grip_from_json(json.dumps(corrupted))
    )
    damping = np.asarray(constrained.damping_factor).T @ np.asarray(
        constrained.damping_factor
    )

    assert damping[0, 0] == pytest.approx(0.0, abs=1e-12)


def test_synthetic_frf_agreement_is_numerical_not_physical(
    dataset: MeasuredGripDataset,
) -> None:
    grip = fit_passive_grip_impedance(dataset)

    def model_fn(omega: float) -> complex:
        return complex(grip_frequency_impedance(grip, omega)[0, 0])

    summary = assess_measured_frf_agreement(
        dataset,
        model_fn,
        max_relative_magnitude_error=0.10,
        max_phase_error_rad=0.15,
        coverage_k=2.0,
        strain_qualified=True,
    )
    assert summary.agreement_qualified is False
    assert summary.passivity_satisfied is True
    assert summary.max_relative_magnitude_error < 0.10
    assert summary.max_phase_error_rad < 0.15
    assert summary.coverage_fraction == pytest.approx(1.0)


def test_frf_agreement_requires_explicit_strain_qualification(
    dataset_payload: dict[str, Any],
) -> None:
    """FRF agreement cannot certify operation without a strain assessment."""
    measured_payload = copy.deepcopy(dataset_payload)
    measured_payload["sources"] = [_measurement_source()]
    dataset = measured_grip_from_json(json.dumps(measured_payload))
    grip = fit_passive_grip_impedance(dataset)

    def model_fn(omega: float) -> complex:
        return complex(grip_frequency_impedance(grip, omega)[0, 0])

    omitted = assess_measured_frf_agreement(
        dataset,
        model_fn,
        max_relative_magnitude_error=0.10,
        max_phase_error_rad=0.15,
    )
    refused = assess_measured_frf_agreement(
        dataset,
        model_fn,
        max_relative_magnitude_error=0.10,
        max_phase_error_rad=0.15,
        strain_qualified=False,
    )
    accepted = assess_measured_frf_agreement(
        dataset,
        model_fn,
        max_relative_magnitude_error=0.10,
        max_phase_error_rad=0.15,
        strain_qualified=True,
    )

    assert omitted.strain_qualified is False
    assert omitted.agreement_qualified is False
    assert refused.strain_qualified is False
    assert refused.agreement_qualified is False
    assert accepted.strain_qualified is True
    assert accepted.agreement_qualified is True


def test_frf_agreement_rejects_non_boolean_strain_qualification(
    dataset: MeasuredGripDataset,
) -> None:
    """Qualification evidence is a Boolean contract, not a truthy flag."""
    with pytest.raises(TypeError, match="strain_qualified"):
        assess_measured_frf_agreement(
            dataset,
            lambda _omega: 1.0 + 0.0j,
            max_relative_magnitude_error=1.0,
            max_phase_error_rad=1.0,
            strain_qualified=1,  # type: ignore[arg-type]
        )


def test_measurement_derived_frf_can_qualify(
    dataset_payload: dict[str, Any],
) -> None:
    """Physical qualification requires a calibrated measurement declaration."""
    measured_payload = copy.deepcopy(dataset_payload)
    measured_payload["sources"] = [_measurement_source()]
    measured = measured_grip_from_json(json.dumps(measured_payload))
    grip = fit_passive_grip_impedance(measured)

    summary = assess_measured_frf_agreement(
        measured,
        lambda omega: complex(grip_frequency_impedance(grip, omega)[0, 0]),
        max_relative_magnitude_error=0.10,
        max_phase_error_rad=0.15,
        coverage_k=2.0,
        strain_qualified=True,
    )

    assert summary.agreement_qualified is True


def test_check_operating_strain_limits() -> None:
    # Linear elastic strain threshold: 0.005 (0.5%)
    curvature = np.array([0.1, 0.2, 0.05])  # 1/m
    radius = 0.007  # m (7mm shaft outer radius)
    axial_strain = 0.0005

    # max strain = 0.2 * 0.007 + 0.0005 = 0.0019 < 0.005 -> Passes
    qualified, max_strain = check_operating_strain_limits(
        curvature, radius, axial_strain, limit=0.005
    )
    assert qualified is True
    assert max_strain == pytest.approx(0.0019)

    # Exceeding limit -> Refuses
    qualified_fail, max_strain_fail = check_operating_strain_limits(
        curvature, radius, axial_strain, limit=0.001
    )
    assert qualified_fail is False
    assert max_strain_fail == pytest.approx(0.0019)


def test_consumer_integration_with_impact_coupling(
    dataset: MeasuredGripDataset,
) -> None:
    boundary = measured_grip_to_boundary(dataset)
    assert isinstance(boundary, GripBoundary)
    assert boundary.effective_mass_kg == pytest.approx(2.0, rel=0.05)
    assert boundary.stiffness_n_m == pytest.approx(20000.0, rel=0.05)
    assert boundary.damping_n_s_m == pytest.approx(40.0, rel=0.05)
    assert "kit-xh-translation-p1" in boundary.provenance

    # Run simulate_coupled_impact with this boundary
    config = CoupledImpactConfig(
        head_mass_kg=0.200,
        shaft_stiffness_n_m=10000.0,
        grip=boundary,
        head_speed_mps=45.0,
    )
    result = simulate_coupled_impact(config)
    assert result.ball_speed_mps > 30.0
    assert result.contact_time_s > 0.0
    assert result.peak_contact_force_n > 5000.0

    # Boundary conversion helpers
    b1 = measured_grip_to_boundary(dataset)
    assert b1.effective_mass_kg == boundary.effective_mass_kg

    fitted_grip = fit_passive_grip_impedance(dataset)
    b2 = passive_impedance_to_boundary(fitted_grip, axis=0)
    assert b2.effective_mass_kg == boundary.effective_mass_kg


def test_rotational_axis_passivity_and_fitting() -> None:
    # Test rotational axis 'ry' (KIT 1000194062 candidate, 10-100 Hz)
    # J = 0.05 kg m^2, C = 0.5 N m s/rad, K = 200.0 N m/rad
    j_true, c_true, k_true = 0.05, 0.5, 200.0
    freqs = [10.0, 25.0, 50.0, 75.0, 100.0]
    samples = []
    for f in freqs:
        omega = 2.0 * math.pi * f
        z = c_true + 1j * (omega * j_true - k_true / omega)
        samples.append(
            GripFrequencySample(
                frequency_hz=f,
                angular_frequency_rad_s=omega,
                impedance_real=float(z.real),
                impedance_imag=float(z.imag),
                magnitude_std=0.02 * abs(z),
                phase_std_rad=0.01,
            )
        )
    source = MeasuredGripSource(
        source_id="kit-1000194062-ry",
        kind="measurement-derived",
        artifact_sha256=hashlib.sha256(b"rotational-artifact").hexdigest(),
        calibration_sha256=hashlib.sha256(b"rotational-cal").hexdigest(),
        method="rotational yh handle excitation, 10-100 Hz",
        uncertainty_note="apparent inertia and torque impedance",
        data_license="CC-BY-4.0",
    )
    dataset = MeasuredGripDataset(
        dataset_id="kit-ry-rotation-p1",
        frame_id="grip",
        axis=GripAxis.ROTATION_Y,
        grip_force_n=40.0,
        push_force_n=15.0,
        frequency_band_hz=(10.0, 100.0),
        sources=(source,),
        samples=tuple(samples),
    )

    audits = audit_grip_passivity(dataset)
    assert all(a.is_passive for a in audits)

    grip = fit_passive_grip_impedance(dataset)
    assert isinstance(grip, PassiveGripImpedance)
    inertia_matrix = np.asarray(grip.inertance_factor).T @ np.asarray(
        grip.inertance_factor
    )
    damp_matrix = np.asarray(grip.damping_factor).T @ np.asarray(grip.damping_factor)
    stiff_matrix = np.asarray(grip.stiffness_factor).T @ np.asarray(
        grip.stiffness_factor
    )

    # ry is index 4
    assert inertia_matrix[4, 4] == pytest.approx(j_true, rel=0.05)
    assert damp_matrix[4, 4] == pytest.approx(c_true, rel=0.05)
    assert stiff_matrix[4, 4] == pytest.approx(k_true, rel=0.05)


def test_assess_coupled_shaft_measured_frf(dataset: MeasuredGripDataset) -> None:
    # Synthetic full and reduced compliance functions
    def full_fn(omega: float) -> complex:
        return 1.0 / (20000.0 - omega**2 * 2.0 + 1j * omega * 40.0)

    def reduced_fn(omega: float) -> complex:
        # 1% perturbation
        return 1.0 / (20000.0 * 1.01 - omega**2 * 2.0 * 1.01 + 1j * omega * 40.0)

    agreed = assess_coupled_shaft_measured_frf(
        dataset, full_fn, reduced_fn, max_reduction_relative_error=0.05
    )
    assert agreed is True

    # Bad reduced model exceeding 5%
    def bad_reduced_fn(omega: float) -> complex:
        return 1.0 / (20000.0 * 1.20 - omega**2 * 2.0 + 1j * omega * 40.0)

    agreed_bad = assess_coupled_shaft_measured_frf(
        dataset, full_fn, bad_reduced_fn, max_reduction_relative_error=0.05
    )
    assert agreed_bad is False


def test_antiresonance_floor_handling(dataset_payload: dict[str, Any]) -> None:
    # Inject an antiresonance (near-zero impedance) at one bin
    payload = copy.deepcopy(dataset_payload)
    payload["samples"][3]["impedance_real"] = 1e-8
    payload["samples"][3]["impedance_imag"] = 1e-8
    payload["samples"][3]["magnitude_std"] = 1e-9
    dataset = measured_grip_from_json(json.dumps(payload))

    def model_fn(omega: float) -> complex:
        return 1e-8 + 1e-8j

    # Should not crash with ZeroDivisionError
    summary = assess_measured_frf_agreement(
        dataset, model_fn, max_relative_magnitude_error=1.0, max_phase_error_rad=1.0
    )
    assert summary.max_relative_magnitude_error >= 0.0
