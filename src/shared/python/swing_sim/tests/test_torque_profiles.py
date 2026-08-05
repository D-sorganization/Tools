"""Contracts for UI-neutral prescribed joint-torque profiles."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

import shared.python.swing_sim.torque_profiles as torque_profiles_module
from shared.python.contracts import ContractViolationError
from shared.python.swing_sim.torque_fitting import fit_torque_polynomial
from shared.python.swing_sim.torque_profiles import (
    COEFFICIENT_ORDER,
    TORQUE_PROFILE_SCHEMA_VERSION,
    TORQUE_UNIT,
    FitMetadata,
    JointTorqueAssignment,
    PrescribedTorqueProfile,
    TorquePolynomial,
    TorqueProfileSource,
    evaluate_ascending_polynomial,
)

pytestmark = pytest.mark.unit


def _profile() -> PrescribedTorqueProfile:
    return PrescribedTorqueProfile(
        profile_id="profile.driver_release.v1",
        model_id="model.double_pendulum.v1",
        name="Driver Release",
        description="Prescribed shoulder and wrist torques for a driver swing.",
        source=TorqueProfileSource.DIRECT,
        source_metadata={"author": "test-suite", "campaign_id": "driver-2026"},
        created_at_utc="2026-08-05T12:00:00Z",
        modified_at_utc="2026-08-05T12:30:00Z",
        time_domain_s=(0.0, 1.25),
        assignments=(
            JointTorqueAssignment(
                joint_id="joint.shoulder",
                polynomial=TorquePolynomial((10.0, -2.0)),
            ),
            JointTorqueAssignment(
                joint_id="joint.wrist",
                polynomial=TorquePolynomial((0.0, 3.0, -0.5)),
            ),
        ),
    )


class TestTorquePolynomial:
    def test_evaluates_explicit_ascending_coefficients(self) -> None:
        polynomial = TorquePolynomial((2.0, 3.0, 4.0))
        assert polynomial.evaluate(2.0) == pytest.approx(24.0)
        assert evaluate_ascending_polynomial((2.0, 3.0, 4.0), 2.0) == pytest.approx(
            24.0
        )

    @pytest.mark.parametrize("coefficients", [(), (1.0, math.nan), (math.inf,)])
    def test_rejects_empty_or_nonfinite_coefficients(
        self, coefficients: tuple[float, ...]
    ) -> None:
        with pytest.raises(ContractViolationError):
            TorquePolynomial(coefficients)

    @pytest.mark.parametrize("time_s", [math.nan, math.inf, -math.inf])
    def test_rejects_nonfinite_evaluation_time(self, time_s: float) -> None:
        with pytest.raises(ContractViolationError):
            evaluate_ascending_polynomial((1.0, 2.0), time_s)

    def test_rejects_fit_degree_inconsistent_with_coefficients(self) -> None:
        metadata = FitMetadata(
            degree=2,
            rmse_nm=0.0,
            max_abs_error_nm=0.0,
            r_squared=1.0,
            condition_number=1.0,
        )
        with pytest.raises(ContractViolationError):
            TorquePolynomial((1.0, 2.0), metadata)

    def test_immutable_polynomial_does_not_revalidate_coefficients_per_sample(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        polynomial = TorquePolynomial((2.0, 3.0))

        def fail_revalidation(*_args: object) -> tuple[float, ...]:
            raise AssertionError("coefficients were revalidated")

        monkeypatch.setattr(torque_profiles_module, "_finite_tuple", fail_revalidation)
        assert polynomial.evaluate(2.0) == pytest.approx(8.0)


class TestPrescribedTorqueProfile:
    def test_pins_schema_units_order_and_source_vocabulary(self) -> None:
        assert TORQUE_PROFILE_SCHEMA_VERSION == 1
        assert TORQUE_UNIT == "N*m"
        assert COEFFICIENT_ORDER == "ascending_c0_first"
        assert {source.value for source in TorqueProfileSource} == {
            "direct",
            "drawn",
            "imported",
            "optimized",
            "fitted_run",
        }

    def test_evaluates_every_joint_by_stable_id(self) -> None:
        values = _profile().evaluate(0.5)
        assert values == pytest.approx({"joint.shoulder": 9.0, "joint.wrist": 1.375})

    def test_rejects_evaluation_outside_time_domain(self) -> None:
        profile = _profile()
        with pytest.raises(ContractViolationError):
            profile.evaluate(-0.001)
        with pytest.raises(ContractViolationError):
            profile.evaluate(1.251)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("profile_id", "contains spaces"),
            ("model_id", ""),
            ("name", "  "),
            ("description", ""),
            ("source_metadata", {"bad key": "value"}),
            ("created_at_utc", "2026-08-05 12:00:00"),
            ("modified_at_utc", "2026-08-05T11:59:59Z"),
            ("time_domain_s", (1.0, 1.0)),
            ("time_domain_s", (0.0, math.inf)),
            ("assignments", ()),
        ],
    )
    def test_rejects_invalid_profile_fields(self, field: str, value: object) -> None:
        values = {
            "profile_id": "profile.valid",
            "model_id": "model.valid",
            "name": "Valid Profile",
            "description": "A valid description.",
            "source": TorqueProfileSource.DIRECT,
            "source_metadata": {"author": "test-suite"},
            "created_at_utc": "2026-08-05T12:00:00Z",
            "modified_at_utc": "2026-08-05T12:30:00Z",
            "time_domain_s": (0.0, 1.0),
            "assignments": (
                JointTorqueAssignment("joint.one", TorquePolynomial((1.0,))),
            ),
        }
        values[field] = value
        with pytest.raises(ContractViolationError):
            PrescribedTorqueProfile(**values)  # type: ignore[arg-type]

    def test_rejects_duplicate_joint_assignment(self) -> None:
        assignment = JointTorqueAssignment("joint.one", TorquePolynomial((1.0,)))
        with pytest.raises(ContractViolationError):
            PrescribedTorqueProfile(
                profile_id="profile.duplicate",
                model_id="model.double",
                name="Duplicate",
                description="Contains duplicate joint identifiers.",
                source=TorqueProfileSource.DIRECT,
                source_metadata={"author": "test-suite"},
                created_at_utc="2026-08-05T12:00:00Z",
                modified_at_utc="2026-08-05T12:30:00Z",
                time_domain_s=(0.0, 1.0),
                assignments=(assignment, assignment),
            )

    def test_json_round_trip_is_lossless(self) -> None:
        profile = _profile()
        assert PrescribedTorqueProfile.loads(profile.dumps()) == profile
        assert profile.to_json_dict()["torque_unit"] == TORQUE_UNIT
        assert profile.to_json_dict()["coefficient_order"] == COEFFICIENT_ORDER
        assert profile.to_json_dict()["source_metadata"]["author"] == "test-suite"

    def test_source_metadata_is_immutable(self) -> None:
        metadata = _profile().source_metadata
        with pytest.raises(TypeError):
            metadata["author"] = "changed"  # type: ignore[index]

    @pytest.mark.parametrize(
        "mutation",
        [
            lambda data: data.update(schema_version=2),
            lambda data: data.update(schema_version=True),
            lambda data: data.update(torque_unit="lbf*ft"),
            lambda data: data.update(coefficient_order="descending"),
            lambda data: data.update(unexpected=True),
            lambda data: data.pop("model_id"),
            lambda data: data["assignments"][0].update(unexpected=True),
        ],
    )
    def test_json_parser_rejects_unknown_missing_or_incompatible_fields(
        self, mutation: object
    ) -> None:
        data = _profile().to_json_dict()
        mutation(data)  # type: ignore[operator]
        with pytest.raises(ContractViolationError):
            PrescribedTorqueProfile.from_json_dict(data)

    def test_loads_rejects_nonobject_json(self) -> None:
        with pytest.raises(ContractViolationError):
            PrescribedTorqueProfile.loads(json.dumps(["not", "an", "object"]))

    def test_loads_rejects_duplicate_json_fields(self) -> None:
        serialized = _profile().dumps()
        duplicated = serialized.replace(
            '"profile_id": "profile.driver_release.v1",',
            '"profile_id": "profile.driver_release.v1",\n'
            '  "profile_id": "profile.duplicate",',
        )
        with pytest.raises(ContractViolationError):
            PrescribedTorqueProfile.loads(duplicated)


class TestPolynomialFit:
    def test_fits_physical_time_ascending_coefficients(self) -> None:
        times_s = np.linspace(2.0, 4.0, 21)
        torque_nm = 5.0 - 3.0 * times_s + 2.0 * times_s**2
        polynomial = fit_torque_polynomial(times_s, torque_nm, degree=2)

        assert polynomial.coefficients == pytest.approx((5.0, -3.0, 2.0))
        assert polynomial.evaluate(3.25) == pytest.approx(
            5.0 - 3.0 * 3.25 + 2.0 * 3.25**2
        )
        assert polynomial.fit_metadata is not None
        assert polynomial.fit_metadata.degree == 2
        assert polynomial.fit_metadata.rmse_nm < 1e-10
        assert polynomial.fit_metadata.max_abs_error_nm < 1e-10
        assert polynomial.fit_metadata.r_squared == pytest.approx(1.0)
        assert polynomial.fit_metadata.condition_number >= 1.0
        assert polynomial.fit_metadata.original_sample_sha256 is not None
        assert len(polynomial.fit_metadata.original_sample_sha256) == 64

    def test_fit_metadata_round_trip(self) -> None:
        metadata = FitMetadata(
            degree=1,
            rmse_nm=0.1,
            max_abs_error_nm=0.2,
            r_squared=0.95,
            condition_number=2.5,
            original_sample_sha256="a" * 64,
        )
        assert FitMetadata.from_json_dict(metadata.to_json_dict()) == metadata

    def test_fit_metadata_rejects_r_squared_above_one(self) -> None:
        with pytest.raises(ContractViolationError):
            FitMetadata(
                degree=1,
                rmse_nm=0.0,
                max_abs_error_nm=0.0,
                r_squared=1.01,
                condition_number=1.0,
            )

    @pytest.mark.parametrize(
        ("times_s", "torque_nm", "degree"),
        [
            ([0.0, 0.5, 0.25], [1.0, 2.0, 3.0], 1),
            ([0.0, 0.5, 1.0], [1.0, math.nan, 3.0], 1),
            ([0.0, 1.0], [1.0], 1),
            ([0.0, 1.0], [1.0, 2.0], 2),
            ([0.0, 1.0], [1.0, 2.0], -1),
        ],
    )
    def test_rejects_invalid_fit_samples(
        self,
        times_s: list[float],
        torque_nm: list[float],
        degree: int,
    ) -> None:
        with pytest.raises(ContractViolationError):
            fit_torque_polynomial(times_s, torque_nm, degree)
