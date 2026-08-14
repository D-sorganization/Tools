"""Registry contract pins + NoiseSpec/VariationPlan validation (#4120 V3)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from shared.python.contracts import ContractViolationError
from shared.python.swing_sim.flight.registry import FlightModelType
from shared.python.swing_sim.variation import (
    CATEGORY_BALL_SETUP,
    CATEGORY_CLUB,
    CATEGORY_DELIVERY,
    CATEGORY_LAUNCH,
    CATEGORY_SWING,
    SWING_DERIVED_KEYS,
    NoiseSpec,
    VariableDef,
    VariationPlan,
    keys_for_mode,
    register_variable,
    variable_registry,
    variables_in_category,
)
from shared.python.swing_sim.variation.spec import PerturbationGroup

pytestmark = pytest.mark.unit

_TEE_HEIGHT = f"{CATEGORY_BALL_SETUP}.tee_height_m"


class TestRegistryContract:
    """Pins the shared vocabulary other packages rely on."""

    def test_delivery_category_pins_the_solver_variable_set(self) -> None:
        names = tuple(d.name for d in variables_in_category(CATEGORY_DELIVERY))
        assert names == (
            "clubhead_speed_mps",
            "club_path_deg",
            "face_angle_deg",
            "attack_angle_deg",
            "dynamic_loft_deg",
            "lie_deg",
            "impact_offset_toe_mm",
            "impact_offset_high_mm",
        )

    def test_swing_category_pins_the_pendulum_variable_set(self) -> None:
        names = tuple(d.name for d in variables_in_category(CATEGORY_SWING))
        assert names == (
            "yaw_deg",
            "side_tilt_deg",
            "forward_tilt_deg",
            "impact_time_offset_s",
            "damping_shoulder",
            "damping_wrist",
            "shoulder_commanded_torque_offset_nm",
            "wrist_commanded_torque_offset_nm",
        )

    def test_club_and_launch_categories_pin(self) -> None:
        club = tuple(d.name for d in variables_in_category(CATEGORY_CLUB))
        launch = tuple(d.name for d in variables_in_category(CATEGORY_LAUNCH))
        assert club == ("head_mass_kg", "head_moi_kg_m2", "cor")
        assert launch == (
            "ball_speed_mph",
            "launch_angle_deg",
            "launch_azimuth_deg",
            "spin_rpm",
            "spin_axis_deg",
        )

    def test_tee_height_is_registered_with_tee_only_applicability(self) -> None:
        definition = variable_registry()[_TEE_HEIGHT]

        assert definition.label == "Tee Height"
        assert definition.unit == "m"
        assert definition.applicability == "tee_only"
        assert _TEE_HEIGHT in keys_for_mode("delivery")
        assert _TEE_HEIGHT in keys_for_mode("swing")
        assert _TEE_HEIGHT not in keys_for_mode("launch")

    def test_every_entry_has_label_unit_guidance_and_scale(self) -> None:
        for definition in variable_registry().values():
            assert definition.label
            assert definition.guidance
            assert definition.typical_scale > 0.0

    def test_swing_mode_excludes_swing_derived_delivery_keys(self) -> None:
        keys = keys_for_mode("swing")
        assert not set(SWING_DERIVED_KEYS) & set(keys)
        assert f"{CATEGORY_DELIVERY}.face_angle_deg" in keys
        assert f"{CATEGORY_SWING}.yaw_deg" in keys

    def test_launch_mode_is_launch_only(self) -> None:
        keys = keys_for_mode("launch")
        assert all(key.startswith(CATEGORY_LAUNCH) for key in keys)

    def test_registry_is_extensible_but_rejects_duplicates(self) -> None:
        definition = VariableDef(
            key="test_pkg.demo.gain",
            label="Demo Gain",
            unit="",
            default=1.0,
            typical_scale=0.1,
            guidance="Test-only entry.",
        )
        register_variable(definition)
        assert variable_registry()["test_pkg.demo.gain"] is definition
        with pytest.raises(ContractViolationError):
            register_variable(definition)


class TestNoiseSpec:
    def test_rejects_unknown_variable_and_bad_scale(self) -> None:
        with pytest.raises(ContractViolationError):
            NoiseSpec(variable_key="nope.nothing")
        with pytest.raises(ContractViolationError):
            NoiseSpec(variable_key=f"{CATEGORY_DELIVERY}.face_angle_deg", scale=0.0)
        with pytest.raises(ContractViolationError):
            NoiseSpec(
                variable_key=f"{CATEGORY_DELIVERY}.face_angle_deg",
                distribution="cauchy",
            )

    def test_rejects_inverted_truncation(self) -> None:
        with pytest.raises(ContractViolationError):
            NoiseSpec(
                variable_key=f"{CATEGORY_DELIVERY}.face_angle_deg",
                lower=2.0,
                upper=-2.0,
            )

    @pytest.mark.parametrize("value", [True, np.bool_(False), "1.0", float("inf")])
    @pytest.mark.parametrize("field", ["scale", "lower", "upper"])
    def test_rejects_coercive_or_nonfinite_numeric_fields(
        self, field: str, value: object
    ) -> None:
        kwargs = {field: value}
        with pytest.raises(ContractViolationError, match=field):
            NoiseSpec(
                f"{CATEGORY_DELIVERY}.face_angle_deg",
                **kwargs,
            )

    @pytest.mark.parametrize(
        "window",
        [
            (False, True),
            (np.bool_(False), 0.1),
            ("0.0", "0.1"),
            (0.0, float("inf")),
        ],
    )
    def test_rejects_coercive_or_nonfinite_time_locus(
        self, window: tuple[object, object]
    ) -> None:
        with pytest.raises(ContractViolationError, match="time_window_s"):
            NoiseSpec(
                f"{CATEGORY_SWING}.yaw_deg",
                time_window_s=window,
            )

    def test_json_round_trip(self) -> None:
        spec = NoiseSpec(
            variable_key=f"{CATEGORY_SWING}.yaw_deg",
            distribution="triangular",
            scale=1.5,
            lower=-4.0,
            upper=4.0,
            spec_id="address-yaw",
            time_window_s=(0.1, 0.3),
            point_ids=("swing.wrist",),
        )
        assert NoiseSpec.from_json_dict(spec.to_json_dict()) == spec

    def test_default_spec_id_preserves_the_stable_v1_stream_key(self) -> None:
        variable_key = f"{CATEGORY_SWING}.yaw_deg"
        assert NoiseSpec(variable_key).spec_id == variable_key

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"spec_id": "  "}, "spec_id"),
            ({"time_window_s": (0.3, 0.1)}, "time_window_s"),
            ({"point_ids": ("swing.wrist", "swing.wrist")}, "point_ids"),
        ],
    )
    def test_rejects_invalid_stable_id_or_locus_metadata(
        self, kwargs: dict[str, object], message: str
    ) -> None:
        with pytest.raises(ContractViolationError, match=message):
            NoiseSpec(f"{CATEGORY_SWING}.yaw_deg", **kwargs)


class TestPerturbationGroup:
    _IDS = ("face", "speed")

    def test_correlation_and_covariance_round_trip(self) -> None:
        for matrix_kind, matrix in (
            ("correlation", ((1.0, 0.6), (0.6, 1.0))),
            ("covariance", ((4.0, 0.3), (0.3, 1.0))),
        ):
            group = PerturbationGroup(
                group_id=f"delivery-{matrix_kind}",
                spec_ids=self._IDS,
                matrix=matrix,
                matrix_kind=matrix_kind,
            )
            assert PerturbationGroup.from_json_dict(group.to_json_dict()) == group

    @pytest.mark.parametrize(
        ("matrix", "kind", "message"),
        [
            (((1.0, 0.2), (0.3, 1.0)), "correlation", "symmetric"),
            (((2.0, 0.2), (0.2, 1.0)), "correlation", "unit diagonal"),
            (((1.0, 2.0), (2.0, 1.0)), "correlation", "positive semidefinite"),
            (((1.0, 0.0), (0.0, -1.0)), "covariance", "positive semidefinite"),
        ],
    )
    def test_rejects_invalid_matrix_semantics(
        self,
        matrix: tuple[tuple[float, ...], ...],
        kind: str,
        message: str,
    ) -> None:
        with pytest.raises(ContractViolationError, match=message):
            PerturbationGroup(
                group_id="delivery-group",
                spec_ids=self._IDS,
                matrix=matrix,
                matrix_kind=kind,
            )

    def test_rejects_ragged_matrix_with_a_contract_error(self) -> None:
        with pytest.raises(ContractViolationError, match="numeric square matrix"):
            PerturbationGroup(
                group_id="delivery-group",
                spec_ids=self._IDS,
                matrix=((1.0, 0.2), (0.2,)),
            )


class TestVariationPlan:
    def _plan(self) -> VariationPlan:
        return VariationPlan(
            mode="delivery",
            base_variables={f"{CATEGORY_DELIVERY}.clubhead_speed_mps": 48.0},
            noise=(
                NoiseSpec(f"{CATEGORY_DELIVERY}.face_angle_deg", scale=1.0),
                NoiseSpec(
                    f"{CATEGORY_CLUB}.cor",
                    distribution="uniform",
                    scale=0.005,
                    lower=0.0,
                    upper=1.0,
                ),
            ),
            n_runs=64,
            seed=7,
        )

    def test_json_round_trip_is_lossless(self) -> None:
        plan = self._plan()
        assert VariationPlan.loads(plan.dumps()) == plan

    @pytest.mark.parametrize("flight_model", [model.value for model in FlightModelType])
    def test_accepts_only_registered_flight_model_identities(
        self, flight_model: str
    ) -> None:
        plan = VariationPlan(
            mode="delivery",
            noise=(NoiseSpec(f"{CATEGORY_DELIVERY}.face_angle_deg"),),
            flight_model=flight_model,
        )

        assert plan.flight_model == flight_model

    @pytest.mark.parametrize(
        "flight_model", ["custom-flight-model", True, None, ["waterloo_penner"]]
    )
    def test_rejects_unregistered_flight_model_identity(
        self, flight_model: object
    ) -> None:
        with pytest.raises(ContractViolationError, match="flight_model.*registered"):
            VariationPlan(
                mode="delivery",
                noise=(NoiseSpec(f"{CATEGORY_DELIVERY}.face_angle_deg"),),
                flight_model=flight_model,  # type: ignore[arg-type]
            )

    def test_v2_grouped_plan_json_round_trip_is_lossless(self) -> None:
        specs = (
            NoiseSpec(
                f"{CATEGORY_DELIVERY}.face_angle_deg",
                scale=2.0,
                spec_id="face",
            ),
            NoiseSpec(
                f"{CATEGORY_DELIVERY}.clubhead_speed_mps",
                scale=1.0,
                spec_id="speed",
            ),
        )
        plan = VariationPlan(
            mode="delivery",
            noise=specs,
            groups=(
                PerturbationGroup(
                    group_id="delivery-correlation",
                    spec_ids=("face", "speed"),
                    matrix=((1.0, -0.4), (-0.4, 1.0)),
                ),
            ),
            n_runs=64,
            seed=7,
        )

        encoded = plan.to_json_dict()

        assert encoded["schema_version"] == 2
        assert VariationPlan.loads(json.dumps(encoded)) == plan

    def test_v1_plan_migrates_with_stable_ids_and_no_groups(self) -> None:
        old = self._plan().to_json_dict()
        old["schema_version"] = 1
        old.pop("groups", None)
        for spec in old["noise"]:
            spec.pop("spec_id", None)
            spec.pop("time_window_s", None)
            spec.pop("point_ids", None)

        migrated = VariationPlan.from_json_dict(old)

        assert migrated.groups == ()
        assert tuple(spec.spec_id for spec in migrated.noise) == tuple(
            spec.variable_key for spec in migrated.noise
        )
        assert migrated.to_json_dict()["schema_version"] == 2

    def test_rejects_variables_illegal_for_the_mode(self) -> None:
        with pytest.raises(ContractViolationError):
            VariationPlan(
                mode="launch",
                noise=(NoiseSpec(f"{CATEGORY_DELIVERY}.face_angle_deg"),),
            )
        with pytest.raises(ContractViolationError):
            VariationPlan(
                mode="swing",
                noise=(NoiseSpec(f"{CATEGORY_DELIVERY}.clubhead_speed_mps"),),
            )

    def test_rejects_duplicate_noise_and_empty_noise(self) -> None:
        spec = NoiseSpec(f"{CATEGORY_DELIVERY}.face_angle_deg")
        with pytest.raises(ContractViolationError):
            VariationPlan(mode="delivery", noise=(spec, spec))
        with pytest.raises(ContractViolationError):
            VariationPlan(mode="delivery", noise=())

    @pytest.mark.parametrize("value", [True, np.bool_(False), "48.0", float("inf")])
    def test_rejects_coercive_or_nonfinite_base_values(self, value: object) -> None:
        with pytest.raises(ContractViolationError, match="base value"):
            VariationPlan(
                mode="delivery",
                base_variables={f"{CATEGORY_DELIVERY}.clubhead_speed_mps": value},
                noise=(NoiseSpec(f"{CATEGORY_DELIVERY}.face_angle_deg"),),
            )

    @pytest.mark.parametrize("field", ["n_runs", "seed"])
    @pytest.mark.parametrize("value", [True, np.bool_(False), "2", 2.0])
    def test_rejects_non_integer_run_and_seed_domains(
        self, field: str, value: object
    ) -> None:
        kwargs = {field: value}
        with pytest.raises(ContractViolationError, match=field):
            VariationPlan(
                mode="delivery",
                noise=(NoiseSpec(f"{CATEGORY_DELIVERY}.face_angle_deg"),),
                **kwargs,
            )

    def test_json_integer_and_float_domains_remain_supported(self) -> None:
        raw = self._plan().to_json_dict()
        raw["base_variables"] = {f"{CATEGORY_DELIVERY}.clubhead_speed_mps": 48}
        raw["noise"][0]["scale"] = 2

        restored = VariationPlan.from_json_dict(raw)

        assert restored.n_runs == 64
        assert restored.seed == 7
        assert (
            restored.base_variables[f"{CATEGORY_DELIVERY}.clubhead_speed_mps"] == 48.0
        )
        assert restored.noise[0].scale == 2.0

    @pytest.mark.parametrize("schema_version", [True, 2.5, "2"])
    def test_json_rejects_coercive_schema_version(self, schema_version: object) -> None:
        raw = self._plan().to_json_dict()
        raw["schema_version"] = schema_version

        with pytest.raises(ContractViolationError, match="schema_version"):
            VariationPlan.from_json_dict(raw)

    def test_rejects_duplicate_spec_ids_and_duplicate_variable_keys(self) -> None:
        face = f"{CATEGORY_DELIVERY}.face_angle_deg"
        speed = f"{CATEGORY_DELIVERY}.clubhead_speed_mps"
        with pytest.raises(ContractViolationError, match="duplicate spec_id"):
            VariationPlan(
                mode="delivery",
                noise=(
                    NoiseSpec(face, spec_id="same"),
                    NoiseSpec(speed, spec_id="same"),
                ),
            )
        with pytest.raises(ContractViolationError, match="duplicate variable_key"):
            VariationPlan(
                mode="delivery",
                noise=(
                    NoiseSpec(face, spec_id="first"),
                    NoiseSpec(face, spec_id="second"),
                ),
            )

    def test_group_references_and_covariance_scales_are_validated(self) -> None:
        specs = (
            NoiseSpec(f"{CATEGORY_DELIVERY}.face_angle_deg", scale=2.0, spec_id="face"),
            NoiseSpec(
                f"{CATEGORY_DELIVERY}.clubhead_speed_mps", scale=1.0, spec_id="speed"
            ),
        )
        with pytest.raises(ContractViolationError, match="unknown spec_id"):
            VariationPlan(
                mode="delivery",
                noise=specs,
                groups=(
                    PerturbationGroup(
                        group_id="bad-ref",
                        spec_ids=("face", "missing"),
                        matrix=((1.0, 0.0), (0.0, 1.0)),
                    ),
                ),
            )
        with pytest.raises(ContractViolationError, match="diagonal.*scale"):
            VariationPlan(
                mode="delivery",
                noise=specs,
                groups=(
                    PerturbationGroup(
                        group_id="bad-covariance",
                        spec_ids=("face", "speed"),
                        matrix=((1.0, 0.0), (0.0, 1.0)),
                        matrix_kind="covariance",
                    ),
                ),
            )

    def test_grouped_specs_must_use_normal_distributions(self) -> None:
        face = NoiseSpec(
            f"{CATEGORY_DELIVERY}.face_angle_deg",
            distribution="uniform",
            spec_id="face",
        )
        speed = NoiseSpec(f"{CATEGORY_DELIVERY}.clubhead_speed_mps", spec_id="speed")
        group = PerturbationGroup(
            group_id="delivery",
            spec_ids=("face", "speed"),
            matrix=((1.0, 0.2), (0.2, 1.0)),
        )
        with pytest.raises(ContractViolationError, match="normal distributions"):
            VariationPlan(mode="delivery", noise=(face, speed), groups=(group,))

    def test_a_spec_cannot_belong_to_overlapping_groups(self) -> None:
        specs = (
            NoiseSpec(f"{CATEGORY_DELIVERY}.face_angle_deg", spec_id="face"),
            NoiseSpec(f"{CATEGORY_DELIVERY}.clubhead_speed_mps", spec_id="speed"),
            NoiseSpec(f"{CATEGORY_DELIVERY}.club_path_deg", spec_id="path"),
        )
        matrix = ((1.0, 0.2), (0.2, 1.0))
        with pytest.raises(ContractViolationError, match="only one group"):
            VariationPlan(
                mode="delivery",
                noise=specs,
                groups=(
                    PerturbationGroup("face-speed", ("face", "speed"), matrix),
                    PerturbationGroup("speed-path", ("speed", "path"), matrix),
                ),
            )

    def test_resolved_base_overlays_defaults(self) -> None:
        base = self._plan().resolved_base()
        assert base[f"{CATEGORY_DELIVERY}.clubhead_speed_mps"] == 48.0
        assert base[f"{CATEGORY_DELIVERY}.dynamic_loft_deg"] == 10.5
        assert base[f"{CATEGORY_CLUB}.cor"] == 0.83
