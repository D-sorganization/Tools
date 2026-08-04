"""Registry contract pins + NoiseSpec/VariationPlan validation (#4120 V3)."""

from __future__ import annotations

import pytest

from shared.python.contracts import ContractViolationError
from shared.python.swing_sim.variation import (
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

pytestmark = pytest.mark.unit


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

    def test_json_round_trip(self) -> None:
        spec = NoiseSpec(
            variable_key=f"{CATEGORY_SWING}.yaw_deg",
            distribution="triangular",
            scale=1.5,
            lower=-4.0,
            upper=4.0,
        )
        assert NoiseSpec.from_json_dict(spec.to_json_dict()) == spec


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

    def test_resolved_base_overlays_defaults(self) -> None:
        base = self._plan().resolved_base()
        assert base[f"{CATEGORY_DELIVERY}.clubhead_speed_mps"] == 48.0
        assert base[f"{CATEGORY_DELIVERY}.dynamic_loft_deg"] == 10.5
        assert base[f"{CATEGORY_CLUB}.cor"] == 0.83
