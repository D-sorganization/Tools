"""Wire and exporter gates for ball_flight_trajectory/1 (ADR-0047 H1).

The refusal gates are the point of a fail-closed wire, so each one
names a specific way a bad record could otherwise reach a viewer: an
unattributable trajectory, an undeclared frame, a time series that
runs backwards, or a channel that exists on some samples and not
others.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import pytest

from shared.python.contracts import PreconditionError
from shared.python.swing_sim.flight import (
    FlightModelRegistry,
    FlightModelType,
    LaunchConditions,
)
from shared.python.swing_sim.flight_interchange import (
    APP_FRAME_ID,
    BALL_FLIGHT_TRAJECTORY_FORMAT,
    FLIGHT_FRAME_ID,
    TOOLS_FLIGHT_FAMILY,
    BallFlightSample,
    BallFlightTrajectory,
    TrajectoryProvenance,
    ball_flight_trajectory_from_json,
    ball_flight_trajectory_to_json,
    flight_model_parameters,
    from_samples,
    parameter_digest,
    trajectory_from_flight_result,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]

_DIGEST = parameter_digest({"cd": 0.22, "cl": 0.24})


def _provenance(family: str = TOOLS_FLIGHT_FAMILY) -> TrajectoryProvenance:
    return TrajectoryProvenance(
        model_family=family,
        model_name="Nathan",
        parameter_digest=_DIGEST,
    )


def _parabola(
    sample_count: int = 6, *, with_velocity: bool = True
) -> BallFlightTrajectory:
    """A closed-form ballistic arc in the flight frame (x fwd, y left, z up)."""
    times = [0.05 * index for index in range(sample_count)]
    positions = [(60.0 * t, 0.0, 30.0 * t - 4.903325 * t * t) for t in times]
    velocities = [(60.0, 0.0, 30.0 - 9.80665 * t) for t in times]
    return from_samples(
        source_id="test:parabola",
        frame_id=FLIGHT_FRAME_ID,
        provenance=_provenance(),
        times_s=times,
        positions_m=positions,
        velocities_mps=velocities if with_velocity else None,
    )


def _payload(trajectory: BallFlightTrajectory) -> dict[str, Any]:
    parsed: dict[str, Any] = json.loads(ball_flight_trajectory_to_json(trajectory))
    return parsed


class TestParameterDigest:
    def test_matches_the_documented_algorithm(self) -> None:
        """The digest recipe is wire contract: other repos reproduce it."""
        parameters = {"cl": 0.24, "cd": 0.22, "model": "nathan"}
        expected = hashlib.sha256(
            json.dumps(
                parameters,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        assert parameter_digest(parameters) == expected

    def test_is_key_order_independent(self) -> None:
        assert parameter_digest({"a": 1.0, "b": 2.0}) == parameter_digest(
            {"b": 2.0, "a": 1.0}
        )

    def test_separates_different_parameter_values(self) -> None:
        assert parameter_digest({"cd": 0.22}) != parameter_digest({"cd": 0.23})

    def test_refuses_empty_and_non_finite_parameters(self) -> None:
        with pytest.raises(PreconditionError, match="nonempty"):
            parameter_digest({})
        with pytest.raises(PreconditionError, match="finite"):
            parameter_digest({"cd": math.inf})
        with pytest.raises(PreconditionError, match="finite number or a string"):
            parameter_digest({"cd": [0.22]})  # type: ignore[dict-item]


class TestRoundTrip:
    def test_round_trip_is_byte_identical(self) -> None:
        trajectory = _parabola()
        text = ball_flight_trajectory_to_json(trajectory)
        reparsed = ball_flight_trajectory_from_json(text)
        assert ball_flight_trajectory_to_json(reparsed) == text
        assert reparsed == trajectory

    def test_serialization_is_deterministic_across_equal_records(self) -> None:
        assert ball_flight_trajectory_to_json(
            _parabola()
        ) == ball_flight_trajectory_to_json(_parabola())

    def test_position_only_records_round_trip(self) -> None:
        trajectory = _parabola(with_velocity=False)
        assert trajectory.channels == ()
        text = ball_flight_trajectory_to_json(trajectory)
        assert '"channels":[]' in text
        assert (
            ball_flight_trajectory_to_json(ball_flight_trajectory_from_json(text))
            == text
        )

    def test_declared_fields_are_exactly_the_wire(self) -> None:
        payload = _payload(_parabola())
        assert set(payload) == {
            "channels",
            "format",
            "frame_id",
            "provenance",
            "samples",
            "source_id",
        }
        assert payload["format"] == BALL_FLIGHT_TRAJECTORY_FORMAT
        assert set(payload["provenance"]) == {
            "model_family",
            "model_name",
            "parameter_digest",
        }
        assert set(payload["samples"][0]) == {"position_m", "time_s", "velocity_mps"}


class TestRefusalGates:
    def test_unknown_top_level_field_is_refused(self) -> None:
        payload = _payload(_parabola())
        payload["notes"] = "extra"
        with pytest.raises(PreconditionError, match="unknown trajectory fields"):
            ball_flight_trajectory_from_json(json.dumps(payload))

    def test_missing_top_level_field_is_refused(self) -> None:
        payload = _payload(_parabola())
        del payload["frame_id"]
        with pytest.raises(PreconditionError, match="missing trajectory fields"):
            ball_flight_trajectory_from_json(json.dumps(payload))

    def test_unknown_sample_field_is_refused(self) -> None:
        payload = _payload(_parabola())
        payload["samples"][0]["carry_m"] = 1.0
        with pytest.raises(PreconditionError, match="sample fields must be exactly"):
            ball_flight_trajectory_from_json(json.dumps(payload))

    def test_undeclared_channel_on_a_sample_is_refused(self) -> None:
        """A ragged channel is the failure the `channels` declaration exists for."""
        payload = _payload(_parabola())
        payload["samples"][2]["spin_rad_s"] = [0.0, -300.0, 0.0]
        with pytest.raises(PreconditionError, match="sample fields must be exactly"):
            ball_flight_trajectory_from_json(json.dumps(payload))

    def test_declared_channel_missing_from_a_sample_is_refused(self) -> None:
        payload = _payload(_parabola())
        del payload["samples"][3]["velocity_mps"]
        with pytest.raises(PreconditionError, match="sample fields must be exactly"):
            ball_flight_trajectory_from_json(json.dumps(payload))

    def test_unknown_channel_name_is_refused(self) -> None:
        payload = _payload(_parabola())
        payload["channels"] = ["acceleration_mps2"]
        with pytest.raises(PreconditionError, match="unknown channel"):
            ball_flight_trajectory_from_json(json.dumps(payload))

    def test_unsorted_channels_are_refused(self) -> None:
        trajectory = from_samples(
            source_id="test:both",
            frame_id=FLIGHT_FRAME_ID,
            provenance=_provenance(),
            times_s=[0.0, 0.1],
            positions_m=[(0.0, 0.0, 0.0), (6.0, 0.0, 2.9)],
            velocities_mps=[(60.0, 0.0, 30.0), (60.0, 0.0, 29.0)],
            spins_rad_s=[(0.0, -300.0, 0.0), (0.0, -299.0, 0.0)],
        )
        assert trajectory.channels == ("spin_rad_s", "velocity_mps")
        payload = _payload(trajectory)
        payload["channels"] = ["velocity_mps", "spin_rad_s"]
        with pytest.raises(PreconditionError, match="channels must be sorted"):
            ball_flight_trajectory_from_json(json.dumps(payload))

    def test_wrong_format_is_refused(self) -> None:
        payload = _payload(_parabola())
        payload["format"] = "swing_sim.ball_flight_trajectory/2"
        with pytest.raises(PreconditionError, match="format must be"):
            ball_flight_trajectory_from_json(json.dumps(payload))

    def test_missing_provenance_is_refused(self) -> None:
        payload = _payload(_parabola())
        payload["provenance"] = None
        with pytest.raises(PreconditionError, match="provenance must be an object"):
            ball_flight_trajectory_from_json(json.dumps(payload))

    def test_partial_provenance_is_refused(self) -> None:
        payload = _payload(_parabola())
        del payload["provenance"]["parameter_digest"]
        with pytest.raises(PreconditionError, match="provenance fields must be"):
            ball_flight_trajectory_from_json(json.dumps(payload))

    def test_blank_model_family_is_refused(self) -> None:
        payload = _payload(_parabola())
        payload["provenance"]["model_family"] = ""
        with pytest.raises(PreconditionError, match="model_family"):
            ball_flight_trajectory_from_json(json.dumps(payload))

    def test_malformed_digest_is_refused(self) -> None:
        with pytest.raises(PreconditionError, match="parameter_digest"):
            TrajectoryProvenance(
                model_family=TOOLS_FLIGHT_FAMILY,
                model_name="Nathan",
                parameter_digest="not-a-digest",
            )
        with pytest.raises(PreconditionError, match="parameter_digest"):
            TrajectoryProvenance(
                model_family=TOOLS_FLIGHT_FAMILY,
                model_name="Nathan",
                parameter_digest=_DIGEST.upper(),
            )

    def test_undeclared_frame_is_refused(self) -> None:
        payload = _payload(_parabola())
        payload["frame_id"] = "world"
        with pytest.raises(PreconditionError, match="frame_id must be one of"):
            ball_flight_trajectory_from_json(json.dumps(payload))

    def test_non_monotone_time_is_refused(self) -> None:
        payload = _payload(_parabola())
        payload["samples"][2]["time_s"] = payload["samples"][1]["time_s"]
        with pytest.raises(PreconditionError, match="strictly increasing"):
            ball_flight_trajectory_from_json(json.dumps(payload))

    def test_reversed_time_is_refused(self) -> None:
        payload = _payload(_parabola())
        payload["samples"].reverse()
        with pytest.raises(PreconditionError, match="strictly increasing"):
            ball_flight_trajectory_from_json(json.dumps(payload))

    def test_negative_time_is_refused(self) -> None:
        with pytest.raises(PreconditionError, match="non-negative"):
            BallFlightSample(time_s=-0.01, position_m=(0.0, 0.0, 0.0))

    def test_single_sample_is_refused(self) -> None:
        with pytest.raises(PreconditionError, match="at least two"):
            BallFlightTrajectory(
                source_id="test:one",
                frame_id=FLIGHT_FRAME_ID,
                provenance=_provenance(),
                samples=(BallFlightSample(time_s=0.0, position_m=(0.0, 0.0, 0.0)),),
            )

    def test_non_finite_position_is_refused(self) -> None:
        with pytest.raises(PreconditionError, match="position_m must be finite"):
            BallFlightSample(time_s=0.0, position_m=(0.0, math.nan, 0.0))

    def test_short_vector_is_refused(self) -> None:
        with pytest.raises(PreconditionError, match="3-vector"):
            BallFlightSample(time_s=0.0, position_m=(0.0, 0.0))  # type: ignore[arg-type]

    def test_blank_source_id_is_refused(self) -> None:
        with pytest.raises(PreconditionError, match="source_id"):
            BallFlightTrajectory(
                source_id="  ",
                frame_id=FLIGHT_FRAME_ID,
                provenance=_provenance(),
                samples=_parabola().samples,
            )


class TestFromSamplesContract:
    def test_length_mismatch_is_refused(self) -> None:
        with pytest.raises(PreconditionError, match="one entry per time"):
            from_samples(
                source_id="test:short",
                frame_id=FLIGHT_FRAME_ID,
                provenance=_provenance(),
                times_s=[0.0, 0.1, 0.2],
                positions_m=[(0.0, 0.0, 0.0), (6.0, 0.0, 2.9)],
            )

    def test_partial_optional_channel_is_refused(self) -> None:
        with pytest.raises(PreconditionError, match="cover every sample"):
            from_samples(
                source_id="test:partial",
                frame_id=FLIGHT_FRAME_ID,
                provenance=_provenance(),
                times_s=[0.0, 0.1, 0.2],
                positions_m=[
                    (0.0, 0.0, 0.0),
                    (6.0, 0.0, 2.9),
                    (12.0, 0.0, 5.8),
                ],
                velocities_mps=[(60.0, 0.0, 30.0)],
            )

    def test_app_frame_is_accepted_and_declared(self) -> None:
        trajectory = from_samples(
            source_id="test:app",
            frame_id=APP_FRAME_ID,
            provenance=_provenance(),
            times_s=[0.0, 0.1],
            positions_m=[(0.0, 0.0, 0.0), (6.0, 2.9, 0.0)],
        )
        assert trajectory.frame_id == APP_FRAME_ID
        assert trajectory.duration_s == pytest.approx(0.1)


class TestFamilyNeutralImport:
    def test_a_foreign_family_record_parses_unchanged(self) -> None:
        """The wire is family-neutral: this is what UD's adapter emits.

        Hand-authored rather than produced here on purpose — it pins
        the shape the UpstreamDrift half is written against, without
        either repository importing the other's runtime.
        """
        digest = parameter_digest(
            {
                "cd0": 0.21,
                "cd1": 0.05,
                "cd2": 0.02,
                "cl0": 0.0,
                "lift_scale": 0.7,
                "lift_exponent": 0.645,
                "cl_max": 0.155,
            }
        )
        payload = {
            "channels": ["velocity_mps"],
            "format": BALL_FLIGHT_TRAJECTORY_FORMAT,
            "frame_id": FLIGHT_FRAME_ID,
            "provenance": {
                "model_family": "ud.flight_models",
                "model_name": "Waterloo/Penner",
                "parameter_digest": digest,
            },
            "samples": [
                {
                    "position_m": [0.0, 0.0, 0.0],
                    "time_s": 0.0,
                    "velocity_mps": [60.0, 0.0, 30.0],
                },
                {
                    "position_m": [6.0, 0.0, 2.95],
                    "time_s": 0.1,
                    "velocity_mps": [59.0, 0.0, 29.0],
                },
            ],
            "source_id": "ud.flight_models:waterloo_penner",
        }
        text = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        trajectory = ball_flight_trajectory_from_json(text)
        assert trajectory.provenance.model_family == "ud.flight_models"
        assert ball_flight_trajectory_to_json(trajectory) == text


def _driver_launch() -> LaunchConditions:
    return LaunchConditions.from_imperial(
        ball_speed_mph=165.0,
        launch_angle_deg=12.0,
        spin_rate_rpm=2600.0,
    )


class TestFlightResultExport:
    @pytest.mark.parametrize(
        "model_type",
        [FlightModelType.WATERLOO_PENNER, FlightModelType.MACDONALD_HANZELY],
    )
    def test_a_real_flight_exports_and_round_trips(
        self, model_type: FlightModelType
    ) -> None:
        FlightModelRegistry.reset()
        model = FlightModelRegistry.get_model(model_type)
        result = model.simulate(_driver_launch())
        trajectory = trajectory_from_flight_result(result, model)

        assert trajectory.provenance.model_family == TOOLS_FLIGHT_FAMILY
        assert trajectory.provenance.model_name == model.name
        assert trajectory.provenance.parameter_digest == parameter_digest(
            flight_model_parameters(model)
        )
        assert trajectory.frame_id == FLIGHT_FRAME_ID
        assert trajectory.channels == ("velocity_mps",)
        assert len(trajectory.samples) == len(result.trajectory)
        assert trajectory.duration_s == pytest.approx(result.flight_time)

        text = ball_flight_trajectory_to_json(trajectory)
        assert (
            ball_flight_trajectory_to_json(ball_flight_trajectory_from_json(text))
            == text
        )
        FlightModelRegistry.reset()

    def test_exported_samples_are_the_retained_integrator_samples(self) -> None:
        """P8 replays these samples; the wire must not resample them."""
        FlightModelRegistry.reset()
        model = FlightModelRegistry.get_model(FlightModelType.WATERLOO_PENNER)
        result = model.simulate(_driver_launch())
        trajectory = trajectory_from_flight_result(result, model)
        for point, sample in zip(result.trajectory, trajectory.samples, strict=True):
            assert sample.time_s == pytest.approx(float(point.time), abs=0.0)
            assert sample.position_m == pytest.approx(tuple(point.position), abs=0.0)
        FlightModelRegistry.reset()

    def test_default_source_id_names_the_family_and_model(self) -> None:
        FlightModelRegistry.reset()
        model = FlightModelRegistry.get_model(FlightModelType.NATHAN)
        result = model.simulate(_driver_launch())
        trajectory = trajectory_from_flight_result(result, model)
        assert trajectory.source_id == f"{TOOLS_FLIGHT_FAMILY}:Nathan"
        assert (
            trajectory_from_flight_result(result, model, source_id="run-17").source_id
            == "run-17"
        )
        FlightModelRegistry.reset()

    def test_mismatched_model_is_refused(self) -> None:
        FlightModelRegistry.reset()
        penner = FlightModelRegistry.get_model(FlightModelType.WATERLOO_PENNER)
        nathan = FlightModelRegistry.get_model(FlightModelType.NATHAN)
        result = penner.simulate(_driver_launch())
        with pytest.raises(PreconditionError, match="same model"):
            trajectory_from_flight_result(result, nathan)
        FlightModelRegistry.reset()

    def test_unnamed_model_parameters_are_refused(self) -> None:
        """A model whose coefficients cannot be named cannot be exported."""

        class _Unregistered:
            name = "Mystery"

        with pytest.raises(PreconditionError, match="unknown flight model"):
            flight_model_parameters(_Unregistered())  # type: ignore[arg-type]

    def test_non_model_argument_is_refused(self) -> None:
        FlightModelRegistry.reset()
        model = FlightModelRegistry.get_model(FlightModelType.NATHAN)
        result = model.simulate(_driver_launch())
        with pytest.raises(PreconditionError, match="model must be a BallFlightModel"):
            trajectory_from_flight_result(result, object())  # type: ignore[arg-type]
        FlightModelRegistry.reset()

    def test_every_registered_model_can_name_its_parameters(self) -> None:
        """No registered model may be unattributable on the wire."""
        FlightModelRegistry.reset()
        for model in FlightModelRegistry.get_all_models():
            parameters = flight_model_parameters(model)
            assert parameters
            assert all(math.isfinite(value) for value in parameters.values())
            assert len(parameter_digest(parameters)) == 64
        FlightModelRegistry.reset()

    def test_two_models_produce_distinguishable_provenance(self) -> None:
        """ADR-0047: families and models stay named, never reconciled."""
        FlightModelRegistry.reset()
        launch = _driver_launch()
        records = []
        for model_type in (
            FlightModelType.WATERLOO_PENNER,
            FlightModelType.MACDONALD_HANZELY,
        ):
            model = FlightModelRegistry.get_model(model_type)
            records.append(trajectory_from_flight_result(model.simulate(launch), model))
        assert records[0].provenance.model_name != records[1].provenance.model_name
        assert (
            records[0].provenance.parameter_digest
            != records[1].provenance.parameter_digest
        )
        FlightModelRegistry.reset()
