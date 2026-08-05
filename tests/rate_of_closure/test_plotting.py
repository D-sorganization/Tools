"""Plotting suite tests (epic #4120 V1): catalog, specs, pipeline, exports.

Covers: the pinned catalog key list (the contract shared with the web
clone through the parity fixture), extractor shapes and finiteness,
PlotSpec validation + JSON round-trip, every builtin rendering
headlessly (Agg) on a reference run, and well-formed CSV/JSON/PNG/SVG
export files.
"""

from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from matplotlib.figure import Figure  # noqa: E402

from rate_of_closure.club import get_club  # noqa: E402
from rate_of_closure.model import ImpactScenario  # noqa: E402
from rate_of_closure.plotting import (  # noqa: E402
    BUILTIN_PLOTS,
    CATALOG,
    CATEGORIES,
    PlotSpec,
    builtin_spec,
    catalog_keys,
    compute_plot_data,
    extract,
    render_plot,
    spec_from_json,
    spec_to_json,
    variables_by_category,
    write_plot_csv,
    write_plot_json,
)
from rate_of_closure.simulation import SimulationConfig, run_simulation  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

#: The pinned catalog contract. The web catalog mirrors this list
#: key-for-key (plotcatalog.fixture.json + plotcatalog.test.ts).
EXPECTED_KEYS: tuple[str, ...] = (
    "input.clubhead_speed_mph",
    "input.omega_plane_dps",
    "input.omega_shaft_dps",
    "input.lie_angle_deg",
    "input.com_to_face_mm",
    "input.impact_offset_toe_mm",
    "input.impact_offset_high_mm",
    "input.contact_duration_us",
    "input.plane_yaw_deg",
    "input.plane_side_tilt_deg",
    "input.plane_forward_tilt_deg",
    "input.impact_time_s",
    "swing.time_s",
    "swing.x_m",
    "swing.y_m",
    "swing.z_m",
    "swing.speed_mps",
    "swing.angular_speed_dps",
    "kinetics.shoulder_torque_nm",
    "kinetics.wrist_torque_nm",
    "kinetics.shoulder_gravity_torque_nm",
    "kinetics.wrist_gravity_torque_nm",
    "kinetics.shoulder_damping_torque_nm",
    "kinetics.wrist_damping_torque_nm",
    "kinetics.shoulder_power_w",
    "kinetics.wrist_power_w",
    "kinetics.shoulder_force_n",
    "kinetics.wrist_force_n",
    "kinetics.clubhead_force_n",
    "impact.clubhead_speed_mps",
    "impact.club_path_deg",
    "impact.attack_angle_deg",
    "impact.spin_loft_deg",
    "impact.face_to_path_deg",
    "impact.spin_axis_tilt_deg",
    "impact.energy_transfer_j",
    "launch.ball_speed_mph",
    "launch.launch_angle_deg",
    "launch.launch_azimuth_deg",
    "launch.spin_rpm",
    "flight.time_s",
    "flight.x_m",
    "flight.y_m",
    "flight.z_m",
    "flight.speed_mps",
    "metric.carry_m",
    "metric.max_height_m",
    "metric.flight_time_s",
    "metric.landing_angle_deg",
    "metric.path_deviation_deg",
    "metric.closure_rate_dps",
)

_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "rate_of_closure"
    / "web"
    / "src"
    / "model"
    / "plotcatalog.fixture.json"
)


@pytest.fixture(scope="module")
def run():  # type: ignore[no-untyped-def]
    """One reference manual-source run shared by the module."""
    return run_simulation(
        SimulationConfig(
            scenario=ImpactScenario(clubhead_speed_mph=113.0),
            club=get_club("Driver 10.5°"),
        )
    )


class TestCatalog:
    def test_catalog_keys_are_pinned(self) -> None:
        assert catalog_keys() == EXPECTED_KEYS

    def test_parity_fixture_matches_the_catalog(self) -> None:
        payload = json.loads(_FIXTURE.read_text(encoding="utf-8"))
        assert payload["format"] == "rate_of_closure.plot_catalog/1"
        assert tuple(payload["keys"]) == catalog_keys()

    def test_every_category_has_variables(self) -> None:
        for category in CATEGORIES:
            assert variables_by_category(category), category

    def test_labels_units_and_prefixes_are_consistent(self) -> None:
        prefixes = {
            "Input": "input.",
            "Swing Sample": "swing.",
            "Kinetics": "kinetics.",
            "Impact": "impact.",
            "Launch": "launch.",
            "Flight": "flight.",
            "Metric": "metric.",
        }
        for key, spec in CATALOG.items():
            assert key == spec.key
            assert key.startswith(prefixes[spec.category]), key
            assert spec.label == spec.label.strip() and spec.label, key
            assert spec.scale in ("linear", "log")

    def test_scalar_extractors_yield_finite_floats(self, run) -> None:  # type: ignore[no-untyped-def]
        for key, spec in CATALOG.items():
            if spec.is_series:
                continue
            value = extract(run, key)
            assert isinstance(value, float), key
            assert np.isfinite(value), key

    def test_series_extractors_match_their_sample_counts(self, run) -> None:  # type: ignore[no-untyped-def]
        n_swing = run.swing_times.shape[0]
        n_flight = run.flight_times.shape[0]
        for key, spec in CATALOG.items():
            if not spec.is_series:
                continue
            values = extract(run, key)
            assert isinstance(values, np.ndarray) and values.ndim == 1, key
            expected = n_flight if spec.category == "Flight" else n_swing
            assert values.shape == (expected,), key
            if spec.category == "Kinetics":
                # The manual reference run has no joint states, so the
                # kinetics extractors yield all-NaN (#4125 H2); the
                # finite double-pendulum case is pinned in
                # test_kinetics.py.
                assert np.isnan(values).all(), key
            else:
                assert np.isfinite(values).all(), key

    def test_unknown_key_is_rejected(self, run) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(Exception, match="unknown catalog key"):
            extract(run, "nope.nothing")


class TestPlotSpec:
    def test_json_round_trip_preserves_every_field(self) -> None:
        spec = PlotSpec(
            kind="sweep",
            x_key="input.omega_shaft_dps",
            y_keys=("metric.path_deviation_deg", "launch.spin_rpm"),
            title="Round Trip",
            x_log=False,
            y_log=True,
            x_start=0.0,
            x_stop=4000.0,
            x_count=11,
        )
        assert PlotSpec.from_json_dict(spec.to_json_dict()) == spec

    def test_file_round_trip(self, tmp_path: Path) -> None:
        spec = builtin_spec("swing_time_series")
        path = tmp_path / "definition.json"
        spec_to_json(spec, path)
        assert spec_from_json(path) == spec

    def test_unknown_keys_and_kinds_are_rejected(self) -> None:
        with pytest.raises(Exception, match="unknown x_key"):
            PlotSpec(kind="line", x_key="bogus.key", y_keys=("swing.speed_mps",))
        with pytest.raises(Exception, match="unknown plot kind"):
            PlotSpec(kind="pie", x_key="swing.time_s", y_keys=("swing.speed_mps",))

    def test_sweep_requires_input_x_and_scalar_y(self) -> None:
        with pytest.raises(Exception, match="Input"):
            PlotSpec(
                kind="sweep",
                x_key="swing.time_s",
                y_keys=("metric.carry_m",),
                x_start=0.0,
                x_stop=1.0,
            )
        with pytest.raises(Exception, match="scalar outputs"):
            PlotSpec(
                kind="sweep",
                x_key="input.omega_shaft_dps",
                y_keys=("swing.speed_mps",),
                x_start=0.0,
                x_stop=1.0,
            )

    def test_line_requires_series_variables(self) -> None:
        with pytest.raises(Exception, match="per-sample"):
            PlotSpec(
                kind="line",
                x_key="input.omega_shaft_dps",
                y_keys=("swing.speed_mps",),
            )

    def test_bad_format_marker_is_rejected(self) -> None:
        with pytest.raises(Exception, match="format"):
            PlotSpec.from_json_dict({"format": "other/9", "kind": "line"})


def _fast(spec: PlotSpec) -> PlotSpec:
    """Shrink sweep grids so the full-simulation sweeps stay quick."""
    if spec.kind == "sweep":
        return dataclasses.replace(spec, x_count=4)
    return spec


class TestPipeline:
    @pytest.mark.parametrize("name", sorted(BUILTIN_PLOTS))
    def test_every_builtin_renders_headlessly(self, run, name: str) -> None:  # type: ignore[no-untyped-def]
        spec = _fast(builtin_spec(name, run))
        data = compute_plot_data(spec, run)
        assert data.x.ndim == 1 and data.x.size >= 2
        for label, values in data.series.items():
            assert values.shape == data.x.shape, label
        figure = Figure()
        render_plot(data, figure)
        axes = figure.axes[0]
        assert axes.get_xlabel() == data.x_label
        assert axes.get_ylabel() == data.y_label
        assert axes.get_title() == spec.title

    def test_builtin_titles_are_labelled(self) -> None:
        for name, (label, _factory) in BUILTIN_PLOTS.items():
            assert label, name
            assert builtin_spec(name).title, name

    def test_histogram_kind_renders(self, run) -> None:  # type: ignore[no-untyped-def]
        spec = PlotSpec(kind="histogram", x_key="flight.speed_mps", title="Hist")
        data = compute_plot_data(spec, run)
        assert data.series == {}
        figure = Figure()
        render_plot(data, figure)
        assert figure.axes[0].get_ylabel() == "Count"

    def test_log_flags_reach_the_axes(self, run) -> None:  # type: ignore[no-untyped-def]
        spec = PlotSpec(
            kind="line",
            x_key="swing.time_s",
            y_keys=("swing.speed_mps",),
            y_log=True,
        )
        figure = Figure()
        render_plot(compute_plot_data(spec, run), figure)
        assert figure.axes[0].get_yscale() == "log"

    def test_sweep_recovers_the_closure_sweep_shape(self, run) -> None:  # type: ignore[no-untyped-def]
        """The migrated closure sweep matches the model's sweep() numbers."""
        from rate_of_closure.model import sweep as model_sweep

        spec = dataclasses.replace(builtin_spec("closure_sweep"), x_count=5)
        data = compute_plot_data(spec, run)
        expected = model_sweep(run.config.scenario, "omega_shaft_dps", data.x)
        np.testing.assert_allclose(
            data.series["Impact-Point Path Deviation"], expected, atol=1e-9
        )


class TestExports:
    def test_csv_export_is_well_formed(self, run, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        data = compute_plot_data(builtin_spec("flight_profile_side"), run)
        path = tmp_path / "plot.csv"
        write_plot_csv(data, path)
        with path.open(encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        assert rows[0] == ["Downrange Distance [m]", "Height"]
        assert len(rows) == data.x.size + 1
        assert float(rows[1][0]) == pytest.approx(float(data.x[0]))

    def test_json_export_carries_spec_and_rows(self, run, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        spec = _fast(builtin_spec("launch_vs_toe_offset"))
        data = compute_plot_data(spec, run)
        path = tmp_path / "plot.json"
        write_plot_json(data, path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["format"] == "rate_of_closure.plot_data/1"
        assert PlotSpec.from_json_dict(payload["spec"]) == spec
        assert len(payload["rows"]) == data.x.size
        assert len(payload["columns"]) == 1 + len(data.series)

    def test_figure_saves_png_and_svg(self, run, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        figure = Figure()
        render_plot(compute_plot_data(builtin_spec("swing_time_series"), run), figure)
        png = tmp_path / "plot.png"
        svg = tmp_path / "plot.svg"
        figure.savefig(png)
        figure.savefig(svg)
        assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert b"<svg" in svg.read_bytes()
