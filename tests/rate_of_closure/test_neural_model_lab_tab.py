"""PyQt neural-model laboratory contracts."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")

from rate_of_closure.ui.pyqt6.neural_model_lab_tab import (
    NeuralModelLabTab,  # noqa: E402
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _write_bundle(path: Path) -> None:
    payload = {
        "schema": "launch-monitor-neural-bundle/v1",
        "modelId": "trackman-demo",
        "vendor": "TrackMan-comparable",
        "createdAt": "2026-08-06T00:00:00Z",
        "features": [
            {
                "name": "ball_speed_mph",
                "unit": "mph",
                "mean": 0.0,
                "scale": 1.0,
                "min": 0.0,
                "max": 200.0,
            },
            {
                "name": "launch_angle_deg",
                "unit": "deg",
                "mean": 0.0,
                "scale": 1.0,
                "min": 0.0,
                "max": 30.0,
            },
        ],
        "outputs": [{"name": "carry_yd", "unit": "yd", "mean": 0.0, "scale": 1.0}],
        "layers": [{"weights": [[1.0, 2.0]], "bias": [0.0], "activation": "linear"}],
        "metrics": {"holdout_rmse": {"carry_yd": 2.5}},
        "learningCurve": [{"epoch": 1, "trainLoss": 3.0, "validationLoss": 4.0}],
        "provenance": {"dataset_sha256": "demo", "row_count": 10},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_tab_imports_safe_model_and_queries_current_row(qtbot, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    tab = NeuralModelLabTab(auto_discover_campaign=False)
    qtbot.addWidget(tab)
    model_path = tmp_path / "trackman.nn.json"
    _write_bundle(model_path)
    tab.set_dataset(
        pd.DataFrame({"ball_speed_mph": [100.0], "launch_angle_deg": [10.0]}),
        source_name="custom.csv",
    )

    tab.import_model(model_path)
    output = tab.predict_current_dataset()

    assert output["predicted_carry_yd"].to_list() == [120.0]
    assert "TrackMan-comparable" in tab.model_summary.toPlainText()
    assert tab.export_predictions_button.isEnabled()


def test_training_controls_are_configurable_and_accessible(qtbot) -> None:  # type: ignore[no-untyped-def]
    tab = NeuralModelLabTab(auto_discover_campaign=False)
    qtbot.addWidget(tab)
    controls = tab.interactive_controls()

    assert {
        tab.controls.activation_combo.itemText(index)
        for index in range(tab.controls.activation_combo.count())
    } >= {
        "relu",
        "tanh",
    }
    assert tab.controls.holdout_spin.value() == pytest.approx(0.2)
    assert all(control.accessibleName() for control in controls)
    assert all(control.toolTip() for control in controls)


def test_training_request_uses_private_cli_and_visible_arguments(
    qtbot, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    tab = NeuralModelLabTab(auto_discover_campaign=False)
    qtbot.addWidget(tab)
    (tmp_path / "campaign.toml").write_text("[campaign]\n", encoding="utf-8")
    tab.campaign_root = tmp_path
    tab.dataset_path = tmp_path / "custom.csv"
    frame = pd.DataFrame(
        {
            "ball_speed_mph": [100.0, 101.0, 102.0],
            "carry_yd": [220.0, 221.0, 222.0],
            "shot_id": ["a", "b", "c"],
        }
    )
    frame.to_csv(tab.dataset_path, index=False)
    tab.set_dataset(frame, source_name="custom.csv")
    tab.controls.feature_list.item(0).setSelected(True)
    tab.controls.target_list.item(1).setSelected(True)
    tab.controls.vendor_combo.addItem("TrackMan-comparable", "TrackMan-comparable")
    tab.controls.vendor_combo.setCurrentIndex(tab.controls.vendor_combo.count() - 1)

    request = tab.training_request(tmp_path / "model.nn.json")

    assert request.program
    assert "neural-train" in request.arguments
    payload = tomllib.loads(request.config_path.read_text(encoding="utf-8"))
    assert payload["surrogate"]["dataset"] == str(tab.dataset_path)
    assert request.working_directory == tmp_path


def test_vendor_without_shot_level_targets_is_disabled_with_reason(qtbot) -> None:  # type: ignore[no-untyped-def]
    tab = NeuralModelLabTab(auto_discover_campaign=False)
    qtbot.addWidget(tab)

    combo = tab.controls.vendor_combo
    foresight = combo.findText("Foresight-comparable")
    assert foresight >= 0
    assert not bool(combo.model().item(foresight).isEnabled())
    assert "shot-level" in combo.itemData(foresight, 3).lower()


def test_modeling_cohort_defaults_are_preselected(qtbot) -> None:  # type: ignore[no-untyped-def]
    tab = NeuralModelLabTab(auto_discover_campaign=False)
    qtbot.addWidget(tab)
    features = {
        "ball_speed_mph",
        "launch_angle_deg",
        "launch_direction_deg",
        "spin_rate_rpm",
        "spin_axis_deg",
    }
    targets = {
        "observed_carry_m",
        "observed_lateral_m",
        "observed_apex_m",
        "observed_landing_angle_deg",
        "observed_flight_time_s",
    }
    frame = pd.DataFrame({name: [1.0, 2.0, 3.0] for name in features | targets})
    frame.insert(0, "shot_id", ["a", "b", "c"])

    tab.set_dataset(frame, source_name="Cohort")

    assert set(tab._selected(tab.controls.feature_list)) == features
    assert set(tab._selected(tab.controls.target_list)) == targets
