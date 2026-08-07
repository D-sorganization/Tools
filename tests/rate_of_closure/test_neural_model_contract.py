"""Safe neural-surrogate bundle and inference contracts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rate_of_closure.neural_model import (
    load_neural_bundle,
    predict_frame,
    predict_records,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _bundle() -> dict[str, object]:
    return {
        "schema": "launch-monitor-neural-bundle/v1",
        "modelId": "trackman-carry-test",
        "vendor": "TrackMan-comparable",
        "createdAt": "2026-08-06T00:00:00Z",
        "features": [
            {
                "name": "ball_speed_mph",
                "unit": "mph",
                "mean": 100.0,
                "scale": 10.0,
                "min": 80.0,
                "max": 120.0,
            },
            {
                "name": "launch_angle_deg",
                "unit": "deg",
                "mean": 10.0,
                "scale": 2.0,
                "min": 6.0,
                "max": 14.0,
            },
        ],
        "outputs": [{"name": "carry_yd", "unit": "yd", "mean": 200.0, "scale": 5.0}],
        "layers": [
            {
                "weights": [[2.0, 3.0]],
                "bias": [1.0],
                "activation": "linear",
            }
        ],
        "metrics": {"holdout_rmse": {"carry_yd": 3.2}},
        "learningCurve": [{"epoch": 1, "trainLoss": 9.0, "validationLoss": 10.0}],
        "provenance": {"dataset_sha256": "abc", "row_count": 1000},
    }


def test_json_bundle_predicts_and_reports_applicability(tmp_path: Path) -> None:
    path = tmp_path / "model.nn.json"
    path.write_text(json.dumps(_bundle()), encoding="utf-8")

    model = load_neural_bundle(path)
    predictions = predict_records(
        model,
        [{"ball_speed_mph": 110.0, "launch_angle_deg": 12.0}],
    )

    assert predictions.values[0, 0] == pytest.approx(230.0)
    assert predictions.warnings == ()
    outside = predict_records(
        model,
        [{"ball_speed_mph": 130.0, "launch_angle_deg": 12.0}],
    )
    assert "ball_speed_mph" in outside.warnings[0]


def test_batch_prediction_preserves_inputs_and_adds_named_outputs() -> None:
    model = load_neural_bundle(_bundle())
    frame = pd.DataFrame(
        {"ball_speed_mph": [100.0, 110.0], "launch_angle_deg": [10.0, 12.0]}
    )

    result = predict_frame(model, frame)

    assert list(result.frame.columns) == [
        "ball_speed_mph",
        "launch_angle_deg",
        "predicted_carry_yd",
    ]
    assert result.frame["predicted_carry_yd"].to_list() == pytest.approx([205.0, 230.0])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda data: data.update(schema="other/v1"), "schema"),
        (lambda data: data["layers"][0].update(weights=[[np.nan, 1.0]]), "finite"),
        (lambda data: data.update(features=data["features"][:1]), "dimension"),
    ],
)
def test_bundle_rejects_unsupported_or_malformed_content(mutation, message) -> None:  # type: ignore[no-untyped-def]
    payload = _bundle()
    mutation(payload)
    with pytest.raises(ValueError, match=message):
        load_neural_bundle(payload)


def test_bundle_loader_never_accepts_pickle(tmp_path: Path) -> None:
    path = tmp_path / "unsafe.pkl"
    path.write_bytes(b"not executable here")
    with pytest.raises(ValueError, match="JSON"):
        load_neural_bundle(path)
