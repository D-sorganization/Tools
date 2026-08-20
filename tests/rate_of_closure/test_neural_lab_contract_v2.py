from __future__ import annotations

import json

import pandas as pd
import pytest

from rate_of_closure.neural_lab_contract import (
    DatasetAuthority,
    TrainingSelection,
    build_training_manifest,
    load_capability_manifest,
    load_portable_model,
    predict_one,
    validate_training_groups,
)

SHA = "a" * 64


def test_current_vendor_capabilities_are_manifest_driven_and_fail_closed() -> None:
    capabilities = load_capability_manifest()
    observed = {item.vendor: item for item in capabilities.vendors}
    assert (observed["TrackMan"].row_count, observed["TrackMan"].strict_row_count) == (
        11_699,
        9_298,
    )
    assert observed["TrackMan"].artifact_state == "retired_non_group_safe"
    assert "approved repeating split group" in " ".join(observed["TrackMan"].blockers)
    assert (
        observed["Foresight"].row_count,
        observed["Foresight"].strict_row_count,
    ) == (4, 2)
    assert (
        observed["FlightScope"].row_count,
        observed["FlightScope"].strict_row_count,
    ) == (2_794, 0)
    assert all(item.state == "unavailable" for item in observed.values())


@pytest.mark.parametrize("column", ["shot_id", "source_row_number", "row_index"])
def test_training_rejects_forbidden_or_non_repeating_split_groups(column: str) -> None:
    frame = pd.DataFrame({column: ["a", "b", "c", "d"], "x": range(4), "y": range(4)})
    with pytest.raises(ValueError, match="split group"):
        validate_training_groups(frame, column, policy_approved=True)


def test_training_requires_policy_approval_three_groups_and_a_repeat() -> None:
    frame = pd.DataFrame({"player": ["a", "a", "b", "c"], "x": range(4), "y": range(4)})
    with pytest.raises(ValueError, match="policy-approved"):
        validate_training_groups(frame, "player", policy_approved=False)
    summary = validate_training_groups(frame, "player", policy_approved=True)
    assert summary.distinct_groups == 3
    assert summary.repeated_groups == 1


def test_training_manifest_contains_reference_not_rows() -> None:
    frame = pd.DataFrame({"player": ["a", "a", "b", "c"], "x": range(4), "y": range(4)})
    manifest = build_training_manifest(
        DatasetAuthority("custom", "private/repo", "b" * 40, "data.csv", SHA, 4),
        frame,
        TrainingSelection("Custom", ("x",), ("y",), "player", True),
    )
    payload = manifest.to_wire()
    assert payload["dataset"]["sha256"] == SHA
    assert "rows" not in json.dumps(payload)
    assert payload["split"]["distinct_groups"] == 3


def test_portable_bundle_requires_manifest_hash_and_warns_ood() -> None:
    training_manifest = {
        "schema": "launch-monitor-neural-training/v2",
        "dataset": {"sha256": SHA},
        "features": ["speed"],
        "targets": ["carry"],
        "split": {"column": "player", "policy_approved": True},
    }
    manifest_sha = __import__("hashlib").sha256(
        json.dumps(training_manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    bundle = load_portable_model(
        {
            "schema": "launch-monitor-neural-bundle/v2",
            "model_id": "trackman-comparable-test",
            "vendor": "TrackMan-Comparable",
            "training_manifest": training_manifest,
            "training_manifest_sha256": manifest_sha,
            "dataset_sha256": SHA,
            "features": [
                {
                    "name": "speed",
                    "unit": "mph",
                    "mean": 100,
                    "scale": 10,
                    "min": 80,
                    "max": 120,
                }
            ],
            "targets": [{"name": "carry", "unit": "yd", "mean": 250, "scale": 20}],
            "layers": [{"activation": "linear", "weights": [[2]], "bias": [0]}],
            "model_card": {"purpose": "descriptive vendor-comparable surrogate"},
            "metrics": [{"target": "carry", "split": "held_out_group", "rmse": 3.2}],
            "residuals": {
                "state": "unavailable",
                "reason": "row-aligned held-out residuals were not exported",
            },
        }
    )
    result = predict_one(bundle, {"speed": 130})
    assert result.values["carry"] == pytest.approx(370)
    assert result.warnings and "outside" in result.warnings[0]
