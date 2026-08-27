"""Governance checks for measured complete-trial retention scaling evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/rate_of_closure/complete_trial_retention_scaling.v1.json"

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_complete_trial_scaling_evidence_is_revision_bound_and_bounded() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    scenarios = evidence["scenarios"]

    assert evidence["schema_version"] == "tools-complete-trial-retention-scaling/v1"
    assert evidence["source_revision"] == (
        "dc1cf1d4f447a11ae5fb1483f80aaaf9a173bf75"  # pragma: allowlist secret
    )
    assert evidence["record_schema"] == "rate-complete-trial/v1"
    assert evidence["durable_schema_version"] == 3
    assert evidence["measurement_command"] == (
        "python -m scripts.measure_complete_trial_retention_scaling"
    )
    assert [item["trial_count"] for item in scenarios] == [16, 64]
    assert {item["chunk_size"] for item in scenarios} == {4}
    assert all(item["archive_bytes"] > 0 for item in scenarios)
    assert all(item["max_chunk_bytes"] < 16_000_000 for item in scenarios)
    assert all(item["peak_python_bytes"] > 0 for item in scenarios)
    assert evidence["observed"]["peak_python_ratio_64_to_16"] <= 2.0
    assert evidence["observed"]["artifact_bytes_per_trial_ratio_64_to_16"] <= 1.25
    assert evidence["scientific_boundary"].startswith(
        "This is software resource-scaling evidence"
    )
