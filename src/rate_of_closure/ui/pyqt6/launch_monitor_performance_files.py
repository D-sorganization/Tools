"""Persistence document builder for the PyQt performance workspace."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from rate_of_closure.launch_monitor_performance import (
    DispersionResult,
    ScoreResult,
    TrendResult,
)
from rate_of_closure.launch_monitor_workspace import DatasetReference


def performance_document(
    reference: DatasetReference,
    settings: dict[str, object],
    dispersion: DispersionResult | None,
    target_error: ScoreResult | None,
    trend: TrendResult | None,
) -> dict[str, object]:
    """Return the fingerprint-bound saved-analysis document."""

    return {
        "contract_version": "launch-monitor-performance/1.0",
        "dataset_sha256": reference.sha256,
        "source_name": reference.source_name,
        "settings": settings,
        "dispersion": asdict(dispersion) if dispersion else None,
        "target_error": asdict(target_error) if target_error else None,
        "trend": asdict(trend) if trend else None,
    }


def load_performance_settings(source: str, expected_sha256: str) -> dict[str, object]:
    """Validate a saved document and return its settings mapping."""

    payload = json.loads(Path(source).read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("dataset_sha256") != expected_sha256
    ):
        raise ValueError("saved analysis references a different dataset")
    settings = payload.get("settings")
    if not isinstance(settings, dict):
        raise ValueError("saved analysis settings are unavailable")
    return settings


__all__ = ["load_performance_settings", "performance_document"]
