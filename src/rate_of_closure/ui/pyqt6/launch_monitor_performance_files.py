"""Persistence document builder for the PyQt performance workspace."""

from __future__ import annotations

import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

from rate_of_closure.launch_monitor_performance import (
    DispersionResult,
    ScoreResult,
    TrendResult,
)
from rate_of_closure.launch_monitor_workspace import DatasetReference
from rate_of_closure.launch_monitor_workspace_v3 import parse_workspace_project


def _aggregate(
    value: DispersionResult | ScoreResult | TrendResult | None,
) -> object | None:
    if value is None:
        return None
    return {
        key: item
        for key, item in asdict(value).items()
        if key not in {"points", "values"}
    }


def _result(
    dispersion: DispersionResult | None,
    target_error: ScoreResult | None,
    trend: TrendResult | None,
) -> dict[str, object]:
    payload = {
        "dispersion": _aggregate(dispersion),
        "target_error": _aggregate(target_error),
        "trend": _aggregate(trend),
    }
    available = any(value is not None for value in payload.values())
    return {
        "status": "available" if available else "unavailable",
        "authority": "offline-compatibility-v1",
        "authority_commit": None,
        "response_sha256": (
            sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
            if available
            else None
        ),
        "payload": payload if available else None,
        "units": {"carry": "yd", "lateral": "yd", "target_error": "yd"},
        "formulas": [
            item.formula
            for item in (dispersion, target_error, trend)
            if item is not None
        ],
        "exclusions": [
            "Per-shot and per-session points remain outside the saved project."
        ],
    }


def _analysis(
    reference: DatasetReference,
    settings: dict[str, object],
    dispersion: DispersionResult | None,
    target_error: ScoreResult | None,
    trend: TrendResult | None,
) -> dict[str, object]:
    return {
        "analysis_id": "performance-summary",
        "operation": "performance_summary",
        "settings": settings,
        "result": _result(dispersion, target_error, trend),
        "backing_join": {
            "algorithm": "sha256-canonical-json-v1",
            "row_count": reference.row_count,
            "sha256": None,
            "status": "available-on-authorized-export",
            "reason": None,
        },
    }


def performance_document(
    reference: DatasetReference,
    settings: dict[str, object],
    dispersion: DispersionResult | None,
    target_error: ScoreResult | None,
    trend: TrendResult | None,
) -> dict[str, object]:
    """Return the fingerprint-bound saved-analysis document."""

    return dict(
        parse_workspace_project(
            {
                "schema_id": "launch-monitor-workspace/v3",
                "schema_version": 3,
                "name": f"{reference.source_name} performance analysis",
                "dataset": {
                    "source_name": reference.source_name,
                    "repository": reference.repository,
                    "revision": reference.revision,
                    "relative_path": reference.relative_path,
                    "content_sha256": reference.sha256,
                    "row_count": reference.row_count,
                    "classification": "restricted",
                    "authority_commit": None,
                    "manifest_sha256": None,
                },
                "identity_evidence": {},
                "analyses": [
                    _analysis(reference, settings, dispersion, target_error, trend)
                ],
                "export_policy": {
                    "persist_rows": False,
                    "backing_rows": "explicit-restricted-approval",
                    "reason": (
                        "Restricted rows remain outside saved projects and browser "
                        "persistence."
                    ),
                },
            }
        )
    )


def load_performance_settings(source: str, expected_sha256: str) -> dict[str, object]:
    """Validate a saved document and return its settings mapping."""

    payload = json.loads(Path(source).read_text(encoding="utf-8"))
    if (
        isinstance(payload, dict)
        and payload.get("schema_id") == "launch-monitor-workspace/v3"
    ):
        workspace = parse_workspace_project(payload)
        if workspace.dataset.content_sha256 != expected_sha256:
            raise ValueError("saved analysis references a different dataset")
        analysis = next(
            item
            for item in workspace.analyses
            if item.operation == "performance_summary"
        )
        return dict(analysis.settings)
    if (
        not isinstance(payload, dict)
        or payload.get("dataset_sha256") != expected_sha256
    ):
        raise ValueError("saved analysis references a different dataset")
    settings = payload.get("settings")
    if not isinstance(settings, dict):
        raise ValueError("saved analysis settings are unavailable")
    return settings


def load_performance_settings_versioned(
    source: str, expected_sha256: str
) -> tuple[dict[str, object], str]:
    """Load settings and expose the active compatibility adapter."""

    payload = json.loads(Path(source).read_text(encoding="utf-8"))
    imported_from = (
        "v3"
        if isinstance(payload, dict)
        and payload.get("schema_id") == "launch-monitor-workspace/v3"
        else "v1-compatibility"
    )
    return load_performance_settings(source, expected_sha256), imported_from


__all__ = [
    "load_performance_settings",
    "load_performance_settings_versioned",
    "performance_document",
]
