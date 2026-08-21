"""Deterministic cross-runtime catalog of all registered run traces."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import fields

import numpy as np

from ..canonical_numeric_json import canonical_numeric_json
from .contract import (
    EXPECTED_UPSTREAM_SOURCE_REVISION,
    MATCHING_RULES,
    MODEL_TIER,
    TORSO_PROFILES,
    RotatingBaseCase,
)
from .loader import EXPECTED_STUDY_SHA256, load_embedded_qualified_study
from .provider import (
    REGISTERED_TORSO_RATES_RAD_S,
    RotatingBaseRunRequest,
    RotatingBaseRunResult,
    registered_run_mapping,
    run_registered_case,
)

RUN_CATALOG_SCHEMA_ID = "swing-sim/rotating-base-run-catalog"
RUN_CATALOG_SCHEMA_VERSION = 1
EXPECTED_RUN_CATALOG_SHA256 = (
    "66493b833955c6492a00eae4a600df79"  # pragma: allowlist secret
    "5df60a6f473f9a11c403084b58e51678"  # pragma: allowlist secret
)
_METRIC_ATOL = 1e-10


def registered_requests() -> tuple[RotatingBaseRunRequest, ...]:
    """Return the complete publication design in stable case-index order."""
    return tuple(
        RotatingBaseRunRequest(profile, rule, rate)
        for rule in MATCHING_RULES
        for profile in TORSO_PROFILES
        for rate in REGISTERED_TORSO_RATES_RAD_S
    )


def _case_matches_authority(
    actual: RotatingBaseCase, expected: RotatingBaseCase
) -> bool:
    if (
        actual.case_index != expected.case_index
        or actual.torso_profile != expected.torso_profile
        or actual.matching_rule != expected.matching_rule
        or actual.initial_torso_rate_rad_s != expected.initial_torso_rate_rad_s
        or actual.valid != expected.valid
        or actual.exclusion_reasons != expected.exclusion_reasons
    ):
        return False
    return all(
        np.isclose(
            getattr(actual.metrics, field.name),
            getattr(expected.metrics, field.name),
            atol=_METRIC_ATOL,
            rtol=0.0,
        )
        for field in fields(expected.metrics)
    )


def registered_run_catalog_mapping(
    results: Sequence[RotatingBaseRunResult],
) -> dict[str, object]:
    """Validate and retain every registered trace without favorable filtering."""
    retained = tuple(results)
    expected_requests = registered_requests()
    if len(retained) != len(expected_requests):
        raise ValueError("results must retain the complete 18-case order")
    authority = load_embedded_qualified_study().study
    for index, (result, request, expected_case) in enumerate(
        zip(retained, expected_requests, authority.cases, strict=True)
    ):
        if not isinstance(result, RotatingBaseRunResult):
            raise TypeError(f"result {index} must be a RotatingBaseRunResult")
        if result.request != request or result.request.case_index != index:
            raise ValueError("results must retain the complete 18-case order")
        if result.source_revision != EXPECTED_UPSTREAM_SOURCE_REVISION:
            raise ValueError("result source revision does not match the authority")
        if not _case_matches_authority(result.case, expected_case):
            raise ValueError(f"result {index} metrics do not match the qualified study")
    return {
        "schema_id": RUN_CATALOG_SCHEMA_ID,
        "schema_version": RUN_CATALOG_SCHEMA_VERSION,
        "source_revision": EXPECTED_UPSTREAM_SOURCE_REVISION,
        "study_sha256": EXPECTED_STUDY_SHA256,
        "model_tier": MODEL_TIER,
        "attempted_run_count": len(retained),
        "runs": [registered_run_mapping(result) for result in retained],
    }


def registered_run_catalog_json(results: Sequence[RotatingBaseRunResult]) -> str:
    """Serialize the complete validated trace catalog canonically."""
    return canonical_numeric_json(registered_run_catalog_mapping(results))


def generate_registered_run_catalog(
    executor: Callable[[RotatingBaseRunRequest], RotatingBaseRunResult] = (
        run_registered_case
    ),
) -> tuple[RotatingBaseRunResult, ...]:
    """Execute every registered case through the one canonical provider."""
    if not callable(executor):
        raise TypeError("executor must be callable")
    return tuple(executor(request) for request in registered_requests())


__all__ = [
    "EXPECTED_RUN_CATALOG_SHA256",
    "RUN_CATALOG_SCHEMA_ID",
    "RUN_CATALOG_SCHEMA_VERSION",
    "generate_registered_run_catalog",
    "registered_requests",
    "registered_run_catalog_json",
    "registered_run_catalog_mapping",
]
