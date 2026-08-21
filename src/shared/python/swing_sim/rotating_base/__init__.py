"""Shared rotating-base provider contracts."""

from .contract import (
    EXPECTED_UPSTREAM_SOURCE_REVISION,
    KILLSWITCH_CHANNELS,
    MATCHING_RULES,
    MODEL_TIER,
    SCHEMA_ID,
    SCHEMA_VERSION,
    RotatingBaseCase,
    RotatingBaseCaseMetrics,
    RotatingBaseProviderResult,
    RotatingBaseStudy,
    SameStateKillswitch,
)

__all__ = [
    "EXPECTED_UPSTREAM_SOURCE_REVISION",
    "KILLSWITCH_CHANNELS",
    "MATCHING_RULES",
    "MODEL_TIER",
    "RotatingBaseCase",
    "RotatingBaseCaseMetrics",
    "RotatingBaseProviderResult",
    "RotatingBaseStudy",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "SameStateKillswitch",
]
