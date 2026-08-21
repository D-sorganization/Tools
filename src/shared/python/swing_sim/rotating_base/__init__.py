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
from .loader import EXPECTED_STUDY_SHA256, load_qualified_study

__all__ = [
    "EXPECTED_UPSTREAM_SOURCE_REVISION",
    "EXPECTED_STUDY_SHA256",
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
    "load_qualified_study",
]
