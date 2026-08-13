"""Strict wire projection for physical regional-ground execution options."""

from __future__ import annotations

from typing import Any

from rate_of_closure.application._regional_ground_execution_job_values import (
    canonical_text,
    integer,
    positive,
    vector,
)
from rate_of_closure.application._workspace_validation import exact_mapping
from shared.python.swing_sim.ground import (
    MAX_REGIONAL_EXECUTION_STEPS,
    MAX_REGIONAL_EXECUTION_TRANSITIONS,
    RegionalGroundExecutionOptions,
    SkidRollSettings,
)

_OPTION_FIELDS = frozenset({"settings", "source_revision"})
_SETTING_FIELDS = frozenset(
    {
        "integration_step_s",
        "max_steps",
        "max_surface_transitions",
        "velocity_tolerance_m_s_decimal",
        "angular_tolerance_rad_s_decimal",
        "slip_tolerance_m_s_decimal",
        "time_tolerance_s_decimal",
        "gravity_m_s2",
        "model_id",
        "model_version",
    }
)


def _settings_to_dict(settings: SkidRollSettings) -> dict[str, Any]:
    if type(settings) is not SkidRollSettings:
        raise TypeError("settings must be an exact SkidRollSettings")
    return {
        "integration_step_s": settings.integration_step_s,
        "max_steps": settings.max_steps,
        "max_surface_transitions": settings.max_surface_transitions,
        "velocity_tolerance_m_s_decimal": _decimal(settings.velocity_tolerance_m_s),
        "angular_tolerance_rad_s_decimal": _decimal(settings.angular_tolerance_rad_s),
        "slip_tolerance_m_s_decimal": _decimal(settings.slip_tolerance_m_s),
        "time_tolerance_s_decimal": _decimal(settings.time_tolerance_s),
        "gravity_m_s2": list(settings.gravity_m_s2),
        "model_id": settings.model_id,
        "model_version": settings.model_version,
    }


def _decimal(value: float) -> str:
    return format(value, ".17f").rstrip("0").rstrip(".")


def _parse_decimal(value: object, name: str) -> float:
    text = canonical_text(value, name)
    try:
        parsed = float(text)
    except ValueError as exc:
        raise ValueError(f"{name} must be decimal text") from exc
    if _decimal(parsed) != text:
        raise ValueError(f"{name} must be canonical decimal text")
    bounded_value: float = positive(parsed, name.removesuffix("_decimal"), 1.0)
    return bounded_value


def _settings_from_dict(value: object) -> SkidRollSettings:
    data = exact_mapping(value, _SETTING_FIELDS, "skid_roll_settings")
    return SkidRollSettings(
        integration_step_s=positive(
            data["integration_step_s"], "integration_step_s", 1.0
        ),
        max_steps=integer(
            data["max_steps"], "max_steps", 1, MAX_REGIONAL_EXECUTION_STEPS
        ),
        max_surface_transitions=integer(
            data["max_surface_transitions"],
            "max_surface_transitions",
            1,
            MAX_REGIONAL_EXECUTION_TRANSITIONS,
        ),
        velocity_tolerance_m_s=_parse_decimal(
            data["velocity_tolerance_m_s_decimal"],
            "velocity_tolerance_m_s_decimal",
        ),
        angular_tolerance_rad_s=_parse_decimal(
            data["angular_tolerance_rad_s_decimal"],
            "angular_tolerance_rad_s_decimal",
        ),
        slip_tolerance_m_s=_parse_decimal(
            data["slip_tolerance_m_s_decimal"],
            "slip_tolerance_m_s_decimal",
        ),
        time_tolerance_s=_parse_decimal(
            data["time_tolerance_s_decimal"], "time_tolerance_s_decimal"
        ),
        gravity_m_s2=vector(data["gravity_m_s2"], "gravity_m_s2"),
        model_id=canonical_text(data["model_id"], "skid/roll model_id"),
        model_version=canonical_text(data["model_version"], "skid/roll model_version"),
    )


def regional_execution_options_to_dict(
    options: RegionalGroundExecutionOptions,
) -> dict[str, Any]:
    """Serialize exact callback-free physical options."""
    if type(options) is not RegionalGroundExecutionOptions:
        raise TypeError("options must be exact RegionalGroundExecutionOptions")
    if options.is_cancelled is not None:
        raise ValueError(
            "execution-job options cannot serialize a cancellation callback"
        )
    return {
        "settings": _settings_to_dict(options.settings),
        "source_revision": options.source_revision,
    }


def regional_execution_options_from_dict(
    value: object,
) -> RegionalGroundExecutionOptions:
    """Parse exact callback-free physical options."""
    data = exact_mapping(value, _OPTION_FIELDS, "regional_execution_options")
    return RegionalGroundExecutionOptions(
        settings=_settings_from_dict(data["settings"]),
        source_revision=canonical_text(data["source_revision"], "source_revision"),
    )


__all__ = [
    "regional_execution_options_from_dict",
    "regional_execution_options_to_dict",
]
