"""UI-neutral execution policy for variation analyses."""

from __future__ import annotations

from typing import Literal, cast

from shared.python.contracts import require

AnalysisExecution = Literal["all_together", "individual", "both"]

ANALYSIS_EXECUTIONS: tuple[AnalysisExecution, ...] = (
    "all_together",
    "individual",
    "both",
)


def validate_analysis_execution(value: object) -> AnalysisExecution:
    """Return a supported policy or fail before launching any analysis."""
    require(
        isinstance(value, str) and value in ANALYSIS_EXECUTIONS,
        "analysis execution must be all_together, individual, or both",
        value,
    )
    return cast(AnalysisExecution, value)


def runs_joint_analysis(value: AnalysisExecution) -> bool:
    """Return whether the jointly enabled Monte Carlo batch executes."""
    return validate_analysis_execution(value) != "individual"


def runs_individual_analysis(value: AnalysisExecution) -> bool:
    """Return whether the one-at-a-time intervention batches execute."""
    return validate_analysis_execution(value) != "all_together"


def planned_analysis_runs(
    n_runs: int, noise_count: int, value: AnalysisExecution
) -> int:
    """Return the exact number of trial evaluations for progress reporting."""
    require(isinstance(n_runs, int) and n_runs >= 2, "n_runs must be at least 2")
    require(
        isinstance(noise_count, int) and noise_count >= 1,
        "noise_count must be positive",
    )
    joint = n_runs if runs_joint_analysis(value) else 0
    individual = n_runs * noise_count if runs_individual_analysis(value) else 0
    return joint + individual


__all__ = [
    "ANALYSIS_EXECUTIONS",
    "AnalysisExecution",
    "planned_analysis_runs",
    "runs_individual_analysis",
    "runs_joint_analysis",
    "validate_analysis_execution",
]
