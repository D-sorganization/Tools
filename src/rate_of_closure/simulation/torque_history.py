"""Convert retained Rate run torques into canonical reusable profiles."""

from __future__ import annotations

from collections.abc import Mapping

from rate_of_closure._contracts import require
from rate_of_closure.simulation.records import SimulationRun
from shared.python.swing_sim.run_config import DOUBLE_PENDULUM_MODEL_ID
from shared.python.swing_sim.torque_export import fit_torque_history_profile
from shared.python.swing_sim.torque_profiles import PrescribedTorqueProfile


def fit_run_torque_profile(
    run: SimulationRun,
    *,
    profile_id: str,
    name: str,
    description: str,
    degree: int,
    source_metadata: Mapping[str, str],
    created_at_utc: str,
    modified_at_utc: str,
    max_condition_number: float = 1.0e8,
) -> PrescribedTorqueProfile:
    """Fit one selected run's retained applied torques into a profile."""
    require(isinstance(run, SimulationRun), "run must be a SimulationRun", run)
    require(
        len(run.swing_joint_ids) > 0 and run.swing_applied_torques_nm.shape[1] > 0,
        "run has no applied joint torque history to fit",
    )
    require(
        run.config.source_kind == "double_pendulum",
        "only double-pendulum torque histories can currently be exported",
        run.config.source_kind,
    )
    histories = {
        joint_id: run.swing_applied_torques_nm[:, index]
        for index, joint_id in enumerate(run.swing_joint_ids)
    }
    return fit_torque_history_profile(
        profile_id=profile_id,
        model_id=DOUBLE_PENDULUM_MODEL_ID,
        name=name,
        description=description,
        source_metadata=source_metadata,
        created_at_utc=created_at_utc,
        modified_at_utc=modified_at_utc,
        times_s=run.swing_times,
        torque_nm_by_joint=histories,
        degree=degree,
        max_condition_number=max_condition_number,
    )


__all__ = ["fit_run_torque_profile"]
