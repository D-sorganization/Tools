"""Direct tag writes, PID auto-tuning and the PID-vs-MPC comparison endpoints.

Extracted from ``main.py`` to keep that module inside the 1200-line module-size
budget. ``main.py`` had grown past its 1440-line grandfathered baseline once this
batch added the E-stop write-seam refusals, the acknowledged-step requirement and
the shutdown sequencing; ``poll_runtime`` had already been split out for the same
reason. These handlers form the one cohesive group left: everything here writes
to, or identifies the dynamics of, a single control loop.

Follows the router-factory pattern already used by
``power_supply_integration.create_power_supply_router`` and
``temperature_integration.create_temperature_router`` — the collaborators are
passed in rather than imported from ``main``, which would be a circular import.

Behaviour is unchanged by the move. In particular:

* ``write_tag`` raising ``NotImplementedError`` still becomes a 501 rather than a
  200 for a write the plant never saw (issue #4015);
* the tuning step still goes out through ``write_pid_setpoint`` and the session
  is marked stepped only once the controller acknowledges it, so identification
  cannot fit gains to a step that never reached the plant (issue #4015);
* every output-writing route is still refused while the E-stop is engaged.
"""

from __future__ import annotations

import logging
import time
from typing import Any, cast

import hardware
from fastapi import APIRouter, Depends, HTTPException

# Imported as a real type, not injected: this module uses
# `from __future__ import annotations`, so every annotation is a *string* at
# runtime and FastAPI resolves it against the module globals. A model passed in
# as a parameter is therefore invisible to it — the annotation stays the literal
# "step_payload_model", the body never binds, and every step request answers 422.
# `models` imports nothing from `main`, so there is no cycle to avoid here.
from models import PIDTuningStepPayload
from mpc import simulate_pid_vs_mpc
from pid_tuning import identify_fopdt_and_tune
from pydantic import BaseModel, field_validator
from pydantic import Field as PydanticField

# Number of PID loops the firmware exposes; mirrors hardware.PID_COUNT.
PID_LOOP_COUNT = 4
# Broker tag count, for the numeric TAG_<n> form accepted by the write route.
TAG_COUNT = 32


class TagWritePayload(BaseModel):
    """Operator-supplied value for a direct tag write.

    ``NaN``/``Infinity`` are valid JSON to pydantic's default ``float`` and
    used to reach the Modbus codec, whose ``ValueError`` was then swallowed by
    the client's I/O handler and reported as a lost PLC link (#3974). Reject
    them here (422) so a bad request body is a bad request, not an outage.
    """

    value: float

    @field_validator("value")
    @classmethod
    def _check_finite(cls, value: float) -> float:
        return float(hardware.require_finite_value(value, "value"))


class MPCSimulatePayload(BaseModel):
    """Bounded plant/tuning parameters for the PID-vs-MPC comparison chart."""

    prediction_horizon: int = PydanticField(10, ge=2, le=30)
    control_horizon: int = PydanticField(3, ge=1, le=10)
    setpoint: float = PydanticField(50.0, ge=0.0, le=100.0)
    rho: float = PydanticField(0.1, ge=0.0, le=10.0)
    process_gain: float = PydanticField(1.2, ge=0.1, le=5.0)
    process_tau: float = PydanticField(5.0, ge=0.5, le=20.0)
    process_delay: float = PydanticField(1.0, ge=0.0, le=5.0)


def create_tuning_router(
    *,
    control_context: Any,
    plc_client: Any,
    backup_simulator: Any,
    reject_output_write_if_estopped: Any,
    require_admin_key: Any,
    logger: logging.Logger,
) -> APIRouter:
    """Build the tag-write / PID-tuning / MPC router.

    Args:
        control_context: The :class:`state.SystemState` owning latest tags,
            the active routing config and the tuning sessions.
        plc_client: Live PLC client (the real Modbus manager in production).
        backup_simulator: Simulated client the bench HMI mirrors commands into.
        reject_output_write_if_estopped: Callable raising 409 while E-stopped.
        require_admin_key: The admin credential dependency.
        logger: Where operator actions are recorded.

    Returns:
        The configured :class:`fastapi.APIRouter`.
    """
    router = APIRouter(tags=["tuning"])

    def _latest_tag_or_http_error(tag_name: str, role: str, pid_index: int) -> float:
        """Return the latest tag value or raise a descriptive tuning error."""
        try:
            return float(control_context.latest_tags[tag_name])
        except KeyError as exc:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"PID loop {pid_index} {role} tag '{tag_name}' is not mapped in "
                    "the latest tag values. Check PLC routing before tuning."
                ),
            ) from exc

    @router.post("/api/tags/{tag_id}", dependencies=[Depends(require_admin_key)])
    async def write_tag_value(tag_id: str, payload: TagWritePayload) -> dict[str, str]:
        """Manually force/write a 32-bit float value directly to a tag register."""
        reject_output_write_if_estopped()
        tag_name = tag_id
        if tag_id.isdigit():
            val_id = int(tag_id)
            if not (0 <= val_id < TAG_COUNT):
                raise HTTPException(
                    status_code=400,
                    detail=f"Tag ID must be between 0 and {TAG_COUNT - 1}.",
                )
            tag_name = f"TAG_{tag_id}"

        if not plc_client.connected:
            success = await backup_simulator.write_tag(tag_name, payload.value)
            if not success:
                raise HTTPException(
                    status_code=400,
                    detail=f"Tag '{tag_name}' not found in simulator registry.",
                )
            control_context.write_tag(tag_name, payload.value)
            return {
                "status": "success",
                "message": (
                    f"Successfully forced simulated tag {tag_name} to {payload.value}."
                ),
            }

        try:
            success = await plc_client.write_tag(tag_name, payload.value)
        except hardware.NonFiniteValueError as exc:
            # Defense in depth behind the payload validator: a precondition
            # failure is the caller's error (400), never a transport fault.
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except NotImplementedError as exc:
            # No host-writable register for this tag (the P1AM TAG_n block is
            # republished by the firmware every scan) — say so rather than answer
            # 200 for a write the plant never saw (issue #4015).
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        await backup_simulator.write_tag(tag_name, payload.value)
        if not success:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to write value {payload.value} to tag {tag_name}.",
            )

        control_context.write_tag(tag_name, payload.value)
        return {
            "status": "success",
            "message": f"Successfully wrote {payload.value} to tag {tag_name}.",
        }

    @router.post(
        "/api/pid/{pid_index}/tuning/start",
        dependencies=[Depends(require_admin_key)],
    )
    async def start_pid_tuning(pid_index: int) -> dict[str, str]:
        """Decouple the loop from automatic control and begin logging history."""
        if not (0 <= pid_index < PID_LOOP_COUNT):
            raise HTTPException(
                status_code=400,
                detail=f"PID index must be between 0 and {PID_LOOP_COUNT - 1}.",
            )

        # Reject a double-start rather than silently overwriting an in-progress
        # session (a double-click or race would otherwise wipe the captured
        # initial PV/CV and step history). The operator must stop it first.
        if pid_index in control_context.tuning_sessions:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Tuning session already active for PID loop {pid_index}; "
                    "stop it before starting a new one."
                ),
            )

        pv_tag = control_context.active_config.pids[pid_index].pv_tag
        cv_tag = control_context.active_config.pids[pid_index].cv_tag
        current_pv = _latest_tag_or_http_error(pv_tag, "PV", pid_index)
        current_cv = _latest_tag_or_http_error(cv_tag, "CV", pid_index)

        control_context.tuning_sessions[pid_index] = {
            "start_time": time.time(),
            "history": [],
            "step_triggered": False,
            "step_time": 0.0,
            "initial_cv": current_cv,
            "initial_pv": current_pv,
            "final_cv": current_cv,
            "final_pv": current_pv,
        }
        logger.info("Started tuning mode for PID loop %s", pid_index)
        return {
            "status": "success",
            "message": f"Tuning mode started for PID loop {pid_index}.",
        }

    @router.post(
        "/api/pid/{pid_index}/tuning/step",
        dependencies=[Depends(require_admin_key)],
    )
    async def step_pid_tuning(
        pid_index: int, payload: PIDTuningStepPayload
    ) -> dict[str, str]:
        """Execute a step change on the loop, through a seam that reaches the PLC.

        The step used to go out via ``write_tag``, which resolves ``TAG_n`` into
        the block the firmware republishes every scan and never reads — the plant
        never saw it, yet identification ran anyway and returned fitted gains as
        ``status="success"`` (issue #4015). It now goes through
        ``write_pid_setpoint``, and the session is marked stepped only once acked.
        """
        if pid_index not in control_context.tuning_sessions:
            raise HTTPException(
                status_code=400, detail="Tuning session not active for this PID loop."
            )

        reject_output_write_if_estopped()
        session = control_context.tuning_sessions[pid_index]
        cv_tag = control_context.active_config.pids[pid_index].cv_tag
        initial_cv = _latest_tag_or_http_error(cv_tag, "CV", pid_index)

        if plc_client.connected:
            stepped = await plc_client.write_pid_setpoint(pid_index, payload.step_value)
            await backup_simulator.write_pid_setpoint(pid_index, payload.step_value)
        else:
            stepped = await backup_simulator.write_pid_setpoint(
                pid_index, payload.step_value
            )
        if not stepped:
            raise HTTPException(
                status_code=502,
                detail=f"PID loop {pid_index} step was not acknowledged; none applied.",
            )

        session["step_triggered"] = True
        session["step_time"] = time.time() - session["start_time"]
        session["initial_cv"] = initial_cv
        session["final_cv"] = payload.step_value

        control_context.write_tag(cv_tag, payload.step_value)

        logger.info(
            "Tuning step triggered on loop %s: CV set to %s",
            pid_index,
            payload.step_value,
        )
        return {
            "status": "success",
            "message": f"Step change applied. CV set to {payload.step_value}.",
        }

    @router.post(
        "/api/pid/{pid_index}/tuning/stop",
        dependencies=[Depends(require_admin_key)],
    )
    async def stop_pid_tuning(pid_index: int) -> dict[str, Any]:
        """Stop the session, identify FOPDT parameters and recommend gains."""
        if pid_index not in control_context.tuning_sessions:
            raise HTTPException(
                status_code=400, detail="Tuning session not active for this PID loop."
            )

        session = control_context.tuning_sessions.pop(pid_index)
        result = identify_fopdt_and_tune(
            session["history"],
            step_triggered=session["step_triggered"],
            initial_pv=session["initial_pv"],
            initial_cv=session["initial_cv"],
            final_cv=session["final_cv"],
            step_time=session["step_time"],
        )
        return cast(dict[str, Any], result.as_response())

    @router.post("/api/mpc/simulate", dependencies=[Depends(require_admin_key)])
    async def simulate_mpc(payload: MPCSimulatePayload) -> dict[str, Any]:
        """Simulate and compare standard PID against Model Predictive Control."""
        return cast(dict[str, Any], simulate_pid_vs_mpc(payload))

    return router
