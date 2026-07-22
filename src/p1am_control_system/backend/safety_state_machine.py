"""Shared safety scaffolding for the P1AM heater/power-supply controllers.

Both :class:`~temperature_controller.TemperatureController` and
:class:`~power_supply.PowerSupplyController` carry byte-for-byte identical
safety machinery around their (very different) control laws: the
IDLE/ARMED/RUNNING/TRIPPED state handling, a one-way E-stop latch, the
permissive toggle with its ARMED<->IDLE / ARMED->RUNNING transitions, trip
latching plus acknowledge, ``_safe_finite`` input coercion, and the
"force the actuator off now" predicate.

This module extracts *only* that genuinely-identical scaffolding into
:class:`SafetyStateMachine`, a small base class parameterised by the concrete
state enum (the two controllers deliberately use distinct enums —
``TemperatureState`` vs ``PowerSupplyState`` — with the same members). Every
controller-specific reset (which setpoint fields to clear, actuator state to
wipe, log wording) is delegated back to the subclass through the ``_on_*``
hooks so the extraction is strictly behaviour-preserving.

Safety-critical: this is conservative on purpose. Only mechanics that are
identical in both controllers live here; each control law, trip check, status
model, and actuator write stays in its own controller.
"""

from __future__ import annotations

import logging
import math
from typing import Generic, TypeVar

from shared.python.compatibility import StrEnum

__all__ = ["SafetyStateMachine"]

StateT = TypeVar("StateT", bound=StrEnum)


class SafetyStateMachine(Generic[StateT]):
    """Shared IDLE/ARMED/RUNNING/TRIPPED + E-stop + trip-latch scaffolding.

    Subclasses supply the concrete state enum members and the actuator-specific
    reset behaviour via the ``_on_*`` hooks. The base owns exactly four pieces
    of state: the current ``_state``, the ``_permissive`` flag, the one-way
    ``_estopped`` latch, and the latched ``_trips`` set.

    Precondition: ``idle``/``armed``/``running``/``tripped`` are the four
    distinct members of a single concrete ``StrEnum`` state type.
    """

    def __init__(
        self,
        *,
        idle: StateT,
        armed: StateT,
        running: StateT,
        tripped: StateT,
        logger: logging.Logger,
    ) -> None:
        self._state_idle = idle
        self._state_armed = armed
        self._state_running = running
        self._state_tripped = tripped
        self._safety_logger = logger
        self._state: StateT = idle
        self._permissive = False
        self._trips: set[str] = set()
        # E-stop latch. While set, the controller forces the actuator off,
        # refuses to arm, and rejects setpoints until clear_estop() is called.
        # This is a one-way kill — it must be explicitly cleared by an operator.
        self._estopped = False

    # -- read-only views -------------------------------------------------

    @property
    def state(self) -> StateT:
        return self._state

    @property
    def permissive(self) -> bool:
        return self._permissive

    @property
    def trips(self) -> list[str]:
        return sorted(self._trips)

    @property
    def estopped(self) -> bool:
        return self._estopped

    # -- subclass hooks (default to no-op resets) ------------------------

    def _on_disarm(self) -> None:
        """Clear controller-specific setpoint state on a permissive-off disarm.

        Called after the base has forced the state machine back to IDLE. The
        base has already reset permissive/state; the subclass wipes its own
        setpoint fields.
        """

    def _on_estop_engaged(self) -> None:
        """Clear controller-specific state when the E-stop latch engages.

        Called after the base has set ``_estopped``, cleared permissive, and
        forced IDLE. The subclass drops setpoints and any actuator/ramp state.
        """

    def _estop_log_message(self) -> str:
        """Return the operator-facing message logged when the E-stop engages.

        Overridden per controller so the wording names the actual actuator
        (heater relay vs power-supply output) without changing the shared
        engage sequence.
        """
        return "E-STOP engaged"

    def _on_estop_cleared(self) -> None:
        """Clear controller-specific setpoint state when the E-stop releases.

        Called after the base has released the latch, cleared permissive, and
        forced IDLE.
        """

    def _on_trip_acknowledged(self) -> None:
        """Clear controller-specific setpoint state on trip acknowledge.

        Called after the base has cleared ``_trips`` but before the state is
        moved out of TRIPPED, mirroring the original in-place reset order.
        """

    # -- shared mechanics -----------------------------------------------

    @staticmethod
    def _safe_finite(value: float) -> float:
        """Coerce a feedback input to a finite float.

        Non-finite / non-numeric values map to 0 so a sensor failure can never
        accidentally hold the actuator on or smuggle a NaN through the
        comparisons.
        """
        if not isinstance(value, int | float) or isinstance(value, bool):
            return 0.0
        v = float(value)
        return v if math.isfinite(v) else 0.0

    def _apply_permissive(self, on: bool) -> None:
        """Shared permissive toggle. A trip latch is not cleared by this.

        Precondition: ``on`` is exactly a bool (no truthy coercion, to catch
        caller bugs).
        Postcondition:
            - permissive == on (except when E-stop is latched: forced False)
            - If on goes False and state was RUNNING / ARMED: state -> IDLE and
              the subclass ``_on_disarm`` hook clears its setpoints.
            - If on goes True and state was IDLE: state -> ARMED
            - If state was TRIPPED: state stays TRIPPED
            - If E-stop is latched: on=True is ignored (stays disarmed).

        Raises:
            TypeError: if ``on`` is not exactly bool.
        """
        if not isinstance(on, bool):
            raise TypeError(f"on must be bool, got {type(on).__name__}")
        if self._estopped:
            # A latched E-stop cannot be armed around. The actuator stays off
            # until the operator explicitly clears the E-stop.
            if on:
                self._safety_logger.warning("permissive ON ignored — E-stop is latched")
            self._permissive = False
            return
        self._permissive = on
        if self._state == self._state_tripped:
            return
        if on:
            if self._state == self._state_idle:
                self._state = self._state_armed
        else:
            self._on_disarm()
            self._state = self._state_idle

    def _latch_trips(self, *, log_context: str) -> None:
        """Flip the state machine to TRIPPED if any trip is latched.

        Subclasses add their concrete trip keys to ``self._trips`` (from their
        own HH / fault checks) and then call this to perform the identical
        latch-and-log transition.

        Precondition: ``self._trips`` reflects the current cycle's checks.
        Postcondition: if any trip is set and state was not already TRIPPED,
        the transition is logged once and state -> TRIPPED.
        """
        if self._trips and self._state != self._state_tripped:
            self._safety_logger.error(
                "trip latched: %s (%s)",
                ",".join(sorted(self._trips)),
                log_context,
            )
            self._state = self._state_tripped

    def _should_force_actuator_off(self) -> bool:
        """All "kill the actuator now" conditions in one place.

        True unless the controller is genuinely running: RUNNING state, armed
        (permissive), no E-stop, and no latched trip.
        """
        return (
            self._estopped
            or self._state != self._state_running
            or not self._permissive
            or bool(self._trips)
        )

    def engage_estop(self) -> None:
        """Latch the emergency stop: force the actuator off and disarm.

        This is the software half of the kill switch. It immediately disarms
        (the actuator-off latch is enforced by the subclass ``tick``), forces
        the state machine to IDLE, and delegates setpoint/actuator clearing to
        ``_on_estop_engaged``. The latch persists — ``set_permissive(True)`` and
        setpoint commands are rejected — until ``clear_estop()`` is called.

        Postcondition: estopped; permissive False; state IDLE; subclass state
        cleared.
        """
        self._estopped = True
        self._permissive = False
        self._state = self._state_idle
        self._on_estop_engaged()
        self._safety_logger.error(self._estop_log_message())

    def clear_estop(self) -> None:
        """Release the emergency-stop latch.

        Leaves the controller IDLE with permissive off, so the operator must
        deliberately re-arm (permissive on) and re-enter a setpoint before the
        actuator can fire again. A no-op when not latched.

        Postcondition: not estopped; permissive False; state IDLE; subclass
        setpoint state cleared.
        """
        if not self._estopped:
            return
        self._estopped = False
        self._permissive = False
        self._state = self._state_idle
        self._on_estop_cleared()
        self._safety_logger.warning("E-stop cleared — controller idle, re-arm required")

    def acknowledge_trip(self) -> bool:
        """Clear latched trips and return controller to safe idle/armed state.

        Postcondition:
            - All trips cleared.
            - Subclass setpoint state cleared.
            - State -> ARMED if permissive is True, else IDLE.

        Returns:
            True if a trip was cleared. False if there were no trips.
        """
        if self._state != self._state_tripped:
            return False
        self._trips.clear()
        self._on_trip_acknowledged()
        self._state = self._state_armed if self._permissive else self._state_idle
        return True
