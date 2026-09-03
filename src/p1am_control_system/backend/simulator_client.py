import random
import time
from typing import Optional

import hardware
from defaults import default_routing_config
from models import RoutingConfig
from plc_interface import BasePLCClient


class SimulatedPLCClient(BasePLCClient):
    """Concrete PLC client simulating the physical hardware and plant process.

    Adheres to the BasePLCClient interface.
    """

    def __init__(self) -> None:
        super().__init__()
        self._connected = True

        # Standard simulated tag storage
        self.simulated_tags: dict[str, float] = {f"TAG_{i}": 0.0 for i in range(32)}

        # Latched emergency-stop state for the simulated controller
        self.e_stop_active = False

        # Discrete coil state (e.g. heater relay) so sim mode models DO writes
        self.coils: dict[int, bool] = {}

        # Active configuration state (shared default — single source of truth)
        self.active_config = default_routing_config()

        # Plant simulation parameters (FOPDT: First Order Plus Dead Time)
        self.fopdt_gain: list[float] = [1.5, 0.8, 1.2, 2.0]
        self.fopdt_tau: list[float] = [5.0, 3.0, 7.0, 4.0]
        self.fopdt_delay: list[float] = [1.2, 0.6, 1.8, 1.0]

        # History buffer of CVs for dead time simulation (at 10Hz, 1 step = 100ms)
        self.cv_history: dict[int, list[float]] = {i: [0.0] * 40 for i in range(4)}

        # PID Integrals and prev errors for simulated closed-loop control
        self.pid_integrals: list[float] = [0.0] * 4
        self.pid_prev_errors: list[float] = [0.0] * 4

    @property
    def connected(self) -> bool:
        return self._connected

    async def connect(self) -> bool:
        self._connected = True
        return True

    async def disconnect(self) -> None:
        self._connected = False

    async def read_tags(self) -> dict[str, float] | None:
        """Perform a single step of the plant simulation and return the tags."""
        async with self.lock:
            if not self._connected:
                return None

            # Simulate custom tags if dynamic tag map is loaded
            if self.tag_map:
                for tag_name in self.tag_map:
                    if tag_name not in self.simulated_tags:
                        self.simulated_tags[tag_name] = 0.0
                    else:
                        # Exclude main loop tags from the simple random walk
                        is_loop_tag = any(
                            tag_name in [pid.pv_tag, pid.cv_tag]
                            for pid in self.active_config.pids
                        )
                        exclude_tags = ["TAG_0", "TAG_9", "TAG_10"]
                        if tag_name not in exclude_tags and not is_loop_tag:
                            val = self.simulated_tags[tag_name] + random.uniform(
                                -0.1, 0.1
                            )
                            self.simulated_tags[tag_name] = round(max(0.0, val), 2)

            for i in range(4):
                pid_cfg = self.active_config.pids[i]
                pv_tag = pid_cfg.pv_tag
                cv_tag = pid_cfg.cv_tag

                # Check if loop is in active tuning mode
                in_tuning = i in self.tuning_sessions

                if not in_tuning:
                    # Closed-loop PID control simulation
                    err = pid_cfg.setpoint - self.simulated_tags.get(pv_tag, 0.0)
                    self.pid_integrals[i] = max(
                        -100.0, min(100.0, self.pid_integrals[i] + err * 0.1)
                    )
                    deriv = (err - self.pid_prev_errors[i]) / 0.1
                    self.pid_prev_errors[i] = err

                    cv_val = (
                        pid_cfg.kp * err
                        + pid_cfg.ki * self.pid_integrals[i]
                        + pid_cfg.kd * deriv
                    )
                    cv_val = max(0.0, min(100.0, cv_val))
                    self.simulated_tags[cv_tag] = cv_val

                # Update FOPDT plant dynamics
                self.cv_history[i].append(self.simulated_tags.get(cv_tag, 0.0))
                if len(self.cv_history[i]) > 40:
                    self.cv_history[i].pop(0)

                delay_steps = int(self.fopdt_delay[i] / 0.1)
                idx = max(0, len(self.cv_history[i]) - 1 - delay_steps)
                delayed_cv = self.cv_history[i][idx]

                noise = random.uniform(-0.05, 0.05)
                y_prev = self.simulated_tags.get(pv_tag, 0.0)
                dy = (self.fopdt_gain[i] * delayed_cv - y_prev) * (
                    0.1 / self.fopdt_tau[i]
                )
                self.simulated_tags[pv_tag] = max(0.0, y_prev + dy + noise)

                # Record tuning session history
                if in_tuning:
                    session = self.tuning_sessions[i]
                    time_offset = time.time() - session["start_time"]
                    session["history"].append(
                        (
                            time_offset,
                            self.simulated_tags.get(cv_tag, 0.0),
                            self.simulated_tags.get(pv_tag, 0.0),
                        )
                    )
            # Set E-stop status and CPU Temp/Cycle Time simulation
            self.simulated_tags["TAG_0"] = 1.0  # Normal safety state
            self.simulated_tags["TAG_9"] = round(
                35.5 + random.uniform(-0.2, 0.2), 1
            )  # CPU Temperature
            self.simulated_tags["TAG_10"] = round(
                0.12 + random.uniform(-0.01, 0.01), 3
            )  # Cycle time

            return dict(self.simulated_tags)

    async def read_routing(self) -> Optional["RoutingConfig"]:
        async with self.lock:
            return self.active_config

    async def write_routing(self, config: "RoutingConfig") -> bool:
        async with self.lock:
            self.active_config = config
            return True

    async def save_to_flash(self) -> bool:
        return True

    async def trigger_estop(self) -> bool:
        async with self.lock:
            self.simulated_tags = {f"TAG_{i}": 0.0 for i in range(32)}
            self.e_stop_active = True
            return True

    async def clear_estop(self) -> bool:
        async with self.lock:
            self.e_stop_active = False
            return True

    async def write_tag(self, tag_name: str, value: float) -> bool:
        """Force a simulated tag.

        Raises:
            TypeError: If ``value`` is not a number.
            hardware.NonFiniteValueError: If ``value`` is NaN/Inf -- the same
                precondition the real client enforces (#3974), so simulator
                mode cannot accept a force the plant would refuse.
        """
        value = hardware.require_finite_value(value, "value")
        async with self.lock:
            if tag_name in self.simulated_tags:
                self.simulated_tags[tag_name] = value
                return True
            return False

    async def write_pid_setpoint(self, pid_index: int, value: float) -> bool:
        """Update a simulated PID loop's setpoint so the sim models the command.

        Mirrors the Modbus client's public seam; the sim's step loop already
        reads ``active_config.pids[i].setpoint``. (Previously the power-supply
        service reached into a private ``_get_client`` that the simulator lacks,
        which spewed harmless-but-noisy errors in simulator mode.)
        """
        async with self.lock:
            if 0 <= pid_index < len(self.active_config.pids):
                self.active_config.pids[pid_index].setpoint = value
                return True
            return False

    async def write_coil(self, address: int, value: bool) -> bool:
        """Record a simulated discrete-coil write (mirrors the Modbus seam)."""
        async with self.lock:
            self.coils[address] = bool(value)
            return True
