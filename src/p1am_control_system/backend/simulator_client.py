import random
import time
from typing import Optional

from models import InterlockConfig, PIDConfig, RoutingConfig
from plc_interface import BasePLCClient


class SimulatedPLCClient(BasePLCClient):
    """Concrete PLC client simulating the physical hardware and plant process.

    Adheres to the BasePLCClient interface.
    """

    def __init__(self) -> None:
        super().__init__()
        self._connected = True

        # Standard simulated tag storage
        self.simulated_tags: list[float] = [0.0] * 32

        # Active configuration state
        self.active_config = RoutingConfig(
            input_routing=[0, 1, 2, 3, 4, 5],
            output_routing=[10, 11],
            pids=[
                PIDConfig(
                    pv_tag_id=1,
                    cv_tag_id=2,
                    setpoint=50.0,
                    kp=1.0,
                    ki=0.5,
                    kd=0.1,
                ),
                PIDConfig(
                    pv_tag_id=3,
                    cv_tag_id=4,
                    setpoint=30.0,
                    kp=1.5,
                    ki=0.2,
                    kd=0.05,
                ),
                PIDConfig(
                    pv_tag_id=5,
                    cv_tag_id=6,
                    setpoint=40.0,
                    kp=2.0,
                    ki=0.8,
                    kd=0.2,
                ),
                PIDConfig(
                    pv_tag_id=7,
                    cv_tag_id=8,
                    setpoint=60.0,
                    kp=0.5,
                    ki=0.1,
                    kd=0.01,
                ),
            ],
            interlocks=[
                InterlockConfig(
                    lolo_limit=0.0,
                    low_limit=5.0,
                    high_limit=95.0,
                    hihi_limit=100.0,
                )
                for _ in range(32)
            ],
        )

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

    async def read_tags(self) -> list[float] | None:
        """Perform a single step of the plant simulation and return the tags."""
        async with self.lock:
            if not self._connected:
                return None

            for i in range(4):
                pid_cfg = self.active_config.pids[i]
                pv_id = pid_cfg.pv_tag_id
                cv_id = pid_cfg.cv_tag_id

                # Check if loop is in active tuning mode
                in_tuning = i in self.tuning_sessions

                if not in_tuning:
                    # Closed-loop PID control simulation
                    err = pid_cfg.setpoint - self.simulated_tags[pv_id]
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
                    self.simulated_tags[cv_id] = cv_val

                # Update FOPDT plant dynamics
                self.cv_history[i].append(self.simulated_tags[cv_id])
                if len(self.cv_history[i]) > 40:
                    self.cv_history[i].pop(0)

                delay_steps = int(self.fopdt_delay[i] / 0.1)
                idx = max(0, len(self.cv_history[i]) - 1 - delay_steps)
                delayed_cv = self.cv_history[i][idx]

                noise = random.uniform(-0.05, 0.05)
                y_prev = self.simulated_tags[pv_id]
                dy = (self.fopdt_gain[i] * delayed_cv - y_prev) * (
                    0.1 / self.fopdt_tau[i]
                )
                self.simulated_tags[pv_id] = max(0.0, y_prev + dy + noise)

                # Record tuning session history
                if in_tuning:
                    session = self.tuning_sessions[i]
                    time_offset = time.time() - session["start_time"]
                    session["history"].append(
                        (
                            time_offset,
                            self.simulated_tags[cv_id],
                            self.simulated_tags[pv_id],
                        )
                    )

            # Set E-stop status and CPU Temp/Cycle Time simulation
            self.simulated_tags[0] = 1.0  # Normal safety state
            self.simulated_tags[9] = round(
                35.5 + random.uniform(-0.2, 0.2), 1
            )  # CPU Temperature
            self.simulated_tags[10] = round(
                0.12 + random.uniform(-0.01, 0.01), 3
            )  # Cycle time

            return list(self.simulated_tags)

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
            self.simulated_tags = [0.0] * 32
            return True

    async def write_tag(self, tag_id: int, value: float) -> bool:
        async with self.lock:
            if 0 <= tag_id < len(self.simulated_tags):
                self.simulated_tags[tag_id] = value
                return True
            return False
