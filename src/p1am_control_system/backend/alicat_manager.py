import asyncio
import logging
import random
import re
from typing import Any

# Configure logging
logger = logging.getLogger("dcs_backend.alicat_manager")

VALID_GASES = ["O2", "N2", "CO2", "He", "H2", "Air"]
PHYSICAL_CONNECTION_TYPES = {"serial", "tcp"}


class AlicatMFC:
    """Represents a single Alicat Mass Flow Controller device.

    Supports simulated loop dynamics or ASCII protocol communication.
    """

    def __init__(
        self,
        device_id: str,
        name: str,
        gas: str = "Air",
        max_flow: float = 50.0,
        connection_type: str = "mock",
        port_or_ip: str | None = None,
    ) -> None:
        if gas not in VALID_GASES:
            raise ValueError(f"Invalid gas: {gas}. Must be one of {VALID_GASES}.")

        self.device_id = device_id
        self.name = name
        self.gas = gas
        self.max_flow = max_flow
        self.connection_type = connection_type  # "mock", "serial", "tcp"
        self.port_or_ip = port_or_ip

        # Live telemetry readings
        self.setpoint: float = 0.0
        self.mass_flow: float = 0.0
        self.volumetric_flow: float = 0.0
        self.pressure: float = 14.7  # PSIA
        self.temperature: float = 23.5  # °C
        self.connection_state: str = (
            "simulated" if connection_type == "mock" else "disconnected"
        )

        # Internals for simulation response curves
        self._target_setpoint: float = 0.0

    def update_setpoint(self, value: float) -> bool:
        """Update target flow setpoint (clamped within range)."""
        next_setpoint = max(0.0, min(value, self.max_flow))
        if self.connection_type == "mock":
            self._target_setpoint = next_setpoint
            self.setpoint = self._target_setpoint
            return True
        if self.connection_type in PHYSICAL_CONNECTION_TYPES:
            self._mark_physical_io_unsupported("setpoint update")
            return False

        self._target_setpoint = next_setpoint
        return True

    def update_gas(self, new_gas: str) -> None:
        """Update active gas calibration."""
        if new_gas not in VALID_GASES:
            raise ValueError(f"Invalid gas: {new_gas}. Must be one of {VALID_GASES}.")
        self.gas = new_gas

    def simulate_step(self) -> None:
        """Simulate physical valve dynamics and sensor noise at 5Hz."""
        # 1st-order system response simulation (valve adjustment time constant)
        # mass flow adjusts toward setpoint target
        diff = self._target_setpoint - self.mass_flow
        self.mass_flow += diff * 0.15  # Adjust factor per step
        if abs(diff) < 0.01:
            self.mass_flow = self._target_setpoint

        # Add minor noise fluctuations
        flow_noise = random.uniform(-0.02, 0.02)
        if self.mass_flow > 0:
            self.mass_flow = max(0.0, self.mass_flow + flow_noise)

        # Pressure noise centered around 14.7 PSIA
        self.pressure = round(14.7 + random.uniform(-0.15, 0.15), 2)

        # Temperature noise centered around 23.5 °C
        self.temperature = round(23.5 + random.uniform(-0.3, 0.3), 1)

        # Volumetric flow tracks mass flow with pressure density ratio
        density_ratio = 14.7 / max(1.0, self.pressure)
        self.volumetric_flow = round(self.mass_flow * density_ratio, 2)

    async def poll_hardware(self) -> None:
        """Perform physical polling query over Serial or TCP socket.

        Mock connections simulate readings. Physical serial/TCP paths are not
        implemented yet and must fail closed instead of silently reporting
        success.
        """
        if self.connection_type in PHYSICAL_CONNECTION_TYPES:
            self._mark_physical_io_unsupported("polling")
            return

        # Pure simulated behavior
        self.simulate_step()
        self.connection_state = "simulated"

    def _mark_physical_io_unsupported(self, operation: str) -> None:
        """Record that real Alicat IO is unavailable for this operation."""
        if self.connection_state != "unsupported":
            logger.error(
                "Alicat MFC %s is configured for %s IO at %s, but %s is not "
                "implemented; refusing to report physical success.",
                self.device_id,
                self.connection_type,
                self.port_or_ip,
                operation,
            )
        self.connection_state = "unsupported"

    def parse_ascii_response(self, response: str) -> None:
        """Parse Alicat ASCII query response.

        Example: 'A 14.65 24.8 12.35 12.35 15.00 Air'
        Tokens: [Address] [Pressure] [Temp] [VolFlow] [MassFlow] [Setpoint] [Gas]
        """
        try:
            tokens = [t for t in re.split(r"\s+", response.strip()) if t]
            if len(tokens) >= 6:
                # Validate address ID matches
                if tokens[0] != self.device_id:
                    return
                self.pressure = float(tokens[1])
                self.temperature = float(tokens[2])
                self.volumetric_flow = float(tokens[3])
                self.mass_flow = float(tokens[4])
                self.setpoint = float(tokens[5])
                if len(tokens) >= 7:
                    self.gas = tokens[6]
        except Exception as parse_err:
            logger.error(
                f"Error parsing Alicat ASCII response '{response}': {parse_err}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert state properties to standard dict for API serialization."""
        return {
            "device_id": self.device_id,
            "name": self.name,
            "gas": self.gas,
            "setpoint": round(self.setpoint, 2),
            "mass_flow": round(self.mass_flow, 2),
            "volumetric_flow": round(self.volumetric_flow, 2),
            "pressure": self.pressure,
            "temperature": self.temperature,
            "max_flow": self.max_flow,
            "connection_type": self.connection_type,
            "port_or_ip": self.port_or_ip,
            "connection_state": self.connection_state,
        }


class AlicatManager:
    """Manages collection of active MFCs and periodic query loops."""

    def __init__(self) -> None:
        self.devices: dict[str, AlicatMFC] = {}
        self.polling_task: asyncio.Task | None = None
        self._running: bool = False

    def add_device(self, mfc: AlicatMFC) -> None:
        """Add mass flow controller to registry."""
        self.devices[mfc.device_id] = mfc

    def get_devices_data(self) -> list[dict[str, Any]]:
        """Return dict lists for all active MFCs."""
        return [dev.to_dict() for dev in self.devices.values()]

    def update_mfc_setpoint(self, device_id: str, setpoint: float) -> bool:
        """Apply target setpoint to specified device ID."""
        if device_id in self.devices:
            return self.devices[device_id].update_setpoint(setpoint)
        return False

    def update_mfc_gas(self, device_id: str, gas: str) -> bool:
        """Apply gas calibration select to specified device ID."""
        if device_id in self.devices:
            try:
                self.devices[device_id].update_gas(gas)
                return True
            except ValueError:
                return False
        return False

    async def polling_loop(self) -> None:
        """Periodic query scheduler running at 5Hz (200ms sleep)."""
        logger.info("Alicat MFC polling loop started.")
        self._running = True
        while self._running:
            try:
                tasks = [dev.poll_hardware() for dev in self.devices.values()]
                if tasks:
                    await asyncio.gather(*tasks)
            except Exception as loop_err:
                logger.error(f"Error in Alicat polling cycle: {loop_err}")
            await asyncio.sleep(0.2)
        logger.info("Alicat MFC polling loop stopped.")

    def start(self) -> None:
        """Launch background update task."""
        if not self._running:
            self.polling_task = asyncio.create_task(self.polling_loop())

    async def stop(self) -> None:
        """Gracefully terminate background update task."""
        self._running = False
        if self.polling_task:
            try:
                await self.polling_task
            except asyncio.CancelledError:
                pass
            self.polling_task = None
