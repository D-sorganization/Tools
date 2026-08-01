import asyncio
import logging
import random
import re
from typing import Any

# Configure logging
logger = logging.getLogger("dcs_backend.alicat_manager")

VALID_GASES = ["O2", "N2", "CO2", "He", "H2", "Air"]
MOCK_CONNECTION_TYPE = "mock"
PHYSICAL_CONNECTION_TYPES = frozenset({"serial", "tcp"})
VALID_CONNECTION_TYPES = frozenset({MOCK_CONNECTION_TYPE}) | PHYSICAL_CONNECTION_TYPES

# PLC drivers that are themselves simulated. Only against one of these may the
# gas subsystem be simulated too; pairing a real PLC with mock MFCs lets an
# operator "establish" a purge that does not exist (issue #4031).
SIMULATED_PLC_DRIVERS = frozenset({"simulator", "simulated"})

# The bench rig's standard MFC complement. Single source of truth so main.py's
# registration block and the tests cannot drift.
DEFAULT_MFC_SPECS: tuple[dict[str, Any], ...] = (
    {"device_id": "A", "name": "Oxygen MFC", "gas": "O2", "max_flow": 50.0},
    {"device_id": "B", "name": "Nitrogen MFC", "gas": "N2", "max_flow": 100.0},
    {"device_id": "C", "name": "Carbon Dioxide MFC", "gas": "CO2", "max_flow": 20.0},
)


def validate_connection_type(connection_type: str) -> str:
    """Normalise and validate an MFC transport name.

    Raises:
        TypeError: If ``connection_type`` is not a str.
        ValueError: If it is not one of :data:`VALID_CONNECTION_TYPES`.
    """
    if not isinstance(connection_type, str):
        raise TypeError(
            f"connection_type must be a str, got {type(connection_type).__name__}"
        )
    normalized = connection_type.strip().lower()
    if normalized not in VALID_CONNECTION_TYPES:
        raise ValueError(
            f"connection_type must be one of {sorted(VALID_CONNECTION_TYPES)}; "
            f"got {connection_type!r}"
        )
    return normalized


def is_simulated_plc_driver(plc_driver: str) -> bool:
    """Whether ``plc_driver`` names a simulated (non-hardware) PLC."""
    if not isinstance(plc_driver, str):
        raise TypeError(f"plc_driver must be a str, got {type(plc_driver).__name__}")
    return plc_driver.strip().lower() in SIMULATED_PLC_DRIVERS


def ensure_gas_control_matches_plc(connection_type: str, plc_driver: str) -> None:
    """Refuse a simulated gas path paired with a real PLC driver (#4031).

    Preconditions:
        ``connection_type`` is already normalised via
        :func:`validate_connection_type`.

    Raises:
        TypeError: If either argument is not a str.
        ValueError: If ``connection_type`` is ``"mock"`` while ``plc_driver``
            drives real hardware.
    """
    if validate_connection_type(connection_type) != MOCK_CONNECTION_TYPE:
        return
    if is_simulated_plc_driver(plc_driver):
        return
    raise ValueError(
        f"Refusing to register a simulated (mock) Alicat MFC while "
        f"PLC_DRIVER={plc_driver!r} drives real hardware: an operator would "
        f"see a purge 'establish' with no gas flowing. Set "
        f"P1AM_ALICAT_CONNECTION_TYPE to 'serial' or 'tcp' (with "
        f"P1AM_ALICAT_PORT_OR_IP), or run the simulated PLC driver."
    )


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
        connection_type: str = MOCK_CONNECTION_TYPE,
        port_or_ip: str | None = None,
    ) -> None:
        """Create an MFC handle.

        Preconditions:
            ``gas`` is in :data:`VALID_GASES`; ``connection_type`` is in
            :data:`VALID_CONNECTION_TYPES`; a physical transport carries a
            ``port_or_ip``.

        Raises:
            TypeError: If ``connection_type`` is not a str.
            ValueError: If ``gas`` or ``connection_type`` is unsupported, or a
                serial/TCP device is missing its ``port_or_ip``.
        """
        if gas not in VALID_GASES:
            raise ValueError(f"Invalid gas: {gas}. Must be one of {VALID_GASES}.")

        connection_type = validate_connection_type(connection_type)
        if (
            connection_type in PHYSICAL_CONNECTION_TYPES
            and not (port_or_ip or "").strip()
        ):
            raise ValueError(
                f"A {connection_type} Alicat MFC requires a port_or_ip "
                f"(serial device or host); got {port_or_ip!r}."
            )

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
            "simulated" if connection_type == MOCK_CONNECTION_TYPE else "disconnected"
        )

        # Internals for simulation response curves
        self._target_setpoint: float = 0.0

    def update_setpoint(self, value: float) -> bool:
        """Update target flow setpoint (clamped within range).

        Returns:
            True if the setpoint was applied. Physical (serial/TCP) transports
            return False because the device IO is not implemented — they must
            never report a purge established that is not.
        """
        if self.connection_type in PHYSICAL_CONNECTION_TYPES:
            self._mark_physical_io_unsupported("setpoint update")
            return False

        # connection_type is validated in __init__, so this is the mock branch.
        self._target_setpoint = max(0.0, min(value, self.max_flow))
        self.setpoint = self._target_setpoint
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
                    self._apply_reported_gas(tokens[6])
        except Exception as parse_err:
            logger.error(
                f"Error parsing Alicat ASCII response '{response}': {parse_err}"
            )

    def _apply_reported_gas(self, gas: str) -> None:
        """Adopt a device-reported gas, going through the VALID_GASES check.

        The previous direct ``self.gas = tokens[6]`` assignment bypassed the
        validation every other write path honours (issue #4031), letting an
        unrecognised gas name into the API payload and the operator's readout.
        """
        try:
            self.update_gas(gas)
        except ValueError:
            logger.error(
                "Alicat MFC %s reported unsupported gas %r; keeping %r. Check "
                "the controller's gas table against VALID_GASES.",
                self.device_id,
                gas,
                self.gas,
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
    """Manages collection of active MFCs and periodic query loops.

    Args:
        plc_driver: The active PLC driver name. Registering a simulated (mock)
            MFC is refused unless this names a simulated PLC (issue #4031).
    """

    def __init__(self, *, plc_driver: str = "simulator") -> None:
        if not isinstance(plc_driver, str):
            raise TypeError(
                f"plc_driver must be a str, got {type(plc_driver).__name__}"
            )
        self.plc_driver = plc_driver
        self.devices: dict[str, AlicatMFC] = {}
        self.polling_task: asyncio.Task | None = None
        self._running: bool = False
        # Why the registry is empty, when startup refused to register devices.
        self.registration_error: str | None = None

    def add_device(self, mfc: AlicatMFC) -> None:
        """Add mass flow controller to registry.

        Raises:
            TypeError: If ``mfc`` is not an :class:`AlicatMFC`.
            ValueError: If a mock device is registered while ``plc_driver``
                drives real hardware.
        """
        if not isinstance(mfc, AlicatMFC):
            raise TypeError(f"mfc must be an AlicatMFC, got {type(mfc).__name__}")
        ensure_gas_control_matches_plc(mfc.connection_type, self.plc_driver)
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


def create_default_manager(
    *,
    connection_type: str,
    plc_driver: str,
    port_or_ip: str | None = None,
) -> AlicatManager:
    """Build the rig's standard MFC complement on the configured transport.

    The transport comes from settings rather than being hardcoded, and a
    simulated gas path against a real PLC driver is refused — the combination
    that let an operator watch a purge "establish" with no gas flowing (issue
    #4031).

    A refused or unbuildable configuration yields an **empty** manager with
    :attr:`AlicatManager.registration_error` set, logged at CRITICAL. Gas
    control is then plainly unavailable (``/api/alicats`` returns nothing)
    rather than silently simulated — and the rest of the SCADA backend, which
    owns the E-stop, heater, and power supply, still comes up.

    Args:
        connection_type: ``"mock"``, ``"serial"``, or ``"tcp"``.
        plc_driver: The active PLC driver name (``settings.plc_driver``).
        port_or_ip: Serial device or host shared by the controllers; required
            for serial/TCP.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``connection_type`` is not a known transport.
    """
    connection_type = validate_connection_type(connection_type)
    manager = AlicatManager(plc_driver=plc_driver)

    try:
        ensure_gas_control_matches_plc(connection_type, plc_driver)
        for spec in DEFAULT_MFC_SPECS:
            manager.add_device(
                AlicatMFC(
                    device_id=str(spec["device_id"]),
                    name=str(spec["name"]),
                    gas=str(spec["gas"]),
                    max_flow=float(spec["max_flow"]),
                    connection_type=connection_type,
                    port_or_ip=port_or_ip,
                )
            )
    except ValueError as exc:
        manager.devices.clear()
        manager.registration_error = str(exc)
        logger.critical(
            "%s NO mass flow controllers were registered: gas control is "
            "UNAVAILABLE until the configuration is corrected.",
            exc,
        )
        return manager

    logger.info(
        "Registered %d Alicat MFCs on the %s transport (plc_driver=%s).",
        len(manager.devices),
        connection_type,
        plc_driver,
    )
    return manager
