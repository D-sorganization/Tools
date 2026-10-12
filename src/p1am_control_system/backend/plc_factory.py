from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from plc_interface import BasePLCClient
from settings import P1AMSettings
from simulator_client import SimulatedPLCClient

logger = logging.getLogger("dcs_backend.plc_factory")


class PLCDriverError(Exception):
    """A PLC driver could not be registered, looked up or constructed."""


@dataclass(frozen=True)
class DriverCapabilities:
    """What a registered PLC driver is, declared once at registration.

    Attributes:
        real_hardware: True when the driver talks to field hardware. Drivers
            with ``real_hardware=False`` are simulators and get the unmissable
            simulated-plant banner on selection.
    """

    real_hardware: bool


DriverFactory = Callable[[P1AMSettings], BasePLCClient]

# name -> (factory, capabilities). Mutated only by ``register_driver``, which is
# deliberately code-only: it is not exposed to any HTTP route or config input.
_DRIVERS: dict[str, tuple[DriverFactory, DriverCapabilities]] = {}


def _normalize(name: object) -> str:
    if not isinstance(name, str):
        raise TypeError(f"driver name must be a str, got {type(name).__name__}")
    key = name.strip().lower()
    if not key:
        raise PLCDriverError("driver name must be a non-empty string")
    return key


def register_driver(
    name: str, factory: DriverFactory, capabilities: DriverCapabilities
) -> None:
    """Register a PLC driver under ``name`` (code-only; never from a request).

    Raises:
        TypeError: ``name`` is not a str, ``factory`` is not callable or
            ``capabilities`` is not a ``DriverCapabilities``.
        PLCDriverError: ``name`` is empty or already registered.
    """
    key = _normalize(name)
    if not callable(factory):
        raise TypeError("driver factory must be callable")
    if not isinstance(capabilities, DriverCapabilities):
        raise TypeError("capabilities must be a DriverCapabilities")
    if key in _DRIVERS:
        raise PLCDriverError(f"PLC driver {key!r} is already registered")
    _DRIVERS[key] = (factory, capabilities)


def registered_drivers() -> dict[str, DriverCapabilities]:
    """Return a snapshot of registered driver names and their capabilities."""
    return {name: caps for name, (_f, caps) in _DRIVERS.items()}


def get_driver(name: str) -> tuple[DriverFactory, DriverCapabilities]:
    """Look up a registered driver.

    Raises:
        PLCDriverError: ``name`` is not registered; the message lists the
            registered drivers.
    """
    key = _normalize(name)
    try:
        return _DRIVERS[key]
    except KeyError:
        raise PLCDriverError(
            f"unknown PLC driver {key!r}; registered drivers: "
            f"{', '.join(sorted(_DRIVERS))}"
        ) from None


def _modbus_factory(settings: P1AMSettings) -> BasePLCClient:
    from modbus_client import AsyncModbusManager

    logger.info(
        "Instantiating Modbus PLC Client at %s:%s",
        settings.plc_ip,
        settings.plc_port,
    )
    return AsyncModbusManager(host=settings.plc_ip, port=settings.plc_port)


def _p1am_factory(settings: P1AMSettings) -> BasePLCClient:
    from modbus_client import AsyncModbusManager

    return AsyncModbusManager(host=settings.plc_ip, port=settings.plc_port)


def _simulator_factory(settings: P1AMSettings) -> BasePLCClient:
    return SimulatedPLCClient()


_REAL = DriverCapabilities(real_hardware=True)
_SIM = DriverCapabilities(real_hardware=False)
register_driver("p1am", _p1am_factory, _REAL)
register_driver("modbus", _modbus_factory, _REAL)
register_driver("simulator", _simulator_factory, _SIM)
register_driver("simulated", _simulator_factory, _SIM)


class PLCFactory:
    """Factory class to resolve the active PLC client based on configuration."""

    @staticmethod
    def create_client(settings: P1AMSettings | None = None) -> BasePLCClient:
        """Create and return a concrete PLC client based on Settings.

        An unrecognised driver name falls back to the simulator with a loud
        banner (unchanged behaviour). A *registered* driver that fails to
        construct raises ``PLCDriverError`` -- it never degrades to the
        simulator.

        Returns:
            BasePLCClient: Resolved concrete client instance.

        Raises:
            PLCDriverError: A registered driver's factory raised.
        """
        settings = settings or P1AMSettings()
        driver = settings.plc_driver
        try:
            factory, caps = get_driver(driver)
        except PLCDriverError:
            PLCFactory._warn_simulated(driver, explicit=False)
            return SimulatedPLCClient()
        try:
            client = factory(settings)
        except Exception as exc:  # noqa: BLE001 - re-raised with driver context
            raise PLCDriverError(
                f"PLC driver {driver!r} failed to construct: {exc}"
            ) from exc
        if not caps.real_hardware:
            PLCFactory._warn_simulated(driver, explicit=True)
        return client

    @staticmethod
    def _warn_simulated(driver: str, *, explicit: bool) -> None:
        """Log an unmissable banner whenever the plant is being simulated.

        A simulated client produces live-looking, fabricated process values that
        an operator cannot distinguish from the real plant on the HMI. That is
        fine on a bench and dangerous anywhere else — and it is exactly what a
        misconfiguration produces, because an unrecognised (or missing) driver
        name silently falls through to the simulator (issue #4030/#4036). One
        ``INFO`` line was far too quiet for that; this is a banner in the boot
        log a human scanning ``journalctl`` cannot miss.
        """
        reason = (
            f"PLC_DRIVER={driver!r} selected the simulator"
            if explicit
            else f"PLC_DRIVER={driver!r} is not a known driver; FELL BACK to "
            "the simulator"
        )
        logger.warning(
            "\n"
            "================================================================\n"
            "  SIMULATED PLC — NO REAL HARDWARE IS CONNECTED\n"
            "  %s.\n"
            "  Every process value the HMI displays is FABRICATED.\n"
            "  For the real plant set P1AM_PLC_DRIVER=modbus and P1AM_PLC_IP.\n"
            "================================================================",
            reason,
        )
