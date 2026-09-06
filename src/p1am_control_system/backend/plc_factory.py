import logging

from plc_interface import BasePLCClient
from settings import P1AMSettings
from simulator_client import SimulatedPLCClient

logger = logging.getLogger("dcs_backend.plc_factory")


class PLCFactory:
    """Factory class to resolve the active PLC client based on configuration."""

    @staticmethod
    def create_client(settings: P1AMSettings | None = None) -> BasePLCClient:
        """Create and return a concrete PLC client based on Settings.

        Returns:
            BasePLCClient: Resolved concrete client instance.
        """
        settings = settings or P1AMSettings()
        driver = settings.plc_driver

        if driver == "p1am":
            from modbus_client import AsyncModbusManager

            return AsyncModbusManager(host=settings.plc_ip, port=settings.plc_port)
        elif driver == "modbus":
            from modbus_client import AsyncModbusManager

            logger.info(
                "Instantiating Modbus PLC Client at %s:%s",
                settings.plc_ip,
                settings.plc_port,
            )
            return AsyncModbusManager(host=settings.plc_ip, port=settings.plc_port)
        elif driver in ("simulator", "simulated"):
            PLCFactory._warn_simulated(driver, explicit=True)
            return SimulatedPLCClient()
        else:
            PLCFactory._warn_simulated(driver, explicit=False)
            return SimulatedPLCClient()

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
