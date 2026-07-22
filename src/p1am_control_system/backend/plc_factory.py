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
        elif driver == "neural":
            # Add src to sys.path to allow importing plant_simulator
            from plant_simulator.neural_simulator_client import NeuralSimulatorClient

            return NeuralSimulatorClient()
        elif driver == "modbus":
            from modbus_client import AsyncModbusManager

            logger.info(
                "Instantiating Modbus PLC Client at %s:%s",
                settings.plc_ip,
                settings.plc_port,
            )
            return AsyncModbusManager(host=settings.plc_ip, port=settings.plc_port)
        elif driver in ("simulator", "simulated"):
            logger.info("Instantiating Simulated PLC Client")
            return SimulatedPLCClient()
        else:
            logger.warning(
                f"Unknown PLC_DRIVER '{driver}'. Defaulting to Simulated PLC Client."
            )
            return SimulatedPLCClient()
