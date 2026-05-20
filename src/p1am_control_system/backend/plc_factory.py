import logging
import os

from plc_interface import BasePLCClient
from simulator_client import SimulatedPLCClient

logger = logging.getLogger("dcs_backend.plc_factory")


class PLCFactory:
    """Factory class to resolve the active PLC client based on configuration."""

    @staticmethod
    def create_client() -> BasePLCClient:
        """Create and return a concrete PLC client based on PLC_DRIVER env var.

        Returns:
            BasePLCClient: Resolved concrete client instance.
        """
        driver = os.getenv("PLC_DRIVER", "simulated").lower()

        if driver == "p1am":
            from modbus_client import AsyncModbusManager

            return AsyncModbusManager(host="192.168.1.100")
        elif driver == "neural":
            # Add src to sys.path to allow importing plant_simulator
            from plant_simulator.neural_simulator_client import NeuralSimulatorClient

            return NeuralSimulatorClient()
        elif driver == "modbus":
            from modbus_client import AsyncModbusManager

            host = os.getenv("PLC_IP", "192.168.1.100")
            try:
                port = int(os.getenv("PLC_PORT", "502"))
            except ValueError:
                logger.warning("Invalid PLC_PORT configuration. Defaulting to 502.")
                port = 502
            logger.info(f"Instantiating Modbus PLC Client at {host}:{port}")
            return AsyncModbusManager(host=host, port=port)
        elif driver == "simulator":
            logger.info("Instantiating Simulated PLC Client")
            return SimulatedPLCClient()
        else:
            logger.warning(
                f"Unknown PLC_DRIVER '{driver}'. Defaulting to Simulated PLC Client."
            )
            return SimulatedPLCClient()
