import asyncio
import logging
import struct

from models import RoutingConfig
from pymodbus.client import AsyncModbusTcpClient
from pymodbus.exceptions import ModbusException

logger = logging.getLogger("dcs_backend.modbus_client")


def registers_to_float(low: int, high: int) -> float:
    """Convert two 16-bit registers to a 32-bit float (IEEE-754)."""
    try:
        packed = struct.pack("<HH", low, high)
        return float(struct.unpack("<f", packed)[0])
    except Exception as e:
        logger.error(f"Error unpacking registers to float: {e}")
        return 0.0


def float_to_registers(val: float) -> list[int]:
    """Convert a 32-bit float to two 16-bit registers (IEEE-754)."""
    try:
        packed = struct.pack("<f", val)
        return list(struct.unpack("<HH", packed))
    except Exception as e:
        logger.error(f"Error packing float to registers: {e}")
        return [0, 0]


class AsyncModbusManager:
    """Manages asynchronous Modbus TCP communication with the P1AM PLC."""

    def __init__(self, host: str, port: int = 502) -> None:
        self.host = host
        self.port = port
        self.client: AsyncModbusTcpClient | None = None
        self.connected = False
        self.lock = asyncio.Lock()

    def _get_client(self) -> AsyncModbusTcpClient:
        """Lazily initialize the AsyncModbusTcpClient instance."""
        if self.client is None:
            self.client = AsyncModbusTcpClient(host=self.host, port=self.port)
        return self.client

    async def connect_with_retry(self) -> bool:
        """Connect to the Modbus server with exponential backoff."""
        if self.connected:
            return True

        backoff = 0.5
        max_backoff = 10.0
        while not self.connected:
            try:
                logger.info(
                    f"Attempting connection to PLC at {self.host}:{self.port}..."
                )
                connected = await self._get_client().connect()
                if connected:
                    self.connected = True
                    logger.info("Connected to PLC successfully.")
                    return True
            except Exception as e:
                logger.warning(f"Connection failed: {e}")

            logger.info(f"Retrying connection in {backoff:.2f} seconds...")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2.0, max_backoff)

        return False

    async def disconnect(self) -> None:
        """Disconnect from the PLC."""
        async with self.lock:
            if self.connected and self.client is not None:
                self.client.close()
                self.connected = False
                logger.info("Disconnected from PLC.")

    async def read_tags(self) -> list[float] | None:
        """Read 32 float tags (holding registers 0-63) from the PLC.

        Returns:
            List[float]: Mapped tag values if successful, None otherwise.
        """
        async with self.lock:
            if not self.connected:
                return None

            try:
                # Read 64 registers (32 tags * 2 regs/tag)
                response = await self._get_client().read_holding_registers(
                    address=0,
                    count=64,
                )
                if response.isError():
                    logger.error(f"Modbus error reading tags: {response}")
                    self.connected = False
                    return None

                tags = []
                for i in range(32):
                    low = response.registers[i * 2]
                    high = response.registers[i * 2 + 1]
                    tags.append(registers_to_float(low, high))
                return tags
            except (ModbusException, Exception) as e:
                logger.error(f"Exception during tag read: {e}")
                self.connected = False
                return None

    async def write_routing(self, config: RoutingConfig) -> bool:
        """Write the routing matrix configuration to the PLC.

        Args:
            config: RoutingConfig configuration model.

        Returns:
            bool: True if writing succeeded, False otherwise.
        """
        async with self.lock:
            if not self.connected:
                logger.error("Cannot write routing: Modbus client disconnected.")
                return False

            try:
                client = self._get_client()

                # 1. Write Input routing (registers 100-105)
                resp = await client.write_registers(
                    address=100,
                    values=config.input_routing,
                )
                if resp.isError():
                    logger.error(f"Error writing input routing: {resp}")
                    return False

                # 2. Write Output routing (registers 110-111)
                resp = await client.write_registers(
                    address=110,
                    values=config.output_routing,
                )
                if resp.isError():
                    logger.error(f"Error writing output routing: {resp}")
                    return False

                # 3. Write PID Config (registers 200-239)
                pid_regs = []
                for pid in config.pids:
                    pid_regs.append(pid.pv_tag_id)
                    pid_regs.append(pid.cv_tag_id)
                    pid_regs.extend(float_to_registers(pid.setpoint))
                    pid_regs.extend(float_to_registers(pid.kp))
                    pid_regs.extend(float_to_registers(pid.ki))
                    pid_regs.extend(float_to_registers(pid.kd))

                resp = await client.write_registers(
                    address=200,
                    values=pid_regs,
                )
                if resp.isError():
                    logger.error(f"Error writing PID configs: {resp}")
                    return False

                # 4. Write Interlock Config (registers 300-427)
                interlock_regs = []
                for interlock in config.interlocks:
                    interlock_regs.extend(float_to_registers(interlock.high_limit))
                    interlock_regs.extend(float_to_registers(interlock.low_limit))

                resp = await client.write_registers(
                    address=300,
                    values=interlock_regs,
                )
                if resp.isError():
                    logger.error(f"Error writing interlock configs: {resp}")
                    return False

                logger.info(
                    "Successfully wrote all routing configs to Modbus registers."
                )
                return True

            except (ModbusException, Exception) as e:
                logger.error(f"Exception writing configuration to PLC: {e}")
                self.connected = False
                return False

    async def save_to_flash(self) -> bool:
        """Trigger the Save to Flash Modbus coil (Coil 0).

        Returns:
            bool: True if writing coil succeeded, False otherwise.
        """
        async with self.lock:
            if not self.connected:
                return False

            try:
                # Write Coil 0 to True
                resp = await self._get_client().write_coil(address=0, value=True)
                if resp.isError():
                    logger.error(f"Error writing Save to Flash coil: {resp}")
                    return False
                logger.info("Triggered Save to Flash Modbus Coil.")
                return True
            except (ModbusException, Exception) as e:
                logger.error(f"Exception saving config to PLC flash: {e}")
                self.connected = False
                return False

    async def trigger_estop(self) -> bool:
        """Write 0.0 to all tag values to force outputs to zero immediately."""
        async with self.lock:
            if not self.connected:
                return False
            try:
                # To E-stop, we can write 0.0 to all tag value registers (0 to 63)
                zeros = []
                for _ in range(32):
                    zeros.extend(float_to_registers(0.0))
                resp = await self._get_client().write_registers(
                    address=0,
                    values=zeros,
                )
                if resp.isError():
                    logger.error(f"Error writing E-stop registers: {resp}")
                    return False
                logger.warning("E-stop command written to tag values successfully.")
                return True
            except (ModbusException, Exception) as e:
                logger.error(f"Exception during E-stop Modbus execution: {e}")
                self.connected = False
                return False

    async def write_tag(self, tag_id: int, value: float) -> bool:
        """Write a 32-bit float directly to a tag register.

        Holding registers starts at tag_id * 2.
        """
        if not (0 <= tag_id < 32):
            logger.error(f"Invalid tag_id for manual write: {tag_id}")
            return False

        async with self.lock:
            if not self.connected:
                return False
            try:
                regs = float_to_registers(value)
                resp = await self._get_client().write_registers(
                    address=tag_id * 2,
                    values=regs,
                )
                if resp.isError():
                    logger.error(f"Error writing to tag {tag_id} registers: {resp}")
                    return False
                logger.info(f"Directly wrote {value} to tag {tag_id} registers.")
                return True
            except (ModbusException, Exception) as e:
                logger.error(f"Exception during direct tag write: {e}")
                self.connected = False
                return False
