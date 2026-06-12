import asyncio
import logging
import struct

from models import InterlockConfig, PIDConfig, RoutingConfig
from plc_interface import BasePLCClient
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


class AsyncModbusManager(BasePLCClient):
    """Manages asynchronous Modbus TCP communication with the P1AM PLC."""

    def __init__(self, host: str, port: int = 502) -> None:
        super().__init__()
        self.host = host
        self.port = port
        self.client: AsyncModbusTcpClient | None = None
        self._connected = False

        # Configurable hardware memory mappings for agnostic flexibility
        self.input_routing_address = 100
        self.output_routing_address = 110
        self.pid_config_address = 200
        self.interlock_config_address = 300
        self.save_to_flash_coil_address = 0
        self.estop_registers_address = 0
        self.estop_reset_coil_address = 1
        self.tag_value_registers_address = 0

    @property
    def connected(self) -> bool:
        return self._connected

    @connected.setter
    def connected(self, value: bool) -> None:
        self._connected = value

    def _get_client(self) -> AsyncModbusTcpClient:
        """Lazily initialize the AsyncModbusTcpClient instance."""
        if self.client is None:
            self.client = AsyncModbusTcpClient(host=self.host, port=self.port)
        return self.client

    async def connect(self) -> bool:
        """Attempt connection to the Modbus PLC."""
        try:
            connected = await self._get_client().connect()
            if connected:
                self._connected = True
                return True
        except Exception as e:
            logger.warning(f"Connection failed: {e}")
        self._connected = False
        return False

    async def connect_with_retry(self) -> bool:
        """Connect to the Modbus server with exponential backoff."""
        if self._connected:
            return True

        backoff = 0.5
        max_backoff = 10.0
        while not self._connected:
            try:
                logger.info(
                    f"Attempting connection to PLC at {self.host}:{self.port}..."
                )
                connected = await self.connect()
                if connected:
                    logger.info("Connected to PLC successfully.")
                    return True
            except Exception as e:
                logger.warning(f"Connection failed: {e}")

            logger.info(f"Retrying connection in {backoff:.2f} seconds...")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2.0, max_backoff)

        return False  # type: ignore[unreachable]

    async def disconnect(self) -> None:
        """Disconnect from the PLC."""
        async with self.lock:
            if self._connected and self.client is not None:
                self.client.close()
                self._connected = False
                logger.info("Disconnected from PLC.")

    async def read_tags(self) -> dict[str, float] | None:
        """Read 32 float tags from the PLC starting at tag_value_registers_address.

        Returns:
            dict[str, float]: Mapped tag values if successful, None otherwise.
        """
        async with self.lock:
            if not self._connected:
                return None

            try:
                # Read 64 registers (32 tags * 2 regs/tag)
                response = await self._get_client().read_holding_registers(
                    address=self.tag_value_registers_address,
                    count=64,
                )
                if response.isError():
                    logger.error(f"Modbus error reading tags: {response}")
                    self._connected = False
                    return None

                tags = {}
                for i in range(32):
                    low = response.registers[i * 2]
                    high = response.registers[i * 2 + 1]
                    tags[f"TAG_{i}"] = registers_to_float(low, high)
                return tags
            except (ModbusException, Exception) as e:
                logger.error(f"Exception during tag read: {e}")
                self._connected = False
                return None

    async def read_routing(self) -> RoutingConfig | None:
        """Read the active routing and PID parameters directly from PLC registers."""
        async with self.lock:
            if not self._connected:
                return None

            try:
                client = self._get_client()

                # 1. Read Input routing
                input_resp = await client.read_holding_registers(
                    address=self.input_routing_address, count=6
                )
                if input_resp.isError():
                    logger.error(f"Error reading input routing: {input_resp}")
                    return None

                # 2. Read Output routing
                output_resp = await client.read_holding_registers(
                    address=self.output_routing_address, count=2
                )
                if output_resp.isError():
                    logger.error(f"Error reading output routing: {output_resp}")
                    return None

                # 3. Read PID configs
                pid_resp = await client.read_holding_registers(
                    address=self.pid_config_address, count=40
                )
                if pid_resp.isError():
                    logger.error(f"Error reading PID configurations: {pid_resp}")
                    return None

                pids = []
                for i in range(4):
                    base = i * 10
                    pv = pid_resp.registers[base]
                    cv = pid_resp.registers[base + 1]
                    sp = registers_to_float(
                        pid_resp.registers[base + 2], pid_resp.registers[base + 3]
                    )
                    kp = registers_to_float(
                        pid_resp.registers[base + 4], pid_resp.registers[base + 5]
                    )
                    ki = registers_to_float(
                        pid_resp.registers[base + 6], pid_resp.registers[base + 7]
                    )
                    kd = registers_to_float(
                        pid_resp.registers[base + 8], pid_resp.registers[base + 9]
                    )
                    pids.append(
                        PIDConfig(
                            pv_tag=f"TAG_{pv}",
                            cv_tag=f"TAG_{cv}",
                            setpoint=sp,
                            kp=kp,
                            ki=ki,
                            kd=kd,
                        )
                    )

                # 4. Read Interlocks (4 limits x 32 tags = 256 regs total,
                # chunked under pymodbus's 125-reg single-request cap; chunk
                # size of 64 keeps every chunk tag-aligned at 8 tags x 8 regs).
                interlock_regs: list[int] = []
                for offset in (0, 64, 128, 192):
                    chunk_resp = await client.read_holding_registers(
                        address=self.interlock_config_address + offset,
                        count=64,
                    )
                    if chunk_resp.isError():
                        logger.error(
                            f"Error reading interlocks (chunk +{offset}): {chunk_resp}"
                        )
                        return None
                    interlock_regs.extend(chunk_resp.registers)

                interlocks = {}
                for i in range(32):
                    base = i * 8
                    lolo = registers_to_float(
                        interlock_regs[base], interlock_regs[base + 1]
                    )
                    low = registers_to_float(
                        interlock_regs[base + 2], interlock_regs[base + 3]
                    )
                    high = registers_to_float(
                        interlock_regs[base + 4], interlock_regs[base + 5]
                    )
                    hihi = registers_to_float(
                        interlock_regs[base + 6], interlock_regs[base + 7]
                    )
                    interlocks[f"TAG_{i}"] = InterlockConfig(
                        lolo_limit=lolo,
                        low_limit=low,
                        high_limit=high,
                        hihi_limit=hihi,
                    )

                return RoutingConfig(
                    input_routing=[f"TAG_{r}" for r in input_resp.registers],
                    output_routing=[f"TAG_{r}" for r in output_resp.registers],
                    pids=pids,
                    interlocks=interlocks,
                )

            except Exception as e:
                logger.error(f"Exception reading routing config from Modbus: {e}")
                self._connected = False
                return None

    async def write_routing(self, config: RoutingConfig) -> bool:
        """Write the routing matrix configuration to the PLC.

        Args:
            config: RoutingConfig configuration model.

        Returns:
            bool: True if writing succeeded, False otherwise.
        """
        async with self.lock:
            try:
                client = self._get_client()

                def _to_idx(t: str) -> int:
                    try:
                        return int(t.split("_")[1])
                    except (IndexError, ValueError):
                        return 0

                input_vals = [_to_idx(x) for x in config.input_routing]
                output_vals = [_to_idx(x) for x in config.output_routing]

                # 1. Write Input routing
                resp = await client.write_registers(
                    address=self.input_routing_address,
                    values=input_vals,
                )
                if resp.isError():
                    logger.error(f"Error writing input routing: {resp}")
                    return False

                # 2. Write Output routing
                resp = await client.write_registers(
                    address=self.output_routing_address,
                    values=output_vals,
                )
                if resp.isError():
                    logger.error(f"Error writing output routing: {resp}")
                    return False

                # 3. Write PID Config
                pid_regs = []
                for pid in config.pids:
                    pid_regs.append(_to_idx(pid.pv_tag))
                    pid_regs.append(_to_idx(pid.cv_tag))
                    pid_regs.extend(float_to_registers(pid.setpoint))
                    pid_regs.extend(float_to_registers(pid.kp))
                    pid_regs.extend(float_to_registers(pid.ki))
                    pid_regs.extend(float_to_registers(pid.kd))

                resp = await client.write_registers(
                    address=self.pid_config_address,
                    values=pid_regs,
                )
                if resp.isError():
                    logger.error(f"Error writing PID configs: {resp}")
                    return False

                # 4. Write Interlock Config
                interlock_regs = []
                for i in range(32):
                    tag_name = f"TAG_{i}"
                    interlock = config.interlocks.get(
                        tag_name,
                        InterlockConfig(
                            lolo_limit=0.0,
                            low_limit=5.0,
                            high_limit=95.0,
                            hihi_limit=100.0,
                        ),
                    )
                    interlock_regs.extend(float_to_registers(interlock.lolo_limit))
                    interlock_regs.extend(float_to_registers(interlock.low_limit))
                    interlock_regs.extend(float_to_registers(interlock.high_limit))
                    interlock_regs.extend(float_to_registers(interlock.hihi_limit))

                # write_multiple_registers (0x10) is capped at 123 regs per
                # request, so chunk the 256-reg interlock block into 4 writes
                # of 64 regs (tag-aligned).
                for offset in (0, 64, 128, 192):
                    resp = await client.write_registers(
                        address=self.interlock_config_address + offset,
                        values=interlock_regs[offset : offset + 64],
                    )
                    if resp.isError():
                        logger.error(
                            f"Error writing interlock configs (chunk +{offset}): {resp}"
                        )
                        return False

                logger.info(
                    "Successfully wrote all routing configs to Modbus registers."
                )
                return True

            except (ModbusException, Exception) as e:
                logger.error(f"Exception writing configuration to PLC: {e}")
                self._connected = False
                return False

    async def save_to_flash(self) -> bool:
        """Trigger the Save to Flash Modbus coil.

        Returns:
            bool: True if writing coil succeeded, False otherwise.
        """
        async with self.lock:
            if not self._connected:
                return False

            try:
                resp = await self._get_client().write_coil(
                    address=self.save_to_flash_coil_address,
                    value=True,
                )
                if resp.isError():
                    logger.error(f"Error writing Save to Flash coil: {resp}")
                    return False
                logger.info("Triggered Save to Flash Modbus Coil.")
                return True
            except (ModbusException, Exception) as e:
                logger.error(f"Exception saving config to PLC flash: {e}")
                self._connected = False
                return False

    async def trigger_estop(self) -> bool:
        """Write 0.0 to all tag values to force outputs to zero immediately."""
        async with self.lock:
            if not self._connected:
                return False
            try:
                zeros = []
                for _ in range(32):
                    zeros.extend(float_to_registers(0.0))
                resp = await self._get_client().write_registers(
                    address=self.estop_registers_address,
                    values=zeros,
                )
                if resp.isError():
                    logger.error(f"Error writing E-stop registers: {resp}")
                    return False
                logger.warning("E-stop command written to tag values successfully.")
                return True
            except (ModbusException, Exception) as e:
                logger.error(f"Exception during E-stop Modbus execution: {e}")
                self._connected = False
                return False

    async def clear_estop(self) -> bool:
        """Pulse the E-stop reset coil so the PLC leaves the latched trip state.

        The E-stop latch lives in the controller, not in this process. Clearing
        it therefore requires an explicit coil write; only a confirmed,
        non-error response means the plant has actually been released.

        Returns:
            bool: True if the reset coil write was acknowledged, False otherwise.
        """
        async with self.lock:
            if not self._connected:
                return False

            try:
                resp = await self._get_client().write_coil(
                    address=self.estop_reset_coil_address,
                    value=True,
                )
                if resp.isError():
                    logger.error(f"Error writing E-stop reset coil: {resp}")
                    return False
                logger.warning("E-stop reset coil written to PLC successfully.")
                return True
            except (ModbusException, Exception) as e:
                logger.error(f"Exception during E-stop reset Modbus execution: {e}")
                self._connected = False
                return False

    async def write_tag(self, tag_name: str, value: float) -> bool:
        """Write a 32-bit float directly to a tag register.

        Supports dynamic tags by name or fallback to 'TAG_idx' format.
        """
        tag_idx: int | None = None
        if tag_name.startswith("TAG_"):
            try:
                tag_idx = int(tag_name.split("_")[1])
            except (ValueError, IndexError):
                pass

        # Fallback register address or dynamically look up
        # from active_config if possible
        address = None

        if tag_idx is not None and (0 <= tag_idx < 32):
            address = tag_idx * 2
        elif hasattr(self, "tag_map") and tag_name in self.tag_map:
            tag_def = self.tag_map[tag_name]
            if tag_def.register_type == "V" and tag_def.register_num is not None:
                address = tag_def.register_num

        if address is None:
            logger.error(
                f"Invalid tag name or no mapped register for write: {tag_name}"
            )
            return False

        async with self.lock:
            if not self._connected:
                return False
            try:
                regs = float_to_registers(value)
                resp = await self._get_client().write_registers(
                    address=address,
                    values=regs,
                )
                if resp.isError():
                    logger.error(
                        f"Error writing to tag {tag_name} at register {address}: {resp}"
                    )
                    return False
                logger.info(
                    f"Directly wrote {value} to tag {tag_name} at register {address}."
                )
                return True
            except (ModbusException, Exception) as e:
                logger.error(f"Exception during direct tag write for {tag_name}: {e}")
                self._connected = False
                return False
