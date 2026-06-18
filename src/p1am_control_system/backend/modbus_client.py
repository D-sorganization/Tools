import asyncio
import logging

import hardware
from modbus_codec import (
    INTERLOCK_CHUNK_OFFSETS,
    TAG_COUNT,
    decode_interlocks,
    decode_pid_configs,
    direct_tag_address,
    encode_interlocks,
    encode_pid_configs,
    encode_tag_indices,
    float_to_registers,
    registers_to_float,
    zero_float_registers,
)
from models import RoutingConfig
from plc_interface import BasePLCClient
from pymodbus.client import AsyncModbusTcpClient
from pymodbus.exceptions import ModbusException

logger = logging.getLogger("dcs_backend.modbus_client")


class AsyncModbusManager(BasePLCClient):
    """Manages asynchronous Modbus TCP communication with the P1AM PLC."""

    def __init__(self, host: str, port: int = 502) -> None:
        super().__init__()
        self.host = host
        self.port = port
        self.client: AsyncModbusTcpClient | None = None
        self._connected = False

        # Hardware register map — single source of truth in hardware.py.
        self.input_routing_address = hardware.INPUT_ROUTING_BASE
        self.output_routing_address = hardware.OUTPUT_ROUTING_BASE
        self.pid_config_address = hardware.PID_CONFIG_BASE
        self.interlock_config_address = hardware.INTERLOCK_BASE
        self.save_to_flash_coil_address = hardware.SAVE_TO_FLASH_COIL
        self.estop_registers_address = hardware.TAG_VALUE_BASE
        self.estop_reset_coil_address = hardware.ESTOP_RESET_COIL
        self.tag_value_registers_address = hardware.TAG_VALUE_BASE

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

                pids = decode_pid_configs(pid_resp.registers)

                # 4. Read Interlocks (4 limits x 32 tags = 256 regs total,
                # chunked under pymodbus's 125-reg single-request cap; chunk
                # size of 64 keeps every chunk tag-aligned at 8 tags x 8 regs).
                interlock_regs: list[int] = []
                for offset in INTERLOCK_CHUNK_OFFSETS:
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

                return RoutingConfig(
                    input_routing=[f"TAG_{r}" for r in input_resp.registers],
                    output_routing=[f"TAG_{r}" for r in output_resp.registers],
                    pids=pids,
                    interlocks=decode_interlocks(interlock_regs),
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

                input_vals = encode_tag_indices(config.input_routing)
                output_vals = encode_tag_indices(config.output_routing)

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
                pid_regs = encode_pid_configs(config.pids)

                resp = await client.write_registers(
                    address=self.pid_config_address,
                    values=pid_regs,
                )
                if resp.isError():
                    logger.error(f"Error writing PID configs: {resp}")
                    return False

                # 4. Write Interlock Config
                interlock_regs = encode_interlocks(config.interlocks)

                # write_multiple_registers (0x10) is capped at 123 regs per
                # request, so chunk the 256-reg interlock block into 4 writes
                # of 64 regs (tag-aligned).
                for offset in INTERLOCK_CHUNK_OFFSETS:
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
        """Zero PID setpoints and tag values so outputs go to zero and hold.

        Best-effort and non-atomic by design: a partial kill is unsafe, so on a
        sub-write failure we do NOT early-return — every register is attempted so
        the output is driven down as far as possible, and ``False`` is returned
        only after all writes so the caller (and the poll-loop re-assert) knows
        to retry. Returns True only when every write was acknowledged.
        """
        async with self.lock:
            if not self._connected:
                return False
            all_ok = True
            try:
                client = self._get_client()
                # PID setpoint is field 3 of each 10-register block.
                for pid_index in range(4):
                    sp_addr = self.pid_config_address + pid_index * 10 + 2
                    resp = await client.write_registers(
                        address=sp_addr, values=float_to_registers(0.0)
                    )
                    if resp.isError():
                        all_ok = False
                        logger.error(
                            f"E-stop: error zeroing PID {pid_index} setpoint: {resp}"
                        )

                # Zero all tag values (covers any directly-driven, non-PID tag).
                resp = await client.write_registers(
                    address=self.estop_registers_address,
                    values=zero_float_registers(TAG_COUNT),
                )
                if resp.isError():
                    all_ok = False
                    logger.error(f"Error writing E-stop registers: {resp}")

                if all_ok:
                    logger.warning("E-stop: PID setpoints and tag values zeroed.")
                else:
                    logger.error("E-stop: one or more zeroing writes FAILED — retry.")
                return all_ok
            except (ModbusException, Exception) as e:
                logger.error(f"Exception during E-stop Modbus execution: {e}")
                self._connected = False
                return False

    async def clear_estop(self) -> bool:
        """Write the E-stop reset coil and return whether the write was accepted.

        NOTE: the current firmware (firmware.ino) only acts on coil 0
        (save-to-flash) and ignores coil 1, so this write is effectively a no-op
        on the device today — the real reset is the controller un-latch in the
        backend plus the operator re-arm. The write is kept (and succeeds at the
        Modbus level) so a future firmware that honors a host reset coil works
        without a backend change. If that firmware treats the coil as
        level-sensitive, add a write-back to False here to make it a true pulse.
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

    async def write_pid_setpoint(self, pid_index: int, value: float) -> bool:
        """Write a PID loop's setpoint register pair (AO pass-through command).

        Public command seam used by the power-supply service so it no longer
        reaches into this client's private connection/lock. Address and float
        encoding come from the shared hardware contract.
        """
        if not self._connected:
            return False
        try:
            address = hardware.pid_setpoint_address(pid_index)
        except ValueError as exc:
            logger.warning("write_pid_setpoint: %s", exc)
            return False
        async with self.lock:
            try:
                resp = await self._get_client().write_registers(
                    address=address, values=float_to_registers(value)
                )
                if resp.isError():
                    logger.error(
                        "write_pid_setpoint(%d, %f) failed: %s", pid_index, value, resp
                    )
                    return False
                return True
            except (ModbusException, Exception) as exc:
                logger.error(
                    "write_pid_setpoint(%d, %f) exception: %s", pid_index, value, exc
                )
                self._connected = False
                return False

    async def write_tag(self, tag_name: str, value: float) -> bool:
        """Write a 32-bit float directly to a tag register.

        Supports dynamic tags by name or fallback to 'TAG_idx' format.
        """
        address = direct_tag_address(tag_name, self.tag_map)
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
