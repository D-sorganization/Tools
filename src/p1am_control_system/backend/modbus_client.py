import asyncio
import logging

import hardware
from modbus_codec import (
    INTERLOCK_CHUNK_OFFSETS,
    decode_interlocks,
    decode_pid_configs,
    direct_tag_address,
    encode_interlocks,
    encode_pid_configs,
    encode_tag_indices,
    float_to_registers,
    registers_to_float,
)
from models import RoutingConfig
from plc_interface import BasePLCClient
from pymodbus.client import AsyncModbusTcpClient

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
        self.estop_reset_coil_address = hardware.ESTOP_RESET_COIL
        self.tag_value_registers_address = hardware.TAG_VALUE_BASE
        self.heater_relay_coil_address = hardware.HEATER_RELAY_COIL
        self.heartbeat_register_address = hardware.HOST_HEARTBEAT_REGISTER

        # Free-running 16-bit host-liveness counter (see write_heartbeat).
        self._heartbeat_counter = 0

        # Defense-in-depth E-stop interlock latch. When set, the low-level write
        # seams (write_coil / write_pid_setpoint) independently force any
        # *energizing* command to the safe OFF/0 direction regardless of what the
        # controller commanded. Deliberately separate from the controllers' own
        # latch so a single missed re-engage cannot re-energize an output.
        self._estop_active = False

    @property
    def estop_active(self) -> bool:
        """Whether the low-level write-seam E-stop interlock is latched."""
        return self._estop_active

    def set_estop_active(self, active: bool) -> None:
        """Set the low-level write-seam E-stop interlock latch.

        Precondition: ``active`` is exactly a bool (no truthy coercion, to catch
        caller bugs on this safety-critical seam).

        Raises:
            TypeError: if ``active`` is not a bool.
        """
        if not isinstance(active, bool):
            raise TypeError(f"active must be a bool, got {type(active).__name__}")
        self._estop_active = active

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
            except Exception as e:  # noqa: BLE001 - any I/O failure drops the connection; poll loop reconnects
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

        Defense-in-depth E-stop interlock: a routing deploy carries the PID
        block, setpoints included, so applying one while the write-seam latch is
        set would re-command an output the E-stop just zeroed. The deploy is
        refused outright rather than partially sanitised — configuration
        deployment is never urgent enough to race a tripped plant, and a partial
        write would leave the PLC holding a half-applied config (issue #4038).

        Args:
            config: RoutingConfig configuration model.

        Returns:
            bool: True if writing succeeded, False otherwise (including when
            refused because the E-stop write-seam latch is set).
        """
        if self._estop_active:
            logger.warning(
                "write_routing refused — E-stop interlock active; a routing "
                "deploy would re-command PID setpoints."
            )
            return False
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

            except Exception as e:  # noqa: BLE001 - any I/O failure drops the connection; poll loop reconnects
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
            except Exception as e:  # noqa: BLE001 - any I/O failure drops the connection; poll loop reconnects
                logger.error(f"Exception saving config to PLC flash: {e}")
                self._connected = False
                return False

    async def trigger_estop(self) -> bool:
        """De-energize the heater relay and zero every PID setpoint.

        Write order is a safety property, not a style choice. The heater relay
        coil (``hardware.HEATER_RELAY_COIL``) is the ONLY thing that commands
        the 110 V element, so it is opened FIRST — before any register write can
        consume the scan budget or fail. Previously this method never touched
        the coil at all, leaving the heater energized until the next
        ``TemperatureService.poll()``: seconds to tens of seconds later, or
        forever if the poll loop is wedged (issue #4000).

        The old 64-register write at ``TAG_VALUE_BASE`` is gone. It was a
        provable no-op: the firmware unconditionally rewrites registers 0..63
        from its broker at the end of every scan and ``SyncModbusToDCS()`` never
        reads that block back, so the host's zeros were overwritten within one
        scan and were never observed by anything.

        Best-effort and non-atomic by design: a partial kill is unsafe, so on a
        sub-write failure we do NOT early-return — every de-energizing write is
        attempted so the plant is driven down as far as possible, and ``False``
        is returned only after all of them so the caller (and the poll-loop
        re-assert) knows to retry.

        Returns:
            bool: True only when EVERY de-energizing write was acknowledged,
            including the heater relay coil. A caller must not report an E-stop
            as successful on ``False`` — the heater may still be closed.
        """
        async with self.lock:
            if not self._connected:
                return False
            all_ok = True
            try:
                client = self._get_client()
                # FIRST: drop the heater relay. Hard error if it is not acked.
                resp = await client.write_coil(
                    address=hardware.HEATER_RELAY_COIL, value=False
                )
                if resp.isError():
                    all_ok = False
                    logger.critical(
                        "E-stop: heater relay coil %d NOT acknowledged (%s) — "
                        "the 110 V element may still be energized.",
                        hardware.HEATER_RELAY_COIL,
                        resp,
                    )

                # Then zero every analog command (PID setpoint pair).
                for pid_index in range(hardware.PID_COUNT):
                    sp_addr = hardware.pid_setpoint_address(pid_index)
                    resp = await client.write_registers(
                        address=sp_addr, values=float_to_registers(0.0)
                    )
                    if resp.isError():
                        all_ok = False
                        logger.error(
                            f"E-stop: error zeroing PID {pid_index} setpoint: {resp}"
                        )

                if all_ok:
                    logger.warning("E-stop: heater relay open, PID setpoints zeroed.")
                else:
                    logger.error("E-stop: one or more kill writes FAILED — retry.")
                return all_ok
            except Exception as e:  # noqa: BLE001 - any I/O failure drops the connection; poll loop reconnects
                logger.error(f"Exception during E-stop Modbus execution: {e}")
                self._connected = False
                return False

    async def write_heartbeat(self) -> bool:
        """Bump the firmware's host-liveness register (see #3999).

        The firmware treats any CHANGE to ``hardware.HOST_HEARTBEAT_REGISTER``
        as proof the host is alive; the value itself carries no meaning, so this
        writes a free-running 16-bit counter. If the firmware sees neither a
        Modbus TCP connection nor a heartbeat change for
        ``hardware.HEARTBEAT_TIMEOUT_S`` it drives all analog outputs to 0 %,
        opens the heater relay, asserts Inhibit and holds the PID loops.

        Call once per successful scan. Deliberately NOT gated by the
        ``_estop_active`` write-seam latch: this is a liveness counter, not a
        plant output. Suppressing it during an operator E-stop would trip the
        firmware watchdog on a host that is in fact alive, and would mask a
        genuine host failure behind a deliberate operator action.

        Returns:
            bool: True if the heartbeat write was acknowledged.
        """
        if not self._connected:
            return False
        self._heartbeat_counter = (self._heartbeat_counter + 1) & 0xFFFF
        async with self.lock:
            try:
                resp = await self._get_client().write_registers(
                    address=hardware.HOST_HEARTBEAT_REGISTER,
                    values=[self._heartbeat_counter],
                )
                if resp.isError():
                    logger.error("Heartbeat write failed: %s", resp)
                    return False
                return True
            except Exception as exc:  # noqa: BLE001 - any I/O failure drops the connection; poll loop reconnects
                logger.error("Heartbeat write exception: %s", exc)
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
            except Exception as e:  # noqa: BLE001 - any I/O failure drops the connection; poll loop reconnects
                logger.error(f"Exception during E-stop reset Modbus execution: {e}")
                self._connected = False
                return False

    async def write_pid_setpoint(self, pid_index: int, value: float) -> bool:
        """Write a PID loop's setpoint register pair (AO pass-through command).

        Public command seam used by the power-supply service so it no longer
        reaches into this client's private connection/lock. Address and float
        encoding come from the shared hardware contract.

        Defense-in-depth E-stop interlock: when the write-seam latch is set
        (``set_estop_active(True)``) any *energizing* (non-zero) setpoint is
        forced to 0 here, independent of the controller's own logic. The
        de-energizing direction (0) is never blocked. A failed de-energize
        write (commanded 0 but the write raises/errors) is retried once and, if
        it still fails, surfaced as ``False`` so the caller escalates a comms
        alarm rather than leaving the output at its last commanded value.
        """
        if not self._connected:
            return False
        try:
            address = hardware.pid_setpoint_address(pid_index)
        except ValueError as exc:
            logger.warning("write_pid_setpoint: %s", exc)
            return False
        # Interlock: force an energizing command to 0 while E-stop is latched.
        if self._estop_active and value != 0.0:
            logger.warning(
                "write_pid_setpoint(%d, %f) forced to 0 — E-stop interlock active",
                pid_index,
                value,
            )
            value = 0.0
        deenergize = value == 0.0
        attempts = 2 if deenergize else 1
        async with self.lock:
            for attempt in range(attempts):
                try:
                    resp = await self._get_client().write_registers(
                        address=address, values=float_to_registers(value)
                    )
                    if not resp.isError():
                        return True
                    logger.error(
                        "write_pid_setpoint(%d, %f) failed: %s",
                        pid_index,
                        value,
                        resp,
                    )
                except Exception as exc:  # noqa: BLE001 - any I/O failure drops the connection; poll loop reconnects
                    logger.error(
                        "write_pid_setpoint(%d, %f) exception: %s",
                        pid_index,
                        value,
                        exc,
                    )
                    self._connected = False
                    if not deenergize:
                        return False
                if deenergize and attempt + 1 < attempts:
                    logger.error(
                        "write_pid_setpoint(%d, 0) de-energize FAILED — retrying",
                        pid_index,
                    )
            if deenergize:
                logger.error(
                    "write_pid_setpoint(%d, 0) de-energize FAILED after retry — "
                    "comms alarm",
                    pid_index,
                )
            return False

    async def write_coil(self, address: int, value: bool) -> bool:
        """Write a single discrete coil (public seam, e.g. the heater relay).

        Defense-in-depth E-stop interlock: when the write-seam latch is set
        (``set_estop_active(True)``) an *energizing* command (``value`` True) is
        forced to ``False`` here, independent of the controller's own logic. The
        de-energizing direction (``False``) is never blocked. A failed
        de-energize write (commanded ``False`` but the write raises/errors) is
        retried once and, if it still fails, surfaced as ``False`` so the caller
        escalates a comms alarm rather than leaving the relay in its last state.
        """
        if not self._connected:
            return False
        if not isinstance(address, int) or isinstance(address, bool):
            raise TypeError(f"address must be an int, got {type(address).__name__}")
        if not isinstance(value, bool):
            raise TypeError(f"value must be a bool, got {type(value).__name__}")
        # Interlock: force an energizing command off while E-stop is latched.
        if self._estop_active and value:
            logger.warning(
                "write_coil(%d, True) forced OFF — E-stop interlock active", address
            )
            value = False
        deenergize = value is False
        attempts = 2 if deenergize else 1
        async with self.lock:
            for attempt in range(attempts):
                try:
                    resp = await self._get_client().write_coil(
                        address=address, value=value
                    )
                    if not resp.isError():
                        return True
                    logger.error("write_coil(%d, %s) failed: %s", address, value, resp)
                except Exception as exc:  # noqa: BLE001 - any I/O failure drops the connection; poll loop reconnects
                    logger.error(
                        "write_coil(%d, %s) exception: %s", address, value, exc
                    )
                    self._connected = False
                    if not deenergize:
                        return False
                if deenergize and attempt + 1 < attempts:
                    logger.error(
                        "write_coil(%d, False) de-energize FAILED — retrying", address
                    )
            if deenergize:
                logger.error(
                    "write_coil(%d, False) de-energize FAILED after retry — "
                    "comms alarm",
                    address,
                )
            return False

    def _is_republished_tag_register(self, address: int) -> bool:
        """Whether ``address`` falls inside the firmware's broker-owned block.

        The firmware rewrites registers ``TAG_VALUE_BASE .. +TAG_COUNT*2`` from
        its tag broker at the end of every scan and never reads them back, so a
        host write there is overwritten within one scan and cannot reach the
        plant.
        """
        base: int = hardware.TAG_VALUE_BASE
        width: int = hardware.TAG_COUNT * 2
        return bool(base <= address < base + width)

    async def write_tag(self, tag_name: str, value: float) -> bool:
        """Write a 32-bit float directly to a tag register.

        Supports dynamic tags mapped to a real V register. ``TAG_n`` names are
        REFUSED: ``modbus_codec.direct_tag_address`` resolves them to holding
        register ``n*2``, inside the block the firmware republishes from its
        broker every scan and never reads back. Such a write cannot influence
        the plant, so claiming success for it is worse than failing — it let the
        API answer 200 for a command the plant never saw and let the PID
        auto-tuner fit gains to a step that never happened (issue #4015). A
        write seam that cannot write must not pretend otherwise, so this raises
        rather than returning a bool the caller may read as a transient fault.

        Defense-in-depth E-stop interlock: when the write-seam latch is set
        (``set_estop_active(True)``) an energizing (non-zero) value is forced to
        0 here, exactly as in ``write_coil`` / ``write_pid_setpoint``. This seam
        previously skipped the latch entirely (issue #4038).

        Raises:
            TypeError: If ``tag_name`` is not a str.
            NotImplementedError: If the tag resolves into the firmware's
                republished broker block, i.e. the write can never take effect.

        Returns:
            bool: True if the write was acknowledged by the PLC.
        """
        if not isinstance(tag_name, str):
            raise TypeError(f"tag_name must be a str, got {type(tag_name).__name__}")

        address = direct_tag_address(tag_name, self.tag_map)
        if address is None:
            logger.error(
                f"Invalid tag name or no mapped register for write: {tag_name}"
            )
            return False

        if self._is_republished_tag_register(address):
            raise NotImplementedError(
                f"write_tag({tag_name!r}) resolves to holding register {address}, "
                "inside the block the P1AM firmware republishes from its broker "
                "every scan and never reads back. This write cannot reach the "
                "plant. Use write_pid_setpoint (PID/AO command) or write_coil "
                "(discrete output) instead."
            )

        # Interlock: force an energizing command to 0 while E-stop is latched.
        if self._estop_active and value != 0.0:
            logger.warning(
                "write_tag(%s, %s) forced to 0 — E-stop interlock active",
                tag_name,
                value,
            )
            value = 0.0

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
            except Exception as e:  # noqa: BLE001 - any I/O failure drops the connection; poll loop reconnects
                logger.error(f"Exception during direct tag write for {tag_name}: {e}")
                self._connected = False
                return False
