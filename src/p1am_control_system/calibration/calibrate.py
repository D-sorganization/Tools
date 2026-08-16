"""Interactive helper for P1AM-100 analog I/O calibration.

See CALIBRATION.md for the full procedure. This script wraps the Modbus
register layout in a CLI so the operator does not have to remember register
addresses, byte order, or the PID-as-pass-through workaround used to drive
AOs from the host.
"""

from __future__ import annotations

import argparse
import logging
import struct
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass

from pymodbus.client import ModbusTcpClient

logger = logging.getLogger("p1am.calibration")

PLC_HOST_DEFAULT = "192.168.1.100"
PLC_PORT_DEFAULT = 502

REG_INPUT_ROUTING_BASE = 100
REG_OUTPUT_ROUTING_BASE = 110
REG_PID_BASE = 200
REG_PID_STRIDE = 10
COIL_SAVE_TO_FLASH = 0
UNMAPPED_TAG = 255

INPUT_SLOT_LABELS = ["TC0", "TC1", "TC2", "TC3", "AI0", "AI1"]
AI_INPUT_SLOT = {0: 4, 1: 5}

AO_TAG = {0: 10, 1: 11}
AI_TAG = {0: 12, 1: 13}
PID_FOR_AO = {0: 0, 1: 1}
PID_PV_TAG = {0: 30, 1: 31}
TAG_DWELL_SECONDS = 0.25
SWEEP_DWELL_SECONDS = 1.0

# An AO is treated as "off" once its tag reads within this many percent of 0
# (0 % == 4 mA). 0.5 pp is ~0.08 mA — well inside the module's own resolution.
AO_ZERO_TOLERANCE_PERCENT = 0.5
# The firmware writes outputs once per scan, so give it a few scans to settle
# before declaring a channel stuck.
AO_ZERO_CONFIRM_ATTEMPTS = 4

# Methods a PLC-like object must expose for the safe-shutdown path to run.
_ZERO_REQUIRED_METHODS = ("set_pid_setpoint", "read_tag")


def _float_to_regs(value: float) -> list[int]:
    lo, hi = struct.unpack("<HH", struct.pack("<f", value))
    return [lo, hi]


def _regs_to_float(lo: int, hi: int) -> float:
    return float(struct.unpack("<f", struct.pack("<HH", lo, hi))[0])


def _percent_to_ma(percent: float) -> float:
    return 4.0 + 0.16 * percent


def _percent_to_volts(percent: float) -> float:
    return 0.05 * percent


@dataclass
class RoutingSnapshot:
    inputs: list[int]
    outputs: list[int]


class PLC:
    """Thin synchronous wrapper around ModbusTcpClient for calibration ops.

    All operations raise SystemExit on Modbus errors so the CLI exits with a
    nonzero status instead of continuing in a half-configured state.
    """

    def __init__(self, host: str, port: int) -> None:
        self._client = ModbusTcpClient(host, port=port, timeout=3)
        self._host = host
        self._port = port

    def connect(self) -> None:
        if not self._client.connect():
            raise SystemExit(f"Cannot connect to PLC at {self._host}:{self._port}")

    def close(self) -> None:
        self._client.close()

    def write_register(self, address: int, value: int) -> None:
        resp = self._client.write_register(address=address, value=value)
        if resp.isError():
            raise SystemExit(f"write_register({address}, {value}) failed: {resp}")

    def write_registers(self, address: int, values: Sequence[int]) -> None:
        resp = self._client.write_registers(address=address, values=list(values))
        if resp.isError():
            raise SystemExit(
                f"write_registers({address}, {list(values)}) failed: {resp}"
            )

    def read_holding(self, address: int, count: int) -> list[int]:
        resp = self._client.read_holding_registers(address=address, count=count)
        if resp.isError():
            raise SystemExit(
                f"read_holding_registers({address}, count={count}) failed: {resp}"
            )
        return list(resp.registers)

    def write_coil(self, address: int, value: bool) -> None:
        resp = self._client.write_coil(address=address, value=value)
        if resp.isError():
            raise SystemExit(f"write_coil({address}, {value}) failed: {resp}")

    def read_tag(self, tag_id: int) -> float:
        regs = self.read_holding(tag_id * 2, 2)
        return _regs_to_float(regs[0], regs[1])

    def read_routing(self) -> RoutingSnapshot:
        return RoutingSnapshot(
            inputs=self.read_holding(REG_INPUT_ROUTING_BASE, 6),
            outputs=self.read_holding(REG_OUTPUT_ROUTING_BASE, 2),
        )

    def set_output_routing(self, channel: int, tag_id: int) -> None:
        self.write_register(REG_OUTPUT_ROUTING_BASE + channel, tag_id)

    def set_input_routing(self, slot: int, tag_id: int) -> None:
        self.write_register(REG_INPUT_ROUTING_BASE + slot, tag_id)

    def configure_pid(
        self,
        pid_index: int,
        pv_tag: int,
        cv_tag: int,
        setpoint: float,
        kp: float,
        ki: float,
        kd: float,
    ) -> None:
        base = REG_PID_BASE + pid_index * REG_PID_STRIDE
        self.write_register(base, pv_tag)
        self.write_register(base + 1, cv_tag)
        self.write_registers(base + 2, _float_to_regs(setpoint))
        self.write_registers(base + 4, _float_to_regs(kp))
        self.write_registers(base + 6, _float_to_regs(ki))
        self.write_registers(base + 8, _float_to_regs(kd))

    def set_pid_setpoint(self, pid_index: int, setpoint: float) -> None:
        base = REG_PID_BASE + pid_index * REG_PID_STRIDE
        self.write_registers(base + 2, _float_to_regs(setpoint))


def _label_for_tag(tag_id: int) -> str:
    return f"TAG_{tag_id}" if tag_id != UNMAPPED_TAG else "(unmapped)"


def cmd_status(_: argparse.Namespace, plc: PLC) -> None:
    routing = plc.read_routing()
    logger.info("Input routing (slot -> tag):")
    for slot, tag in enumerate(routing.inputs):
        label = INPUT_SLOT_LABELS[slot] if slot < len(INPUT_SLOT_LABELS) else f"#{slot}"
        logger.info("  slot %d (%s): %s", slot, label, _label_for_tag(tag))
    logger.info("Output routing (AO channel -> tag):")
    for ch, tag in enumerate(routing.outputs):
        logger.info("  AO %d: %s", ch, _label_for_tag(tag))
    logger.info("Calibration tag readings:")
    for ch, tag in AO_TAG.items():
        value = plc.read_tag(tag)
        logger.info(
            "  AO %d <- TAG_%d = %6.3f%%  (~%5.3f mA expected)",
            ch,
            tag,
            value,
            _percent_to_ma(value),
        )
    for ch, tag in AI_TAG.items():
        value = plc.read_tag(tag)
        logger.info(
            "  AI %d -> TAG_%d = %6.3f%%  (~%5.3f mA at terminal)",
            ch,
            tag,
            value,
            _percent_to_ma(value),
        )


def cmd_setup(_: argparse.Namespace, plc: PLC) -> None:
    logger.info("Configuring output routing...")
    for ch, tag in AO_TAG.items():
        plc.set_output_routing(ch, tag)
        logger.info("  AO %d <- TAG_%d", ch, tag)

    logger.info("Configuring input routing...")
    for ch, tag in AI_TAG.items():
        slot = AI_INPUT_SLOT[ch]
        plc.set_input_routing(slot, tag)
        logger.info("  AI %d (slot %d) -> TAG_%d", ch, slot, tag)

    logger.info("Configuring pass-through PIDs for AO drive...")
    for ch, pid_index in PID_FOR_AO.items():
        cv_tag = AO_TAG[ch]
        pv_tag = PID_PV_TAG[ch]
        plc.configure_pid(
            pid_index=pid_index,
            pv_tag=pv_tag,
            cv_tag=cv_tag,
            setpoint=0.0,
            kp=1.0,
            ki=0.0,
            kd=0.0,
        )
        logger.info(
            "  PID %d: pv=TAG_%d (unused) cv=TAG_%d sp=0 kp=1.0 ki=0 kd=0",
            pid_index,
            pv_tag,
            cv_tag,
        )
    logger.info("Setup complete. AOs are at 0%s (4 mA).", "%")


def _require_zeroable(plc: object) -> None:
    """Precondition check for the safe-shutdown path (DbC).

    Raises:
        TypeError: If ``plc`` does not expose the calibration command surface
            (``set_pid_setpoint`` and ``read_tag``).
    """
    missing = [
        name
        for name in _ZERO_REQUIRED_METHODS
        if not callable(getattr(plc, name, None))
    ]
    if missing:
        raise TypeError(
            f"plc must expose callable {', '.join(_ZERO_REQUIRED_METHODS)}; "
            f"missing {', '.join(missing)}"
        )


def drive_outputs_to_zero(plc: PLC) -> list[tuple[int, float]]:
    """Command every calibration AO to 0 % and read the tag back to confirm.

    The firmware keeps writing a routed tag's value every scan, so simply
    unmapping the pass-through PID freezes the AO at its last commanded value
    instead of releasing it (issue #3997). The commanded zero must therefore be
    *confirmed* at the tag before anything is unmapped.

    Preconditions:
        ``plc`` is connected and exposes ``set_pid_setpoint``/``read_tag``.

    Returns:
        ``(channel, last_reading)`` for every channel that did NOT confirm 0 %
        within :data:`AO_ZERO_CONFIRM_ATTEMPTS`. An empty list means every AO
        is confirmed off.

    Raises:
        TypeError: If ``plc`` lacks the calibration command surface.
    """
    _require_zeroable(plc)

    unconfirmed: list[tuple[int, float]] = []
    for channel, pid_index in PID_FOR_AO.items():
        ao_tag = AO_TAG[channel]
        plc.set_pid_setpoint(pid_index, 0.0)
        actual = float("nan")
        for _ in range(AO_ZERO_CONFIRM_ATTEMPTS):
            time.sleep(TAG_DWELL_SECONDS)
            actual = plc.read_tag(ao_tag)
            if abs(actual) <= AO_ZERO_TOLERANCE_PERCENT:
                break
        else:
            unconfirmed.append((channel, actual))
            logger.error(
                "AO %d (TAG_%d) did not reach 0%s after %d attempts; last read "
                "%.3f%s (~%.3f mA)",
                channel,
                ao_tag,
                "%",
                AO_ZERO_CONFIRM_ATTEMPTS,
                actual,
                "%",
                _percent_to_ma(actual),
            )
            continue
        logger.info("AO %d (TAG_%d) confirmed at %.3f%s", channel, ao_tag, actual, "%")
    return unconfirmed


def zero_analog_outputs(plc: PLC) -> None:
    """Drive every calibration AO to 0 % and fail loudly if one will not go.

    Raises:
        TypeError: If ``plc`` lacks the calibration command surface.
        SystemExit: If any AO could not be confirmed at 0 %.
    """
    unconfirmed = drive_outputs_to_zero(plc)
    if unconfirmed:
        raise SystemExit(_unconfirmed_message(unconfirmed))


def _unconfirmed_message(unconfirmed: Sequence[tuple[int, float]]) -> str:
    detail = ", ".join(f"AO {ch} last read {value:.3f}%" for ch, value in unconfirmed)
    return (
        f"Analog outputs did not confirm 0% ({detail}). The channels were "
        "unmapped so the firmware drives them to 0% itself — VERIFY AT THE "
        "TERMINALS before energizing the rig."
    )


def _unmap_calibration_pids(plc: PLC) -> None:
    logger.info("Unmapping calibration PIDs...")
    for pid_index in PID_FOR_AO.values():
        plc.configure_pid(
            pid_index=pid_index,
            pv_tag=UNMAPPED_TAG,
            cv_tag=UNMAPPED_TAG,
            setpoint=0.0,
            kp=0.0,
            ki=0.0,
            kd=0.0,
        )
        logger.info("  PID %d: pv=cv=unmapped, gains=0", pid_index)


def _release_output_routing(plc: PLC) -> None:
    """Unmap the AO channels so the firmware's own 0 % safe path takes over.

    ``SignalBroker::WriteHardwareOutputs`` writes the routed *tag* while a
    channel is mapped and only calls ``WriteAnalogOutput(i, 0.0f)`` once the
    CHANNEL itself is unmapped. Releasing the routing is therefore the only
    state in which a stale tag cannot re-energize the output.
    """
    logger.info("Releasing AO output routing...")
    for channel in AO_TAG:
        plc.set_output_routing(channel, UNMAPPED_TAG)
        logger.info("  AO %d: unmapped (firmware holds it at 0%s)", channel, "%")


def cmd_teardown(_: argparse.Namespace, plc: PLC) -> None:
    logger.info("Driving calibration AOs to 0%s before releasing them...", "%")
    unconfirmed = drive_outputs_to_zero(plc)
    _unmap_calibration_pids(plc)
    _release_output_routing(plc)
    if unconfirmed:
        raise SystemExit(_unconfirmed_message(unconfirmed))
    logger.info(
        "Teardown complete. AOs confirmed at 0%s (4 mA) and unmapped; run "
        "`setup` (or deploy a routing config from the backend) before driving "
        "them again.",
        "%",
    )


def cmd_ao(args: argparse.Namespace, plc: PLC) -> None:
    if args.channel not in AO_TAG:
        raise SystemExit(f"AO channel must be 0 or 1; got {args.channel}")
    if not 0.0 <= args.percent <= 100.0:
        raise SystemExit(f"--percent must be in [0, 100]; got {args.percent}")

    pid_index = PID_FOR_AO[args.channel]
    plc.set_pid_setpoint(pid_index, args.percent)
    time.sleep(TAG_DWELL_SECONDS)
    actual = plc.read_tag(AO_TAG[args.channel])
    expected_ma = _percent_to_ma(args.percent)
    expected_v = _percent_to_volts(args.percent)
    logger.info(
        "AO %d set to %.2f%% (TAG_%d reads %.3f%%) -> expect %.3f mA, "
        "%.3f V at calibrated sig-cond output",
        args.channel,
        args.percent,
        AO_TAG[args.channel],
        actual,
        expected_ma,
        expected_v,
    )


def cmd_ai(args: argparse.Namespace, plc: PLC) -> None:
    if args.channel not in AI_TAG:
        raise SystemExit(f"AI channel must be 0 or 1; got {args.channel}")
    tag = AI_TAG[args.channel]
    value = plc.read_tag(tag)
    logger.info(
        "AI %d -> TAG_%d = %.3f%%  (~%.3f mA at PLC AI terminal)",
        args.channel,
        tag,
        value,
        _percent_to_ma(value),
    )


def cmd_sweep(args: argparse.Namespace, plc: PLC) -> None:
    if args.channel not in AO_TAG:
        raise SystemExit(f"channel must be 0 or 1; got {args.channel}")

    pid_index = PID_FOR_AO[args.channel]
    ao_tag = AO_TAG[args.channel]
    ai_tag = AI_TAG[args.channel]
    steps = [0.0, 25.0, 50.0, 75.0, 100.0, 75.0, 50.0, 25.0, 0.0]

    logger.info(
        "Sweeping AO %d (TAG_%d) via PID %d; reading AI %d (TAG_%d)",
        args.channel,
        ao_tag,
        pid_index,
        args.channel,
        ai_tag,
    )
    logger.info(" setpoint%s | AO actual | expected mA | AI actual | AI err (pp)", "%")
    logger.info(" ---------+-----------+-------------+-----------+------------")
    for sp in steps:
        plc.set_pid_setpoint(pid_index, sp)
        time.sleep(SWEEP_DWELL_SECONDS)
        ao = plc.read_tag(ao_tag)
        ai = plc.read_tag(ai_tag)
        err = ai - sp
        logger.info(
            " %8.2f | %9.3f | %11.3f | %9.3f | %+11.3f",
            sp,
            ao,
            _percent_to_ma(sp),
            ai,
            err,
        )


def cmd_save(args: argparse.Namespace, plc: PLC) -> None:
    if not args.yes:
        reply = input(
            "Save the current routing + PID config to PLC flash? "
            "This persists across reboots. [y/N] "
        )
        if reply.strip().lower() not in ("y", "yes"):
            logger.info("Aborted.")
            return
    plc.write_coil(COIL_SAVE_TO_FLASH, True)
    logger.info(
        "Save-to-flash coil triggered. Power-cycle the PLC and run "
        "`status` to confirm persistence."
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="calibrate.py",
        description="P1AM-100 analog I/O calibration helper. See CALIBRATION.md.",
    )
    parser.add_argument(
        "--host", default=PLC_HOST_DEFAULT, help="PLC IP (default %(default)s)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=PLC_PORT_DEFAULT,
        help="Modbus TCP port (default %(default)s)",
    )

    sub = parser.add_subparsers(dest="cmd", required=True, metavar="COMMAND")
    sub.add_parser("status", help="Show current routing and key tag values")
    sub.add_parser(
        "setup",
        help="Configure routing + pass-through PIDs (run once before calibration)",
    )
    sub.add_parser(
        "teardown",
        help="Unmap the calibration PIDs (run when calibration is done)",
    )

    ao = sub.add_parser("ao", help="Drive an analog output to a specific percent")
    ao.add_argument("--channel", type=int, required=True, choices=[0, 1])
    ao.add_argument(
        "--percent",
        type=float,
        required=True,
        help="0..100 (0%% = 4 mA, 100%% = 20 mA)",
    )

    ai = sub.add_parser("ai", help="Read an analog input")
    ai.add_argument("--channel", type=int, required=True, choices=[0, 1])

    sw = sub.add_parser("sweep", help="Walk an AO through 0..100..0 in 25%% steps")
    sw.add_argument("--channel", type=int, required=True, choices=[0, 1])

    save = sub.add_parser("save", help="Persist active config to PLC flash")
    save.add_argument(
        "--yes", "-y", action="store_true", help="Skip confirmation prompt"
    )
    return parser


_COMMANDS = {
    "status": cmd_status,
    "setup": cmd_setup,
    "teardown": cmd_teardown,
    "ao": cmd_ao,
    "ai": cmd_ai,
    "sweep": cmd_sweep,
    "save": cmd_save,
}


def _emergency_zero_outputs(plc: PLC) -> None:
    """Best-effort AO shutdown on an abnormal exit; never masks the cause.

    Every :class:`PLC` method raises ``SystemExit`` on a Modbus error, so an
    error part-way through ``sweep`` used to unwind with the AO parked at
    whatever step it had reached (issue #3997). This runs on *any* exit path
    that is not a clean return, and swallows its own failures so the original
    exception still propagates — it only logs, loudly.
    """
    logger.warning("Aborting: driving analog outputs to 0%s...", "%")
    try:
        zero_analog_outputs(plc)
    except (Exception, SystemExit) as exc:  # noqa: BLE001 - must not mask cause
        logger.error(
            "FAILED to drive the analog outputs to 0%s during abort (%s). The "
            "AO channels may still be energized at up to 20 mA — check the "
            "terminals and power down the rig before touching it.",
            "%",
            exc,
        )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    args = _build_parser().parse_args(argv)

    plc = PLC(args.host, args.port)
    plc.connect()
    try:
        _COMMANDS[args.cmd](args, plc)
    except BaseException:
        # SystemExit and KeyboardInterrupt are BaseException, and both are
        # routine here (Modbus errors raise SystemExit; the operator Ctrl-Cs a
        # sweep). Neither may leave an output energized.
        _emergency_zero_outputs(plc)
        raise
    finally:
        plc.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
