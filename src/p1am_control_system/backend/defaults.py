"""Shared startup defaults for the control system.

Single source of truth for the default routing/PID/interlock config so the
FastAPI app and the simulator can't drift (they previously each defined it
verbatim).
"""

from __future__ import annotations

import copy

import hardware
from models import InterlockConfig, PIDConfig, RoutingConfig

# A pass-through PID is unity-gain feed-forward: the loop's commanded value is
# written straight to its CV (the AO) with no PV feedback term. kp=1, ki=kd=0.
PASSTHROUGH_KP = 1.0
PASSTHROUGH_KI = 0.0
PASSTHROUGH_KD = 0.0


# Default high-side alarm/trip band for a ROUTED input tag, in percent of span
# (95 % of a 1400 C type-K full scale is 1330 C). Only the high side is set:
# a low limit would trip a routed thermocouple at room temperature (25 C is
# 1.8 %), and an unrouted tag at 0.0 -- so the low side defaults to disabled.
DEFAULT_INPUT_HIGH_LIMIT = 95.0
DEFAULT_INPUT_HIHI_LIMIT = 100.0


def default_interlock_for(tag: str, routed_inputs: frozenset[str]) -> InterlockConfig:
    """The startup interlock for ``tag``.

    Routed inputs get a high-side-only band; every other tag is fully disabled
    (all four limits ``None``), which the firmware skips entirely (#4001).
    """
    if tag in routed_inputs:
        return InterlockConfig(
            high_limit=DEFAULT_INPUT_HIGH_LIMIT, hihi_limit=DEFAULT_INPUT_HIHI_LIMIT
        )
    return InterlockConfig()


def default_routing_config() -> RoutingConfig:
    """The startup default routing/PID/interlock configuration.

    Invariant (contract-tested): deploying this config to a freshly booted PLC
    must not trip the interlock -- no tag holding 0.0 (unrouted) or a
    room-temperature thermocouple reading violates any limit in it.
    """
    input_routing = [hardware.tag_name(i) for i in range(6)]
    routed_inputs = frozenset(input_routing)
    return RoutingConfig(
        input_routing=input_routing,
        output_routing=[hardware.tag_name(10), hardware.tag_name(11)],
        pids=[
            PIDConfig(
                pv_tag="TAG_1", cv_tag="TAG_2", setpoint=50.0, kp=1.0, ki=0.5, kd=0.1
            ),
            PIDConfig(
                pv_tag="TAG_3", cv_tag="TAG_4", setpoint=30.0, kp=1.5, ki=0.2, kd=0.05
            ),
            PIDConfig(
                pv_tag="TAG_5", cv_tag="TAG_6", setpoint=40.0, kp=2.0, ki=0.8, kd=0.2
            ),
            PIDConfig(
                pv_tag="TAG_7", cv_tag="TAG_8", setpoint=60.0, kp=0.5, ki=0.1, kd=0.01
            ),
        ],
        interlocks={
            hardware.tag_name(i): default_interlock_for(
                hardware.tag_name(i), routed_inputs
            )
            for i in range(hardware.TAG_COUNT)
        },
    )


def is_pid_passthrough(config: RoutingConfig, pid_index: int, command_tag: str) -> bool:
    """Return whether ``pid_index`` is wired as a pass-through to ``command_tag``.

    The power-supply controller writes the *setpoint* of its PID loop and relies
    on that loop being a unity-gain pass-through (cv -> the command AO, kp=1,
    ki=kd=0) so the command reaches the analog output. After an NVRAM reset the
    PLC can come up with PID0 unmapped (e.g. cv=TAG_255, kp=0), which silently
    swallows the command — "no current flows" with no error (issue #3550).

    Returns False (rather than raising) for an out-of-range ``pid_index`` so the
    caller can treat "not a pass-through" uniformly.
    """
    if not (0 <= pid_index < len(config.pids)):
        return False
    pid = config.pids[pid_index]
    return all(
        (
            pid.cv_tag == command_tag,
            pid.kp == PASSTHROUGH_KP,
            pid.ki == PASSTHROUGH_KI,
            pid.kd == PASSTHROUGH_KD,
        )
    )


def ensure_pid_passthrough(
    config: RoutingConfig, pid_index: int, command_tag: str
) -> tuple[RoutingConfig, bool]:
    """Return ``config`` with ``pid_index`` forced to a pass-through if needed.

    Args:
        config: The routing config read from (or destined for) the PLC.
        pid_index: The PID loop the power-supply AO command flows through.
        command_tag: The CV/AO tag the loop must drive (e.g. the PS command_tag).

    Returns:
        A ``(config, repaired)`` tuple. When already a pass-through, the original
        ``config`` is returned unchanged with ``repaired=False``. Otherwise a
        deep-copied config with the loop rewired (preserving the existing
        setpoint) is returned with ``repaired=True``.

    Raises:
        ValueError: If ``pid_index`` is outside the config's PID range.
    """
    if not (0 <= pid_index < len(config.pids)):
        raise ValueError(f"pid index {pid_index} out of range [0, {len(config.pids)})")
    if is_pid_passthrough(config, pid_index, command_tag):
        return config, False

    repaired = copy.deepcopy(config)
    pid = repaired.pids[pid_index]
    pid.cv_tag = command_tag
    pid.kp = PASSTHROUGH_KP
    pid.ki = PASSTHROUGH_KI
    pid.kd = PASSTHROUGH_KD
    return repaired, True
