"""Shared startup defaults for the control system.

Single source of truth for the default routing/PID/interlock config so the
FastAPI app and the simulator can't drift (they previously each defined it
verbatim).
"""

from __future__ import annotations

import hardware
from models import InterlockConfig, PIDConfig, RoutingConfig


def default_routing_config() -> RoutingConfig:
    """The startup default routing/PID/interlock configuration."""
    return RoutingConfig(
        input_routing=[hardware.tag_name(i) for i in range(6)],
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
            hardware.tag_name(i): InterlockConfig(
                lolo_limit=0.0, low_limit=5.0, high_limit=95.0, hihi_limit=100.0
            )
            for i in range(hardware.TAG_COUNT)
        },
    )
