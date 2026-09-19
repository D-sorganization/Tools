"""Contract tests for GSPro Open Connect v1 wire format (Tools #5228).

Verifies wire format against https://gsprogolf.com/GSProConnectV1.html and ensures
putting_launch_monitor and upstream simulator adapters produce identical wire bytes.
"""

from __future__ import annotations

import json

import pytest

from putting_launch_monitor.gspro import (
    PuttShot,
)
from putting_launch_monitor.gspro import (
    encode_heartbeat as putting_encode_heartbeat,
)
from putting_launch_monitor.gspro import (
    encode_shot as putting_encode_shot,
)
from putting_launch_monitor.gspro import (
    parse_reply as putting_parse_reply,
)
from shared.python.launch_monitor.gspro_connect import (
    GSProBallData,
    GSProClubData,
    GSProShot,
    ResponseCategory,
    encode_heartbeat,
    encode_shot,
    parse_reply,
)

pytestmark = [pytest.mark.contract, pytest.mark.headless_safe]


def test_putt_wire_bytes_contract_golden() -> None:
    """Pin the exact wire bytes of a putt to ensure zero drift."""
    ball = GSProBallData(
        speed_mph=4.47,
        hla_deg=3.3,
        vla_deg=0.0,
        total_spin_rpm=0.0,
        spin_axis_deg=0.0,
    )
    raw_bytes = encode_shot(ball, device_id="cam", shot_number=7, ready=True)

    # Decode and re-verify exact canonical schema
    payload = json.loads(raw_bytes.decode("utf-8"))
    expected = {
        "DeviceID": "cam",
        "Units": "Yards",
        "ShotNumber": 7,
        "APIversion": "1",
        "BallData": {
            "Speed": 4.47,
            "SpinAxis": 0.0,
            "TotalSpin": 0.0,
            "BackSpin": 0.0,
            "SideSpin": 0.0,
            "HLA": 3.3,
            "VLA": 0.0,
        },
        "ShotDataOptions": {
            "ContainsBallData": True,
            "ContainsClubData": False,
            "LaunchMonitorIsReady": True,
            "LaunchMonitorBallDetected": True,
            "IsHeartBeat": False,
        },
    }
    assert payload == expected

    # Verify putting_launch_monitor produces identical bytes
    putting_bytes = putting_encode_shot(
        PuttShot(4.47, 3.3), device_id="cam", shot_number=7, ready=True
    )
    assert raw_bytes == putting_bytes


def test_full_swing_wire_bytes_contract_golden() -> None:
    """Pin wire bytes for a full swing containing both ball data and club data."""
    ball = GSProBallData(
        speed_mph=160.14,
        spin_axis_deg=-1.25,
        total_spin_rpm=2387.0,
        back_spin_rpm=2380.0,
        side_spin_rpm=-150.0,
        hla_deg=0.5,
        vla_deg=12.1,
    )
    club = GSProClubData(
        speed_mph=100.66,
        angle_of_attack_deg=-3.5,
        path_deg=1.2,
        face_to_target_deg=-0.4,
    )
    shot = GSProShot(ball_data=ball, club_data=club, ready=True, ball_detected=True)
    raw_bytes = encode_shot(shot, device_id="UpstreamDrift", shot_number=42)

    payload = json.loads(raw_bytes.decode("utf-8"))
    assert payload["DeviceID"] == "UpstreamDrift"
    assert payload["Units"] == "Yards"
    assert payload["ShotNumber"] == 42
    assert payload["APIversion"] == "1"
    assert payload["ShotDataOptions"]["ContainsBallData"] is True
    assert payload["ShotDataOptions"]["ContainsClubData"] is True
    assert payload["ShotDataOptions"]["IsHeartBeat"] is False

    assert payload["BallData"]["Speed"] == 160.14
    assert payload["BallData"]["VLA"] == 12.1
    assert payload["BallData"]["HLA"] == 0.5
    assert payload["BallData"]["TotalSpin"] == 2387.0
    assert payload["BallData"]["SpinAxis"] == -1.25

    assert payload["ClubData"] == {
        "Speed": 100.66,
        "AngleOfAttack": -3.5,
        "Path": 1.2,
        "FaceToTarget": -0.4,
    }


def test_heartbeat_wire_bytes_contract_golden() -> None:
    """Pin the heartbeat wire format."""
    shared_bytes = encode_heartbeat(device_id="cam", shot_number=3, ready=True)
    putting_bytes = putting_encode_heartbeat(device_id="cam", shot_number=3, ready=True)
    assert shared_bytes == putting_bytes

    payload = json.loads(shared_bytes.decode("utf-8"))
    assert payload["DeviceID"] == "cam"
    assert payload["ShotNumber"] == 3
    assert payload["APIversion"] == "1"
    assert payload["BallData"] == {}
    assert payload["ShotDataOptions"] == {
        "ContainsBallData": False,
        "ContainsClubData": False,
        "LaunchMonitorIsReady": True,
        "LaunchMonitorBallDetected": False,
        "IsHeartBeat": True,
    }


def test_reply_parity_contract() -> None:
    """Verify that reply parsing produces equivalent results through both interfaces."""
    raw_json = (
        '{"Code":201,"Message":"Player info",'
        '"Player":{"Handed":"RH","Club":"PT","DistanceToTarget":12.5}}'
    )
    p_reply = putting_parse_reply(raw_json)
    s_reply = parse_reply(raw_json)

    assert p_reply.code == s_reply.code == 201
    assert p_reply.ok is True
    assert s_reply.ok is True
    assert s_reply.category == ResponseCategory.PLAYER_UPDATE
    assert p_reply.player is not None and s_reply.player is not None
    assert p_reply.player.club == s_reply.player.club == "PT"
    assert (
        p_reply.player.distance_to_target == s_reply.player.distance_to_target == 12.5
    )
    assert p_reply.player.is_putting() is True
    assert s_reply.player.is_putting() is True
