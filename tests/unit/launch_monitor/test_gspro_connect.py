"""Unit tests for the shared GSPro Open Connect v1 codec (Tools #5228).

Follows TDD, DbC, LoD and DRY principles.
"""

from __future__ import annotations

import json

import pytest

from shared.python.launch_monitor.gspro_connect import (
    API_VERSION,
    GSProBallData,
    GSProClubData,
    GSProShot,
    ResponseCategory,
    categorize_code,
    encode_heartbeat,
    encode_heartbeat_payload,
    encode_shot,
    encode_shot_payload,
    parse_reply,
    split_objects,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_categorize_code() -> None:
    assert categorize_code(200) == ResponseCategory.CONFIRMED_ACCEPTED
    assert categorize_code(201) == ResponseCategory.PLAYER_UPDATE
    assert categorize_code(500) == ResponseCategory.ERROR_REJECTED
    assert categorize_code(501) == ResponseCategory.ERROR_REJECTED
    assert categorize_code(599) == ResponseCategory.ERROR_REJECTED
    assert categorize_code(404) == ResponseCategory.UNKNOWN
    assert categorize_code(777) == ResponseCategory.UNKNOWN


def test_ball_data_invariants() -> None:
    ball = GSProBallData(
        speed_mph=100.0, hla_deg=2.5, vla_deg=11.0, total_spin_rpm=2400.0
    )
    assert ball.speed_mph == 100.0
    assert ball.hla_deg == 2.5
    assert ball.vla_deg == 11.0

    # speed must be positive and finite
    with pytest.raises(ValueError, match="positive"):
        GSProBallData(speed_mph=0.0)
    with pytest.raises(ValueError, match="positive"):
        GSProBallData(speed_mph=-5.0)
    with pytest.raises(ValueError, match="finite"):
        GSProBallData(speed_mph=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        GSProBallData(speed_mph=float("inf"))

    # HLA within (-90, 90)
    with pytest.raises(ValueError, match="HLA"):
        GSProBallData(speed_mph=10.0, hla_deg=90.0)
    with pytest.raises(ValueError, match="HLA"):
        GSProBallData(speed_mph=10.0, hla_deg=-90.0)

    # VLA within (-90, 90)
    with pytest.raises(ValueError, match="VLA"):
        GSProBallData(speed_mph=10.0, vla_deg=90.0)

    # Spin must be non-negative
    with pytest.raises(ValueError, match="TotalSpin"):
        GSProBallData(speed_mph=10.0, total_spin_rpm=-1.0)


def test_encode_shot_payload_ball_only() -> None:
    ball = GSProBallData(
        speed_mph=4.47,
        hla_deg=3.3,
        vla_deg=0.0,
        total_spin_rpm=0.0,
        spin_axis_deg=0.0,
    )
    payload = encode_shot_payload(ball, device_id="Cam1", shot_number=7)
    assert payload["DeviceID"] == "Cam1"
    assert payload["Units"] == "Yards"
    assert payload["ShotNumber"] == 7
    assert payload["APIversion"] == API_VERSION
    assert payload["BallData"] == {
        "Speed": 4.47,
        "SpinAxis": 0.0,
        "TotalSpin": 0.0,
        "BackSpin": 0.0,
        "SideSpin": 0.0,
        "HLA": 3.3,
        "VLA": 0.0,
    }
    assert payload["ShotDataOptions"] == {
        "ContainsBallData": True,
        "ContainsClubData": False,
        "LaunchMonitorIsReady": True,
        "LaunchMonitorBallDetected": True,
        "IsHeartBeat": False,
    }
    assert "ClubData" not in payload


def test_encode_shot_payload_with_club_data() -> None:
    ball = GSProBallData(
        speed_mph=155.25,
        hla_deg=-1.5,
        vla_deg=12.4,
        total_spin_rpm=2450.0,
        spin_axis_deg=3.2,
    )
    club = GSProClubData(
        speed_mph=105.4,
        angle_of_attack_deg=-2.1,
        path_deg=1.8,
    )
    shot = GSProShot(ball_data=ball, club_data=club)
    payload = encode_shot_payload(shot, device_id="UD", shot_number=12)

    assert payload["ShotDataOptions"]["ContainsClubData"] is True
    assert "ClubData" in payload
    c_data = payload["ClubData"]
    assert c_data["Speed"] == 105.4
    assert c_data["AngleOfAttack"] == -2.1
    assert c_data["Path"] == 1.8
    # Unmeasured fields must NOT be present
    assert "FaceToTarget" not in c_data
    assert "Lie" not in c_data


def test_encode_shot_bytes_no_trailing_newline() -> None:
    ball = GSProBallData(speed_mph=10.0, hla_deg=0.0)
    raw = encode_shot(ball, device_id="LM", shot_number=1)
    assert isinstance(raw, bytes)
    assert not raw.endswith(b"\n")
    data = json.loads(raw.decode("utf-8"))
    assert data["DeviceID"] == "LM"
    assert data["ShotNumber"] == 1


def test_encode_shot_preconditions() -> None:
    ball = GSProBallData(speed_mph=10.0)
    with pytest.raises(ValueError, match="device id"):
        encode_shot(ball, device_id="", shot_number=1)
    with pytest.raises(ValueError, match="shot number"):
        encode_shot(ball, device_id="LM", shot_number=0)


def test_encode_heartbeat() -> None:
    raw = encode_heartbeat(device_id="HeartMonitor", shot_number=5, ready=True)
    assert not raw.endswith(b"\n")
    data = json.loads(raw.decode("utf-8"))
    assert data["DeviceID"] == "HeartMonitor"
    assert data["ShotNumber"] == 5
    assert data["APIversion"] == API_VERSION
    assert data["BallData"] == {}
    assert data["ShotDataOptions"] == {
        "ContainsBallData": False,
        "ContainsClubData": False,
        "LaunchMonitorIsReady": True,
        "LaunchMonitorBallDetected": False,
        "IsHeartBeat": True,
    }

    payload = encode_heartbeat_payload(device_id="HeartMonitor", shot_number=0)
    assert payload["ShotNumber"] == 0
    assert payload["ShotDataOptions"]["IsHeartBeat"] is True

    with pytest.raises(ValueError, match="device id"):
        encode_heartbeat(device_id="", shot_number=1)
    with pytest.raises(ValueError, match="shot number"):
        encode_heartbeat(device_id="dev", shot_number=-1)


def test_parse_reply() -> None:
    # 200 OK
    reply_200 = parse_reply(b'{"Code":200,"Message":"OK"}')
    assert reply_200.code == 200
    assert reply_200.category == ResponseCategory.CONFIRMED_ACCEPTED
    assert reply_200.ok is True
    assert reply_200.message == "OK"
    assert reply_200.player is None

    # 201 Player
    reply_201 = parse_reply(
        '{"Code":201,"Message":"Player info",'
        '"Player":{"Handed":"RH","Club":"PT","DistanceToTarget":15.0}}'
    )
    assert reply_201.code == 201
    assert reply_201.category == ResponseCategory.PLAYER_UPDATE
    assert reply_201.ok is True
    assert reply_201.player is not None
    assert reply_201.player.handed == "RH"
    assert reply_201.player.club == "PT"
    assert reply_201.player.distance_to_target == 15.0
    assert reply_201.player.is_putting() is True

    # 201 with Handedness key (seen in some simulator payloads)
    reply_201_alt = parse_reply('{"Code":201,"Player":{"Handedness":"LH","Club":"DR"}}')
    assert reply_201_alt.player is not None
    assert reply_201_alt.player.handed == "LH"
    assert reply_201_alt.player.club == "DR"
    assert reply_201_alt.player.is_putting() is False

    # 501 Error
    reply_501 = parse_reply(b'{"Code":501,"Message":"Invalid lie"}')
    assert reply_501.code == 501
    assert reply_501.category == ResponseCategory.ERROR_REJECTED
    assert reply_501.ok is False

    # Unknown
    reply_custom = parse_reply(b'{"Code":999,"Message":"Custom code"}')
    assert reply_custom.code == 999
    assert reply_custom.category == ResponseCategory.UNKNOWN
    assert reply_custom.ok is False

    # Invalid input
    with pytest.raises(ValueError, match="not JSON"):
        parse_reply(b"garbage not json")
    with pytest.raises(ValueError, match="Code"):
        parse_reply(b'{"Message":"missing code"}')
    with pytest.raises(ValueError, match="object"):
        parse_reply(b"[1, 2, 3]")


def test_split_objects() -> None:
    raw = b'{"Code":200}{"Code":201,"Message":"glued } object"}{"Code":5'
    objs, remainder = split_objects(raw)
    assert len(objs) == 2
    assert json.loads(objs[0])["Code"] == 200
    assert json.loads(objs[1])["Code"] == 201
    assert remainder == b'{"Code":5'

    # Empty
    assert split_objects(b"") == ([], b"")

    # Escaped quote
    esc_objs, esc_rest = split_objects(b'{"key":"val\\"with quote"}')
    assert len(esc_objs) == 1
    assert esc_rest == b""
