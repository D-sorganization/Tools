"""GSPro Open Connect v1: the codec against the published spec, and the client."""

from __future__ import annotations

import json

import pytest

from putting_launch_monitor.gspro import (
    CODE_OK,
    CODE_PLAYER,
    GsproClient,
    PuttShot,
    encode_heartbeat,
    encode_shot,
    parse_reply,
    split_objects,
)

pytestmark = pytest.mark.unit


def test_shot_encodes_every_required_field_per_spec() -> None:
    data = json.loads(encode_shot(PuttShot(4.47, 3.3), device_id="cam", shot_number=7))
    assert data["DeviceID"] == "cam" and data["Units"] == "Yards"
    assert data["ShotNumber"] == 7 and data["APIversion"] == "1"
    ball = data["BallData"]
    assert ball["Speed"] == 4.47 and ball["HLA"] == 3.3 and ball["VLA"] == 0.0
    assert ball["TotalSpin"] == 0.0 and ball["SpinAxis"] == 0.0
    assert data["ShotDataOptions"] == {
        "ContainsBallData": True,
        "ContainsClubData": False,
        "LaunchMonitorIsReady": True,
        "LaunchMonitorBallDetected": True,
        "IsHeartBeat": False,
    }
    assert not encode_shot(PuttShot(1, 0), device_id="c", shot_number=1).endswith(b"\n")


def test_heartbeat_and_contracts() -> None:
    hb = json.loads(encode_heartbeat(device_id="cam", shot_number=3))
    assert hb["ShotDataOptions"]["IsHeartBeat"] is True
    assert hb["ShotDataOptions"]["ContainsBallData"] is False and hb["ShotNumber"] == 3
    with pytest.raises(ValueError, match="positive"):
        PuttShot(0.0, 0.0)
    with pytest.raises(ValueError, match="HLA"):
        PuttShot(2.0, 95.0)
    with pytest.raises(ValueError, match="device id"):
        encode_shot(PuttShot(2.0, 0.0), device_id="", shot_number=1)
    with pytest.raises(ValueError, match="starts at 1"):
        encode_shot(PuttShot(2.0, 0.0), device_id="c", shot_number=0)


def test_replies_parse_and_a_201_names_the_putter() -> None:
    ok = parse_reply(b'{"Code":200,"Message":"OK"}')
    assert ok.code == CODE_OK and ok.ok and ok.player is None
    info = parse_reply(
        '{"Code":201,"Message":"GSPro Player Information",'
        '"Player":{"Handed":"RH","Club":"PT","DistanceToTarget":12.5}}'
    )
    assert info.code == CODE_PLAYER and info.player is not None
    assert info.player.is_putting() and info.player.distance_to_target == 12.5
    driver = parse_reply('{"Code":201,"Player":{"Handed":"RH","Club":"DR"}}').player
    assert driver is not None and not driver.is_putting()
    assert not parse_reply('{"Code":501,"Message":"bad"}').ok
    with pytest.raises(ValueError, match="not JSON"):
        parse_reply(b"nope")
    with pytest.raises(ValueError, match="Code"):
        parse_reply(b'{"Message":"x"}')


def test_framing_splits_glued_objects_and_keeps_the_remainder() -> None:
    stream = b'{"Code":200}{"Code":201,"Message":"a } b","Player":{"Club":"PT"}}{"Co'
    objects, rest = split_objects(stream)
    assert [json.loads(o)["Code"] for o in objects] == [200, 201]
    assert rest == b'{"Co'
    assert split_objects(b"") == ([], b"")
    assert split_objects(b'{"m":"\\"}"}')[0] == [b'{"m":"\\"}"}']  # escaped quote


class FakeSocket:
    def __init__(self, replies: list[bytes]) -> None:
        self.sent: list[bytes] = []
        self.replies = replies
        self.closed = False

    def sendall(self, data: bytes) -> None:
        self.sent.append(data)

    def recv(self, n: int) -> bytes:
        return self.replies.pop(0) if self.replies else b""

    def close(self) -> None:
        self.closed = True


def test_client_counts_shots_learns_the_club_and_recovers_from_a_drop() -> None:
    sock = FakeSocket(
        [
            b'{"Code":200,"Message":"OK"}'
            b'{"Code":201,"Message":"GSPro Player Information",'
            b'"Player":{"Handed":"RH","Club":"PT"}}',
            b'{"Code":200,"Message":"OK"}',
        ]
    )
    client = GsproClient(device_id="cam", socket_factory=lambda: sock)
    assert not client.connected and not client.putting_mode
    reply = client.send_shot(PuttShot(5.0, -1.0))
    assert reply.code == 200 and client.shot_number == 1 and client.putting_mode
    assert json.loads(sock.sent[0])["ShotNumber"] == 1
    client.send_shot(PuttShot(3.0, 0.5))
    assert client.shot_number == 2
    # The peer goes away: the client closes its socket and raises, so the
    # next call reconnects through the factory.
    with pytest.raises(OSError, match="closed"):
        client.heartbeat()
    assert not client.connected and sock.closed
    with pytest.raises(ValueError):
        GsproClient(device_id="cam", port=70000)


def test_putter_codes_are_a_setting() -> None:
    reply = parse_reply('{"Code":201,"Player":{"Handed":"LH","Club":"PUTTER"}}')
    assert reply.player is not None
    assert not reply.player.is_putting()
    assert reply.player.is_putting(frozenset({"PUTTER"}))
