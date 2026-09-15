"""GSPro Open Connect v1 — the wire protocol, and a client that speaks it.

Verified against https://gsprogolf.com/GSProConnectV1.html: newline-free JSON
objects over a plain TCP socket to ``127.0.0.1:921``. GSPro answers every
message with a JSON object carrying ``Code`` (200 shot accepted, 201 player
information, 5xx failure). The launch monitor owns ``ShotNumber`` and must
increment it. A putter shows up in the 201 ``Player.Club`` field; the
official document never names the code, and connectors in the wild key
putting mode off ``"PT"``, so that string is a setting here, not a constant
carved into the protocol.

The codec functions are pure so they are unit-tested without a socket; the
client wraps them with a socket, a heartbeat and reconnection.
"""

from __future__ import annotations

import json
import socket
import time
from dataclasses import dataclass, field
from typing import Any

from shared.python.contracts import require

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 921
API_VERSION = "1"
DEFAULT_PUTTER_CODES: frozenset[str] = frozenset({"PT"})
CODE_OK = 200
CODE_PLAYER = 201


@dataclass(frozen=True)
class PuttShot:
    """What a putt is to GSPro: speed in mph and a horizontal launch angle.

    Invariants: ``speed_mph > 0``; ``abs(hla_deg) < 90``. Spin is zero for a
    rolling putt and the vertical launch angle is zero.
    """

    speed_mph: float
    hla_deg: float

    def __post_init__(self) -> None:
        require(self.speed_mph > 0, "putt speed must be positive", self.speed_mph)
        require(abs(self.hla_deg) < 90, "HLA out of range", self.hla_deg)


@dataclass(frozen=True)
class PlayerInfo:
    """The 201 message: who is up and what they are holding."""

    handed: str
    club: str
    distance_to_target: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def is_putting(self, putter_codes: frozenset[str] = DEFAULT_PUTTER_CODES) -> bool:
        return self.club.upper() in putter_codes


@dataclass(frozen=True)
class Reply:
    code: int
    message: str
    player: PlayerInfo | None = None

    @property
    def ok(self) -> bool:
        return self.code in (CODE_OK, CODE_PLAYER)


def encode_shot(
    shot: PuttShot, *, device_id: str, shot_number: int, ready: bool = True
) -> bytes:
    """The JSON bytes GSPro expects for a putt.

    Preconditions: non-empty ``device_id``; ``shot_number >= 1``.
    Postcondition: a single JSON object, no trailing newline (the spec
    frames messages by JSON object, not by line).
    """
    require(bool(device_id), "device id must be non-empty")
    require(shot_number >= 1, "shot number starts at 1", shot_number)
    payload = {
        "DeviceID": device_id,
        "Units": "Yards",
        "ShotNumber": int(shot_number),
        "APIversion": API_VERSION,
        "BallData": {
            "Speed": round(float(shot.speed_mph), 2),
            "SpinAxis": 0.0,
            "TotalSpin": 0.0,
            "BackSpin": 0.0,
            "SideSpin": 0.0,
            "HLA": round(float(shot.hla_deg), 2),
            "VLA": 0.0,
        },
        "ShotDataOptions": {
            "ContainsBallData": True,
            "ContainsClubData": False,
            "LaunchMonitorIsReady": bool(ready),
            "LaunchMonitorBallDetected": True,
            "IsHeartBeat": False,
        },
    }
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def encode_heartbeat(*, device_id: str, shot_number: int, ready: bool = True) -> bytes:
    """A heartbeat: keeps the connection alive and GSPro's status green.

    Precondition: non-empty ``device_id``; ``shot_number >= 0`` (the count so
    far — a heartbeat does not consume a number).
    """
    require(bool(device_id), "device id must be non-empty")
    require(shot_number >= 0, "shot number", shot_number)
    payload = {
        "DeviceID": device_id,
        "Units": "Yards",
        "ShotNumber": int(shot_number),
        "APIversion": API_VERSION,
        "BallData": {},
        "ShotDataOptions": {
            "ContainsBallData": False,
            "ContainsClubData": False,
            "LaunchMonitorIsReady": bool(ready),
            "LaunchMonitorBallDetected": False,
            "IsHeartBeat": True,
        },
    }
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def parse_reply(raw: bytes | str) -> Reply:
    """Decode one GSPro reply. Raises ``ValueError`` on malformed input."""
    text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"GSPro reply is not JSON: {text[:120]!r}") from exc
    require(isinstance(data, dict), "GSPro reply must be an object")
    require("Code" in data, "GSPro reply lacks Code", data)
    code = int(data["Code"])
    message = str(data.get("Message", ""))
    player = None
    if code == CODE_PLAYER and isinstance(data.get("Player"), dict):
        p = data["Player"]
        dist = p.get("DistanceToTarget")
        player = PlayerInfo(
            handed=str(p.get("Handed", "")),
            club=str(p.get("Club", "")),
            distance_to_target=float(dist) if dist is not None else None,
            raw=dict(p),
        )
    return Reply(code=code, message=message, player=player)


def split_objects(buffer: bytes) -> tuple[list[bytes], bytes]:
    """Split a byte stream into complete top-level JSON objects.

    GSPro can send several objects back to back (a 200 followed by a 201)
    with no separator, so framing is by brace depth, string-aware.
    Postcondition: every returned object parses; the remainder is an
    incomplete prefix (possibly empty).
    """
    objects: list[bytes] = []
    depth, start, in_str, esc = 0, -1, False, False
    for i, ch in enumerate(buffer):
        c = chr(ch)
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                objects.append(buffer[start : i + 1])
                start = -1
    rest = buffer[start:] if start >= 0 else b""
    return objects, rest


class GsproClient:
    """One connection to GSPro's Open Connect listener.

    Owns the shot counter, tracks the last player information (so callers
    know whether GSPro is in putting mode) and reconnects on demand. The
    socket is injectable for tests.
    """

    def __init__(
        self,
        *,
        device_id: str,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        putter_codes: frozenset[str] = DEFAULT_PUTTER_CODES,
        timeout_s: float = 2.0,
        socket_factory: Any = None,
    ) -> None:
        require(bool(device_id), "device id must be non-empty")
        require(0 < port < 65536, "port", port)
        require(timeout_s > 0, "timeout", timeout_s)
        self.device_id = device_id
        self.host, self.port = host, port
        self.putter_codes = putter_codes
        self.timeout_s = timeout_s
        self._factory = socket_factory or self._tcp
        self._sock: Any = None
        self._buffer = b""
        self.shot_number = 0
        self.player: PlayerInfo | None = None
        self.last_reply: Reply | None = None

    # -- state ---------------------------------------------------------------------
    @property
    def connected(self) -> bool:
        return self._sock is not None

    @property
    def putting_mode(self) -> bool:
        """True once GSPro has told us the player holds a putter."""
        return self.player is not None and self.player.is_putting(self.putter_codes)

    # -- lifecycle -------------------------------------------------------------------
    def _tcp(self) -> socket.socket:
        s = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
        s.settimeout(self.timeout_s)
        return s

    def connect(self) -> None:
        if self._sock is None:
            self._sock = self._factory()
            self._buffer = b""

    def close(self) -> None:
        sock, self._sock = self._sock, None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    # -- traffic -----------------------------------------------------------------------
    def send_shot(self, shot: PuttShot) -> Reply:
        """Deliver a putt. Postcondition: ``shot_number`` advanced by one."""
        self.connect()
        self.shot_number += 1
        data = encode_shot(shot, device_id=self.device_id, shot_number=self.shot_number)
        return self._exchange(data)

    def heartbeat(self) -> Reply:
        self.connect()
        return self._exchange(
            encode_heartbeat(device_id=self.device_id, shot_number=self.shot_number)
        )

    def _exchange(self, data: bytes) -> Reply:
        assert self._sock is not None
        try:
            self._sock.sendall(data)
            reply = self._read_reply()
        except (OSError, ValueError):
            self.close()
            raise
        return reply

    def _read_reply(self) -> Reply:
        """The first complete object; any trailing 201 updates the player."""
        assert self._sock is not None
        deadline = time.monotonic() + self.timeout_s
        first: Reply | None = None
        while time.monotonic() < deadline:
            objects, self._buffer = split_objects(self._buffer)
            for obj in objects:
                reply = parse_reply(obj)
                self._absorb(reply)
                if first is None:
                    first = reply
            if first is not None and not self._buffer:
                return first
            chunk = self._sock.recv(4096)
            if not chunk:
                raise OSError("GSPro closed the connection")
            self._buffer += chunk
        if first is not None:
            return first
        raise TimeoutError("no reply from GSPro")

    def _absorb(self, reply: Reply) -> None:
        self.last_reply = reply
        if reply.player is not None:
            self.player = reply.player
