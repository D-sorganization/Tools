"""GSPro Open Connect v1 — the wire protocol, and a client that speaks it.

Verified against https://gsprogolf.com/GSProConnectV1.html: newline-free JSON
objects over a plain TCP socket to ``127.0.0.1:921``. GSPro answers every
message with a JSON object carrying ``Code`` (200 shot accepted, 201 player
information, 5xx failure). The launch monitor owns ``ShotNumber`` and must
increment it. A putter shows up in the 201 ``Player.Club`` field; the
official document never names the code, and connectors in the wild key
putting mode off ``"PT"``, so that string is a setting here, not a constant
carved into the protocol.

The protocol-level codec is delegated to ``shared.python.launch_monitor.gspro_connect``;
the client here wraps it with a socket, a heartbeat and reconnection.
"""

from __future__ import annotations

import socket
import time
from dataclasses import dataclass
from typing import Any

from shared.python.contracts import require
from shared.python.launch_monitor.gspro_connect import (
    API_VERSION,
    CODE_OK,
    CODE_PLAYER,
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_PUTTER_CODES,
    GSProBallData,
    encode_heartbeat,
    parse_reply,
    split_objects,
)
from shared.python.launch_monitor.gspro_connect import (
    GSProPlayerInfo as PlayerInfo,
)
from shared.python.launch_monitor.gspro_connect import (
    GSProReply as Reply,
)
from shared.python.launch_monitor.gspro_connect import (
    encode_shot as _shared_encode_shot,
)

__all__ = [
    "API_VERSION",
    "CODE_OK",
    "CODE_PLAYER",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "DEFAULT_PUTTER_CODES",
    "GsproClient",
    "PlayerInfo",
    "PuttShot",
    "Reply",
    "encode_heartbeat",
    "encode_shot",
    "parse_reply",
    "split_objects",
]


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


def encode_shot(
    shot: PuttShot, *, device_id: str, shot_number: int, ready: bool = True
) -> bytes:
    """The JSON bytes GSPro expects for a putt.

    Preconditions: non-empty ``device_id``; ``shot_number >= 1``.
    Postcondition: a single JSON object, no trailing newline (the spec
    frames messages by JSON object, not by line).
    """
    ball = GSProBallData(
        speed_mph=shot.speed_mph,
        hla_deg=shot.hla_deg,
        vla_deg=0.0,
        total_spin_rpm=0.0,
        spin_axis_deg=0.0,
    )
    raw = _shared_encode_shot(
        ball,
        device_id=device_id,
        shot_number=shot_number,
        ready=ready,
    )
    assert isinstance(raw, bytes)
    return raw


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
