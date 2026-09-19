"""Shared GSPro Open Connect v1 codec (Tools #5228).

Pure protocol-level codec and wire definitions for GSPro Open Connect v1,
verified against https://gsprogolf.com/GSProConnectV1.html.

Follows TDD, DbC, Law of Demeter, and DRY.
All functions in this module are pure with zero network, subprocess, or
GUI side effects.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from shared.python.contracts import require

DEFAULT_HOST: str = "127.0.0.1"
DEFAULT_PORT: int = 921
API_VERSION: str = "1"
DEFAULT_PUTTER_CODES: frozenset[str] = frozenset({"PT"})

CODE_OK: int = 200
CODE_PLAYER: int = 201


class ResponseCategory(str, Enum):
    """Categorized response type from GSPro."""

    CONFIRMED_ACCEPTED = "confirmed_accepted"
    PLAYER_UPDATE = "player_update"
    ERROR_REJECTED = "error_rejected"
    UNKNOWN = "unknown"


def categorize_code(code: int) -> ResponseCategory:
    """Categorize an integer response code from GSPro Open Connect."""
    if code == CODE_OK:
        return ResponseCategory.CONFIRMED_ACCEPTED
    if code == CODE_PLAYER:
        return ResponseCategory.PLAYER_UPDATE
    if 500 <= code < 600:
        return ResponseCategory.ERROR_REJECTED
    return ResponseCategory.UNKNOWN


@dataclass(frozen=True)
class GSProBallData:
    """Ball launch data in wire units (mph, degrees, rpm)."""

    speed_mph: float
    spin_axis_deg: float = 0.0
    total_spin_rpm: float = 0.0
    back_spin_rpm: float = 0.0
    side_spin_rpm: float = 0.0
    hla_deg: float = 0.0
    vla_deg: float = 0.0
    carry_distance_yards: float | None = None

    def __post_init__(self) -> None:
        require(math.isfinite(self.speed_mph), "speed must be finite", self.speed_mph)
        require(self.speed_mph > 0, "speed must be positive", self.speed_mph)
        require(math.isfinite(self.hla_deg), "HLA must be finite", self.hla_deg)
        require(abs(self.hla_deg) < 90, "HLA out of range", self.hla_deg)
        require(math.isfinite(self.vla_deg), "VLA must be finite", self.vla_deg)
        require(abs(self.vla_deg) < 90, "VLA out of range", self.vla_deg)
        require(
            math.isfinite(self.total_spin_rpm),
            "TotalSpin must be finite",
            self.total_spin_rpm,
        )
        require(
            self.total_spin_rpm >= 0,
            "TotalSpin must be non-negative",
            self.total_spin_rpm,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert ball data to GSPro wire representation."""
        payload: dict[str, Any] = {
            "Speed": round(float(self.speed_mph), 2),
            "SpinAxis": round(float(self.spin_axis_deg), 2),
            "TotalSpin": round(float(self.total_spin_rpm), 2),
            "BackSpin": round(float(self.back_spin_rpm), 2),
            "SideSpin": round(float(self.side_spin_rpm), 2),
            "HLA": round(float(self.hla_deg), 2),
            "VLA": round(float(self.vla_deg), 2),
        }
        if self.carry_distance_yards is not None:
            payload["CarryDistance"] = round(float(self.carry_distance_yards), 2)
        return payload


@dataclass(frozen=True)
class GSProClubData:
    """Club impact data in wire units (mph, degrees).

    Only fields with measured values should be populated. Unmeasured fields
    remain None and are excluded from the wire payload (do not zero-fill).
    """

    speed_mph: float | None = None
    angle_of_attack_deg: float | None = None
    face_to_target_deg: float | None = None
    lie_deg: float | None = None
    loft_deg: float | None = None
    path_deg: float | None = None
    speed_at_impact_mph: float | None = None
    vertical_face_impact: float | None = None
    horizontal_face_impact: float | None = None
    closure_rate: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert measured club fields to GSPro wire keys without zero-filling."""
        out: dict[str, Any] = {}
        mapping = (
            ("Speed", self.speed_mph),
            ("AngleOfAttack", self.angle_of_attack_deg),
            ("FaceToTarget", self.face_to_target_deg),
            ("Lie", self.lie_deg),
            ("Loft", self.loft_deg),
            ("Path", self.path_deg),
            ("SpeedAtImpact", self.speed_at_impact_mph),
            ("VerticalFaceImpact", self.vertical_face_impact),
            ("HorizontalFaceImpact", self.horizontal_face_impact),
            ("ClosureRate", self.closure_rate),
        )
        for wire_key, val in mapping:
            if val is not None:
                out[wire_key] = round(float(val), 2)
        return out


@dataclass(frozen=True)
class GSProShot:
    """Full shot envelope for GSPro Open Connect."""

    ball_data: GSProBallData
    club_data: GSProClubData | None = None
    ready: bool = True
    ball_detected: bool = True


@dataclass(frozen=True)
class GSProPlayerInfo:
    """Player information returned by GSPro in a 201 message."""

    handed: str = ""
    club: str = ""
    distance_to_target: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def is_putting(self, putter_codes: frozenset[str] = DEFAULT_PUTTER_CODES) -> bool:
        """Return True if the player's selected club is a putter."""
        return self.club.upper() in putter_codes


@dataclass(frozen=True)
class GSProReply:
    """Decoded response from GSPro."""

    code: int
    message: str = ""
    category: ResponseCategory = ResponseCategory.UNKNOWN
    player: GSProPlayerInfo | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True if the response represents success (200) or player update (201)."""
        return self.category in (
            ResponseCategory.CONFIRMED_ACCEPTED,
            ResponseCategory.PLAYER_UPDATE,
        )


def encode_shot_payload(
    shot: GSProShot | GSProBallData,
    *,
    device_id: str,
    shot_number: int,
    units: str = "Yards",
    ready: bool = True,
) -> dict[str, Any]:
    """Encode shot parameters into a GSPro Open Connect v1 payload dictionary."""
    require(bool(device_id), "device id must be non-empty")
    require(shot_number >= 1, "shot number starts at 1", shot_number)

    if isinstance(shot, GSProBallData):
        ball_data = shot
        club_data = None
        ball_detected = True
        is_ready = ready
    else:
        ball_data = shot.ball_data
        club_data = shot.club_data
        ball_detected = shot.ball_detected
        is_ready = shot.ready

    has_club = club_data is not None and bool(club_data.to_dict())

    payload: dict[str, Any] = {
        "DeviceID": str(device_id),
        "Units": str(units),
        "ShotNumber": int(shot_number),
        "APIversion": API_VERSION,
        "BallData": ball_data.to_dict(),
        "ShotDataOptions": {
            "ContainsBallData": True,
            "ContainsClubData": has_club,
            "LaunchMonitorIsReady": bool(is_ready),
            "LaunchMonitorBallDetected": bool(ball_detected),
            "IsHeartBeat": False,
        },
    }

    if has_club and club_data is not None:
        payload["ClubData"] = club_data.to_dict()

    return payload


def encode_shot(
    shot: GSProShot | GSProBallData,
    *,
    device_id: str,
    shot_number: int,
    units: str = "Yards",
    ready: bool = True,
) -> bytes:
    """Encode a shot into GSPro Open Connect v1 JSON bytes without a newline."""
    payload = encode_shot_payload(
        shot,
        device_id=device_id,
        shot_number=shot_number,
        units=units,
        ready=ready,
    )
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def encode_heartbeat_payload(
    *,
    device_id: str,
    shot_number: int = 0,
    units: str = "Yards",
    ready: bool = True,
) -> dict[str, Any]:
    """Encode a minimal heartbeat payload dictionary."""
    require(bool(device_id), "device id must be non-empty")
    require(shot_number >= 0, "shot number must be non-negative", shot_number)

    return {
        "DeviceID": str(device_id),
        "Units": str(units),
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


def encode_heartbeat(
    *,
    device_id: str,
    shot_number: int = 0,
    units: str = "Yards",
    ready: bool = True,
) -> bytes:
    """Encode a heartbeat into JSON bytes without a trailing newline."""
    payload = encode_heartbeat_payload(
        device_id=device_id,
        shot_number=shot_number,
        units=units,
        ready=ready,
    )
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def parse_reply(raw: bytes | str | dict[str, Any]) -> GSProReply:
    """Decode and categorize one GSPro reply object.

    Raises ValueError on malformed input.
    """
    if isinstance(raw, dict):
        data = raw
    else:
        text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"GSPro reply is not JSON: {text[:120]!r}") from exc

    require(isinstance(data, dict), "GSPro reply must be an object", type(data))
    require("Code" in data, "GSPro reply lacks Code", data)

    code = int(data["Code"])
    message = str(data.get("Message", ""))
    category = categorize_code(code)

    player: GSProPlayerInfo | None = None
    if isinstance(data.get("Player"), dict):
        p = data["Player"]
        handed_val = p.get("Handed", p.get("Handedness", ""))
        dist = p.get("DistanceToTarget")
        player = GSProPlayerInfo(
            handed=str(handed_val) if handed_val is not None else "",
            club=str(p.get("Club", "")),
            distance_to_target=float(dist) if dist is not None else None,
            raw=dict(p),
        )

    return GSProReply(
        code=code,
        message=message,
        category=category,
        player=player,
        raw=dict(data),
    )


def split_objects(buffer: bytes) -> tuple[list[bytes], bytes]:
    """Split a byte stream into complete top-level JSON objects.

    GSPro can send several objects back-to-back without separators.
    Framing is by brace depth, string and escape aware.
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


__all__ = [
    "API_VERSION",
    "CODE_OK",
    "CODE_PLAYER",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "DEFAULT_PUTTER_CODES",
    "GSProBallData",
    "GSProClubData",
    "GSProPlayerInfo",
    "GSProReply",
    "GSProShot",
    "ResponseCategory",
    "categorize_code",
    "encode_heartbeat",
    "encode_heartbeat_payload",
    "encode_shot",
    "encode_shot_payload",
    "parse_reply",
    "split_objects",
]
