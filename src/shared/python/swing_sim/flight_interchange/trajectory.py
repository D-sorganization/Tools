"""The ``swing_sim.ball_flight_trajectory/1`` wire (ADR-0047 H1, UD #9350).

One versioned, fail-closed, byte-deterministic record of a single ball
flight, in the idiom of
:mod:`shared.python.swing_sim.delivery_interchange` and the
``swing_sim.putting_result/2`` wire: a declared ``format``, a declared
frame, mandatory model provenance, strictly increasing timestamps,
finite SI values, sorted-keys compact JSON, and unknown fields refused.

Why the record exists
---------------------
Two independent flight-model families produce trajectories that no
viewer could previously share (ADR-0047): UpstreamDrift's named
published models (``physics/flight_models.py`` — Waterloo/Penner,
MacDonald-Hanzely, and the constant-coefficient set) and this repo's
:mod:`shared.python.swing_sim.flight`. Both stay separate and named;
what they gain is a common export format, so a Waterloo/Penner curve
and a ``swing_sim`` flight can sit on the same axes *because each is
labelled*, never because they were forced through one implementation.

Frames (declared, not assumed)
------------------------------
``frame_id`` must be one of two documented right-handed frames:

- :data:`FLIGHT_FRAME_ID` — the UpstreamDrift flight frame both model
  families integrate in: ``x`` forward (downrange), ``y`` left, ``z``
  up. Ground is ``z = 0``.
- :data:`APP_FRAME_ID` — the AffineDrift app frame the Tools viewers
  draw in: ``x`` target, ``y`` up, ``z`` right. Ground is ``y = 0``.

The two are related by ``app = (flight_x, flight_z, -flight_y)``; see
:mod:`shared.python.swing_sim.flight.frames`. A free-form frame string
is refused: a consumer plotting two families together must be able to
*interpret* the axes, and an undeclared frame silently mirrors a shot.

Units
-----
SI throughout and non-negotiable: ``time_s`` seconds, ``position_m``
metres, ``velocity_mps`` metres per second, ``spin_rad_s`` radians per
second (a spin *vector*, not an RPM magnitude). No degrees, no RPM, no
yards anywhere in the wire.

Optional channels are declared, not sniffed
-------------------------------------------
Velocity and spin are optional, but *per record*, never per sample:
``channels`` lists the optional channels every sample carries, and a
sample carrying more or fewer keys than the declaration is refused. A
ragged record — velocity on the first sample and not the fortieth —
would pass a consumer that inspected only ``samples[0]`` and then fail
mid-render, so the wire cannot express one.

Provenance is mandatory
-----------------------
Every record names the family that produced it, the model within that
family, and a digest of the parameters that model integrated with (see
:func:`parameter_digest`). A trajectory with no attributable physics is
exactly the confusion ADR-0045 and ADR-0047 exist to prevent, so there
is no default and no "unknown" sentinel.

The digest is meaningful **only within a declared family**. The same
physical coefficient is legitimately named ``cl1`` in one family and
``lift_scale`` in the other, so equal digests across families mean
nothing and unequal digests across families prove nothing. Compare
digests only when ``model_family`` matches.

Runtime boundary
----------------
This module imports nothing from either flight family — it is a wire,
constructible from plain sequences through :func:`from_samples`, which
is the seam a producer in another repository writes against without
importing this package. The JSON codec is :mod:`.serialization`; the
Tools-side exporter is :mod:`.adapters`.

TypeScript twin: **none**, deliberately, mirroring
:mod:`shared.python.swing_sim.delivery_interchange`, which also has no
twin. The wire is produced and consumed by Python flight producers;
the web surfaces read trajectories through the P8 playback transport,
whose sample-to-frame mapping is already golden-pinned across both
runtimes. A twin lands with the first TypeScript *producer*, not
before.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from shared.python.contracts import require

BALL_FLIGHT_TRAJECTORY_FORMAT = "swing_sim.ball_flight_trajectory/1"

FLIGHT_FRAME_ID = "flight_xfwd_yleft_zup"
"""UpstreamDrift flight frame: x forward, y left, z up; ground at z = 0."""

APP_FRAME_ID = "app_xtarget_yup_zright"
"""AffineDrift app frame: x target, y up, z right; ground at y = 0."""

FRAME_IDS = (APP_FRAME_ID, FLIGHT_FRAME_ID)
"""Every frame v1 can declare, sorted."""

OPTIONAL_CHANNELS = ("spin_rad_s", "velocity_mps")
"""Optional per-sample channels, sorted; each is a 3-vector in SI units."""

__all__ = [
    "APP_FRAME_ID",
    "BALL_FLIGHT_TRAJECTORY_FORMAT",
    "FLIGHT_FRAME_ID",
    "FRAME_IDS",
    "OPTIONAL_CHANNELS",
    "BallFlightSample",
    "BallFlightTrajectory",
    "TrajectoryProvenance",
    "from_samples",
    "parameter_digest",
]


def _identifier(value: object, name: str) -> str:
    """Return a trimmed nonempty string, or refuse."""
    require(
        isinstance(value, str) and value.strip() == value and value != "",
        f"{name} must be a trimmed nonempty string",
    )
    return str(value)


def _finite_triplet(value: object, name: str) -> tuple[float, float, float]:
    """Return a finite SI 3-vector, or refuse."""
    require(isinstance(value, (tuple, list)), f"{name} must be a 3-vector")
    items: tuple[Any, ...] = tuple(cast("tuple[Any, ...] | list[Any]", value))
    require(len(items) == 3, f"{name} must be a 3-vector")
    for item in items:
        require(
            not isinstance(item, bool) and isinstance(item, (int, float)),
            f"{name} components must be numbers",
        )
    values = tuple(float(item) for item in items)
    require(all(math.isfinite(item) for item in values), f"{name} must be finite")
    return (values[0], values[1], values[2])


def parameter_digest(parameters: Mapping[str, float | int | str]) -> str:
    """Return the SHA-256 provenance digest of a model's parameter set.

    The algorithm is part of the wire contract, so a producer in another
    repository can reproduce it byte-for-byte without importing this
    module::

        payload = json.dumps(
            dict(parameters),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    Args:
        parameters: A flat, nonempty mapping of parameter name to a
            finite number or a string. Nested containers are refused —
            a digest whose input shape is open-ended is not a contract.

    Returns:
        The 64-character lowercase hex digest.

    Raises:
        ContractViolationError: If the mapping is empty, has a
            non-string key, or carries a value that is neither a finite
            number nor a string.
    """
    require(isinstance(parameters, Mapping), "parameters must be a mapping")
    require(len(parameters) > 0, "parameters must be nonempty")
    payload: dict[str, float | str] = {}
    for key, value in parameters.items():
        name = _identifier(key, "parameter name")
        if isinstance(value, str):
            payload[name] = value
            continue
        require(
            not isinstance(value, bool) and isinstance(value, (int, float)),
            f"parameter {name!r} must be a finite number or a string",
        )
        number = float(value)
        require(math.isfinite(number), f"parameter {name!r} must be finite")
        payload[name] = number
    text = json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TrajectoryProvenance:
    """Which family, which model, and which parameters produced a record.

    Attributes:
        model_family: The producing family's stable identifier, e.g.
            ``"swing_sim.flight"`` or ``"ud.flight_models"``. Families
            are never merged, so this is the primary label a viewer
            shows beside a curve.
        model_name: The model's own display name within that family,
            e.g. ``"Waterloo/Penner"``.
        parameter_digest: 64-character lowercase hex SHA-256 from
            :func:`parameter_digest`, comparable only within a family.
    """

    model_family: str
    model_name: str
    parameter_digest: str

    def __post_init__(self) -> None:
        """Refuse an unattributable or malformed provenance."""
        _identifier(self.model_family, "model_family")
        _identifier(self.model_name, "model_name")
        digest = self.parameter_digest
        require(
            isinstance(digest, str)
            and len(digest) == 64
            and all(char in "0123456789abcdef" for char in digest),
            "parameter_digest must be a 64-character lowercase hex SHA-256",
        )


@dataclass(frozen=True)
class BallFlightSample:
    """One time-stamped ball state in the record's declared frame."""

    time_s: float
    position_m: tuple[float, float, float]
    velocity_mps: tuple[float, float, float] | None = None
    spin_rad_s: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        """Coerce and validate the sample's SI values."""
        require(
            not isinstance(self.time_s, bool)
            and isinstance(self.time_s, (int, float))
            and math.isfinite(self.time_s),
            "time_s must be finite",
        )
        require(float(self.time_s) >= 0.0, "time_s must be non-negative", self.time_s)
        object.__setattr__(self, "time_s", float(self.time_s))
        object.__setattr__(
            self, "position_m", _finite_triplet(self.position_m, "position_m")
        )
        for name in OPTIONAL_CHANNELS:
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite_triplet(value, name))

    @property
    def channels(self) -> tuple[str, ...]:
        """The optional channels this sample carries, sorted."""
        return tuple(
            name for name in OPTIONAL_CHANNELS if getattr(self, name) is not None
        )


@dataclass(frozen=True)
class BallFlightTrajectory:
    """A validated ball flight in a declared frame with model provenance."""

    source_id: str
    frame_id: str
    provenance: TrajectoryProvenance
    samples: tuple[BallFlightSample, ...]

    def __post_init__(self) -> None:
        """Refuse an unusable record: see the module docstring for why."""
        _identifier(self.source_id, "source_id")
        require(
            self.frame_id in FRAME_IDS,
            f"frame_id must be one of {list(FRAME_IDS)}",
            self.frame_id,
        )
        require(
            isinstance(self.provenance, TrajectoryProvenance),
            "provenance must be TrajectoryProvenance and is mandatory",
        )
        require(
            isinstance(self.samples, tuple)
            and len(self.samples) >= 2
            and all(isinstance(item, BallFlightSample) for item in self.samples),
            "samples must be a tuple of at least two BallFlightSample records",
        )
        times = [sample.time_s for sample in self.samples]
        pairs = zip(times, times[1:], strict=False)
        require(
            all(later > earlier for earlier, later in pairs),
            "sample times must be strictly increasing",
        )
        declared = self.samples[0].channels
        require(
            all(sample.channels == declared for sample in self.samples),
            "every sample must carry the same optional channels",
        )

    @property
    def channels(self) -> tuple[str, ...]:
        """The optional channels every sample carries, sorted."""
        return self.samples[0].channels

    @property
    def duration_s(self) -> float:
        """Physical timestamp of the last retained sample [s]."""
        return self.samples[-1].time_s


def from_samples(
    *,
    source_id: str,
    frame_id: str,
    provenance: TrajectoryProvenance,
    times_s: Sequence[float],
    positions_m: Sequence[Sequence[float]],
    velocities_mps: Sequence[Sequence[float]] | None = None,
    spins_rad_s: Sequence[Sequence[float]] | None = None,
) -> BallFlightTrajectory:
    """Build a record from parallel sample sequences — the producer seam.

    This is the constructor every flight producer targets, including
    producers that never import this package and instead write the
    documented JSON directly (see the module docstring for the wire).

    Contract:

    - **Units are SI.** ``times_s`` seconds, ``positions_m`` metres,
      ``velocities_mps`` metres per second, ``spins_rad_s`` radians per
      second as a spin *vector*. Converting from RPM or degrees is the
      producer's job and must happen before this call.
    - **Frame is declared, not inferred.** ``frame_id`` must be
      :data:`FLIGHT_FRAME_ID` (x forward, y left, z up) or
      :data:`APP_FRAME_ID` (x target, y up, z right), and every vector
      in the call — position, velocity, and spin alike — must already
      be expressed in it.
    - **Provenance is mandatory.** ``provenance`` names the family, the
      model, and the :func:`parameter_digest` of the parameters that
      model integrated with. There is no unattributed record.
    - **At least two samples**, with ``times_s`` strictly increasing and
      non-negative; every value finite.
    - **Optional channels are all-or-nothing.** Pass a velocity (or
      spin) sequence covering every sample, or pass ``None``; a partial
      sequence is refused rather than padded.
    - Sequence lengths must all match, and every 3-vector must have
      exactly three finite components.

    Args:
        source_id: Trimmed nonempty identifier of the producing run,
            e.g. ``"ud.flight_models:waterloo_penner"``. Free-form: it
            names the run, while ``provenance`` names the physics.
        frame_id: One of :data:`FRAME_IDS`.
        provenance: Mandatory model provenance.
        times_s: Strictly increasing non-negative sample times [s].
        positions_m: One ``(x, y, z)`` position per time [m].
        velocities_mps: Optional velocity per time [m/s].
        spins_rad_s: Optional spin vector per time [rad/s].

    Returns:
        The validated :class:`BallFlightTrajectory`.

    Raises:
        ContractViolationError: If any part of the contract above fails.
    """
    require(isinstance(times_s, Sequence), "times_s must be a sequence")
    require(isinstance(positions_m, Sequence), "positions_m must be a sequence")
    count = len(times_s)
    require(
        len(positions_m) == count,
        "positions_m must have one entry per time",
        (len(positions_m), count),
    )
    optional: dict[str, Sequence[Sequence[float]] | None] = {
        "velocity_mps": velocities_mps,
        "spin_rad_s": spins_rad_s,
    }
    for name, series in optional.items():
        if series is None:
            continue
        require(isinstance(series, Sequence), f"{name} series must be a sequence")
        require(
            len(series) == count,
            f"{name} must cover every sample or be omitted entirely",
            (len(series), count),
        )
    samples = tuple(
        BallFlightSample(
            time_s=times_s[index],
            position_m=_finite_triplet(positions_m[index], "position_m"),
            velocity_mps=(
                None
                if velocities_mps is None
                else _finite_triplet(velocities_mps[index], "velocity_mps")
            ),
            spin_rad_s=(
                None
                if spins_rad_s is None
                else _finite_triplet(spins_rad_s[index], "spin_rad_s")
            ),
        )
        for index in range(count)
    )
    return BallFlightTrajectory(
        source_id=source_id,
        frame_id=frame_id,
        provenance=provenance,
        samples=samples,
    )
