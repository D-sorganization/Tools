"""Shared 'how parameters vary' vocabulary for the variation engine.

Epic #4120 (V3). This module is the single, namespaced registry of
variables that any package in the repo can perturb, plus the frozen
:class:`NoiseSpec` / :class:`VariationPlan` value types that describe a
reproducible Monte-Carlo study over them.

Prior art (surveyed, credited)
------------------------------
- UpstreamDrift ``physics/aerodynamics/_config.py`` ``RandomizationConfig``
  and ``_environment.py`` ``EnvironmentRandomizer``: per-quantity Gaussian
  scales with ad-hoc clamping. Generalized here into per-variable
  :class:`NoiseSpec` with a distribution choice and explicit truncation.
- UpstreamDrift ``perturbation/config.py`` ``PerturbationConfig``: one
  global scalar ``noise_amplitude`` + noise-colour string. The per-variable
  spec list replaces that single knob so different inputs can carry
  different, unit-aware scales — the epic's core "one shared theme".
- Variable names and defaults come from
  :mod:`shared.python.swing_sim.solver.goals` (delivery + swing variables),
  :mod:`shared.python.swing_sim.impact` (club constants), and
  :mod:`shared.python.swing_sim.flight` (launch conditions), so the solver,
  the simulation session, and the variation engine speak one vocabulary.

Registry scheme
---------------
Keys are namespaced ``<category>.<name>`` strings, e.g.
``swing_sim.impact.delivery.face_angle_deg``. Built-in categories:

- ``swing_sim.impact.delivery`` — clubhead delivery front-end variables;
- ``swing_sim.swing`` — double-pendulum swing-plane tilts, impact timing,
  and joint damping (delivery speed/path/attack derived from the swing);
- ``swing_sim.club`` — clubhead mass / MOI / COR fed to the impact solve;
- ``swing_sim.flight.launch`` — direct launch conditions (flight only).

Other packages adopt the same scheme by calling :func:`register_variable`
with their own category prefix; the engine only ever sees registry keys.

JSON round-trip: :meth:`VariationPlan.to_json_dict` /
:meth:`VariationPlan.from_json_dict` (schema mirrored bit-for-bit by the
web port in ``rate_of_closure/web/src/model/variation.ts``).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from shared.python.contracts import require

from .registry import (
    CATEGORY_CLUB,
    CATEGORY_DELIVERY,
    CATEGORY_LAUNCH,
    CATEGORY_SWING,
    MODE_CATEGORIES,
    MODES,
    SWING_DERIVED_KEYS,
    VariableDef,
    keys_for_mode,
    register_variable,
    variable_registry,
    variables_in_category,
)

DISTRIBUTIONS: tuple[str, ...] = ("normal", "uniform", "triangular")
"""Supported sampling distributions (see :class:`NoiseSpec.scale`)."""

_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class NoiseSpec:
    """How one registry variable varies, run to run.

    Noise is additive about the plan's base value ``b``:

    - ``normal``: ``b + Normal(0, scale)`` (``scale`` = standard deviation);
    - ``uniform``: ``Uniform(b - scale, b + scale)`` (``scale`` = half-width);
    - ``triangular``: ``Triangular(b - scale, b, b + scale)`` (mode at the
      base value, ``scale`` = half-width).

    ``lower`` / ``upper`` (optional) truncate the sampled *absolute* value
    by clipping — samples never leave ``[lower, upper]``. Clipping (rather
    than resampling) keeps the draw count deterministic per stream, which
    the one-at-a-time sensitivity analysis relies on.
    """

    variable_key: str
    distribution: str = "normal"
    scale: float = 1.0
    lower: float | None = None
    upper: float | None = None

    def __post_init__(self) -> None:
        require(
            self.variable_key in variable_registry(),
            "variable_key must be a registered variable",
            self.variable_key,
        )
        require(
            self.distribution in DISTRIBUTIONS,
            f"distribution must be one of {DISTRIBUTIONS}",
            self.distribution,
        )
        require(
            math.isfinite(self.scale) and self.scale > 0.0,
            "scale must be finite and > 0",
            self.scale,
        )
        for name in ("lower", "upper"):
            value = getattr(self, name)
            require(
                value is None or math.isfinite(float(value)),
                f"{name} must be finite when given",
                value,
            )
        if self.lower is not None and self.upper is not None:
            require(
                self.lower < self.upper,
                "truncation bounds must satisfy lower < upper",
                (self.lower, self.upper),
            )

    def to_json_dict(self) -> dict[str, Any]:
        """Plain-JSON representation (schema shared with the web port)."""
        return {
            "variable_key": self.variable_key,
            "distribution": self.distribution,
            "scale": self.scale,
            "lower": self.lower,
            "upper": self.upper,
        }

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any]) -> NoiseSpec:
        """Inverse of :meth:`to_json_dict` (DbC-validated)."""
        return cls(
            variable_key=str(data["variable_key"]),
            distribution=str(data.get("distribution", "normal")),
            scale=float(data.get("scale", 1.0)),
            lower=None if data.get("lower") is None else float(data["lower"]),
            upper=None if data.get("upper") is None else float(data["upper"]),
        )


@dataclass(frozen=True)
class VariationPlan:
    """A reproducible N-run variation study.

    Attributes:
        mode: Pipeline slice — ``"delivery"`` (delivery → impact → flight),
            ``"swing"`` (pendulum → impact → flight), or ``"launch"``
            (launch → flight only).
        base_variables: Registry-key → base-value overrides; unlisted
            variables take their registry defaults.
        noise: One :class:`NoiseSpec` per varied variable (unique keys).
        n_runs: Number of Monte-Carlo runs (>= 1).
        seed: Master RNG seed; the engine derives one independent,
            subset-stable stream per noise spec from it.
        flight_model: Registry flight-model name (kept on the plan so a
            serialized study replays identically).
    """

    mode: str
    base_variables: Mapping[str, float] = field(default_factory=dict)
    noise: tuple[NoiseSpec, ...] = ()
    n_runs: int = 200
    seed: int = 0
    flight_model: str = "waterloo_penner"

    def __post_init__(self) -> None:
        require(self.mode in MODES, f"mode must be one of {MODES}", self.mode)
        require(self.n_runs >= 1, "n_runs must be >= 1", self.n_runs)
        require(self.seed >= 0, "seed must be >= 0", self.seed)
        require(len(self.noise) > 0, "plan must vary at least one variable", None)
        legal = set(keys_for_mode(self.mode))
        base = {str(k): float(v) for k, v in self.base_variables.items()}
        for key, value in base.items():
            require(key in legal, f"base variable not legal in {self.mode} mode", key)
            require(math.isfinite(value), "base value must be finite", (key, value))
        specs = tuple(self.noise)
        seen: set[str] = set()
        for spec in specs:
            require(
                isinstance(spec, NoiseSpec), "noise entries must be NoiseSpec", spec
            )
            require(
                spec.variable_key in legal,
                f"noise variable not legal in {self.mode} mode",
                spec.variable_key,
            )
            require(
                spec.variable_key not in seen,
                "duplicate noise spec for variable",
                spec.variable_key,
            )
            seen.add(spec.variable_key)
        object.__setattr__(self, "base_variables", MappingProxyType(base))
        object.__setattr__(self, "noise", specs)

    def resolved_base(self) -> dict[str, float]:
        """Full base mapping: registry defaults overlaid with overrides."""
        registry = variable_registry()
        values = {key: registry[key].default for key in keys_for_mode(self.mode)}
        values.update(self.base_variables)
        return values

    def to_json_dict(self) -> dict[str, Any]:
        """Plain-JSON representation (schema shared with the web port)."""
        return {
            "schema_version": _SCHEMA_VERSION,
            "mode": self.mode,
            "base_variables": dict(self.base_variables),
            "noise": [spec.to_json_dict() for spec in self.noise],
            "n_runs": self.n_runs,
            "seed": self.seed,
            "flight_model": self.flight_model,
        }

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any]) -> VariationPlan:
        """Inverse of :meth:`to_json_dict` (DbC-validated)."""
        version = int(data.get("schema_version", _SCHEMA_VERSION))
        require(version == _SCHEMA_VERSION, "unsupported schema_version", version)
        return cls(
            mode=str(data["mode"]),
            base_variables={
                str(k): float(v)
                for k, v in dict(data.get("base_variables", {})).items()
            },
            noise=tuple(
                NoiseSpec.from_json_dict(entry) for entry in data.get("noise", [])
            ),
            n_runs=int(data.get("n_runs", 200)),
            seed=int(data.get("seed", 0)),
            flight_model=str(data.get("flight_model", "waterloo_penner")),
        )

    def dumps(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_json_dict(), indent=2, sort_keys=True)

    @classmethod
    def loads(cls, text: str) -> VariationPlan:
        """Parse a plan from a JSON string."""
        return cls.from_json_dict(json.loads(text))


__all__ = [
    "CATEGORY_CLUB",
    "CATEGORY_DELIVERY",
    "CATEGORY_LAUNCH",
    "CATEGORY_SWING",
    "DISTRIBUTIONS",
    "MODES",
    "MODE_CATEGORIES",
    "SWING_DERIVED_KEYS",
    "NoiseSpec",
    "VariableDef",
    "VariationPlan",
    "keys_for_mode",
    "register_variable",
    "variable_registry",
    "variables_in_category",
]
