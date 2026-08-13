"""Shared, namespaced perturbation vocabulary for variation studies.

The frozen :class:`NoiseSpec`, :class:`PerturbationGroup`, and
:class:`VariationPlan` types describe reproducible independent or jointly
normal studies. Version 2 adds stable spec IDs, locus metadata, and validated
correlation/covariance groups while migrating version-1 plans losslessly.

Registry keys remain ``<category>.<name>`` strings shared with the solver,
impact, and flight packages. Other packages may extend the vocabulary through
:func:`register_variable`. The web port consumes the same v2 schema and stable
spec/group identifiers.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from shared.python.contracts import require
from shared.python.swing_sim._numeric_contracts import finite_real, integer

from .group_spec import PerturbationGroup
from .identity_contracts import stable_id as require_stable_id
from .identity_contracts import stable_id_array, strict_string
from .registry import (
    CATEGORY_BALL_SETUP,
    CATEGORY_CLUB,
    CATEGORY_DELIVERY,
    CATEGORY_LAUNCH,
    CATEGORY_SWING,
    LOCALIZED_TORQUE_VARIABLE_JOINTS,
    MODE_CATEGORIES,
    MODES,
    SWING_DERIVED_KEYS,
    VariableDef,
    keys_for_mode,
    register_variable,
    variable_registry,
    variables_in_category,
)

MAX_SAFE_INTEGER = 9_007_199_254_740_991
"""Largest integer represented exactly by both Python and JavaScript plan wires."""


def _normalize_json_signed_zero(value: Any) -> Any:
    """Return a JSON-shaped value with floating signed zero canonicalized."""
    if isinstance(value, float):
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        return {key: _normalize_json_signed_zero(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_json_signed_zero(item) for item in value]
    return value


DISTRIBUTIONS: tuple[str, ...] = ("normal", "uniform", "triangular")
"""Supported sampling distributions (see :class:`NoiseSpec.scale`)."""

SCHEMA_VERSION = 2
_SUPPORTED_SCHEMA_VERSIONS = (1, 2)


def _normalize_locus(
    time_window_s: tuple[float, float] | None,
    point_ids: tuple[str, ...],
) -> tuple[tuple[float, float] | None, tuple[str, ...]]:
    """Validate and normalize optional temporal/spatial locus metadata."""
    raw_window = cast(object, time_window_s)
    require(
        raw_window is None
        or (isinstance(raw_window, (tuple, list)) and len(raw_window) == 2),
        "time_window_s must contain finite start < end",
        raw_window,
    )
    normalized_window: tuple[float, float] | None = None
    if raw_window is not None:
        window = cast(tuple[object, object] | list[object], raw_window)
        start = finite_real(window[0], "time_window_s start")
        end = finite_real(window[1], "time_window_s end")
        require(
            start < end,
            "time_window_s must contain finite start < end",
            window,
        )
        normalized_window = (start, end)
    points = tuple(point_ids)
    for point in points:
        require_stable_id(point, "point_ids")
    require(
        len(set(points)) == len(points),
        "point_ids must be unique, non-empty stable IDs",
        points,
    )
    return normalized_window, points


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
    spec_id: str | None = None
    time_window_s: tuple[float, float] | None = None
    point_ids: tuple[str, ...] = ()

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
        scale = finite_real(cast(object, self.scale), "scale")
        require(scale > 0.0, "scale must be finite and > 0", self.scale)
        bounds: dict[str, float | None] = {}
        for name in ("lower", "upper"):
            value = getattr(self, name)
            bounds[name] = (
                None if value is None else finite_real(cast(object, value), name)
            )
        lower, upper = bounds["lower"], bounds["upper"]
        if lower is not None and upper is not None:
            require(
                lower < upper,
                "truncation bounds must satisfy lower < upper",
                (lower, upper),
            )
        resolved_spec_id = self.variable_key if self.spec_id is None else self.spec_id
        resolved_spec_id = require_stable_id(resolved_spec_id, "spec_id")
        window, points = _normalize_locus(self.time_window_s, self.point_ids)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "spec_id", resolved_spec_id)
        object.__setattr__(self, "time_window_s", window)
        object.__setattr__(self, "point_ids", points)

    @property
    def is_global(self) -> bool:
        """Whether the scalar evaluator can apply this perturbation globally."""
        return self.time_window_s is None and not self.point_ids

    def to_json_dict(self) -> dict[str, Any]:
        """Plain-JSON representation (schema shared with the web port)."""
        return {
            "variable_key": self.variable_key,
            "distribution": self.distribution,
            "scale": self.scale,
            "lower": self.lower,
            "upper": self.upper,
            "spec_id": self.spec_id,
            "time_window_s": (
                None if self.time_window_s is None else list(self.time_window_s)
            ),
            "point_ids": list(self.point_ids),
        }

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any]) -> NoiseSpec:
        """Inverse of :meth:`to_json_dict` (DbC-validated)."""
        return cls(
            variable_key=strict_string(data["variable_key"], "variable_key"),
            distribution=strict_string(
                data.get("distribution", "normal"), "distribution"
            ),
            scale=cast(float, data.get("scale", 1.0)),
            lower=cast(float | None, data.get("lower")),
            upper=cast(float | None, data.get("upper")),
            spec_id=(
                None
                if data.get("spec_id") is None
                else require_stable_id(data["spec_id"], "spec_id")
            ),
            time_window_s=cast(
                tuple[float, float] | None,
                data.get("time_window_s"),
            ),
            point_ids=stable_id_array(data.get("point_ids", []), "point_ids"),
        )


def _validate_plan_groups(
    specs: tuple[NoiseSpec, ...], groups: tuple[PerturbationGroup, ...]
) -> None:
    """Validate group references, disjointness, marginals, and scale semantics."""
    by_id = {spec.spec_id: spec for spec in specs}
    known_ids = set(by_id)
    assigned: set[str] = set()
    group_ids: set[str] = set()
    for group in groups:
        require(
            isinstance(group, PerturbationGroup),
            "groups entries must be PerturbationGroup",
            group,
        )
        require(group.group_id not in group_ids, "duplicate group_id", group.group_id)
        members = set(group.spec_ids)
        require(
            members <= known_ids,
            "group references unknown spec_id",
            members - known_ids,
        )
        require(
            not members & assigned,
            "a spec_id may belong to only one group",
            members & assigned,
        )
        member_specs = tuple(by_id[spec_id] for spec_id in group.spec_ids)
        require(
            all(spec.distribution == "normal" for spec in member_specs),
            "grouped specs must use normal distributions",
            group.spec_ids,
        )
        if group.matrix_kind == "covariance":
            diagonal = np.diag(np.asarray(group.matrix, dtype=float))
            expected = np.square([spec.scale for spec in member_specs])
            require(
                np.allclose(diagonal, expected, rtol=1e-9, atol=1e-12),
                "covariance diagonal must equal each NoiseSpec scale squared",
                (diagonal, expected),
            )
        assigned.update(members)
        group_ids.add(group.group_id)


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
    groups: tuple[PerturbationGroup, ...] = ()

    def __post_init__(self) -> None:
        require(self.mode in MODES, f"mode must be one of {MODES}", self.mode)
        n_runs = integer(cast(object, self.n_runs), "n_runs", minimum=1)
        seed = integer(cast(object, self.seed), "seed", minimum=0)
        require(n_runs <= MAX_SAFE_INTEGER, "n_runs must be a safe integer", n_runs)
        require(seed <= MAX_SAFE_INTEGER, "seed must be a safe integer", seed)
        require(len(self.noise) > 0, "plan must vary at least one variable", None)
        legal = set(keys_for_mode(self.mode))
        base = {
            str(k): finite_real(cast(object, v), "base value")
            for k, v in self.base_variables.items()
        }
        for key, value in base.items():
            require(key in legal, f"base variable not legal in {self.mode} mode", key)
            require(math.isfinite(value), "base value must be finite", (key, value))
        specs = tuple(self.noise)
        seen_variables: set[str] = set()
        seen_spec_ids: set[str] = set()
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
                spec.variable_key not in seen_variables,
                "duplicate variable_key would overwrite a scalar evaluator input",
                spec.variable_key,
            )
            require(
                spec.spec_id not in seen_spec_ids,
                "duplicate spec_id",
                spec.spec_id,
            )
            seen_variables.add(spec.variable_key)
            assert spec.spec_id is not None
            seen_spec_ids.add(spec.spec_id)
        groups = tuple(self.groups)
        _validate_plan_groups(specs, groups)
        object.__setattr__(self, "base_variables", MappingProxyType(base))
        object.__setattr__(self, "noise", specs)
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "n_runs", n_runs)
        object.__setattr__(self, "seed", seed)

    def resolved_base(self) -> dict[str, float]:
        """Full base mapping: registry defaults overlaid with overrides."""
        registry = variable_registry()
        values = {key: registry[key].default for key in keys_for_mode(self.mode)}
        values.update(self.base_variables)
        return values

    def to_json_dict(self) -> dict[str, Any]:
        """Plain-JSON representation (schema shared with the web port)."""
        return cast(
            dict[str, Any],
            _normalize_json_signed_zero(
                {
                    "schema_version": SCHEMA_VERSION,
                    "mode": self.mode,
                    "base_variables": dict(self.base_variables),
                    "noise": [spec.to_json_dict() for spec in self.noise],
                    "n_runs": self.n_runs,
                    "seed": self.seed,
                    "flight_model": self.flight_model,
                    "groups": [group.to_json_dict() for group in self.groups],
                }
            ),
        )

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any]) -> VariationPlan:
        """Inverse of :meth:`to_json_dict` (DbC-validated)."""
        version = integer(
            cast(object, data.get("schema_version", 1)),
            "schema_version",
            minimum=1,
        )
        require(
            version in _SUPPORTED_SCHEMA_VERSIONS,
            "unsupported schema_version",
            version,
        )
        return cls(
            mode=strict_string(data["mode"], "mode"),
            base_variables=dict(data.get("base_variables", {})),
            noise=tuple(
                NoiseSpec.from_json_dict(entry) for entry in data.get("noise", [])
            ),
            n_runs=cast(int, data.get("n_runs", 200)),
            seed=cast(int, data.get("seed", 0)),
            flight_model=strict_string(
                data.get("flight_model", "waterloo_penner"), "flight_model"
            ),
            groups=(
                ()
                if version == 1
                else tuple(
                    PerturbationGroup.from_json_dict(entry)
                    for entry in data.get("groups", [])
                )
            ),
        )

    def dumps(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_json_dict(), indent=2, sort_keys=True)

    @classmethod
    def loads(cls, text: str) -> VariationPlan:
        """Parse a plan from a JSON string."""
        return cls.from_json_dict(json.loads(text))


__all__ = [
    "CATEGORY_BALL_SETUP",
    "CATEGORY_CLUB",
    "CATEGORY_DELIVERY",
    "CATEGORY_LAUNCH",
    "CATEGORY_SWING",
    "LOCALIZED_TORQUE_VARIABLE_JOINTS",
    "MAX_SAFE_INTEGER",
    "DISTRIBUTIONS",
    "MODES",
    "MODE_CATEGORIES",
    "SCHEMA_VERSION",
    "SWING_DERIVED_KEYS",
    "NoiseSpec",
    "PerturbationGroup",
    "VariableDef",
    "VariationPlan",
    "keys_for_mode",
    "register_variable",
    "variable_registry",
    "variables_in_category",
]
