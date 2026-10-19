"""Every pendulum model defaults gravity to standard gravity (#5635, part 2).

Owner decision (2026-10-18): the default ``g`` is 9.80665 m/s^2 (standard
gravity), matching UpstreamDrift, replacing the former 9.81.  This is a
deliberate spec change, not a weakening: the literal below is the contract.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
from pathlib import Path
from types import ModuleType


from double_pendulum_golf.constants import GRAVITY_MSS, GRAVITY_STANDARD
from double_pendulum_golf.physics import PendulumParams
from double_pendulum_golf.physics_triple import TriplePendulumParams

STANDARD = 9.80665
_NATIVE_PY = (
    Path(__file__).resolve().parents[1]
    / "pendulum-core"
    / "python"
    / "physics_native.py"
)


def test_constants_are_standard_gravity() -> None:
    assert GRAVITY_STANDARD == STANDARD
    assert GRAVITY_MSS == GRAVITY_STANDARD


def test_double_pendulum_default_g() -> None:
    assert PendulumParams(m1=5.0, m2=0.11, L1=0.65, L2=1.1).g == STANDARD


def test_triple_pendulum_default_g() -> None:
    p = TriplePendulumParams(m1=1.0, m2=1.0, m3=1.0, L1=1.0, L2=1.0, L3=1.0)
    assert p.g == STANDARD


def test_jax_golfer_params_default_g() -> None:
    """JAX is optional (and mocked in some suites), so pin the source default."""
    path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "double_pendulum_golf"
        / "physics_golfer_jax.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "GolferParamsJAX"
    )
    field = next(
        n
        for n in cls.body
        if isinstance(n, ast.AnnAssign) and getattr(n.target, "id", "") == "g"
    )
    assert isinstance(field.value, ast.Name) and field.value.id == "GRAVITY_STANDARD"


def _load_native() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_physics_native_5635", _NATIVE_PY)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_native_python_wrappers_default_g() -> None:
    native = _load_native()
    assert native.STANDARD_GRAVITY == STANDARD
    assert native.DoublePendulumParams(1.0, 1.0, 1.0, 1.0).g == STANDARD
    for cls in (native.DoublePendulum, native.Golfer, native.GolferParams):
        assert inspect.signature(cls.__init__).parameters["g"].default == STANDARD
    assert (
        inspect.signature(native.DoublePendulumParams.__init__).parameters["g"].default
        == STANDARD
    )
