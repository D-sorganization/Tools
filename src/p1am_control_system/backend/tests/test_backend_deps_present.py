"""CI guard: assert the p1am backend test dependencies are actually installed.

The safety-critical backend tests (E-stop, alarm/historian, routing) open with
``pytest.importorskip`` so they degrade gracefully on a dev box without the
backend runtime deps. The failure mode that motivated #3534 was that CI *also*
lacked those deps, so the whole safety suite silently SKIPPED (= passed) instead
of gating.

This test closes that gap: when running under CI it performs HARD imports of the
backend runtime deps and the ``main`` app module, FAILING (not skipping) if any
is missing. Locally (no ``CI`` env var) it skips, so a dev without the deps is
unaffected. ``tools_core`` is intentionally NOT required — ``main`` falls back to
the pure-Python ``scada_fallback`` when the Rust wheel is absent.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

_IN_CI = os.environ.get("CI", "").lower() in {"1", "true", "yes"}

# Runtime deps the safety-critical backend suite needs in order to RUN.
_REQUIRED_DEPS = ("fastapi", "httpx", "pymodbus", "requests", "sqlmodel")


@pytest.mark.skipif(not _IN_CI, reason="dep-presence gate only enforced in CI")
@pytest.mark.parametrize("module_name", _REQUIRED_DEPS)
def test_backend_dependency_importable_in_ci(module_name: str) -> None:
    """Each backend runtime dep must HARD-import in CI (no silent skip)."""
    try:
        importlib.import_module(module_name)
    except ImportError:  # pragma: no cover - only on a misconfigured CI
        pytest.fail(
            f"Backend dependency {module_name!r} is not installed in CI. The "
            f"safety-critical p1am backend tests would silently skip. Install it "
            f"in the ci-standard tests job (see #3534)."
        )


@pytest.mark.skipif(not _IN_CI, reason="app-import gate only enforced in CI")
def test_backend_app_imports_in_ci(monkeypatch: pytest.MonkeyPatch) -> None:
    """The FastAPI ``main`` app must import in CI so its endpoint tests run."""
    # monkeypatch, not os.environ: a bare setdefault leaks into every suite that
    # runs after this one and makes their auth posture collection-order
    # dependent (#4061).
    monkeypatch.setenv("PLC_DRIVER", "simulator")
    monkeypatch.setenv("P1AM_DEV_NO_AUTH", "1")
    sys.path.insert(0, str(Path(__file__).parent.parent))
    try:
        main = importlib.import_module("main")
    except Exception as exc:  # pragma: no cover - only on a broken CI env
        pytest.fail(f"Backend `main` app failed to import in CI: {exc}")
    assert main.app is not None
