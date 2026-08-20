"""Package-wide isolation guards for the P1AM backend suite.

`main` exposes `control_context` as a module-level singleton, so every test in
this package shares one safety state. Once the suite began running under xdist
(#4548) tests from different modules interleave on the same worker, and a test
that latches E-stop without clearing it silently changes the expected response
of everything scheduled after it -- the failure reads as an unrelated
assertion, in an unrelated file, only on some runs.

This is the same class of defect as the import-time `P1AM_DEV_NO_AUTH`
assignment (#4061): correct in isolation, order-dependent in a suite. The fix
there was a per-test fixture, and it is the fix here.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest


@pytest.fixture(autouse=True)
def _clear_shared_estop_latch() -> Generator[None, None, None]:
    """Guarantee each test starts *and* leaves the E-stop latch cleared.

    Clearing on both sides is deliberate. Teardown alone would make the first
    test of a module depend on whichever test the worker happened to run
    before it -- exactly the ordering coupling this fixture exists to remove.
    A test that latches E-stop still observes it for the whole of its own body;
    only the boundaries are normalised.

    Import is deferred and failure tolerated because this package's modules are
    collected on interpreters where the FastAPI/httpx stack may be absent --
    those modules `importorskip` themselves, and this guard must not turn that
    skip into a collection error.
    """
    try:
        from main import control_context
    except Exception:  # pragma: no cover - stack absent; module skips itself
        yield
        return
    control_context.clear_estop()
    try:
        yield
    finally:
        control_context.clear_estop()
