"""Memory profiling respects ownership of the process-global trace session."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from utils import debug_memory, debug_utils


@pytest.fixture
def tracing(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    state = SimpleNamespace(active=False)
    allocations = [SimpleNamespace(traceback=f"fixture:{i}", size=i) for i in range(12)]
    snapshot = SimpleNamespace(statistics=Mock(return_value=allocations))
    tracer = SimpleNamespace(
        is_tracing=lambda: state.active,
        start=Mock(side_effect=lambda: setattr(state, "active", True)),
        stop=Mock(side_effect=lambda: setattr(state, "active", False)),
        get_traced_memory=Mock(return_value=(2 * 1024**2, 4 * 1024**2)),
        take_snapshot=Mock(return_value=snapshot),
        state=state,
    )
    monkeypatch.setattr(debug_memory, "tracemalloc", tracer)
    monkeypatch.setattr(debug_memory.gc, "collect", Mock())
    monkeypatch.setattr(
        debug_memory, "get_memory_usage", Mock(side_effect=[10.0, 12.5])
    )
    monkeypatch.setattr(debug_utils, "is_debug_mode", lambda: False)
    return tracer


@pytest.mark.parametrize("already_tracing", [False, True])
def test_tracker_only_stops_tracing_it_started(
    tracing: SimpleNamespace, already_tracing: bool
) -> None:
    tracing.state.active = already_tracing
    with debug_memory.memory_tracker(log_result=False) as stats:
        assert tracing.state.active
        assert stats.top_allocations == []
    assert tracing.state.active is already_tracing
    assert tracing.start.call_count == (0 if already_tracing else 1)
    assert tracing.stop.call_count == (0 if already_tracing else 1)
    assert stats.current_mb == 2
    assert stats.peak_mb == 4
    assert stats.diff_mb == 2.5
    assert stats.top_allocations == [(f"fixture:{i}", i) for i in range(10)]


@pytest.mark.parametrize("already_tracing", [False, True])
def test_body_failure_preserves_error_and_trace_ownership(
    tracing: SimpleNamespace, already_tracing: bool
) -> None:
    tracing.state.active = already_tracing
    error = ValueError("fixture body failure")
    with pytest.raises(ValueError) as caught:
        with debug_memory.memory_tracker(log_result=False):
            raise error
    assert caught.value is error
    assert tracing.state.active is already_tracing


def test_measurement_setup_failure_cleans_up_owned_tracing(
    monkeypatch: pytest.MonkeyPatch, tracing: SimpleNamespace
) -> None:
    error = RuntimeError("fixture measurement failure")
    monkeypatch.setattr(debug_memory, "get_memory_usage", Mock(side_effect=error))
    with pytest.raises(RuntimeError) as caught:
        with debug_memory.memory_tracker():
            pytest.fail("setup failure must prevent entering the block")
    assert caught.value is error
    assert not tracing.state.active
    tracing.stop.assert_called_once_with()


@pytest.mark.parametrize("already_tracing", [False, True])
def test_snapshot_failure_still_obeys_trace_ownership(
    tracing: SimpleNamespace, already_tracing: bool
) -> None:
    tracing.state.active = already_tracing
    tracing.take_snapshot.side_effect = RuntimeError("fixture snapshot failure")
    with pytest.raises(RuntimeError, match="snapshot failure"):
        with debug_memory.memory_tracker():
            pass
    assert tracing.state.active is already_tracing


def test_real_nested_trackers_leave_outer_trace_alive() -> None:
    import tracemalloc

    previously_active = tracemalloc.is_tracing()
    try:
        with debug_memory.memory_tracker(log_result=False):
            with debug_memory.memory_tracker(log_result=False):
                allocation = bytearray(1024)
            assert tracemalloc.is_tracing()
            assert len(allocation) == 1024
        assert tracemalloc.is_tracing() is previously_active
    finally:
        if not previously_active and tracemalloc.is_tracing():
            tracemalloc.stop()


@pytest.mark.parametrize(
    ("log_result", "debug", "expected"),
    [(False, True, 0), (True, False, 0), (True, True, 1)],
)
def test_logging_requires_both_flags(
    monkeypatch: pytest.MonkeyPatch,
    tracing: SimpleNamespace,
    log_result: bool,
    debug: bool,
    expected: int,
) -> None:
    monkeypatch.setattr(debug_utils, "is_debug_mode", lambda: debug)
    logger = Mock()
    monkeypatch.setattr(debug_memory, "logger", logger)
    with debug_memory.memory_tracker("fixture block", log_result=log_result):
        pass
    assert logger.debug.call_count == expected


def test_missing_name_is_rejected_before_tracing(tracing: SimpleNamespace) -> None:
    with pytest.raises(ValueError, match="name"):
        with debug_memory.memory_tracker(None):  # type: ignore[arg-type]
            pass
    tracing.start.assert_not_called()


@pytest.mark.parametrize(
    ("platform", "rss", "expected"),
    [("linux", 2048, 2.0), ("darwin", 2 * 1024**2, 2.0)],
)
def test_resource_units_are_platform_specific(
    monkeypatch: pytest.MonkeyPatch, platform: str, rss: int, expected: float
) -> None:
    monkeypatch.setattr(debug_memory, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setitem(
        sys.modules,
        "resource",
        SimpleNamespace(
            RUSAGE_SELF=0, getrusage=lambda who: SimpleNamespace(ru_maxrss=rss)
        ),
    )
    assert debug_memory.get_memory_usage() == expected


@pytest.mark.parametrize("psutil_available", [False, True])
def test_optional_windows_memory_fallback(
    monkeypatch: pytest.MonkeyPatch, psutil_available: bool
) -> None:
    monkeypatch.setitem(sys.modules, "resource", None)
    process = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=3 * 1024**2))
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(Process=lambda: process) if psutil_available else None,
    )
    assert debug_memory.get_memory_usage() == (3.0 if psutil_available else 0.0)


def test_decorator_retains_return_value_and_metadata(tracing: SimpleNamespace) -> None:
    @debug_memory.memory_profile
    def decorated(value: int, *, offset: int) -> int:
        """Fixture documentation."""
        return value + offset

    assert decorated(4, offset=3) == 7
    assert decorated.__name__ == "decorated"
    assert decorated.__doc__ == "Fixture documentation."
    assert not tracing.state.active
