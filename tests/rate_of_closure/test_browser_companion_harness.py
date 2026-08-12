"""Contract tests for the out-of-browser production-companion harness."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest

from tests.rate_of_closure import browser_companion_harness as harness_module
from tests.rate_of_closure.browser_companion_harness import (
    BrowserCompanionHarness,
    HarnessProtocolError,
    run_command_stream,
)

_ASSET_MANIFEST = (
    Path(__file__).parents[2]
    / "src"
    / "rate_of_closure"
    / "web"
    / "dist"
    / "rate-of-closure-assets.v1.json"
)


@dataclass
class _Process:
    running: bool = True

    def poll(self) -> int | None:
        return None if self.running else 9

    def kill(self) -> None:
        self.running = False

    def wait(self, timeout: float) -> int:
        assert 0 < timeout <= 10.0
        self.running = False
        return 9


@dataclass
class _Authority:
    process: _Process
    token: str
    port: int


class _Runtime:
    def __init__(self) -> None:
        self.url = "http://127.0.0.1:43123/"
        self.authority = _Authority(_Process(), "private-token-a", 49123)
        self.closed = False

    def close(self) -> None:
        self.closed = True
        self.authority.process.running = False


def _assert_no_private_authority_values(payload: object) -> None:
    wire = json.dumps(payload)
    assert "private-token" not in wire
    assert "49123" not in wire
    assert "49234" not in wire
    assert "authority_port" not in wire
    assert "authority_url" not in wire
    assert "pid" not in wire.lower()


def test_ready_event_exposes_only_public_origin_and_opaque_control() -> None:
    runtime = _Runtime()
    harness = BrowserCompanionHarness(
        cast(Any, runtime), control_id="opaque-control-id"
    )

    event = harness.ready_event()

    assert event == {
        "event": "ready",
        "gateway_url": runtime.url,
        "authority_running": True,
        "control_id": "opaque-control-id",
    }
    _assert_no_private_authority_values(event)


def test_hard_loss_and_replacement_emit_only_boolean_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _Runtime()
    harness = BrowserCompanionHarness(
        cast(Any, runtime), control_id="opaque-control-id"
    )
    command = {"command": "authority_hard_loss", "control_id": "opaque-control-id"}

    stopped = harness.dispatch(command)

    assert stopped == {"event": "authority_stopped", "authority_stopped": True}
    replacement = _Authority(_Process(), "private-token-b", 49234)

    def replace_authority(_gateway_url: str) -> None:
        runtime.authority = replacement

    monkeypatch.setattr(
        harness_module, "_request_authority_replacement", replace_authority
    )
    observed = harness.dispatch(
        {"command": "observe_replacement", "control_id": "opaque-control-id"}
    )
    assert observed == {
        "event": "authority_replaced",
        "authority_replaced": True,
        "authority_running": True,
        "token_changed": True,
        "port_changed": True,
    }
    _assert_no_private_authority_values((stopped, observed))


@pytest.mark.parametrize(
    "command",
    [
        {},
        {"command": "shutdown", "control_id": "wrong"},
        {"command": "shutdown", "control_id": 123},
        {"command": "unknown", "control_id": "opaque-control-id"},
        {"command": "shutdown", "control_id": "opaque-control-id", "extra": True},
    ],
)
def test_dispatch_rejects_malformed_or_unauthorized_commands(command: object) -> None:
    harness = BrowserCompanionHarness(
        cast(Any, _Runtime()), control_id="opaque-control-id"
    )

    with pytest.raises(HarnessProtocolError):
        harness.dispatch(command)


def test_command_stream_is_strict_ndjson_and_closes_on_shutdown() -> None:
    runtime = _Runtime()
    harness = BrowserCompanionHarness(
        cast(Any, runtime), control_id="opaque-control-id"
    )
    source = io.StringIO(
        json.dumps({"command": "shutdown", "control_id": "opaque-control-id"}) + "\n"
    )
    sink = io.StringIO()
    ready = harness.ready_event()

    exit_code = run_command_stream(harness, source, sink)

    assert exit_code == 0
    assert runtime.closed is True
    events = [json.loads(line) for line in sink.getvalue().splitlines()]
    assert events == [ready, {"event": "stopped", "stopped": True}]
    _assert_no_private_authority_values(events)


def test_command_stream_sanitizes_protocol_errors_and_continues() -> None:
    runtime = _Runtime()
    harness = BrowserCompanionHarness(
        cast(Any, runtime), control_id="opaque-control-id"
    )
    source = io.StringIO(
        "not-json\n"
        + json.dumps({"command": "shutdown", "control_id": "opaque-control-id"})
        + "\n"
    )
    sink = io.StringIO()

    assert run_command_stream(harness, source, sink) == 0

    events = [json.loads(line) for line in sink.getvalue().splitlines()]
    assert events[1] == {"event": "error", "code": "invalid_command"}
    assert events[-1] == {"event": "stopped", "stopped": True}


def test_start_harness_uses_headless_companion_and_bounded_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _Runtime()
    captured: dict[str, object] = {}

    def fake_start_companion(**kwargs: object) -> Any:
        captured.update(kwargs)
        return runtime

    monkeypatch.setattr(harness_module, "start_companion", fake_start_companion)
    bundle = object()
    monkeypatch.setattr(harness_module, "_release_bundle", lambda: bundle)

    harness = harness_module.start_browser_harness(tmp_path, "cancellable")

    assert isinstance(harness, BrowserCompanionHarness)
    assert captured == {
        "bundle": bundle,
        "state_root": tmp_path,
        "open_browser": False,
        "authority_app_factory": (
            "tests.rate_of_closure.test_regional_ground_real_loopback:"
            "create_cancellable_authority_app"
        ),
    }


def test_start_harness_rejects_unknown_fixture(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported browser authority fixture"):
        harness_module.start_browser_harness(tmp_path, "not-a-fixture")


def test_main_sanitizes_startup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_start(_root: Path, _mode: str) -> BrowserCompanionHarness:
        raise RuntimeError("private-token-a at authority port 49123")

    monkeypatch.setattr(harness_module, "start_browser_harness", fail_start)

    exit_code = harness_module.main(["--state-root", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert json.loads(captured.out) == {"event": "fatal", "code": "startup_failed"}
    assert captured.err == ""
    _assert_no_private_authority_values(captured.out)


@pytest.mark.integration
@pytest.mark.headless_safe
def test_real_harness_replaces_authority_without_disclosing_identity(
    tmp_path: Path,
) -> None:
    if not _ASSET_MANIFEST.is_file():
        pytest.skip("exact production web bundle must be built before this test")
    harness = harness_module.start_browser_harness(tmp_path, "fast")
    try:
        ready = harness.ready_event()
        stopped = harness.dispatch(
            {
                "command": "authority_hard_loss",
                "control_id": ready["control_id"],
            }
        )
        replaced = harness.dispatch(
            {
                "command": "observe_replacement",
                "control_id": ready["control_id"],
            }
        )
        assert stopped["authority_stopped"] is True
        assert replaced["authority_replaced"] is True
        assert replaced["token_changed"] is True
        assert replaced["port_changed"] is True
        _assert_no_private_authority_values((ready, stopped, replaced))
    finally:
        harness.close()
