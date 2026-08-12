"""Secret-free NDJSON control harness for production-browser qualification.

This module is test support, not a second application launcher.  It starts the
real production companion and keeps native process control outside browser
state so Playwright can qualify hard-loss behavior without learning authority
credentials or its private listener.
"""

from __future__ import annotations

import argparse
import json
import secrets
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from http.client import HTTPConnection
from pathlib import Path
from typing import Final, Protocol, TextIO, cast
from urllib.parse import urlsplit

from rate_of_closure.web_authority.api import CAPABILITY_PATH
from rate_of_closure.web_companion.bundle import (
    CompanionWebBundle,
    build_companion_bundle,
)
from rate_of_closure.web_companion.runtime import CompanionRuntime, start_companion
from rate_of_closure.web_distribution.asset_resolver import resolve_web_assets

_CONTROL_ID_BYTES: Final = 24
_CONTROL_ID_MAX_LENGTH: Final = 128
_PROCESS_WAIT_S: Final = 10.0
_PROBE_TIMEOUT_S: Final = 10.0
_MAX_PROBE_BYTES: Final = 4_097
_COMMAND_FIELDS: Final = frozenset({"command", "control_id"})
_AUTHORITY_FACTORIES: Final = {
    "fast": (
        "tests.rate_of_closure.test_regional_ground_real_loopback:"
        "create_durable_test_authority_app"
    ),
    "cancellable": (
        "tests.rate_of_closure.test_regional_ground_real_loopback:"
        "create_cancellable_authority_app"
    ),
    "blocking": (
        "tests.rate_of_closure.test_regional_ground_real_loopback:"
        "create_durable_blocking_authority_app"
    ),
}
_RELEASE_ROOT: Final = (
    Path(__file__).parents[2] / "src" / "rate_of_closure" / "web" / "dist"
)
_ASSET_MANIFEST_NAME: Final = "rate-of-closure-assets.v1.json"


def _release_bundle() -> CompanionWebBundle:
    """Resolve the exact browser-built bundle through production validators."""
    manifest = (_RELEASE_ROOT / _ASSET_MANIFEST_NAME).read_bytes()
    return build_companion_bundle(resolve_web_assets(_RELEASE_ROOT, manifest))


class _Process(Protocol):
    """Minimum child-process control used by the test harness."""

    def poll(self) -> int | None: ...

    def kill(self) -> None: ...

    def wait(self, timeout: float) -> int: ...


class _Authority(Protocol):
    """Private authority facts retained solely in the harness process."""

    process: _Process
    token: str
    port: int


class _Companion(Protocol):
    """Production companion surface required by the browser harness."""

    url: str

    @property
    def authority(self) -> _Authority: ...

    def close(self) -> None: ...


class HarnessProtocolError(ValueError):
    """A sanitized failure caused by an invalid control command."""


@dataclass(frozen=True, slots=True)
class _AuthorityIdentity:
    """Private identity used only to compare authority generations."""

    token: str = field(repr=False)
    port: int = field(repr=False)


def _authority_running(authority: _Authority) -> bool:
    """Return whether the authority process has not reached a terminal state."""
    return authority.process.poll() is None


def _request_authority_replacement(gateway_url: str) -> None:
    """Trigger the production supervisor through its public same-origin route."""
    parsed = urlsplit(gateway_url)
    if parsed.scheme != "http" or parsed.hostname != "127.0.0.1":
        raise HarnessProtocolError("invalid gateway origin")
    if parsed.port is None or parsed.path != "/":
        raise HarnessProtocolError("invalid gateway origin")
    connection = HTTPConnection(parsed.hostname, parsed.port, timeout=_PROBE_TIMEOUT_S)
    try:
        connection.request("GET", CAPABILITY_PATH)
        response = connection.getresponse()
        response.read(_MAX_PROBE_BYTES)
    except (OSError, TimeoutError) as error:
        raise HarnessProtocolError("gateway replacement probe failed") from error
    finally:
        connection.close()


@dataclass(slots=True)
class BrowserCompanionHarness:
    """Own one real companion and expose a bounded, secret-free control plane."""

    runtime: _Companion = field(repr=False)
    control_id: str = field(repr=False)
    _lost_identity: _AuthorityIdentity | None = field(default=None, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the opaque control identity at the IPC trust boundary."""
        if (
            type(self.control_id) is not str
            or not self.control_id
            or self.control_id != self.control_id.strip()
            or len(self.control_id) > _CONTROL_ID_MAX_LENGTH
        ):
            raise ValueError("control_id must be bounded nonempty trimmed text")

    def ready_event(self) -> dict[str, object]:
        """Return the complete sanitized startup record for Playwright."""
        return {
            "event": "ready",
            "gateway_url": self.runtime.url,
            "authority_running": _authority_running(self.runtime.authority),
            "control_id": self.control_id,
        }

    def dispatch(self, source: object) -> dict[str, object]:
        """Validate and execute one exact control command."""
        command = self._command(source)
        if command == "authority_hard_loss":
            return self._hard_loss()
        if command == "observe_replacement":
            return self._observe_replacement()
        if command == "shutdown":
            self.close()
            return {"event": "stopped", "stopped": True}
        raise HarnessProtocolError("unsupported command")

    def close(self) -> None:
        """Close and reap the companion exactly once within production bounds."""
        if not self._closed:
            self.runtime.close()
            self._closed = True

    def _command(self, source: object) -> str:
        """Decode one command without permitting ambient control fields."""
        if not isinstance(source, Mapping) or set(source) != _COMMAND_FIELDS:
            raise HarnessProtocolError("invalid command shape")
        control_id = source.get("control_id")
        if type(control_id) is not str or not secrets.compare_digest(
            control_id, self.control_id
        ):
            raise HarnessProtocolError("invalid control identity")
        command = source.get("command")
        if type(command) is not str:
            raise HarnessProtocolError("invalid command name")
        return command

    def _hard_loss(self) -> dict[str, object]:
        """Kill one live authority while retaining only private comparison facts."""
        authority = self.runtime.authority
        if not _authority_running(authority):
            raise HarnessProtocolError("authority is not running")
        self._lost_identity = _AuthorityIdentity(authority.token, authority.port)
        authority.process.kill()
        authority.process.wait(timeout=_PROCESS_WAIT_S)
        return {"event": "authority_stopped", "authority_stopped": True}

    def _observe_replacement(self) -> dict[str, object]:
        """Trigger replacement and publish only boolean generation comparisons."""
        lost = self._lost_identity
        if lost is None:
            raise HarnessProtocolError("authority hard loss was not requested")
        _request_authority_replacement(self.runtime.url)
        authority = self.runtime.authority
        token_changed = not secrets.compare_digest(lost.token, authority.token)
        port_changed = lost.port != authority.port
        running = _authority_running(authority)
        return {
            "event": "authority_replaced",
            "authority_replaced": running and token_changed and port_changed,
            "authority_running": running,
            "token_changed": token_changed,
            "port_changed": port_changed,
        }


def _write_event(sink: TextIO, event: Mapping[str, object]) -> None:
    """Write and flush one compact NDJSON event."""
    sink.write(json.dumps(event, separators=(",", ":"), sort_keys=True) + "\n")
    sink.flush()


def _decode_command(line: str) -> object:
    """Decode one bounded command line into an untrusted JSON value."""
    if len(line) > 1_024:
        raise HarnessProtocolError("command exceeds maximum size")
    try:
        return json.loads(line)
    except json.JSONDecodeError as error:
        raise HarnessProtocolError("command is not valid JSON") from error


def run_command_stream(
    harness: BrowserCompanionHarness, source: TextIO, sink: TextIO
) -> int:
    """Serve control commands until EOF or an authenticated shutdown request."""
    _write_event(sink, harness.ready_event())
    try:
        for line in source:
            try:
                event = harness.dispatch(_decode_command(line))
            except HarnessProtocolError:
                event = {"event": "error", "code": "invalid_command"}
            except (OSError, RuntimeError, subprocess.SubprocessError):
                event = {"event": "error", "code": "control_failed"}
            _write_event(sink, event)
            if event.get("event") == "stopped":
                return 0
        return 0
    finally:
        harness.close()


def _authority_factory(authority_mode: str) -> str:
    """Resolve one closed fixture identity without accepting import strings."""
    try:
        return _AUTHORITY_FACTORIES[authority_mode]
    except KeyError as error:
        raise ValueError("unsupported browser authority fixture") from error


def start_browser_harness(
    state_root: Path, authority_mode: str = "fast"
) -> BrowserCompanionHarness:
    """Start the production companion with one closed browser-test fixture."""
    if not state_root.is_absolute() or not state_root.is_dir():
        raise ValueError("state_root must be an existing absolute directory")
    factory = _authority_factory(authority_mode)
    runtime: CompanionRuntime = start_companion(
        bundle=_release_bundle(),
        state_root=state_root,
        open_browser=False,
        authority_app_factory=factory,
    )
    return BrowserCompanionHarness(
        cast(_Companion, runtime), secrets.token_urlsafe(_CONTROL_ID_BYTES)
    )


def _parser() -> argparse.ArgumentParser:
    """Create the closed command-line contract used by Playwright."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--authority-mode",
        choices=tuple(_AUTHORITY_FACTORIES),
        default="fast",
    )
    parser.add_argument("--state-root", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Start the companion and serve strict NDJSON on standard streams."""
    arguments = _parser().parse_args(argv)
    try:
        if arguments.state_root is not None:
            root = arguments.state_root.resolve()
            root.mkdir(parents=True, exist_ok=True)
            harness = start_browser_harness(root, arguments.authority_mode)
            return run_command_stream(harness, sys.stdin, sys.stdout)
        with tempfile.TemporaryDirectory(prefix="roc-browser-authority-") as temporary:
            harness = start_browser_harness(
                Path(temporary).resolve(), arguments.authority_mode
            )
            return run_command_stream(harness, sys.stdin, sys.stdout)
    except BrokenPipeError:
        return 1
    except Exception:
        _write_event(sys.stdout, {"event": "fatal", "code": "startup_failed"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
