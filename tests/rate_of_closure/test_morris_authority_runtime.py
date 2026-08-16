"""Real child-process lifecycle smoke for the Morris authority runtime."""

from __future__ import annotations

import http.client
import io
from pathlib import Path

import pytest

from rate_of_closure.application.morris import runtime as authority_runtime
from rate_of_closure.application.morris.host import CAPABILITY_PATH
from rate_of_closure.application.morris.runtime import MorrisAuthorityRuntime


@pytest.mark.integration
def test_runtime_starts_authenticated_loopback_child_and_reaps_it() -> None:
    source_root = Path(__file__).resolve().parents[2] / "src"
    runtime = MorrisAuthorityRuntime.start(source_root, startup_timeout_s=15.0)
    process = runtime.process
    try:
        assert runtime.base_url.startswith("http://127.0.0.1:")
        assert runtime.vite_env == {
            "ROC_MORRIS_AUTHORITY_URL": runtime.base_url,
            "ROC_MORRIS_AUTHORITY_TOKEN": runtime.token,
        }
        assert runtime.token not in repr(runtime)
        port = int(runtime.base_url.rsplit(":", 1)[1])
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2.0)
        connection.request("GET", CAPABILITY_PATH)
        unauthorized = connection.getresponse()
        unauthorized.read()
        assert unauthorized.status == 401
        assert unauthorized.getheader("WWW-Authenticate") == "Bearer"
        connection.close()
    finally:
        runtime.close()
    assert process.poll() is not None
    assert process.returncode == 0
    assert process.stdout is not None
    assert process.stdout.closed


def test_runtime_rejects_unbounded_startup_timeout() -> None:
    with pytest.raises(ValueError, match="startup_timeout_s"):
        MorrisAuthorityRuntime.start(startup_timeout_s=0.0)


def test_runtime_fails_closed_when_interpreter_path_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(authority_runtime.sys, "executable", str(tmp_path / "missing"))
    with pytest.raises(RuntimeError, match="interpreter path"):
        MorrisAuthorityRuntime.start(startup_timeout_s=1.0)


class _CountingPipe(io.StringIO):
    def __init__(self, fail_close: bool = False) -> None:
        super().__init__()
        self.close_count = 0
        self.fail_close = fail_close

    def close(self) -> None:
        self.close_count += 1
        if self.fail_close:
            self.fail_close = False
            raise OSError("pipe close failed")
        super().close()


class _FakeProcess:
    def __init__(
        self, *, fail_terminate: bool = False, fail_close: bool = False
    ) -> None:
        self.stdout = _CountingPipe(fail_close)
        self.returncode: int | None = None
        self.terminate_count = 0
        self.kill_count = 0
        self.fail_terminate = fail_terminate

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminate_count += 1
        if self.fail_terminate:
            raise OSError("terminate failed")
        self.returncode = -15

    def kill(self) -> None:
        self.kill_count += 1
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        assert self.returncode is not None
        return self.returncode


@pytest.mark.parametrize("interruption", [KeyboardInterrupt(), SystemExit(7)])
def test_start_reaps_child_and_closes_pipe_on_base_exception(
    monkeypatch: pytest.MonkeyPatch,
    interruption: BaseException,
) -> None:
    process = _FakeProcess()
    diagnostics = io.StringIO()
    monkeypatch.setattr(
        authority_runtime, "_spawn_child", lambda *_args: (process, diagnostics)
    )

    def interrupt(*_args: object) -> int:
        raise interruption

    monkeypatch.setattr(authority_runtime, "_ready_port", interrupt)
    with pytest.raises(type(interruption)) as caught:
        MorrisAuthorityRuntime.start(startup_timeout_s=1.0)
    assert caught.value is interruption
    assert process.terminate_count == 1
    assert process.kill_count == 0
    assert process.stdout.close_count == 1


def test_runtime_close_is_idempotent_and_closes_completed_child_pipe() -> None:
    process = _FakeProcess()
    process.returncode = 0
    runtime = MorrisAuthorityRuntime(process, "http://127.0.0.1:1234", "private-token")  # type: ignore[arg-type]

    runtime.close()
    runtime.close()

    assert process.stdout.close_count == 1


def test_start_preserves_original_interrupt_when_cleanup_itself_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess(fail_terminate=True, fail_close=True)
    interruption = KeyboardInterrupt()
    diagnostics = io.StringIO()
    monkeypatch.setattr(
        authority_runtime, "_spawn_child", lambda *_args: (process, diagnostics)
    )

    def interrupt(*_args: object) -> int:
        raise interruption

    monkeypatch.setattr(authority_runtime, "_ready_port", interrupt)
    with pytest.raises(KeyboardInterrupt) as caught:
        MorrisAuthorityRuntime.start(startup_timeout_s=1.0)
    assert caught.value is interruption
    assert process.terminate_count == 1
    assert process.stdout.close_count == 1
