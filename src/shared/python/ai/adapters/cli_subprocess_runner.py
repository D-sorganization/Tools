"""Shared ``Popen`` runner for subscription CLI adapters (Tools #5727 wave 1.2)."""

from __future__ import annotations

import contextlib
import errno
import os
import signal
import subprocess
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from enum import StrEnum
else:
    from shared.python.compatibility import StrEnum
from typing import IO

from shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_OUTPUT_BYTE_CAP = 512 * 1024
DEFAULT_GRACE_KILL_SECONDS = 2.0
_WINDOWS_ARGV_PROMPT_LIMIT = 30_000
_WIN_CMD_UNSAFE_CHARS = ("\r", "\n", '"', "%", "^", "&", "<", ">", "|")
_TRUNCATION_MARKER = "[truncated {n} bytes]"


class PromptDelivery(StrEnum):
    """How a prompt is delivered to the CLI child process."""

    STDIN = "stdin"
    ARGV = "argv"
    TEMP_FILE = "temp_file"
    AUTO = "auto"


@dataclass(frozen=True)
class CliSubprocessResult:
    """Captured outcome from a bounded CLI subprocess invocation."""

    returncode: int
    stdout: str
    stderr: str
    stdout_truncated_bytes: int = 0
    stderr_truncated_bytes: int = 0
    prompt_temp_path: str | None = None


class CliSubprocessTimeoutError(subprocess.TimeoutExpired):
    """Raised when a CLI subprocess exceeds its wall-clock timeout."""


def scrub_cli_subprocess_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return a copy of ``env`` stripped of vault secrets and raw credential tokens."""
    source = os.environ if env is None else env
    sensitive_suffixes = ("_SECRET", "_PASSWORD", "_PRIVATE_KEY", "_TOKEN")
    return {
        key: value
        for key, value in source.items()
        if not key.upper().startswith("VAULT_SECRET_")
        and not any(key.upper().endswith(suffix) for suffix in sensitive_suffixes)
    }


def terminate_cli_process_group(pid: int | None) -> bool:
    """Best-effort kill of the CLI process group on timeout or cancellation."""
    if pid is None or pid <= 0:
        return False
    if os.name == "nt":
        return _windows_kill_process_tree(pid)
    if not hasattr(os, "killpg"):
        return False
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
        return True
    except OSError:
        return False


def run_cli_subprocess(
    argv: Sequence[str],
    *,
    prompt: str | None = None,
    prompt_delivery: PromptDelivery = PromptDelivery.STDIN,
    timeout: float,
    env: Mapping[str, str] | None = None,
    output_byte_cap: int = DEFAULT_OUTPUT_BYTE_CAP,
    grace_kill_seconds: float = DEFAULT_GRACE_KILL_SECONDS,
    encoding: str = "utf-8",
) -> CliSubprocessResult:
    """Run ``argv`` with bounded I/O, closed stdin (unless prompting), and tree kill."""
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if not argv:
        raise ValueError("argv must be non-empty")
    launch_argv, prompt_file, stdin_payload = _prepare_argv_and_prompt(
        list(argv), prompt=prompt, prompt_delivery=prompt_delivery
    )
    child_env = scrub_cli_subprocess_env(env)
    if prompt_file is not None:
        child_env = {**child_env, "SIDEKICK_CLI_PROMPT_FILE": prompt_file}
    creationflags = _creation_flags()
    stdin_handle: int | IO[bytes] | None
    if stdin_payload is not None:
        stdin_handle = subprocess.PIPE
    else:
        stdin_handle = subprocess.DEVNULL
    proc = subprocess.Popen(
        launch_argv,
        stdin=stdin_handle,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=child_env,
        creationflags=creationflags,
        start_new_session=(os.name != "nt"),
    )
    temp_paths = [prompt_file] if prompt_file else []
    try:
        readers = _start_bounded_readers(proc, output_byte_cap)
        deadline = time.monotonic() + timeout
        stdin_writer = _start_stdin_writer(proc, stdin_payload)
        try:
            _wait_for_process(proc, deadline)
        except subprocess.TimeoutExpired as exc:
            _graceful_then_forced_kill(proc, grace_kill_seconds=grace_kill_seconds)
            timeout_error = CliSubprocessTimeoutError(
                launch_argv, timeout, output=None, stderr=None
            )
            timeout_error.pid = proc.pid  # type: ignore[attr-defined]
            raise timeout_error from exc
        finally:
            if stdin_writer is not None:
                stdin_writer.join(timeout=1.0)
        assert readers.stdout is not None and readers.stderr is not None
        stdout_raw, stdout_extra = readers.stdout.result()
        stderr_raw, stderr_extra = readers.stderr.result()
        stdout, stdout_trunc = _decode_with_marker(stdout_raw, stdout_extra, encoding)
        stderr, stderr_trunc = _decode_with_marker(stderr_raw, stderr_extra, encoding)
        return CliSubprocessResult(
            returncode=int(proc.returncode or 0),
            stdout=stdout,
            stderr=stderr,
            stdout_truncated_bytes=stdout_trunc,
            stderr_truncated_bytes=stderr_trunc,
            prompt_temp_path=prompt_file,
        )
    finally:
        _cleanup_temp_paths(temp_paths)
        if proc.poll() is None:
            _graceful_then_forced_kill(proc, grace_kill_seconds=grace_kill_seconds)


def _prepare_argv_and_prompt(
    argv: list[str],
    *,
    prompt: str | None,
    prompt_delivery: PromptDelivery,
) -> tuple[list[str], str | None, bytes | None]:
    if prompt is None or prompt == "":
        return argv, None, None
    delivery = _resolve_prompt_delivery(argv, prompt, prompt_delivery)
    if delivery is PromptDelivery.STDIN:
        return _argv_without_prompt_placeholders(argv), None, prompt.encode("utf-8")
    if delivery is PromptDelivery.ARGV:
        argv_copy = list(argv)
        if "-p" in argv_copy:
            index = argv_copy.index("-p")
            if index + 1 < len(argv_copy):
                argv_copy[index + 1] = prompt
                return argv_copy, None, None
        return [*argv_copy, prompt], None, None
    prompt_path = _write_temp_prompt(prompt)
    return _argv_without_prompt_placeholders(argv), prompt_path, None


def _resolve_prompt_delivery(
    argv: Sequence[str], prompt: str, delivery: PromptDelivery
) -> PromptDelivery:
    if delivery is not PromptDelivery.AUTO:
        return delivery
    if len(prompt) > _WINDOWS_ARGV_PROMPT_LIMIT:
        return PromptDelivery.STDIN
    if (
        os.name == "nt"
        and argv
        and argv[0].lower().endswith((".cmd", ".bat"))
        and any(ch in prompt for ch in _WIN_CMD_UNSAFE_CHARS)
    ):
        return PromptDelivery.STDIN
    argv_style = "-p" in argv or "exec" in argv
    return PromptDelivery.ARGV if argv_style else PromptDelivery.STDIN


def _argv_without_prompt_placeholders(argv: Sequence[str]) -> list[str]:
    if "-p" not in argv:
        return list(argv)
    cleaned: list[str] = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg == "-p":
            skip_next = True
            continue
        cleaned.append(arg)
    return cleaned


def _write_temp_prompt(prompt: str) -> str:
    handle, path = tempfile.mkstemp(prefix="sidekick-cli-prompt-", suffix=".txt")
    os.close(handle)
    Path(path).write_text(prompt, encoding="utf-8")
    return path


def _creation_flags() -> int:
    if os.name != "nt":
        return 0
    flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    if hasattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB"):
        flags |= subprocess.CREATE_BREAKAWAY_FROM_JOB
    return int(flags)


def _start_stdin_writer(
    proc: subprocess.Popen[bytes], stdin_payload: bytes | None
) -> threading.Thread | None:
    stdin_stream = proc.stdin
    if stdin_payload is None or stdin_stream is None:
        return None
    payload = stdin_payload

    def _run() -> None:
        stdin_stream.write(payload)
        stdin_stream.close()

    thread = threading.Thread(
        target=_run, name="cli-subprocess-stdin-writer", daemon=True
    )
    thread.start()
    return thread


def _wait_for_process(proc: subprocess.Popen[bytes], deadline: float) -> None:
    while proc.poll() is None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(proc.args, 0)
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired as exc:
            if time.monotonic() >= deadline:
                raise exc


def _start_bounded_readers(
    proc: subprocess.Popen[bytes],
    cap: int,
) -> _BoundedReaders:
    assert proc.stdout is not None
    assert proc.stderr is not None
    readers = _BoundedReaders()
    readers.stdout = _spawn_reader(proc.stdout, cap)
    readers.stderr = _spawn_reader(proc.stderr, cap)
    return readers


@dataclass
class _BoundedReaders:
    stdout: _ReaderHandle | None = None
    stderr: _ReaderHandle | None = None


@dataclass
class _ReaderHandle:
    thread: threading.Thread
    payload: list[tuple[bytes, int]]

    def result(self) -> tuple[bytes, int]:
        self.thread.join()
        return self.payload[0]


def _spawn_reader(pipe: IO[bytes], cap: int) -> _ReaderHandle:
    payload: list[tuple[bytes, int]] = []

    def _run() -> None:
        payload.append(_read_bounded(pipe, cap))

    thread = threading.Thread(target=_run, name="cli-subprocess-reader", daemon=True)
    thread.start()
    return _ReaderHandle(thread=thread, payload=payload)


def _read_bounded(pipe: IO[bytes], cap: int) -> tuple[bytes, int]:
    chunks: list[bytes] = []
    kept = 0
    truncated = 0
    while True:
        piece = pipe.read(65536)
        if not piece:
            break
        if kept < cap:
            allowed = min(len(piece), cap - kept)
            chunks.append(piece[:allowed])
            kept += allowed
            truncated += len(piece) - allowed
        else:
            truncated += len(piece)
    return b"".join(chunks), truncated


def _decode_with_marker(raw: bytes, truncated: int, encoding: str) -> tuple[str, int]:
    text = raw.decode(encoding, errors="replace")
    if truncated > 0:
        text = f"{text}{_TRUNCATION_MARKER.format(n=truncated)}"
    return text, truncated


def _graceful_then_forced_kill(
    proc: subprocess.Popen[bytes], *, grace_kill_seconds: float
) -> None:
    pid = proc.pid
    if os.name == "nt":
        _windows_kill_process_tree(
            pid, graceful=True, grace_kill_seconds=grace_kill_seconds
        )
        return
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    deadline = time.monotonic() + grace_kill_seconds
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.05)
    terminate_cli_process_group(pid)


def _windows_kill_process_tree(
    pid: int, *, graceful: bool = False, grace_kill_seconds: float = 0.0
) -> bool:
    if graceful and grace_kill_seconds > 0:
        with contextlib.suppress(Exception):
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T"],
                check=False,
                capture_output=True,
                timeout=grace_kill_seconds,
            )
        time.sleep(grace_kill_seconds)
    with contextlib.suppress(Exception):
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            check=False,
            capture_output=True,
            timeout=max(grace_kill_seconds, 2.0),
        )
    return True


def _cleanup_temp_paths(paths: Sequence[str | None]) -> None:
    for path in paths:
        if not path:
            continue
        with contextlib.suppress(OSError):
            Path(path).unlink()


def ensure_executable(binary_path: str) -> None:
    """Raise when ``binary_path`` is missing or not executable."""
    path = Path(binary_path)
    if not path.exists():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(path))
    if not os.access(path, os.X_OK):
        raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), str(path))


def invoke_cli_subprocess_for_adapter(
    binary: str,
    argv: Sequence[str],
    *,
    provider: str,
    prompt: str | None = None,
    prompt_delivery: PromptDelivery = PromptDelivery.STDIN,
    timeout: float,
) -> CliSubprocessResult:
    """Run a subscription CLI binary, mapping filesystem errors to provider errors."""
    executable = binary
    try:
        ensure_executable(executable)
    except FileNotFoundError as exc:
        from shared.python.ai.exceptions import AIConnectionError

        raise AIConnectionError(
            f"CLI binary not found at {binary}",
            provider=provider,
        ) from exc
    except PermissionError as exc:
        from shared.python.ai.exceptions import AIConnectionError

        raise AIConnectionError(
            f"CLI binary is not executable at {binary}",
            provider=provider,
        ) from exc
    launch_argv = [executable, *list(argv)[1:]]
    try:
        return run_cli_subprocess(
            launch_argv,
            prompt=prompt,
            prompt_delivery=prompt_delivery,
            timeout=timeout,
        )
    except CliSubprocessTimeoutError as exc:
        from shared.python.ai.exceptions import AITimeoutError

        terminate_cli_process_group(getattr(exc, "pid", None))
        raise AITimeoutError(
            f"{provider} CLI timed out after {timeout}s",
            provider=provider,
        ) from exc
