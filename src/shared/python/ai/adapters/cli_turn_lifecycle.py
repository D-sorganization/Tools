"""Turn lifecycle, cancellation, idle timeout, and exit classification (#5727)."""

from __future__ import annotations

import contextlib
import ctypes
import re
import signal
import subprocess
import sys
import threading
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from shared.python.ai.exceptions import (
    AIAuthenticationError,
    AICancelledError,
    AIConnectionError,
    AIIdleTimeoutError,
    AIProviderError,
    AIRateLimitError,
    AITimeoutError,
    AIUsageExhaustedError,
)
from shared.python.lifecycle import LifecycleFailure, LifecycleSnapshot, LifecycleStatus

if TYPE_CHECKING:
    from shared.python.ai.adapters.cli_subprocess_runner import (
        CliSubprocessResult,
        PromptDelivery,
    )

_AUTH_RE = re.compile(
    r"not\s+logged\s+in|login\s+required|please\s+run\s+['\"]?\w+\s+login|"
    r"authentication\s+failed|invalid\s+api\s+key|\bunauthorized\b|\b401\b",
    re.IGNORECASE,
)
_USAGE_RE = re.compile(
    r"usage\s+limit\s+exhausted|insufficient_quota|quota\s+(?:exhausted|exceeded)|"
    r"billing\s+cycle|plan\s+usage\s+limit|credit\s+balance",
    re.IGNORECASE,
)
_RATE_RE = re.compile(r"too\s+many\s+requests|rate[\s_-]?limit|\b429\b", re.IGNORECASE)
_RETRY_AFTER_RE = re.compile(r"retry\s+after\s+(\d+(?:\.\d+)?)\s*s?", re.IGNORECASE)


class CliSubprocessCancelledError(RuntimeError):
    """Raised when a CLI subprocess is cancelled before or during execution."""

    def __init__(
        self,
        msg: str = "CLI subprocess cancelled",
        *,
        pid: int | None = None,
        partial_output: str = "",
    ) -> None:
        super().__init__(msg)
        self.pid = pid
        self.partial_output = partial_output


class CliSubprocessIdleTimeoutError(subprocess.TimeoutExpired):
    """Raised when a CLI subprocess emits no stdout/stderr past ``idle_timeout``."""

    def __init__(
        self,
        cmd: Any,
        timeout: float,
        *,
        partial_output: str = "",
        pid: int | None = None,
    ) -> None:
        super().__init__(cmd, timeout)
        self.partial_output = partial_output
        self.is_partial = bool(partial_output)
        self.pid = pid


class CliCancelToken:
    """Thread-safe, idempotent cancellation token for a single subprocess turn."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._pid: int | None = None
        self._killer: Callable[[int | None], bool] | None = None

    @property
    def is_cancelled(self) -> bool:
        """Return ``True`` if cancellation has been requested."""
        return self._event.is_set()

    def attach(
        self, pid: int | None, killer: Callable[[int | None], bool] | None = None
    ) -> bool:
        """Bind an active subprocess PID unless already cancelled."""
        with self._lock:
            if self._event.is_set():
                return False
            self._pid, self._killer = pid, killer
            return True

    def clear(self) -> None:
        """Clear bound process metadata once subprocess execution ends."""
        with self._lock:
            self._pid, self._killer = None, None

    def cancel(self) -> bool:
        """Cancel the turn idempotently and terminate any bound process tree."""
        with self._lock:
            if self._event.is_set():
                return False
            self._event.set()
            pid, killer = self._pid, self._killer
        if pid is not None and killer is not None:
            with contextlib.suppress(Exception):
                killer(pid)
        return True


def attach_kill_on_parent_exit(proc: subprocess.Popen[bytes]) -> int | None:
    """Attach a Windows Job Object with KILL_ON_JOB_CLOSE to ``proc``."""
    proc_handle = getattr(proc, "_handle", None)
    if sys.platform != "win32" or not proc_handle:
        return None
    try:
        from ctypes import wintypes

        class _BasicLimit(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class _IoCounters(ctypes.Structure):
            _fields_ = [
                (k, ctypes.c_uint64) for k in ("RO", "WO", "OO", "RT", "WT", "OT")
            ]

        class _ExtendedLimit(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _BasicLimit),
                ("IoInfo", _IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        k32 = ctypes.windll.kernel32
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        ]
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        job = k32.CreateJobObjectW(None, None)
        if not job:
            return None
        info = _ExtendedLimit()
        info.BasicLimitInformation.LimitFlags = (
            0x00002000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        if k32.SetInformationJobObject(
            job, 9, ctypes.byref(info), ctypes.sizeof(info)
        ) and k32.AssignProcessToJobObject(job, int(proc_handle)):
            return int(job)
        k32.CloseHandle(job)
        return None
    except (OSError, AttributeError, ValueError):
        return None


def close_parent_exit_guard(guard: int | None) -> None:
    """Close parent-exit Job Object handle after child process termination."""
    if sys.platform == "win32" and guard:
        with contextlib.suppress(OSError, AttributeError):
            ctypes.windll.kernel32.CloseHandle(guard)


def posix_pdeathsig_preexec() -> Callable[[], None] | None:
    """Return a Linux ``preexec_fn`` setting PR_SET_PDEATHSIG(SIGTERM)."""

    def _set_pdeathsig() -> None:
        with contextlib.suppress(Exception):
            ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM, 0, 0, 0)

    return _set_pdeathsig if sys.platform.startswith("linux") else None


def classify_cli_nonzero_exit(
    provider: str, returncode: int, stderr: str, stdout: str = ""
) -> tuple[AIProviderError, str]:
    """Map CLI stderr/stdout to a distinct typed error and lifecycle outcome."""
    combined = f"{stderr or ''}\n{stdout or ''}"
    tail = (stderr or stdout or "")[-500:].strip()
    if _AUTH_RE.search(combined):
        return (
            AIAuthenticationError(
                f"{provider} CLI not logged in: {tail}", provider=provider
            ),
            "not_logged_in",
        )
    if _USAGE_RE.search(combined):
        return (
            AIUsageExhaustedError(
                f"{provider} CLI usage exhausted: {tail}", provider=provider
            ),
            "usage_exhausted",
        )
    if _RATE_RE.search(combined):
        m = _RETRY_AFTER_RE.search(combined)
        return (
            AIRateLimitError(
                f"{provider} CLI rate limited: {tail}",
                provider=provider,
                retry_after=float(m.group(1)) if m else None,
            ),
            "rate_limited",
        )
    return (
        AIProviderError(
            f"{provider} CLI exited {returncode}: {tail}", provider=provider
        ),
        "provider_error",
    )


def build_turn_lifecycle_snapshot(
    status: LifecycleStatus, outcome: str, *, message: str | None = None
) -> LifecycleSnapshot:
    """Build a canonical ``LifecycleSnapshot`` for a finished CLI turn."""
    failure = (
        LifecycleFailure(code=outcome, stage="cli_turn", message=message or outcome)
        if status is LifecycleStatus.FAILED
        else None
    )
    return LifecycleSnapshot(status=status, outcome=outcome, failure=failure)


def run_adapter_cli_turn(
    adapter: Any,
    argv: Sequence[str],
    *,
    runner_fn: Callable[..., CliSubprocessResult],
    prompt: str | None,
    prompt_delivery: PromptDelivery,
    cancel_token: CliCancelToken | None = None,
    missing_binary_message: str = "CLI binary not found on PATH.",
) -> CliSubprocessResult:
    """Run an adapter CLI turn via ``runner_fn`` and record ``last_turn_lifecycle``."""
    binary: str | None = adapter.binary
    provider: str = adapter.PROVIDER_NAME
    timeout: float = adapter.timeout
    idle_timeout: float | None = getattr(adapter, "idle_timeout", None)
    token = cancel_token or CliCancelToken()
    adapter._active_cancel_token = token

    def _record(status: LifecycleStatus, outcome: str, msg: str | None = None) -> None:
        adapter.last_turn_lifecycle = build_turn_lifecycle_snapshot(
            status, outcome, message=msg
        )

    try:
        if token.is_cancelled:
            _record(LifecycleStatus.CANCELLED, "cancelled")
            raise AICancelledError(
                f"{provider} CLI request cancelled", provider=provider
            )
        if binary is None:
            _record(LifecycleStatus.FAILED, "connection_error", missing_binary_message)
            raise AIConnectionError(missing_binary_message, provider=provider)
        try:
            is_real = getattr(runner_fn, "__module__", "").endswith(
                "cli_subprocess_runner"
            )
            kwargs: dict[str, Any] = {
                "provider": provider,
                "prompt": prompt,
                "prompt_delivery": prompt_delivery,
                "timeout": timeout,
            }
            if is_real:
                kwargs.update(idle_timeout=idle_timeout, cancel_token=token)
            result = runner_fn(binary, argv, **kwargs)
        except AIConnectionError as exc:
            _record(LifecycleStatus.FAILED, "connection_error", str(exc))
            raise
        except CliSubprocessCancelledError as exc:
            _record(LifecycleStatus.CANCELLED, "cancelled")
            raise AICancelledError(
                str(exc), provider=provider, partial_output=exc.partial_output
            ) from exc
        except CliSubprocessIdleTimeoutError as exc:
            _record(LifecycleStatus.FAILED, "idle_timeout", str(exc))
            raise AIIdleTimeoutError(
                f"{provider} CLI idle timed out after {idle_timeout}s",
                provider=provider,
                timeout=idle_timeout,
                partial_output=exc.partial_output,
                is_partial=exc.is_partial,
            ) from exc
        except AITimeoutError as exc:
            _record(LifecycleStatus.FAILED, "timeout", str(exc))
            raise
        if result.returncode != 0:
            err, outcome = classify_cli_nonzero_exit(
                provider, result.returncode, result.stderr, result.stdout
            )
            _record(LifecycleStatus.FAILED, outcome, str(err))
            raise err
        _record(LifecycleStatus.SUCCEEDED, "completed")
        return result
    finally:
        token.clear()
        adapter._active_cancel_token = None
