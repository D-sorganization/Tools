"""Tests for the interactive scripting environment."""

from __future__ import annotations

from pathlib import Path

import pytest

from shared.python.scripting.scripting_env import (
    _BLOCKED_BUILTINS,
    _BLOCKED_IMPORT_MODULES,
    ConsoleEnvironment,
)

# ---------------------------------------------------------------------------
# Sandbox / security tests (issue #2471)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_blocked_builtins_not_present_in_namespace() -> None:
    """Restricted builtins must not appear in the sandbox __builtins__ dict."""
    env = ConsoleEnvironment()
    sandbox_builtins = env.namespace["__builtins__"]
    assert isinstance(sandbox_builtins, dict), "__builtins__ must be a restricted dict"
    for name in _BLOCKED_BUILTINS:
        assert (
            name not in sandbox_builtins
        ), f"Blocked builtin '{name}' leaked into sandbox"


@pytest.mark.unit
def test_import_is_replaced_not_removed() -> None:
    """``__import__`` must be present in __builtins__ but be a restricted wrapper."""
    env = ConsoleEnvironment()
    sandbox_builtins = env.namespace["__builtins__"]
    assert (
        "__import__" in sandbox_builtins
    ), "__import__ must exist for C-extension sub-imports"
    # The wrapper must not be the real builtins.__import__
    import builtins as _builtins

    assert (
        sandbox_builtins["__import__"] is not _builtins.__import__
    ), "__import__ must be the restricted wrapper, not the real one"


@pytest.mark.unit
def test_open_is_blocked_in_sandbox(tmp_path: Path) -> None:
    """User code must not be able to call ``open()`` directly."""
    env = ConsoleEnvironment()
    _, err = env.execute("open('/etc/passwd', 'r')")
    assert err, "Expected an error when calling open() in the sandbox"
    assert "NameError" in err or "open" in err.lower()


@pytest.mark.unit
def test_os_system_is_blocked_via_import_restriction(tmp_path: Path) -> None:
    """``import os; os.system(...)`` must be blocked — os is in the deny-list.

    Regression test for issue #2471: arbitrary host-process commands must not
    be executable from the scripting sandbox.
    """
    env = ConsoleEnvironment()
    _, err = env.execute("import os; os.system('echo pwned')")
    # The restricted __import__ wrapper raises ImportError for os.
    assert err, "Expected an error when attempting os.system() via import"
    assert (
        "ImportError" in err or "blocked" in err.lower()
    ), f"Unexpected error message: {err!r}"


@pytest.mark.unit
def test_blocked_import_modules_are_denied() -> None:
    """Every module in _BLOCKED_IMPORT_MODULES must raise ImportError in the sandbox."""
    env = ConsoleEnvironment()
    # Spot-check a representative subset to keep the test fast.
    spot_check = {"os", "subprocess", "sys", "socket"}
    assert spot_check.issubset(
        _BLOCKED_IMPORT_MODULES
    ), "spot-check set must be a subset"
    for module_name in spot_check:
        _, err = env.execute(f"import {module_name}")
        assert err, f"Expected ImportError when importing '{module_name}'"
        assert (
            "ImportError" in err or "blocked" in err.lower()
        ), f"import of '{module_name}' did not produce expected error: {err!r}"


@pytest.mark.unit
def test_exec_builtin_is_blocked_in_sandbox() -> None:
    """``exec`` must not be callable from within user code."""
    env = ConsoleEnvironment()
    _, err = env.execute("exec('x = 1')")
    assert err, "Expected an error when calling exec() in the sandbox"
    assert "NameError" in err


@pytest.mark.unit
def test_eval_builtin_is_blocked_in_sandbox() -> None:
    """``eval`` must not be callable from within user code."""
    env = ConsoleEnvironment()
    _, err = env.execute("eval('1+1')")
    assert err, "Expected an error when calling eval() in the sandbox"
    assert "NameError" in err


@pytest.mark.unit
def test_compile_builtin_is_blocked_in_sandbox() -> None:
    """``compile`` must not be callable from within user code."""
    env = ConsoleEnvironment()
    _, err = env.execute("compile('x=1', '<s>', 'exec')")
    assert err, "Expected an error when calling compile() in the sandbox"
    assert "NameError" in err


@pytest.mark.unit
def test_max_execution_time_negative_raises() -> None:
    """Negative max_execution_time must be rejected at construction time."""
    with pytest.raises(ValueError, match="max_execution_time"):
        ConsoleEnvironment(max_execution_time=-1)


@pytest.mark.unit
def test_legitimate_code_still_executes_in_sandbox() -> None:
    """Basic numpy/math operations must remain functional after sandboxing."""
    env = ConsoleEnvironment()
    out, err = env.execute("np.array([1, 2, 3]).sum()")
    assert not err, f"Unexpected error: {err}"
    assert "6" in out


@pytest.mark.unit
def test_reset_reinstalls_restricted_builtins() -> None:
    """``reset()`` must reinstall the restricted __builtins__ dict."""
    env = ConsoleEnvironment()
    # Manually corrupt the namespace
    env.namespace["__builtins__"] = __builtins__
    env.reset()
    sandbox_builtins = env.namespace["__builtins__"]
    assert isinstance(sandbox_builtins, dict)
    for name in _BLOCKED_BUILTINS:
        assert name not in sandbox_builtins


# ---------------------------------------------------------------------------
# Pre-existing tests
# ---------------------------------------------------------------------------


def test_refresh_user_functions_propagates_system_level_errors(tmp_path: Path) -> None:
    """System-level failures in saved user code should not be hidden."""
    user_library = tmp_path / "user_library.py"
    user_library.write_text(
        "raise MemoryError('simulated exhaustion')\n", encoding="utf-8"
    )

    with pytest.raises(MemoryError, match="simulated exhaustion"):
        ConsoleEnvironment(user_lib_path=str(user_library))


def test_refresh_user_functions_reports_expected_user_code_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Expected user-code failures are reported without crashing the host."""
    user_library = tmp_path / "user_library.py"
    user_library.write_text(
        "raise ValueError('bad saved function')\n", encoding="utf-8"
    )

    ConsoleEnvironment(user_lib_path=str(user_library))

    captured = capsys.readouterr()
    assert "Error loading user library: bad saved function" in captured.err
