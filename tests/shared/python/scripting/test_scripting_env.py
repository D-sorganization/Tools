"""Tests for the scripting console AST escape screen (issue #3180).

Covers the in-process defense-in-depth screen added to
``ConsoleEnvironment`` that rejects attribute-introspection sandbox
escapes (``__class__``/``__subclasses__`` traversal) and
runtime-constructed dunder names built via ``getattr``/``type``/etc.
before any user source reaches ``compile``/``exec``.

The real (OS-level) trust boundary is documented in the module
docstring; these tests pin the in-process screen behavior.
"""

from __future__ import annotations

import threading

import pytest

from shared.python.scripting.scripting_env import (
    _BLOCKED_BUILTINS,
    _BLOCKED_IMPORT_MODULES,
    ConsoleEnvironment,
    SecurityError,
    _screen_source_for_escapes,
)

# ---------------------------------------------------------------------------
# _screen_source_for_escapes — direct unit tests
# ---------------------------------------------------------------------------

ESCAPE_SOURCES = [
    # Classic subclasses-traversal escape.
    "().__class__.__bases__[0].__subclasses__()",
    # Direct dunder attribute access.
    "x.__class__",
    "x.__globals__",
    "x.__bases__",
    "x.__subclasses__()",
    "(lambda: 0).__globals__['__builtins__']",
    # Runtime-constructed dunder name via getattr.
    "getattr((), '__class__')",
    "getattr((), '__cl' + 'ass__')",
    "getattr((), chr(95) * 2 + 'class' + chr(95) * 2)",
    # Other introspection gadgets with non-literal / dunder targets.
    "setattr(x, '__class__', y)",
    "vars(x)['__class__']",
    "type(x).__bases__",
    "delattr(x, '__dict__')",
]


@pytest.mark.unit
@pytest.mark.parametrize("source", ESCAPE_SOURCES)
def test_screen_rejects_escape_attempts(source: str) -> None:
    """Each known escape gadget must raise ``SecurityError``."""
    with pytest.raises(SecurityError):
        _screen_source_for_escapes(source)


SAFE_SOURCES = [
    "1 + 1",
    "x = [1, 2, 3]; sum(x)",
    "np.array([1, 2, 3]).mean()",
    "math.sqrt(16)",
    "s = 'hello'; s.upper()",
    "d = {'a': 1}; d.get('a')",
    "getattr(np, 'array')",  # literal, non-dunder attr name
    "type(5)",  # bare type() call is allowed
    "[i * 2 for i in range(3)]",
    "def f(a, b):\n    return a + b\nf(1, 2)",
]


@pytest.mark.unit
@pytest.mark.parametrize("source", SAFE_SOURCES)
def test_screen_allows_safe_sources(source: str) -> None:
    """Ordinary safe console operations must pass the screen unchanged."""
    # Should not raise.
    _screen_source_for_escapes(source)


# ---------------------------------------------------------------------------
# ConsoleEnvironment.execute — integration: escapes blocked, safe ops work
# ---------------------------------------------------------------------------


@pytest.fixture
def env() -> ConsoleEnvironment:
    return ConsoleEnvironment(max_execution_time=0)


@pytest.mark.unit
def test_execute_blocks_subclasses_escape(env: ConsoleEnvironment) -> None:
    """The subclasses traversal escape is reported, never reaching os."""
    out, err = env.execute("().__class__.__bases__[0].__subclasses__()")
    assert out == ""
    assert "SecurityError" in err
    # The gadget never executed, so no subclasses listing leaked to stdout.
    assert "subprocess" not in out


@pytest.mark.unit
def test_execute_blocks_getattr_constructed_dunder(
    env: ConsoleEnvironment,
) -> None:
    """A chr/concatenation-constructed dunder name is blocked."""
    out, err = env.execute("getattr((), chr(95)*2 + 'class' + chr(95)*2)")
    assert "SecurityError" in err
    assert out == ""


@pytest.mark.unit
def test_execute_allows_safe_expression(env: ConsoleEnvironment) -> None:
    """A plain expression still evaluates and prints its repr."""
    out, err = env.execute("2 + 3")
    assert err == ""
    assert out.strip() == "5"


@pytest.mark.unit
def test_execute_allows_safe_attribute_method(
    env: ConsoleEnvironment,
) -> None:
    """Non-dunder attribute/method access (e.g. numpy) still works."""
    out, err = env.execute("int(np.array([1, 2, 3]).sum())")
    assert err == ""
    assert out.strip() == "6"


# ---------------------------------------------------------------------------
# Restricted builtins / restricted import enforcement (issue #3700)
# ---------------------------------------------------------------------------
# reset() installs _make_restricted_builtins() as the namespace __builtins__
# (the "primary blast-radius guard"). These tests assert through execute()
# that the guard actually denies file I/O, code injection, and host-module
# imports — the load-bearing-but-previously-untested security layer.


@pytest.mark.unit
@pytest.mark.parametrize("blocked", sorted(_BLOCKED_BUILTINS))
def test_blocked_builtins_absent_from_namespace(
    env: ConsoleEnvironment, blocked: str
) -> None:
    """Each removed builtin (open/exec/eval/compile/breakpoint) is a NameError."""
    out, err = env.execute(f"{blocked}")
    assert out == ""
    assert "NameError" in err


@pytest.mark.unit
def test_execute_open_is_nameerror(env: ConsoleEnvironment) -> None:
    """File I/O via open() must not resolve — it is removed from builtins."""
    out, err = env.execute("open('some-file')")
    assert out == ""
    assert "NameError" in err
    assert "open" in err


@pytest.mark.unit
@pytest.mark.parametrize("module_name", sorted(_BLOCKED_IMPORT_MODULES))
def test_blocked_modules_rejected_via_import_statement(
    env: ConsoleEnvironment, module_name: str
) -> None:
    """Every module in the blocklist is rejected by an ``import`` statement."""
    out, err = env.execute(f"import {module_name}")
    assert out == ""
    assert "ImportError" in err
    assert "blocked in the scripting sandbox" in err


@pytest.mark.unit
@pytest.mark.parametrize("module_name", ["os", "subprocess", "sys", "socket", "ctypes"])
def test_blocked_modules_rejected_via_dunder_import_call(
    env: ConsoleEnvironment, module_name: str
) -> None:
    """``__import__('os')`` is also blocked by the restricted import wrapper.

    The dunder name literal is screened first; that screen itself blocks the
    call before the import wrapper is reached, so either a SecurityError or
    an ImportError is an acceptable denial — the key invariant is that the
    host module never loads (no stdout leak).
    """
    out, err = env.execute(f"__import__({module_name!r})")
    assert out == ""
    assert "Error" in err


@pytest.mark.unit
def test_benign_numpy_import_and_use_still_works(
    env: ConsoleEnvironment,
) -> None:
    """The blocklist must not break legitimate scientific imports."""
    out, err = env.execute(
        "import numpy as _np\nprint(int(_np.array([1, 2, 3]).sum()))"
    )
    assert err == ""
    assert out.strip() == "6"


@pytest.mark.unit
def test_blocked_module_submodule_import_rejected(
    env: ConsoleEnvironment,
) -> None:
    """A blocked top-level module cannot be reached via a submodule import."""
    out, err = env.execute("import os.path")
    assert out == ""
    assert "ImportError" in err


# ---------------------------------------------------------------------------
# Timeout async-exception delivery race (issue #3702)
# ---------------------------------------------------------------------------
# On Windows (and any non-main thread) the timeout is enforced by a daemon
# threading.Timer that injects a KeyboardInterrupt via the CPython C API.
# A late-firing timer must never leak a KeyboardInterrupt past the context
# boundary into unrelated host code.


@pytest.mark.unit
def test_quick_executes_never_leak_keyboard_interrupt() -> None:
    """Many fast executes under a live timeout never surface a stray KI.

    This drives the daemon-thread fallback path repeatedly: each execute()
    starts and cancels a Timer. If the cancel/fire handshake were racy, a
    borderline cancellation would occasionally inject a KeyboardInterrupt
    that escapes execute(). It must not.
    """
    # Force the daemon-thread fallback by running off the main thread.
    leaked: list[BaseException] = []

    def worker() -> None:
        environment = ConsoleEnvironment(max_execution_time=1)
        try:
            for _ in range(300):
                out, err = environment.execute("1 + 1")
                assert err == ""
                assert out.strip() == "2"
        except BaseException as exc:  # noqa: BLE001 — record any leak
            leaked.append(exc)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=30)
    assert not thread.is_alive(), "worker hung — possible interrupt deadlock"
    assert not leaked, f"execute() leaked an exception: {leaked!r}"


@pytest.mark.unit
def test_genuine_timeout_reported_as_timeouterror_not_interrupt() -> None:
    """A real timeout surfaces deterministically as TimeoutError, not bare KI.

    Runs an infinite loop on a non-main thread (the daemon-thread fallback
    path). The interrupt injected by the timer must be absorbed inside the
    timeout context and reported as a TimeoutError on stderr — it must not
    propagate out of execute() as a KeyboardInterrupt.
    """
    result: dict[str, tuple[str, str]] = {}
    escaped: list[BaseException] = []

    def worker() -> None:
        environment = ConsoleEnvironment(max_execution_time=1)
        try:
            result["value"] = environment.execute("n = 0\nwhile True:\n    n += 1")
        except BaseException as exc:  # noqa: BLE001 — record any escape
            escaped.append(exc)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=15)
    assert not thread.is_alive(), "timeout did not interrupt the busy loop"
    assert not escaped, f"timeout escaped execute(): {escaped!r}"
    out, err = result["value"]
    assert out == ""
    assert "TimeoutError" in err
