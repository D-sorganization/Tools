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

import pytest

from shared.python.scripting.scripting_env import (
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
