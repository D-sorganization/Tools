"""Tests for codemap.parsers."""

from __future__ import annotations

from codemap import parsers

PY_SOURCE = '''\
"""Module doc."""
import os
from typing import List

CONST = 1

class WGSReactor:
    """Water-gas shift."""

    def shift(self, x: int) -> int:
        return self.helper(x)

    def helper(self, y):
        return y + 1

def top():
    WGSReactor().shift(1)
'''

TS_SOURCE = """\
import { Foo } from './foo';

export class Bar {
  doThing(): number { return 1; }
}

export function topLevel(x: number) {
  return x + 1;
}

const arrow = (n: number) => n * 2;
"""


def test_extract_python_finds_class_methods_and_functions() -> None:
    r = parsers.dispatch("x.py", PY_SOURCE)
    assert r is not None
    assert r.language == "python"
    assert "os" in r.imports
    quals = {s.qualified: s for s in r.symbols}
    assert "WGSReactor" in quals
    assert quals["WGSReactor"].kind == "class"
    assert "WGSReactor.shift" in quals
    assert quals["WGSReactor.shift"].kind == "method"
    assert "top" in quals
    # calls_out captured.
    assert any("shift" in c for c in quals["top"].calls_out)
    # docstrings captured.
    assert quals["WGSReactor"].docstring.startswith("Water-gas")


def test_extract_typescript_finds_class_function_and_arrow() -> None:
    r = parsers.dispatch("x.ts", TS_SOURCE)
    assert r is not None
    assert r.language == "typescript"
    quals = {s.qualified for s in r.symbols}
    assert "Bar" in quals
    assert "Bar.doThing" in quals
    assert "topLevel" in quals
    assert "arrow" in quals
    assert any(imp == "./foo" for imp in r.imports)


def test_dispatch_unknown_extension_returns_none() -> None:
    assert parsers.dispatch("x.unknown", "blah") is None


def test_language_for_known_and_unknown() -> None:
    assert parsers.language_for("a.py") == "python"
    assert parsers.language_for("a.tsx") == "tsx"
    assert parsers.language_for("a.rs") == "rust"
    assert parsers.language_for("a.md") == "markdown"
    assert parsers.language_for("a.xyz") is None
