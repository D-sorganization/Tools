"""Tests for the size debt register generator (issue #3261)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "generate_size_debt_register.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "generate_size_debt_register", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_collect_only_returns_large_source_files(tmp_path: Path) -> None:
    module = _load_module()
    src = tmp_path / "src"
    (src / "pkg").mkdir(parents=True)
    (src / "pkg" / "big.py").write_text("x = 1\n" * 900, encoding="utf-8")
    (src / "pkg" / "small.py").write_text("x = 1\n" * 10, encoding="utf-8")
    (src / "pkg" / "notes.txt").write_text("y\n" * 900, encoding="utf-8")
    # excluded directory must be ignored even when large
    vendored = src / "node_modules"
    vendored.mkdir()
    (vendored / "huge.js").write_text("a\n" * 5000, encoding="utf-8")

    rows = module.collect(src)
    files = [rel for _, rel in rows]
    assert any(f.endswith("big.py") for f in files)
    assert not any("small.py" in f for f in files)
    assert not any("notes.txt" in f for f in files)
    assert not any("node_modules" in f for f in files)


def test_render_classes_and_counts() -> None:
    module = _load_module()
    rows = [(1500, "src/a.py"), (900, "src/b.py")]
    out = module.render(rows)
    assert "Files at/above 800 LOC: **2**" in out
    assert "CRITICAL (at/above 1000 LOC): **1**" in out
    assert "| 1 | 1500 | CRITICAL | `src/a.py` |" in out
    assert "| 2 | 900 | HIGH | `src/b.py` |" in out


def test_rows_sorted_descending_by_loc() -> None:
    module = _load_module()
    rows = [(810, "src/c.py"), (1500, "src/a.py"), (900, "src/b.py")]
    out = module.render(sorted(rows, key=lambda r: (-r[0], r[1])))
    a = out.index("src/a.py")
    b = out.index("src/b.py")
    c = out.index("src/c.py")
    assert a < b < c


def test_committed_register_is_in_sync() -> None:
    """The committed register file must match the current source tree."""
    module = _load_module()
    assert module.main(["--check"]) == 0


def test_register_is_ascii_safe() -> None:
    """The generated doc must be UTF-8/ASCII-safe (regression for em-dash bug)."""
    module = _load_module()
    content = module.render(module.collect())
    content.encode("utf-8")  # must not raise
    assert all(ord(ch) < 128 for ch in content)
