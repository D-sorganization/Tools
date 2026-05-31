"""Regression tests for the Sidekick per-file coverage gate (issue #3139).

The gate must not pass vacuously when zero Sidekick files are checked, and a
changed Sidekick production file absent from the coverage XML must fail loudly.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "check_sidekick_coverage.py"
)


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "check_sidekick_coverage", SCRIPT_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_EMPTY_COVERAGE_XML = """<?xml version="1.0" ?>
<coverage>
  <sources><source>{root}</source></sources>
  <packages>
    <package name="other">
      <classes>
        <class filename="src/other/module.py" name="module.py">
          <lines><line number="1" hits="1"/></lines>
        </class>
      </classes>
    </package>
  </packages>
</coverage>
"""


def _covered_sidekick_xml(root: str, rate_hits: int, rate_total: int) -> str:
    lines = "".join(
        f'<line number="{i + 1}" hits="{1 if i < rate_hits else 0}"/>'
        for i in range(rate_total)
    )
    return f"""<?xml version="1.0" ?>
<coverage>
  <sources><source>{root}</source></sources>
  <packages>
    <package name="sidekick">
      <classes>
        <class filename="src/shared/python/sidekick/latex_renderer.py"
               name="latex_renderer.py">
          <lines>{lines}</lines>
        </class>
      </classes>
    </package>
  </packages>
</coverage>
"""


def test_zero_files_checked_fails(tmp_path: Path) -> None:
    """An enforced run that checks zero Sidekick files must exit non-zero."""
    module = _load_module()
    coverage = tmp_path / "coverage.xml"
    coverage.write_text(_EMPTY_COVERAGE_XML.format(root=tmp_path), encoding="utf-8")

    rc = module.check_sidekick_coverage(coverage, None)

    assert rc != 0, "Gate must fail when zero Sidekick files are checked"


def test_changed_sidekick_file_missing_from_coverage_fails(
    tmp_path: Path, capsys: Any
) -> None:
    """A changed Sidekick file absent from coverage XML must fail and name it."""
    module = _load_module()
    coverage = tmp_path / "coverage.xml"
    coverage.write_text(_EMPTY_COVERAGE_XML.format(root=tmp_path), encoding="utf-8")
    changed = tmp_path / "changed.txt"
    missing = "src/shared/python/sidekick/ui/tools_sidebar/runtime_tabs.py"
    changed.write_text(missing + "\n", encoding="utf-8")

    rc = module.check_sidekick_coverage(coverage, changed)
    out = capsys.readouterr()

    assert rc != 0, "Gate must fail for a changed Sidekick file missing coverage"
    assert "runtime_tabs.py" in (out.out + out.err), "Gate must name the missing file"


def test_zero_files_ok_when_no_sidekick_changed(tmp_path: Path) -> None:
    """A non-Sidekick PR (changed-files given, none Sidekick) may check zero."""
    module = _load_module()
    coverage = tmp_path / "coverage.xml"
    coverage.write_text(_EMPTY_COVERAGE_XML.format(root=tmp_path), encoding="utf-8")
    changed = tmp_path / "changed.txt"
    changed.write_text("src/signal_toolkit/filters.py\n", encoding="utf-8")

    rc = module.check_sidekick_coverage(coverage, changed)

    assert rc == 0, "Non-Sidekick PR with no Sidekick changes must not fail the gate"


def test_changed_sidekick_file_above_threshold_passes(tmp_path: Path) -> None:
    """A changed Sidekick file present and above 50% must pass."""
    module = _load_module()
    sidekick_dir = tmp_path / "src" / "shared" / "python" / "sidekick"
    sidekick_dir.mkdir(parents=True)
    (sidekick_dir / "latex_renderer.py").write_text("x = 1\n", encoding="utf-8")
    coverage = tmp_path / "coverage.xml"
    coverage.write_text(
        _covered_sidekick_xml(str(tmp_path), rate_hits=8, rate_total=10),
        encoding="utf-8",
    )
    changed = tmp_path / "changed.txt"
    changed.write_text(
        "src/shared/python/sidekick/latex_renderer.py\n", encoding="utf-8"
    )

    rc = module.check_sidekick_coverage(coverage, changed)

    assert rc == 0, "Gate must pass for a changed Sidekick file above 50%"
