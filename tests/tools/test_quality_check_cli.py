from __future__ import annotations

"""CLI tests for the repository quality-check script."""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
QUALITY_CHECK = REPO_ROOT / "scripts" / "quality-check.py"


def _run_quality_check(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(QUALITY_CHECK), *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_quality_check_blocks_when_findings_exist(tmp_path: Path) -> None:
    sample = tmp_path / "sample.py"
    sample.write_text("def needs_work():\n    pass\n", encoding="utf-8")

    result = _run_quality_check(str(sample))

    assert result.returncode == 1
    assert "Quality check FAILED" in result.stderr
    assert "Empty pass statement" in result.stderr


def test_quality_check_report_only_exits_zero_with_findings(tmp_path: Path) -> None:
    sample = tmp_path / "sample.py"
    sample.write_text("def needs_work():\n    pass\n", encoding="utf-8")

    result = _run_quality_check("--report-only", str(sample))

    assert result.returncode == 0
    assert "Quality check FAILED" in result.stderr
    assert "Empty pass statement" in result.stderr
