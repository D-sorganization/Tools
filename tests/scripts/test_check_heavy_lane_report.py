"""Tests for scripts/check_heavy_lane_report.py (issue #5921)."""

from __future__ import annotations

from pathlib import Path

from scripts.check_heavy_lane_report import DEFAULT_MANIFEST, check, load_manifest, main

REPO_ROOT = Path(__file__).resolve().parents[2]
A = '<testcase classname="t.m.C" name="test_a"/>'
B = '<testcase classname="t.m.C" name="test_b"/>'
SKIP_A = '<testcase classname="t.m.C" name="test_a"><skipped message="x"/></testcase>'
FAIL_A = (
    '<testcase classname="t.m.C" name="test_a"><failure message="boom"/></testcase>'
)
ERR_A = '<testcase classname="t.m.C" name="test_a"><error message="boom"/></testcase>'
OPT_SKIP = (
    '<testcase classname="t.m.C" name="test_opt"><skipped message="x"/></testcase>'
)


def _files(
    tmp_path: Path, cases: str, required: str = "t.m.C::test_a\n"
) -> tuple[Path, Path]:
    report = tmp_path / "r.xml"
    report.write_text(f"<testsuites><testsuite>{cases}</testsuite></testsuites>")
    manifest = tmp_path / "m.txt"
    manifest.write_text("# comment\n" + required)
    return report, manifest


def test_required_passes_with_optional_skip(tmp_path: Path) -> None:
    assert check(*_files(tmp_path, A + OPT_SKIP)) == []


def test_missing_report(tmp_path: Path) -> None:
    _, manifest = _files(tmp_path, A)
    assert check(tmp_path / "absent.xml", manifest) == [
        f"report not found: {tmp_path / 'absent.xml'}"
    ]


def test_unparseable_report(tmp_path: Path) -> None:
    report, manifest = _files(tmp_path, A)
    report.write_text("<testsuites")
    assert check(report, manifest)[0].startswith("report unparseable")


def test_zero_cases(tmp_path: Path) -> None:
    assert check(*_files(tmp_path, "")) == ["report contains zero test cases"]


def test_all_skipped(tmp_path: Path) -> None:
    problems = check(*_files(tmp_path, SKIP_A + OPT_SKIP))
    assert "all 2 selected cases skipped" in problems
    assert "required case skipped: t.m.C::test_a" in problems


def test_required_case_renamed_or_deleted(tmp_path: Path) -> None:
    assert check(*_files(tmp_path, B)) == ["required case missing: t.m.C::test_a"]


def test_required_case_failed_or_errored(tmp_path: Path) -> None:
    assert check(*_files(tmp_path, FAIL_A + B)) == [
        "required case failed: t.m.C::test_a"
    ]
    assert check(*_files(tmp_path, ERR_A + B)) == [
        "required case failed: t.m.C::test_a"
    ]


def test_empty_manifest_fails(tmp_path: Path) -> None:
    problems = check(*_files(tmp_path, A, required=""))
    assert problems and problems[0].startswith("required-case manifest is empty")


def test_main_exit_codes(tmp_path: Path) -> None:
    report, manifest = _files(tmp_path, A)
    assert main([str(report), "--manifest", str(manifest)]) == 0
    report.write_text("<testsuites><testsuite></testsuite></testsuites>")
    assert main([str(report), "--manifest", str(manifest)]) == 1


def test_repo_manifest_is_nonempty_and_well_formed() -> None:
    required = load_manifest(REPO_ROOT / DEFAULT_MANIFEST)
    assert len(required) >= 10
    assert all("::" in case_id for case_id in required)
    assert len(required) == len(set(required))
