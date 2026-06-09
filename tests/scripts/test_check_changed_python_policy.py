from __future__ import annotations

from pathlib import Path

import pytest

from scripts.check_changed_python_policy import (
    find_policy_violations,
    load_allowlist,
)


def test_flags_print_and_sys_path_in_changed_production_files(tmp_path: Path) -> None:
    source = tmp_path / "src" / "tool" / "runner.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "\n".join(
            [
                "import sys",
                "",
                "def main() -> None:",
                '    print("debug")',
                '    sys.path.insert(0, "vendor")',
            ]
        ),
        encoding="utf-8",
    )

    violations = find_policy_violations(
        tmp_path,
        ["src/tool/runner.py", "README.md"],
        allowlist={},
    )

    assert [(item.policy, item.line_number) for item in violations] == [
        ("print", 4),
        ("sys_path", 5),
    ]


def test_allows_documented_stdout_contract(tmp_path: Path) -> None:
    script = tmp_path / "scripts" / "status.py"
    script.parent.mkdir(parents=True)
    script.write_text('print("machine-readable status")\n', encoding="utf-8")
    allowlist_path = tmp_path / "allowlist.txt"
    allowlist_path.write_text(
        "scripts/status.py | CLI stdout is parsed by release automation.\n",
        encoding="utf-8",
    )

    assert (
        find_policy_violations(
            tmp_path,
            ["scripts/status.py"],
            allowlist=load_allowlist(allowlist_path),
        )
        == []
    )


def test_skips_tests_even_when_changed(tmp_path: Path) -> None:
    test_file = tmp_path / "tests" / "test_runner.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text('print("captured output")\n', encoding="utf-8")

    assert (
        find_policy_violations(
            tmp_path,
            ["tests/test_runner.py"],
            allowlist={},
        )
        == []
    )


def test_allowlist_entries_must_document_reason(tmp_path: Path) -> None:
    allowlist_path = tmp_path / "allowlist.txt"
    allowlist_path.write_text("scripts/status.py\n", encoding="utf-8")

    with pytest.raises(ValueError, match="documented reason"):
        load_allowlist(allowlist_path)
