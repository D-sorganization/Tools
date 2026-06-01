from pathlib import Path

from scripts.check_sidekick_coverage import (
    _is_sidekick_production_path,
    check_sidekick_coverage,
)


def _write_coverage(path: Path, classes: list[tuple[str, list[int]]]) -> None:
    def _line_xml(hits_by_line: list[int]) -> str:
        return "".join(
            f'<line number="{idx}" hits="{hits}" />'
            for idx, hits in enumerate(hits_by_line, 1)
        )

    class_xml = "\n".join(
        f"""
        <class filename="{filename}">
          <lines>
            {_line_xml(hits_by_line)}
          </lines>
        </class>
        """
        for filename, hits_by_line in classes
    )
    path.write_text(
        f"""<?xml version="1.0" ?>
<coverage>
  <sources><source>{path.parent}</source></sources>
  <packages><package><classes>{class_xml}</classes></package></packages>
</coverage>
""",
        encoding="utf-8",
    )


def test_sidekick_coverage_fails_when_zero_files_checked(tmp_path: Path) -> None:
    coverage_file = tmp_path / "coverage.xml"
    _write_coverage(
        coverage_file,
        [("src/shared/python/not_sidekick.py", [1, 1, 1])],
    )

    assert check_sidekick_coverage(coverage_file) == 1


def test_sidekick_coverage_fails_when_changed_file_missing_from_xml(
    tmp_path: Path,
) -> None:
    coverage_file = tmp_path / "coverage.xml"
    changed_file = tmp_path / "changed.txt"
    changed_file.write_text(
        "src/shared/python/sidekick/ui/tools_sidebar/runtime_tabs.py\n",
        encoding="utf-8",
    )
    _write_coverage(
        coverage_file,
        [("src/shared/python/sidekick/notes_store.py", [1, 1, 1])],
    )

    assert check_sidekick_coverage(coverage_file, changed_file) == 1


def test_sidekick_coverage_passes_when_enforced_files_have_coverage(
    tmp_path: Path,
) -> None:
    coverage_file = tmp_path / "coverage.xml"
    changed_file = tmp_path / "changed.txt"
    changed_file.write_text(
        "src/shared/python/sidekick/ui/tools_sidebar/runtime_tabs.py\n",
        encoding="utf-8",
    )
    _write_coverage(
        coverage_file,
        [
            ("src/shared/python/sidekick/notes_store.py", [1, 1, 1]),
            ("src/shared/python/sidekick/ui/tools_sidebar/runtime_tabs.py", [1, 0]),
        ],
    )

    assert check_sidekick_coverage(coverage_file, changed_file) == 0


def test_sidekick_production_path_contract_excludes_tests() -> None:
    assert _is_sidekick_production_path(
        "src/shared/python/sidekick/ui/tools_sidebar/runtime_tabs.py"
    )
    assert _is_sidekick_production_path(
        "/work/repo/src/shared/python/sidekick/ui/tools_sidebar/runtime_tabs.py"
    )
    assert not _is_sidekick_production_path(
        "tests/unit/sidekick/test_tab_collection_alias_contract.py"
    )
    assert not _is_sidekick_production_path(
        "src/shared/python/sidekick/tests/test_notes_store.py"
    )


def test_sidekick_coverage_ignores_changed_sidekick_tests(
    tmp_path: Path,
) -> None:
    coverage_file = tmp_path / "coverage.xml"
    changed_file = tmp_path / "changed.txt"
    changed_file.write_text(
        "\n".join(
            [
                "tests/unit/sidekick/test_tab_collection_alias_contract.py",
                "src/shared/python/sidekick/tests/test_notes_store.py",
            ]
        ),
        encoding="utf-8",
    )
    _write_coverage(
        coverage_file,
        [("src/shared/python/not_sidekick.py", [1, 1, 1])],
    )

    assert check_sidekick_coverage(coverage_file, changed_file) == 0
