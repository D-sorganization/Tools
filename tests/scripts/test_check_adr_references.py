"""Contracts for scripts/check_adr_references.py (ADR-0049 fleet ADR home)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import check_adr_references as mod

ROOT = Path(__file__).resolve().parents[2]
OLD_STATUS = "Accepted for contract implementation"


def _seed(tmp_path: Path, *, adrs: dict[str, str], citations: str) -> Path:
    adr_dir = tmp_path / "docs" / "adr"
    adr_dir.mkdir(parents=True)
    for name, body in adrs.items():
        (adr_dir / name).write_text(body, encoding="utf-8")
    (adr_dir / "README.md").write_text(
        f"# ADRs\n\n## Records\n\n{mod.START_MARKER}\n{mod.END_MARKER}\n",
        encoding="utf-8",
    )
    src = tmp_path / "src" / "pkg"
    src.mkdir(parents=True)
    (src / "module.py").write_text(citations, encoding="utf-8")
    return tmp_path


def test_repository_citations_all_resolve_and_index_is_fresh() -> None:
    assert mod.main(["--root", str(ROOT)]) == 0


def test_repository_has_no_duplicate_adr_numbers() -> None:
    records = mod.adr_files(ROOT / "docs" / "adr")
    assert "007" in records
    assert "008" in records
    for number in ("0045", "0046", "0047", "0048", "0049"):
        assert records[number].is_file()


def test_unresolved_citation_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _seed(
        tmp_path,
        adrs={"ADR-0001-a.md": "# ADR-0001: A\n\n- Status: Accepted\n"},
        citations='"""See ADR-0001 and ADR-0002."""\n',
    )
    assert mod.main(["--root", str(root)]) == 1
    assert "ADR-0002" in capsys.readouterr().err


def test_write_then_check_round_trips_and_marks_mirrors(tmp_path: Path) -> None:
    mirror = (
        "# ADR-0002: Mirror\n\n"
        "> **Mirrored ADR (fleet ADR home: ADR-0049).**\n"
        "> Source: UpstreamDrift docs/adr/0002-mirror.md\n\n"
        "- Status: Proposed | ratified later\n"
    )
    root = _seed(
        tmp_path,
        adrs={
            "ADR-001-old.md": f"# ADR-001: Old\n\nStatus: {OLD_STATUS}\n",
            "ADR-0002-mirror.md": mirror,
        },
        citations="# ADR-0002 applies\n",
    )
    assert mod.main(["--root", str(root)]) == 1  # index stale
    assert mod.main(["--root", str(root), "--write"]) == 0
    assert mod.main(["--root", str(root)]) == 0
    index = (root / "docs" / "adr" / "README.md").read_text(encoding="utf-8")
    assert f"| [ADR-001](ADR-001-old.md) | {OLD_STATUS} | Tools | Old |" in index
    assert (
        "| [ADR-0002](ADR-0002-mirror.md) | Proposed | mirror (UpstreamDrift) "
        "| Mirror |"
    ) in index


def test_duplicate_number_is_rejected(tmp_path: Path) -> None:
    root = _seed(
        tmp_path,
        adrs={"ADR-007-a.md": "# ADR-007: A\n", "ADR-007-b.md": "# ADR-007: B\n"},
        citations="",
    )
    with pytest.raises(SystemExit, match="duplicate ADR number 007"):
        mod.adr_files(root / "docs" / "adr")


def test_citation_pattern_ignores_three_digit_and_longer_numbers() -> None:
    found = mod.CITATION_PATTERN.findall("ADR-003 ADR-0046 ADR-00461 XADR-0047")
    assert found == ["0046"]
