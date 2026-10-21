"""Security floor contracts for the repository's PDF reader dependency."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from pypdf import PdfWriter

ROOT = Path(__file__).resolve().parents[2]
PDF_RENAMER_SOURCE = ROOT / "src" / "document_processing" / "pdf_renamer" / "src"
REQUIREMENT_FILES = (
    ROOT / "requirements.txt",
    ROOT / "src" / "document_processing" / "pdf_renamer" / "requirements.txt",
)


def _pypdf_requirement(path: Path) -> Requirement:
    for line in path.read_text(encoding="utf-8").splitlines():
        declaration = line.split("#", maxsplit=1)[0].strip()
        if declaration.lower().startswith("pypdf"):
            requirement = Requirement(declaration)
            if requirement.name.lower() == "pypdf":
                return requirement
    raise AssertionError(f"No pypdf requirement found in {path}")


@pytest.mark.parametrize("path", REQUIREMENT_FILES, ids=("root", "pdf-renamer"))
def test_requirement_files_reject_affected_releases_and_accept_patch(
    path: Path,
) -> None:
    requirement = _pypdf_requirement(path)

    assert not requirement.specifier.contains("6.20.1")
    assert not requirement.specifier.contains("6.21.0")
    assert requirement.specifier.contains("6.21.1")


def test_root_requirement_retains_existing_next_major_upper_bound() -> None:
    requirement = _pypdf_requirement(ROOT / "requirements.txt")

    assert ">=6.21.1" in str(requirement.specifier)
    assert "<7.0.0" in str(requirement.specifier)


def test_standalone_requirement_keeps_its_existing_unbounded_upper_range() -> None:
    requirement = _pypdf_requirement(REQUIREMENT_FILES[1])

    assert str(requirement.specifier) == ">=6.21.1"


def test_reproducible_lock_pins_the_security_patch_exactly() -> None:
    lock_path = ROOT / "requirements-lock.txt"
    locked = [
        line.split("#", maxsplit=1)[0].strip()
        for line in lock_path.read_text(encoding="utf-8").splitlines()
        if line.split("#", maxsplit=1)[0].strip().lower().startswith("pypdf")
    ]

    assert locked == ["pypdf==6.21.1"]


@pytest.mark.integration
@pytest.mark.parametrize("title", ["Patched PDF Metadata Contract", None])
def test_pdf_renamer_reads_real_pdf_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    title: str | None,
) -> None:
    monkeypatch.syspath_prepend(str(PDF_RENAMER_SOURCE))
    extractors = importlib.import_module("pdf_renamer.extractors")
    pdf_path = tmp_path / "metadata.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    if title is not None:
        writer.add_metadata({"/Title": title})
    with pdf_path.open("wb") as pdf_file:
        writer.write(pdf_file)

    result = extractors.title_from_metadata(pdf_path)

    assert result.title == title
    assert result.method == "metadata"
