"""Security tests for the zip-extraction defenses in ``project_import``.

Project import ingests an operator-supplied ZIP and wipes/rebuilds the plant
database, so a malicious archive is a direct attack surface. ``project_import``
guards extraction with four budgets and a path check:

- ``_validate_member_path`` blocks path traversal (``../``) and absolute
  member paths (Zip-Slip),
- ``_validate_member_budget`` blocks an oversized single member and a
  suspicious compression ratio (zip bomb),
- ``_safe_extract_zip`` blocks too-many-members and total-uncompressed-budget
  overflow,
- and ``_extract_project_zip`` maps a corrupt/non-zip payload to HTTP 400.

These tests build tiny in-memory malicious archives with the stdlib
:mod:`zipfile` module (or hand-crafted :class:`zipfile.ZipInfo` records for the
size/ratio checks, so no gigabyte-scale test data is needed) and assert the
exact fail-closed status code for each defense. The module-level budget
constants are monkeypatched down where a real archive is required, keeping the
suite fast while exercising the production code path.
"""

from __future__ import annotations

import io
import sys
import tempfile
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import project_import  # noqa: E402
from fastapi import HTTPException  # noqa: E402


def _zip_from(members: list[tuple[str, bytes]]) -> zipfile.ZipFile:
    """Return an open, in-memory zip containing ``members`` (deflated)."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in members:
            archive.writestr(name, data)
    buffer.seek(0)
    return zipfile.ZipFile(buffer, "r")


@pytest.fixture
def dest_root() -> Iterator[Path]:
    """A real, resolved destination directory for extraction/path checks."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir).resolve()


# --------------------------------------------------------------------------- #
# _validate_member_path — Zip-Slip / path traversal                           #
# --------------------------------------------------------------------------- #


def test_validate_member_path_accepts_safe_relative(dest_root: Path) -> None:
    # A nested, non-escaping path is allowed (returns None, no raise).
    assert project_import._validate_member_path("sub/dir/ok.txt", dest_root) is None


def test_validate_member_path_rejects_parent_escape(dest_root: Path) -> None:
    with pytest.raises(HTTPException) as excinfo:
        project_import._validate_member_path("../escape.txt", dest_root)
    assert excinfo.value.status_code == 400


def test_validate_member_path_rejects_nested_parent_escape(
    dest_root: Path,
) -> None:
    with pytest.raises(HTTPException) as excinfo:
        project_import._validate_member_path("a/../../escape.txt", dest_root)
    assert excinfo.value.status_code == 400


def test_validate_member_path_rejects_absolute(dest_root: Path) -> None:
    with pytest.raises(HTTPException) as excinfo:
        project_import._validate_member_path("/etc/passwd", dest_root)
    assert excinfo.value.status_code == 400


# --------------------------------------------------------------------------- #
# _validate_member_budget — oversized member and zip-bomb ratio               #
# --------------------------------------------------------------------------- #


def test_validate_member_budget_rejects_oversized_member() -> None:
    info = zipfile.ZipInfo("big.bin")
    info.file_size = project_import.MAX_IMPORT_MEMBER_BYTES + 1
    info.compress_size = info.file_size  # ratio 1.0, so only size trips
    with pytest.raises(HTTPException) as excinfo:
        project_import._validate_member_budget(info)
    assert excinfo.value.status_code == 413


def test_validate_member_budget_rejects_zip_bomb_ratio() -> None:
    info = zipfile.ZipInfo("bomb.bin")
    # Well under the per-member size cap, but an absurd compression ratio.
    info.file_size = 10_000
    info.compress_size = 1
    assert (
        info.file_size / info.compress_size
        > project_import.MAX_IMPORT_COMPRESSION_RATIO
    )
    with pytest.raises(HTTPException) as excinfo:
        project_import._validate_member_budget(info)
    assert excinfo.value.status_code == 413


def test_validate_member_budget_accepts_normal_member() -> None:
    info = zipfile.ZipInfo("ok.txt")
    info.file_size = 100
    info.compress_size = 50  # ratio 2.0, well under the cap
    assert project_import._validate_member_budget(info) == 100


# --------------------------------------------------------------------------- #
# _safe_extract_zip — member count and total-budget overflow                  #
# --------------------------------------------------------------------------- #


def test_safe_extract_zip_rejects_too_many_members(
    monkeypatch: pytest.MonkeyPatch, dest_root: Path
) -> None:
    monkeypatch.setattr(project_import, "MAX_IMPORT_MEMBERS", 2)
    archive = _zip_from([("a.txt", b"x"), ("b.txt", b"y"), ("c.txt", b"z")])
    with pytest.raises(HTTPException) as excinfo:
        project_import._safe_extract_zip(archive, dest_root)
    assert excinfo.value.status_code == 413


def test_safe_extract_zip_rejects_total_budget_overflow(
    monkeypatch: pytest.MonkeyPatch, dest_root: Path
) -> None:
    monkeypatch.setattr(project_import, "MAX_IMPORT_TOTAL_BYTES", 10)
    archive = _zip_from([("a.txt", b"x" * 20), ("b.txt", b"y" * 20)])
    with pytest.raises(HTTPException) as excinfo:
        project_import._safe_extract_zip(archive, dest_root)
    assert excinfo.value.status_code == 413


def test_safe_extract_zip_rejects_traversal_member(dest_root: Path) -> None:
    archive = _zip_from([("../escape.txt", b"nope")])
    with pytest.raises(HTTPException) as excinfo:
        project_import._safe_extract_zip(archive, dest_root)
    assert excinfo.value.status_code == 400


def test_safe_extract_zip_extracts_clean_archive(dest_root: Path) -> None:
    archive = _zip_from([("sub/ok.txt", b"hello")])
    project_import._safe_extract_zip(archive, dest_root)
    assert (dest_root / "sub" / "ok.txt").read_bytes() == b"hello"


# --------------------------------------------------------------------------- #
# _extract_project_zip — corrupt / non-zip payload is mapped to HTTP 400      #
# --------------------------------------------------------------------------- #


def test_extract_project_zip_maps_bad_zip_to_400(dest_root: Path) -> None:
    bad_zip = dest_root / "corrupt.zip"
    bad_zip.write_bytes(b"this is definitely not a zip archive")
    with pytest.raises(HTTPException) as excinfo:
        project_import._extract_project_zip(bad_zip, dest_root)
    assert excinfo.value.status_code == 400


def test_extract_project_zip_preserves_httpexception(dest_root: Path) -> None:
    # A traversal member surfaced during extraction must keep its 400 status,
    # not be re-wrapped by the BadZipFile handler.
    zip_path = dest_root / "evil.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("../escape.txt", b"nope")
    with pytest.raises(HTTPException) as excinfo:
        project_import._extract_project_zip(zip_path, dest_root)
    assert excinfo.value.status_code == 400
