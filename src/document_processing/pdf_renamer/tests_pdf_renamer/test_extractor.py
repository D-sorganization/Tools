"""Tests for PDF metadata and content extractors."""

from unittest.mock import MagicMock, patch

from pdf_renamer.extractor import extract_metadata


def test_extract_metadata(tmp_path: object) -> None:
    # Create a dummy PDF file (empty)
    # mypy handling for tmp_path fixture
    from pathlib import Path

    assert isinstance(tmp_path, Path)
    pdf_path = tmp_path / "dummy.pdf"
    pdf_path.touch()

    # Mock pdfplumber.open
    with patch("pdfplumber.open") as mock_open:
        mock_pdf = MagicMock()
        mock_pdf.metadata = {"Author": "Alice", "Title": "Wonderland"}
        mock_open.return_value.__enter__.return_value = mock_pdf

        author, title = extract_metadata(pdf_path)

        assert author == "Alice"
        assert title == "Wonderland"


def test_extract_metadata_missing_keys(tmp_path: object) -> None:
    from pathlib import Path

    assert isinstance(tmp_path, Path)
    pdf_path = tmp_path / "dummy.pdf"
    pdf_path.touch()

    with patch("pdfplumber.open") as mock_open:
        mock_pdf = MagicMock()
        mock_pdf.metadata = {}  # Empty metadata
        mock_open.return_value.__enter__.return_value = mock_pdf

        author, title = extract_metadata(pdf_path)

        assert author is None
        assert title is None
