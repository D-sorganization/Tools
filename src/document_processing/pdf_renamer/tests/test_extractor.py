from pathlib import Path
from unittest.mock import MagicMock, patch

from pdf_renamer.extractors import title_from_first_page, title_from_metadata


def test_title_from_metadata(tmp_path: Path) -> None:
    pdf_path = tmp_path / "test.pdf"
    pdf_path.touch()

    with patch("pypdf.PdfReader") as mock_reader_cls:
        mock_reader = MagicMock()
        mock_reader.metadata = MagicMock()
        mock_reader.metadata.title = "Metadata Title"
        mock_reader_cls.return_value = mock_reader


        result = title_from_metadata(pdf_path)
        assert result.title == "Metadata Title"
        assert result.method == "metadata"


def test_title_from_metadata_missing(tmp_path: Path) -> None:
    pdf_path = tmp_path / "test.pdf"
    pdf_path.touch()

    with patch("pypdf.PdfReader") as mock_reader_cls:
        mock_reader = MagicMock()
        mock_reader.metadata = {}  # Empty metadata
        mock_reader_cls.return_value = mock_reader


        result = title_from_metadata(pdf_path)
        assert result.title is None
        assert result.method == "metadata"


def test_title_from_first_page(tmp_path: Path) -> None:
    pdf_path = tmp_path / "test.pdf"
    pdf_path.touch()

    with patch("fitz.open") as mock_open:
        mock_doc = MagicMock()
        mock_doc.page_count = 1
        mock_page = MagicMock()
        mock_doc.load_page.return_value = mock_page

        # Mock struct: blocks -> lines -> spans
        # This simulates layout extraction
        mock_page.get_text.return_value = {
            "blocks": [
                {
                    "lines": [
                        {
                            "spans": [
                                {
                                    "text": "Big Title",
                                    "size": 24.0,
                                    "bbox": [0, 0, 100, 20],
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        mock_page.rect.height = 1000
        mock_open.return_value = mock_doc

        result = title_from_first_page(pdf_path)
        assert result.title == "Big Title"
        assert result.method == "heuristic"


def test_title_from_first_page_empty(tmp_path: Path) -> None:
    pdf_path = tmp_path / "test.pdf"
    pdf_path.touch()

    with patch("fitz.open") as mock_fitz:
        mock_doc = MagicMock()
        mock_doc.page_count = 0
        mock_fitz.return_value = mock_doc

        result = title_from_first_page(pdf_path)
        assert result.title is None
        assert result.method == "heuristic"
