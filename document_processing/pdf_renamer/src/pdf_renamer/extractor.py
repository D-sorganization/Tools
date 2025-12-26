import logging
from pathlib import Path

import pdfplumber

logger = logging.getLogger(__name__)


def extract_metadata(file_path: Path) -> tuple[str | None, str | None]:
    """
    Extracts Author and Title from PDF metadata.
    Returns (Author, Title) or (None, None) if extraction fails.
    """
    try:
        with pdfplumber.open(file_path) as pdf:
            metadata = pdf.metadata or {}

            # Metadata keys can be Title/Author or title/author
            # We try standard keys
            title = metadata.get("Title") or metadata.get("title")
            author = metadata.get("Author") or metadata.get("author")

            # Clean up empty strings
            if title and not title.strip():
                title = None
            if author and not author.strip():
                author = None

            # Fallback to text extraction if metadata is missing or sparse
            if not title or not author:
                try:
                    import re

                    if len(pdf.pages) > 0:
                        first_page = pdf.pages[0]
                        text = first_page.extract_text() if first_page else ""

                        # Simple heuristic for Title: First non-empty line
                        if not title:
                            lines = [
                                line.strip()
                                for line in text.split("\n")
                                if line.strip()
                            ]
                            if lines:
                                title = lines[0]

                        # Simple heuristic for Author: Look for "by [Author]"
                        if not author:
                            match = re.search(r"by ([A-Z][a-zA-Z\s]+)", text)
                            if match:
                                possible_author = match.group(1).strip()
                                # Ensure extracted author name isn't unreasonably long
                                if len(possible_author) < 50:
                                    author = possible_author
                except Exception as e:
                    logger.debug(f"Text fallback failed for {file_path}: {e}")

            return author, title

    except Exception as e:
        logger.warning(f"Failed to extract metadata from {file_path}: {e}")
        return None, None
