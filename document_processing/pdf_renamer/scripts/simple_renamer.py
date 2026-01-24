"""
Simple PDF Renamer script using PyPDF2.
Acts as a secondary/fallback option to the main PDFRenamer tool.
"""

import logging

# Use shared logging utility
try:
    from utils.logging_utils import init_default_logging
except ImportError:
    # Fallback
    def init_default_logging():
        init_default_logging()
import os
import re
import sys

from PyPDF2 import PdfReader

# Configure logging
init_default_logging()s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def extract_title_author(pdf_path: str) -> tuple[str, str]:
    """Extract author (last name) and title from PDF metadata or text."""
    try:
        reader = PdfReader(pdf_path)
        # Try metadata first
        info = reader.metadata
        title = info.title if info and info.title else None
        author = info.author if info and info.author else None

        # Fallback: try first page text
        if not title or not author:
            try:
                if len(reader.pages) > 0:
                    first_page = reader.pages[0]
                    text = first_page.extract_text() if first_page else ""
                else:
                    text = ""
            except Exception:
                text = ""

            # Try to extract title (first non-empty line)
            if not title:
                lines = [line.strip() for line in text.split("\n") if line.strip()]
                title = lines[0] if lines else "Unknown Title"
            # Try to extract author (look for 'by <Author>' or similar)
            if not author:
                match = re.search(r"by ([A-Z][a-zA-Z\-]+)", text)
                author = match.group(1) if match else "Unknown"

        # Use only last name if possible
        if author and author != "Unknown":
            author_last = author.split()[-1]
        else:
            author_last = "Unknown"

        # Clean title for filename (remove forbidden and control characters)
        def clean(s: str) -> str:
            forbidden = r'[\\/:*?"<>|\r\n]'
            s = re.sub(forbidden, "", s)
            # Only allow ASCII letters, numbers, space, dash, underscore
            s = re.sub(r"[^A-Za-z0-9 _\-]", "", s)
            s = re.sub(r"\s+", " ", s).strip()  # Collapse spaces
            return s

        title_clean = clean(title if title else "Unknown Title")
        author_last_clean = clean(author_last)

        # Ensure non-empty values
        if not author_last_clean:
            author_last_clean = "Unknown"
        if not title_clean:
            title_clean = "Unknown Title"

        # Limit length to avoid Windows path issues
        author_last_clean = author_last_clean[:40]
        title_clean = title_clean[:80]

        return author_last_clean, title_clean

    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {e}")
        return "Unknown", "Unknown Title"


def rename_pdfs(root_folder: str) -> None:
    """Recursively rename PDFs in the given folder."""
    for dirpath, _, filenames in os.walk(root_folder):
        # Sort filenames alphabetically
        for filename in sorted(filenames, key=lambda x: x.lower()):
            if filename.lower().endswith(".pdf"):
                full_path = Path(dirpath) / filename
                author, title = extract_title_author(full_path)
                new_name = f"{author} - {title}.pdf"
                new_path = Path(dirpath) / new_name

                # Avoid overwriting by appending a number if needed
                base_new_name = new_name
                count = 1
                while Path(new_path).exists() and new_path != full_path:
                    name_wo_ext, ext = os.path.splitext(base_new_name)
                    new_name = f"{name_wo_ext} ({count}){ext}"
                    new_path = Path(dirpath) / new_name
                    count += 1

                if new_path != full_path:
                    logger.info(f"Renaming: {filename} -> {new_name}")
                    try:
                        os.rename(full_path, new_path)
                    except OSError as e:
                        logger.error(f"Failed to rename {filename}: {e}")


def main() -> None:
    """Main entry point."""
    if len(sys.argv) > 1:
        root = sys.argv[1]
    else:
        root = os.getcwd()

    logger.info(f"Scanning directory: {root}")
    rename_pdfs(root)


if __name__ == "__main__":
    main()
