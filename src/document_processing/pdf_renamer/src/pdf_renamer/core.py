"""Core title extraction logic for PDF documents."""

from __future__ import annotations

import logging
from pathlib import Path

from .extractors import TitleLLM, title_from_first_page, title_from_metadata
from .types import TitleResult

logger = logging.getLogger(__name__)


def extract_title(pdf_path: Path, llm: TitleLLM | None = None) -> TitleResult:
    """
    Extracts title using a layered approach:
    0. Metadata (fastest, high precision)
    1. Heuristic (layout-aware, fast)
    2. LLM (slowest, highest recall, optional)
    """

    # 0) Metadata
    r0 = title_from_metadata(pdf_path)
    # title_from_metadata returns 0.95 conf if it passes looks_like_title
    if r0.title:
        logger.debug(f"Found title via metadata: {r0.title}")
        return r0

    # 1) Heuristic
    r1 = title_from_first_page(pdf_path)
    if r1.title and r1.confidence >= 0.7:
        logger.debug(f"Found title via heuristic: {r1.title} ({r1.confidence:.2f})")
        return r1

    # 2) LLM Fallback
    if llm is None:
        logger.debug("No LLM provided, returning heuristic result.")
        # Return whatever we got from r1, even if low confidence, or empty
        return r1

    logger.info(f"Escalating {pdf_path.name} to LLM...")
    r2 = llm.extract_title(pdf_path)
    return r2
