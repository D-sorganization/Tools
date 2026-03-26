import logging
import sqlite3
from datetime import datetime
from pathlib import Path

from .types import TitleResult

logger = logging.getLogger(__name__)


class ResultCache:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._initialize_database()

    def _initialize_database(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS results (
                    sha256 TEXT PRIMARY KEY,
                    file_path TEXT,
                    title TEXT,
                    confidence REAL,
                    method TEXT,
                    provider TEXT,
                    model TEXT,
                    timestamp DATETIME,
                    error TEXT
                )
                """)

    def get(self, sha256: str) -> TitleResult | None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute(
                    "SELECT title, confidence, method, error FROM results " "WHERE sha256 = ?",
                    (sha256,),
                )
                row = cur.fetchone()
                if row:
                    title, conf, method, error = row
                    return TitleResult(title, conf, method, error or "")
        except sqlite3.Error as e:
            logger.debug(f"Cache get error: {e}")
        return None

    def save(
        self,
        sha256: str,
        path: Path,
        result: TitleResult,
        provider: str = "",
        model: str = "",
    ) -> None:
        try:
            error_msg = result.details if result.confidence == 0.0 else ""
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO results (
                        sha256, file_path, title, confidence, method,
                        provider, model, timestamp, error
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        sha256,
                        str(path),
                        result.title,
                        result.confidence,
                        result.method,
                        provider,
                        model,
                        datetime.now(),
                        error_msg,
                    ),
                )
        except sqlite3.Error as e:
            logger.debug(f"Cache save error: {e}")
