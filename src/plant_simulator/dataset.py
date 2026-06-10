"""SCADA dataset loader for the (experimental) neural plant simulator.

.. warning::

   This package is **experimental**. It previously fabricated random training
   data silently (issue #3295), which made ``train.py`` "train" on noise. That
   path has been removed: :class:`SCADADataset` now loads real ``TagLog`` rows
   from the SCADA SQLite database and raises ``ValueError`` when the database
   does not contain enough data. Synthetic data is only produced when the caller
   *explicitly* opts in via ``allow_synthetic=True`` (for tests/groundwork).
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class SCADADataset(Dataset):
    """PyTorch Dataset of sliding windows of SCADA Tag values.

    Loads logged tag values from the SCADA SQLite database (the ``TagLog`` table
    written by ``p1am_control_system.backend``) and pivots them into an array of
    shape ``(num_timesteps, num_tags)`` for sequence modelling.

    Args:
        db_path: Path to the SCADA SQLite database file.
        sequence_length: Window length for the input sequence.
        num_tags: Number of tag columns expected in the pivoted array.
        allow_synthetic: If True, and the database is missing/empty, generate
            deterministic synthetic data instead of raising. **Off by default** —
            real training must never silently run on fabricated data.

    Raises:
        ValueError: If the database has insufficient data and ``allow_synthetic``
            is False, or if ``sequence_length``/``num_tags`` are non-positive.
    """

    def __init__(
        self,
        db_path: str,
        sequence_length: int = 10,
        num_tags: int = 32,
        *,
        allow_synthetic: bool = False,
    ) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be a positive integer")
        if num_tags <= 0:
            raise ValueError("num_tags must be a positive integer")
        self.sequence_length = sequence_length
        self.num_tags = num_tags
        self.data = self._load_data(db_path, allow_synthetic=allow_synthetic)

    def _load_data(self, db_path: str, *, allow_synthetic: bool) -> NDArray[np.float32]:
        """Load and pivot ``TagLog`` rows into ``(num_timesteps, num_tags)``.

        Reads the ``taglog`` table directly via sqlite3 (no ORM dependency),
        groups rows by timestamp, and fills a dense matrix indexed by tag.
        Raises ``ValueError`` when there are not enough rows to form a single
        training window, unless ``allow_synthetic`` is explicitly set.
        """
        path = Path(db_path)
        min_rows = self.sequence_length + 1

        if not path.exists():
            if allow_synthetic:
                logger.warning(
                    "SCADA DB %s not found; generating synthetic data "
                    "(allow_synthetic=True).",
                    db_path,
                )
                return self._synthetic(min_rows)
            raise ValueError(
                f"SCADA database not found at {db_path!r}. Provide a populated "
                "TagLog database, or pass allow_synthetic=True for groundwork."
            )

        matrix = self._read_taglog(path)
        if matrix is None or matrix.shape[0] < min_rows:
            have = 0 if matrix is None else matrix.shape[0]
            if allow_synthetic:
                logger.warning(
                    "SCADA DB %s has insufficient rows (%d < %d); generating "
                    "synthetic data (allow_synthetic=True).",
                    db_path,
                    have,
                    min_rows,
                )
                return self._synthetic(min_rows)
            raise ValueError(
                f"SCADA database {db_path!r} has insufficient data: need at "
                f"least {min_rows} timesteps, found {have}."
            )
        return matrix

    def _read_taglog(self, path: Path) -> NDArray[np.float32] | None:
        """Read and pivot the ``taglog`` table, or return None if unavailable."""
        try:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        except sqlite3.Error as exc:  # pragma: no cover - environment dependent
            logger.error("Could not open SCADA DB %s: %s", path, exc)
            return None
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND lower(name)='taglog'"
            )
            if cur.fetchone() is None:
                logger.error("SCADA DB %s has no 'taglog' table.", path)
                return None
            cur.execute(
                "SELECT timestamp, tag_id, value FROM taglog ORDER BY timestamp ASC"
            )
            rows = cur.fetchall()
        except sqlite3.Error as exc:
            logger.error("Failed to query taglog in %s: %s", path, exc)
            return None
        finally:
            conn.close()

        if not rows:
            return None

        # Group by timestamp preserving order of first appearance.
        timestamps: list[object] = []
        ts_index: dict[object, int] = {}
        for ts, _tag_id, _value in rows:
            if ts not in ts_index:
                ts_index[ts] = len(timestamps)
                timestamps.append(ts)

        matrix: NDArray[np.float32] = np.zeros(
            (len(timestamps), self.num_tags), dtype=np.float32
        )
        for ts, tag_id, value in rows:
            col = self._tag_column(tag_id)
            if col is None or not (0 <= col < self.num_tags):
                continue
            try:
                matrix[ts_index[ts], col] = float(value)
            except (TypeError, ValueError):
                continue
        return matrix

    @staticmethod
    def _tag_column(tag_id: object) -> int | None:
        """Map a tag identifier (int, ``"5"``, or ``"TAG_5"``) to a column index."""
        if isinstance(tag_id, bool):
            return None
        if isinstance(tag_id, int):
            return tag_id
        if isinstance(tag_id, str):
            s = tag_id.strip()
            if s.upper().startswith("TAG_"):
                s = s[4:]
            try:
                return int(s)
            except ValueError:
                return None
        return None

    def _synthetic(self, min_rows: int) -> NDArray[np.float32]:
        """Deterministic synthetic data (opt-in only) for tests/groundwork."""
        rng = np.random.default_rng(seed=0)
        rows = max(1000, min_rows)
        return cast(
            NDArray[np.float32],
            rng.random((rows, self.num_tags)).astype(np.float32),
        )

    def __len__(self) -> int:
        return max(0, len(self.data) - self.sequence_length)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(x, y)`` window: inputs ``[t, t+L)`` and target at ``t+L``."""
        x = self.data[idx : idx + self.sequence_length]
        y = self.data[idx + self.sequence_length]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
