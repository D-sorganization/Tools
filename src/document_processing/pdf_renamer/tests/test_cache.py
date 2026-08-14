"""Regression tests for ``ResultCache`` SQLite connection lifecycle.

``sqlite3.connect(...)`` used as a context manager only manages the
*transaction*; it never closes the connection. Every ``ResultCache`` operation
therefore leaked a connection (and its file descriptor), so a long-running
rename session could exhaust the process descriptor limit or leave the cache
database locked. ``contextlib.closing`` is what actually closes it.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest
from pdf_renamer.cache import ResultCache
from pdf_renamer.types import TitleResult


class _ConnectionSpy:
    """Record every connection ``pdf_renamer.cache`` opens."""

    def __init__(self) -> None:
        # Bind the real factory up front: the patch target below is the shared
        # ``sqlite3`` module attribute, so calling ``sqlite3.connect`` from here
        # would re-enter this spy.
        self._connect = sqlite3.connect
        self.connections: list[sqlite3.Connection] = []

    def __call__(self, *args: Any, **kwargs: Any) -> sqlite3.Connection:
        conn = self._connect(*args, **kwargs)
        self.connections.append(conn)
        return conn

    def all_closed(self) -> bool:
        for conn in self.connections:
            try:
                conn.execute("SELECT 1")
            except sqlite3.ProgrammingError:
                continue  # "Cannot operate on a closed database" — as intended.
            return False
        return True


@pytest.fixture
def connection_spy(monkeypatch: pytest.MonkeyPatch) -> _ConnectionSpy:
    spy = _ConnectionSpy()
    monkeypatch.setattr("pdf_renamer.cache.sqlite3.connect", spy)
    return spy


def test_initialize_database_closes_its_connection(
    tmp_path: Path, connection_spy: _ConnectionSpy
) -> None:
    ResultCache(tmp_path / "cache.db")

    assert len(connection_spy.connections) == 1
    assert connection_spy.all_closed()


def test_save_and_get_close_every_connection(
    tmp_path: Path, connection_spy: _ConnectionSpy
) -> None:
    cache = ResultCache(tmp_path / "cache.db")
    cache.save(
        "abc123",
        tmp_path / "doc.pdf",
        TitleResult("A Title", 0.9, "metadata"),
        provider="test",
        model="test-model",
    )

    result = cache.get("abc123")

    # Behaviour is preserved: the round trip still returns the saved record.
    assert result is not None
    assert result.title == "A Title"
    assert result.method == "metadata"

    # One connection per operation (init + save + get), all of them closed.
    assert len(connection_spy.connections) == 3
    assert connection_spy.all_closed()


def test_get_miss_closes_its_connection(
    tmp_path: Path, connection_spy: _ConnectionSpy
) -> None:
    cache = ResultCache(tmp_path / "cache.db")

    assert cache.get("no-such-sha") is None
    assert connection_spy.all_closed()


def test_repeated_operations_do_not_accumulate_open_connections(
    tmp_path: Path, connection_spy: _ConnectionSpy
) -> None:
    cache = ResultCache(tmp_path / "cache.db")
    for index in range(25):
        cache.save(
            f"sha-{index}",
            tmp_path / f"doc-{index}.pdf",
            TitleResult(f"Title {index}", 0.5, "heuristic"),
        )
        assert cache.get(f"sha-{index}") is not None

    assert len(connection_spy.connections) == 51
    assert connection_spy.all_closed()
