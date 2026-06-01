from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2] / "src" / "python" / "shared"),
)

from performance_utils import (  # noqa: E402
    FastHasher,
    MemoryOptimizedProcessor,
    OptimizedFileScanner,
)


def test_scan_directory_cache_invalidates_when_root_mtime_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scanner = OptimizedFileScanner(max_workers=1)
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("first", encoding="utf-8")
    first_mtime = 1_800_000_000
    second_mtime = first_mtime + 5
    os.utime(tmp_path, (first_mtime, first_mtime))
    monotonic_values = iter([100.0, 101.0])
    monkeypatch.setattr(
        "performance_utils.time.monotonic",
        lambda: next(monotonic_values),
    )

    initial = {path.name for path in scanner.scan_directory_parallel(tmp_path, "*.txt")}
    second.write_text("second", encoding="utf-8")
    os.utime(tmp_path, (second_mtime, second_mtime))
    rescanned = {
        path.name for path in scanner.scan_directory_parallel(tmp_path, "*.txt")
    }

    assert initial == {"first.txt"}
    assert rescanned == {"first.txt", "second.txt"}


def test_scan_directory_cache_reuses_fresh_unchanged_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scanner = OptimizedFileScanner(max_workers=1)
    file_path = tmp_path / "first.txt"
    file_path.write_text("first", encoding="utf-8")
    stable_mtime = 1_800_000_000
    os.utime(tmp_path, (stable_mtime, stable_mtime))
    monotonic_values = iter([200.0, 205.0])
    monkeypatch.setattr(
        "performance_utils.time.monotonic",
        lambda: next(monotonic_values),
    )

    first_scan = list(scanner.scan_directory_parallel(tmp_path, "*.txt"))
    file_path.unlink()
    os.utime(tmp_path, (stable_mtime, stable_mtime))
    cached_scan = list(scanner.scan_directory_parallel(tmp_path, "*.txt"))

    assert first_scan == [file_path]
    assert cached_scan == [file_path]


def test_scan_directory_cache_expires_after_ttl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scanner = OptimizedFileScanner(max_workers=1)
    first = tmp_path / "first.txt"
    first.write_text("first", encoding="utf-8")
    stable_mtime = 1_800_000_000
    os.utime(tmp_path, (stable_mtime, stable_mtime))
    monotonic_values = iter([300.0, 361.0])
    monkeypatch.setattr(
        "performance_utils.time.monotonic",
        lambda: next(monotonic_values),
    )

    first_scan = list(scanner.scan_directory_parallel(tmp_path, "*.txt"))
    first.unlink()
    os.utime(tmp_path, (stable_mtime, stable_mtime))
    expired_scan = list(scanner.scan_directory_parallel(tmp_path, "*.txt"))

    assert first_scan == [first]
    assert expired_scan == []


def test_scan_directory_parallel_branch_suppresses_worker_os_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ["a", "b"]:
        subdir = tmp_path / name
        subdir.mkdir()
        (subdir / f"{name}.txt").write_text(name, encoding="utf-8")

    class FailingFuture:
        def result(self) -> list[Path]:
            raise OSError("worker failed")

    class FakeExecutor:
        def __init__(self, *, max_workers: int) -> None:
            self.max_workers = max_workers

        def __enter__(self) -> FakeExecutor:
            return self

        def __exit__(self, *exc_info: object) -> None:
            return None

        def submit(self, *args: object, **kwargs: object) -> FailingFuture:
            return FailingFuture()

    monkeypatch.setattr("performance_utils.ThreadPoolExecutor", FakeExecutor)
    monkeypatch.setattr(
        "performance_utils.as_completed",
        lambda futures: list(futures),
    )

    found = list(OptimizedFileScanner(max_workers=4).scan_directory_parallel(tmp_path))

    assert found == []


def test_scan_directory_handles_missing_and_inaccessible_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scanner = OptimizedFileScanner(max_workers=1)
    missing = tmp_path / "missing"

    assert list(scanner.scan_directory_parallel(missing)) == []

    real_iterdir = Path.iterdir

    def fail_iterdir(path: Path) -> Any:
        if path == tmp_path:
            raise PermissionError("blocked")
        return real_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", fail_iterdir)

    assert list(scanner.scan_directory_parallel(tmp_path)) == []


def test_fast_hash_handles_empty_small_large_and_unreadable_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty = tmp_path / "empty.bin"
    small = tmp_path / "small.bin"
    large = tmp_path / "large.bin"
    missing = tmp_path / "missing.bin"
    empty.write_bytes(b"")
    small.write_bytes(b"abc")
    large.write_bytes(b"a" * (FastHasher.FAST_CHUNK_SIZE * 2 + 1))

    assert FastHasher.fast_hash(empty).startswith("empty_")
    assert FastHasher.fast_hash(small) == FastHasher.full_hash(small)
    assert FastHasher.fast_hash(large).startswith("fast_")
    assert FastHasher.fast_hash(missing) == "error_missing.bin"
    assert FastHasher.full_hash(missing) == "error_missing.bin"

    real_open = Path.open

    def fail_on_large(path: Path, *args: object, **kwargs: object) -> object:
        if path == large:
            raise OSError("blocked")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_on_large)

    assert FastHasher.fast_hash(large) == "error_large.bin"


def test_chunked_processing_yields_chunks_or_processed_results() -> None:
    processor = MemoryOptimizedProcessor()

    assert list(processor.chunked_processing([1, 2, 3, 4, 5], chunk_size=2)) == [
        [1, 2],
        [3, 4],
        [5],
    ]
    assert list(
        processor.chunked_processing(
            [1, 2, 3, 4, 5],
            chunk_size=2,
            processor_func=sum,
        )
    ) == [3, 7, 5]


def test_lazy_file_reader_yields_chunks_and_suppresses_os_errors(
    tmp_path: Path,
) -> None:
    processor = MemoryOptimizedProcessor()
    file_path = tmp_path / "data.txt"
    file_path.write_text("abcdef", encoding="utf-8")

    assert list(processor.lazy_file_reader(file_path, chunk_size=2)) == [
        "ab",
        "cd",
        "ef",
    ]
    assert list(processor.lazy_file_reader(tmp_path / "missing.txt")) == []
