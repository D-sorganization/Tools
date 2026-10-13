"""Durable on-disk spool for store-and-forward historian delivery (issue #5541, H2).

One responsibility: buffer scan records to a bounded append-only spool when the
downstream sink fails, and replay them in FIFO order from a durable checkpoint
when downstream recovers.

Design by Contract & Safety:
* :meth:`submit` is non-blocking and safe to call from the poll loop; if the spool
  is full or unwritable, the record is dropped with an incremented counter and
  never raises into the poll loop.
* Bounded spool capacity: records exceeding ``max_bytes`` are dropped to prevent
  unbounded disk growth.
* Replay happens off the event loop via :meth:`drain_once` in batches up to
  ``batch_max``.
* Corrupt or truncated records at the tail of the spool are skipped with a warning
  log without crashing the runtime.
* Checkpoints track byte offsets into the spool file so restarts resume replay
  without duplicating already-delivered records.
"""

from __future__ import annotations

import asyncio
import json
import logging
import struct
import threading
import time
import zlib
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

from historian import HistorianRecord
from historian_sink import HistorianSink

logger = logging.getLogger("dcs_backend.historian_spool")

__all__ = ["SpoolingSink"]

MAGIC = b"\xaa\x55"
HEADER_FORMAT = ">2sII"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
DEFAULT_MAX_BYTES = 10 * 1024 * 1024
DEFAULT_BATCH_MAX = 32


class SpoolingSink:
    """Store-and-forward sink decorator buffering records on downstream failure."""

    def __init__(
        self,
        downstream: HistorianSink,
        spool_path: Path | str,
        checkpoint_path: Path | str | None = None,
        max_bytes: int = DEFAULT_MAX_BYTES,
        batch_max: int = DEFAULT_BATCH_MAX,
        to_thread: Callable[..., Awaitable[Any]] = asyncio.to_thread,
    ) -> None:
        """Initialize spooling sink with downstream target and storage paths."""
        if not isinstance(downstream, HistorianSink):
            cls_name = type(downstream).__name__
            raise TypeError(f"downstream must implement HistorianSink, got {cls_name}")
        if not isinstance(max_bytes, int) or isinstance(max_bytes, bool):
            raise TypeError(f"max_bytes must be an int, got {type(max_bytes).__name__}")
        if max_bytes < 1:
            raise ValueError(f"max_bytes must be >= 1, got {max_bytes}")
        if not isinstance(batch_max, int) or isinstance(batch_max, bool):
            raise TypeError(f"batch_max must be an int, got {type(batch_max).__name__}")
        if batch_max < 1:
            raise ValueError(f"batch_max must be >= 1, got {batch_max}")
        if not callable(to_thread):
            raise TypeError("to_thread must be callable")

        self._downstream = downstream
        self._spool_path = Path(spool_path)
        self._checkpoint_path = (
            Path(checkpoint_path)
            if checkpoint_path is not None
            else self._spool_path.with_suffix(self._spool_path.suffix + ".checkpoint")
        )
        self._max_bytes = max_bytes
        self._batch_max = batch_max
        self._to_thread = to_thread

        self._spool_path.parent.mkdir(parents=True, exist_ok=True)
        self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        self._spooled_records: int = 0
        self._replayed_records: int = 0
        self._dropped_records: int = 0
        self._quarantined_files: int = 0
        self._closed: bool = False
        self._lock = threading.RLock()

        self._checkpoint: int = self._read_checkpoint()

    # -- Persistence helpers --------------------------------------------------

    def _read_checkpoint(self) -> int:
        """Read saved checkpoint offset from disk."""
        if not self._checkpoint_path.exists():
            return 0
        try:
            content = self._checkpoint_path.read_text(encoding="utf-8").strip()
            offset = int(content) if content else 0
            return max(0, offset)
        except (OSError, ValueError) as exc:
            logger.warning(
                "Failed to read checkpoint from %s, defaulting to 0: %s",
                self._checkpoint_path,
                exc,
            )
            return 0

    def _write_checkpoint(self, offset: int) -> None:
        """Persist checkpoint offset to disk and memory."""
        self._checkpoint = offset
        try:
            self._checkpoint_path.write_text(f"{offset}\n", encoding="utf-8")
        except OSError as exc:
            logger.warning(
                "Failed to write checkpoint to %s: %s", self._checkpoint_path, exc
            )

    def _get_spool_size(self) -> int:
        """Return the current on-disk size of the spool file."""
        try:
            return self._spool_path.stat().st_size if self._spool_path.exists() else 0
        except OSError:
            return 0

    def has_pending(self) -> bool:
        """Return True if un-replayed records remain in the spool."""
        return self._get_spool_size() > self._checkpoint

    def _reset_spool(self) -> None:
        """Truncate the spool file and reset checkpoint to zero."""
        with self._lock:
            try:
                with open(self._spool_path, "wb"):
                    pass
            except OSError as exc:
                logger.warning(
                    "Failed to truncate spool file %s: %s", self._spool_path, exc
                )
            self._write_checkpoint(0)

    def _quarantine_spool(self) -> Path | None:
        """Quarantine a corrupted/tampered spool file by moving it aside."""
        with self._lock:
            if not self._spool_path.exists() or self._spool_path.stat().st_size == 0:
                self._reset_spool()
                return None
            timestamp = int(time.time())
            quarantine_path = self._spool_path.with_name(
                f"{self._spool_path.name}.quarantine.{timestamp}"
            )
            counter = 0
            while quarantine_path.exists():
                counter += 1
                quarantine_path = self._spool_path.with_name(
                    f"{self._spool_path.name}.quarantine.{timestamp}_{counter}"
                )
            try:
                self._spool_path.rename(quarantine_path)
                logger.warning(
                    "Quarantined corrupt spool file %s -> %s",
                    self._spool_path,
                    quarantine_path,
                )
                self._quarantined_files += 1
            except OSError as exc:
                logger.warning(
                    "Failed to quarantine spool file %s: %s", self._spool_path, exc
                )
            self._reset_spool()
            return quarantine_path

    # -- Submit (producer side) -----------------------------------------------

    def submit(self, record: HistorianRecord) -> bool:
        """Submit a scan record without blocking.

        If previous records are pending in the spool or the downstream sink fails,
        the record is appended to the spool to preserve FIFO ordering.
        If the spool is full or unwritable, the record is dropped and the drop
        counter increments.
        """
        if not isinstance(record, HistorianRecord):
            raise TypeError(
                f"record must be a HistorianRecord, got {type(record).__name__}"
            )
        if self._closed:
            return False

        if self.has_pending():
            return self._append_to_spool(record)

        try:
            res = self._downstream.submit(record)
            if res:
                return True
            return False
        except Exception as exc:
            logger.warning("Downstream sink failed, spooling record: %s", exc)
            return self._append_to_spool(record)

    def _append_to_spool(self, record: HistorianRecord) -> bool:
        """Encode and append a single record to the spool file under lock."""
        try:
            payload = json.dumps(record.to_dict(), separators=(",", ":")).encode(
                "utf-8"
            )
        except Exception as exc:
            logger.warning("Failed to serialize historian record: %s", exc)
            self._dropped_records += 1
            return False

        length = len(payload)
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        header = struct.pack(HEADER_FORMAT, MAGIC, length, crc)
        frame = header + payload

        with self._lock:
            current_size = self._get_spool_size()
            if current_size + len(frame) > self._max_bytes:
                self._dropped_records += 1
                logger.warning(
                    "Spool file full (%d + %d > %d bytes); dropping record",
                    current_size,
                    len(frame),
                    self._max_bytes,
                )
                return False

            try:
                with open(self._spool_path, "ab") as f:
                    f.write(frame)
                    f.flush()
                self._spooled_records += 1
                return True
            except OSError as exc:
                self._dropped_records += 1
                logger.warning(
                    "Failed to append record to spool %s: %s", self._spool_path, exc
                )
                return False

    # -- Drain / replay (consumer side) ---------------------------------------

    def _read_batch(
        self, start_offset: int, batch_max: int
    ) -> tuple[list[tuple[HistorianRecord, int]], bool, int]:
        """Read up to batch_max records from spool starting at start_offset.

        Returns:
            Tuple of (records_with_end_offsets, corruption_detected, new_checkpoint).
        """
        with self._lock:
            if not self._spool_path.exists():
                return [], False, start_offset

            file_size = self._get_spool_size()
            if start_offset >= file_size:
                return [], False, start_offset

            batch: list[tuple[HistorianRecord, int]] = []
            corruption_detected = False
            current_offset = start_offset

            try:
                with open(self._spool_path, "rb") as f:
                    f.seek(start_offset)
                    while len(batch) < batch_max:
                        offset_before = f.tell()
                        header_bytes = f.read(HEADER_SIZE)
                        if not header_bytes:
                            break
                        if len(header_bytes) < HEADER_SIZE:
                            corruption_detected = True
                            current_offset = file_size
                            break

                        magic, length, expected_crc = struct.unpack(
                            HEADER_FORMAT, header_bytes
                        )
                        if magic != MAGIC:
                            corruption_detected = True
                            current_offset = file_size
                            break

                        if offset_before + HEADER_SIZE + length > file_size:
                            corruption_detected = True
                            current_offset = file_size
                            break

                        payload = f.read(length)
                        if len(payload) < length:
                            corruption_detected = True
                            current_offset = file_size
                            break

                        computed_crc = zlib.crc32(payload) & 0xFFFFFFFF
                        if computed_crc != expected_crc:
                            corruption_detected = True
                            current_offset = file_size
                            break

                        try:
                            decoded = json.loads(payload.decode("utf-8"))
                            record = HistorianRecord.from_dict(decoded)
                        except Exception as exc:
                            logger.warning(
                                "Corrupt or tampered record payload in spool %s: %s",
                                self._spool_path,
                                exc,
                            )
                            corruption_detected = True
                            current_offset = file_size
                            break

                        current_offset = f.tell()
                        batch.append((record, current_offset))

            except OSError as exc:
                logger.warning("Error reading spool file %s: %s", self._spool_path, exc)
                return batch, False, current_offset

            return batch, corruption_detected, current_offset

    async def drain_once(self) -> int:
        """Replay one batch of spooled records to downstream."""
        if self._closed:
            return 0

        file_size = self._get_spool_size()
        if file_size == 0 or self._checkpoint >= file_size:
            if file_size > 0:
                await self._to_thread(self._reset_spool)
            return 0

        batch, corruption_detected, new_checkpoint = await self._to_thread(
            self._read_batch, self._checkpoint, self._batch_max
        )

        replayed = 0
        for record, record_end_offset in batch:
            try:
                res = self._downstream.submit(record)
                if not res:
                    break
            except Exception as exc:
                logger.warning("Downstream sink failed during replay: %s", exc)
                break

            replayed += 1
            self._replayed_records += 1
            self._write_checkpoint(record_end_offset)

        if corruption_detected:
            logger.warning(
                "Corrupt/tampered record in spool %s: quarantining file",
                self._spool_path,
            )
            await self._to_thread(self._quarantine_spool)
            self._write_checkpoint(0)

        file_size = self._get_spool_size()
        if self._checkpoint >= file_size:
            await self._to_thread(self._reset_spool)

        return replayed

    # -- Diagnostics & Lifecycle ----------------------------------------------

    def stats(self) -> Mapping[str, int]:
        """Return diagnostic counters and spool status."""
        base_stats: dict[str, int] = {
            "spooled_records": self._spooled_records,
            "replayed_records": self._replayed_records,
            "dropped_records": self._dropped_records,
            "quarantined_files": self._quarantined_files,
            "spool_bytes": self._get_spool_size(),
        }
        stats_fn = getattr(self._downstream, "stats", None)
        if callable(stats_fn):
            try:
                res = stats_fn()
                if isinstance(res, Mapping):
                    for k, v in res.items():
                        if k not in base_stats:
                            try:
                                base_stats[k] = int(v)
                            except (ValueError, TypeError):
                                pass
            except Exception:
                pass
        return base_stats

    def close(self) -> None:
        """Release resources idempotently."""
        if self._closed:
            return
        self._closed = True
        closer = getattr(self._downstream, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception as exc:
                logger.warning("Failed to close downstream sink: %s", exc)
