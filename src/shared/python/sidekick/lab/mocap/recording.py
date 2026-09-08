"""Crash-safe recording writer and reader for multi-camera session streams."""

from __future__ import annotations

import json
import os
import zlib
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ._validation import (
    require_nonnegative_integer,
    require_text,
)
from .acquisition import FramePacket
from .enums import SessionState
from .serialization import dumps_canonical, load_session_manifest
from .session import MocapSessionManifest


@dataclass(frozen=True, slots=True)
class FrameIndexEntry:
    """Index record for one recorded frame chunk in an append-only stream."""

    stream_id: str
    sequence_number: int
    timestamp_ns: int
    byte_offset: int
    payload_bytes: int
    checksum_crc32: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "stream_id", require_text(self.stream_id, "stream_id"))
        object.__setattr__(
            self,
            "sequence_number",
            require_nonnegative_integer(self.sequence_number, "sequence_number"),
        )
        object.__setattr__(
            self,
            "timestamp_ns",
            require_nonnegative_integer(self.timestamp_ns, "timestamp_ns"),
        )
        object.__setattr__(
            self,
            "byte_offset",
            require_nonnegative_integer(self.byte_offset, "byte_offset"),
        )
        object.__setattr__(
            self,
            "payload_bytes",
            require_nonnegative_integer(self.payload_bytes, "payload_bytes"),
        )
        chk = require_text(self.checksum_crc32, "checksum_crc32")
        if len(chk) != 8 or any(c not in "0123456789abcdef" for c in chk.lower()):
            raise ValueError("checksum_crc32 must be 8 hexadecimal characters")
        object.__setattr__(self, "checksum_crc32", chk.lower())

    def to_json(self) -> str:
        data = {
            "stream_id": self.stream_id,
            "sequence_number": self.sequence_number,
            "timestamp_ns": self.timestamp_ns,
            "byte_offset": self.byte_offset,
            "payload_bytes": self.payload_bytes,
            "checksum_crc32": self.checksum_crc32,
        }
        return json.dumps(data, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FrameIndexEntry:
        return cls(
            stream_id=data["stream_id"],
            sequence_number=data["sequence_number"],
            timestamp_ns=data["timestamp_ns"],
            byte_offset=data["byte_offset"],
            payload_bytes=data["payload_bytes"],
            checksum_crc32=data["checksum_crc32"],
        )


@dataclass(frozen=True, slots=True)
class RecordingIntegrityReport:
    """Validation report examining recorded frame payloads and index integrity."""

    total_frames: int
    corrupt_frames: int
    missing_frames: int
    details: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return self.corrupt_frames == 0 and self.missing_frames == 0


class RecordingWriter:
    """Crash-safe append-only session recording writer with atomic manifest."""

    def __init__(self, session_dir: Path | str, manifest: MocapSessionManifest) -> None:
        if not isinstance(manifest, MocapSessionManifest):
            raise TypeError("manifest must be a MocapSessionManifest")
        self._session_dir = Path(session_dir)
        self._manifest = manifest
        self._streams_dir = self._session_dir / "streams"
        self._manifest_file = self._session_dir / "session_manifest.json"
        self._index_file = self._session_dir / "index.jsonl"
        self._stream_handles: dict[str, Any] = {}
        self._stream_offsets: dict[str, int] = {}
        self._is_open = False

    def initialize(self) -> None:
        """Create directories and atomically write initial manifest."""
        self._session_dir.mkdir(parents=True, exist_ok=True)
        self._streams_dir.mkdir(parents=True, exist_ok=True)
        self._atomic_write_manifest(self._manifest)
        self._is_open = True

    def write_frame(self, packet: FramePacket) -> FrameIndexEntry:
        """Append a frame packet to its stream file and update index."""
        if not self._is_open:
            raise RuntimeError(
                "RecordingWriter must be initialized before writing frames"
            )
        if not isinstance(packet, FramePacket):
            raise TypeError("packet must be a FramePacket")

        # Enforce recording policy
        if self._manifest.recording_policy.no_store and packet.image_bytes:
            raise ValueError("no_store policy forbids writing raw image bytes to disk")

        s_id = packet.source_id
        if s_id not in self._stream_handles:
            stream_path = self._streams_dir / f"{s_id}.bin"
            self._stream_handles[s_id] = open(stream_path, "a+b")
            self._stream_offsets[s_id] = stream_path.stat().st_size

        handle = self._stream_handles[s_id]
        offset = self._stream_offsets[s_id]
        payload = packet.image_bytes
        payload_len = len(payload)

        # Compute CRC32 checksum
        crc_val = zlib.crc32(payload) & 0xFFFFFFFF
        crc_str = f"{crc_val:08x}"

        # Write payload
        handle.write(payload)
        handle.flush()
        self._stream_offsets[s_id] += payload_len

        entry = FrameIndexEntry(
            stream_id=s_id,
            sequence_number=packet.sequence_number,
            timestamp_ns=packet.timestamp_ns,
            byte_offset=offset,
            payload_bytes=payload_len,
            checksum_crc32=crc_str,
        )

        with open(self._index_file, "a", encoding="utf-8", newline="\n") as f_idx:
            f_idx.write(entry.to_json() + "\n")

        return entry

    def finalize(self) -> None:
        """Flush and close all streams, then transition manifest to FINALIZED."""
        if not self._is_open:
            return

        for handle in self._stream_handles.values():
            handle.flush()
            handle.close()
        self._stream_handles.clear()

        # Update manifest to FINALIZED atomically
        finalized_manifest = replace(self._manifest, state=SessionState.FINALIZED)
        self._atomic_write_manifest(finalized_manifest)
        self._manifest = finalized_manifest
        self._is_open = False

    def close(self) -> None:
        """Close any remaining open handles."""
        if self._is_open:
            for handle in self._stream_handles.values():
                handle.close()
            self._stream_handles.clear()
            self._is_open = False

    def _atomic_write_manifest(self, manifest: MocapSessionManifest) -> None:
        """Write manifest to a temporary file and atomically rename it."""
        raw_json = dumps_canonical(manifest)
        tmp_file = self._manifest_file.with_suffix(".tmp")
        with open(tmp_file, "w", encoding="utf-8", newline="\n") as f:
            f.write(raw_json)
        # On Windows, os.replace guarantees atomic replacement if target exists
        os.replace(tmp_file, self._manifest_file)


class RecordingReader:
    """Reads recorded sessions, parses index entries, and verifies data integrity."""

    def __init__(self, session_dir: Path | str) -> None:
        self._session_dir = Path(session_dir)
        self._manifest_file = self._session_dir / "session_manifest.json"
        self._index_file = self._session_dir / "index.jsonl"
        self._streams_dir = self._session_dir / "streams"

    def read_manifest(self) -> MocapSessionManifest:
        """Load and validate session manifest from disk."""
        if not self._manifest_file.is_file():
            raise FileNotFoundError(f"manifest file not found at {self._manifest_file}")
        with open(self._manifest_file, encoding="utf-8") as f:
            return load_session_manifest(f.read())

    def read_index(self) -> list[FrameIndexEntry]:
        """Read all index entries in recorded order."""
        if not self._index_file.is_file():
            return []
        entries: list[FrameIndexEntry] = []
        with open(self._index_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    entries.append(FrameIndexEntry.from_dict(data))
        return entries

    def read_frames(self) -> Iterator[FramePacket]:
        """Reconstruct FramePackets from stream logs and index entries."""
        entries = self.read_index()

        # Cache open file handles during read
        handles: dict[str, Any] = {}
        try:
            for entry in entries:
                s_id = entry.stream_id
                if s_id not in handles:
                    stream_path = self._streams_dir / f"{s_id}.bin"
                    handles[s_id] = open(stream_path, "rb")

                f = handles[s_id]
                f.seek(entry.byte_offset)
                payload = f.read(entry.payload_bytes)

                # Construct FramePacket
                yield FramePacket(
                    source_id=s_id,
                    sequence_number=entry.sequence_number,
                    timestamp_ns=entry.timestamp_ns,
                    host_monotonic_ns=entry.timestamp_ns,
                    image_bytes=payload,
                    pixel_format="GRAY8",
                    resolution_px=(16, 16),
                )
        finally:
            for h in handles.values():
                h.close()

    def verify_integrity(self) -> RecordingIntegrityReport:
        """Verify file existence and CRC32 payload checksums for all indexed frames."""
        entries = self.read_index()
        corrupt = 0
        missing = 0
        details: list[str] = []

        handles: dict[str, Any] = {}
        try:
            for entry in entries:
                s_id = entry.stream_id
                stream_path = self._streams_dir / f"{s_id}.bin"
                if not stream_path.is_file():
                    missing += 1
                    details.append(f"Missing stream file for {s_id}")
                    continue

                if s_id not in handles:
                    handles[s_id] = open(stream_path, "rb")

                f = handles[s_id]
                f.seek(entry.byte_offset)
                payload = f.read(entry.payload_bytes)

                if len(payload) != entry.payload_bytes:
                    corrupt += 1
                    details.append(
                        f"Truncated frame {entry.sequence_number} on {s_id}: "
                        f"expected {entry.payload_bytes}b, got {len(payload)}b"
                    )
                    continue

                crc_val = zlib.crc32(payload) & 0xFFFFFFFF
                crc_str = f"{crc_val:08x}"
                if crc_str != entry.checksum_crc32:
                    corrupt += 1
                    details.append(
                        f"CRC mismatch on {s_id} frame {entry.sequence_number}: "
                        f"expected {entry.checksum_crc32}, got {crc_str}"
                    )
        finally:
            for h in handles.values():
                h.close()

        return RecordingIntegrityReport(
            total_frames=len(entries),
            corrupt_frames=corrupt,
            missing_frames=missing,
            details=tuple(details),
        )


__all__ = [
    "FrameIndexEntry",
    "RecordingIntegrityReport",
    "RecordingReader",
    "RecordingWriter",
]
