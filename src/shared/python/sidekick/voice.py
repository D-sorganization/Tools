"""Shared local Whisper voice-to-text transcriber for Sidekick & host apps (#5730)."""

from __future__ import annotations

import importlib.util
import io
import os
import shutil
import struct
import subprocess
import tempfile
import threading
import uuid
import wave
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ALLOWED_AUDIO_MIMES: frozenset[str] = frozenset(
    {"audio/webm", "audio/wav", "audio/x-wav", "audio/ogg", "audio/mp4", "audio/mpeg"}
)
MAX_AUDIO_BYTES = 10 * 1024 * 1024
MAX_AUDIO_DURATION_S = 60.0
NO_SPEECH_PROB_THRESHOLD = 0.65
TEMP_PREFIX = "sidekick-voice-"
HALLUCINATION_PHRASES: frozenset[str] = frozenset(
    {
        "you",
        "thank you",
        "thanks for watching",
        "thank you for watching",
        "subtitles by the amara.org community",
        "[blank_audio]",
    }
)


class VoiceUnavailableError(RuntimeError):
    """Raised when no local Whisper transcription engine is available."""


class VoiceValidationError(ValueError):
    """Raised when audio MIME, byte size, or duration violates DbC invariants."""


@dataclass(frozen=True)
class VoiceCapability:
    available: bool
    engine: str
    model: str
    detail: str


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    duration_s: float
    no_speech_prob: float = 0.0
    engine: str = "faster-whisper"


def sweep_stale_voice_temp_files() -> int:
    removed = 0
    for base in {Path("/dev/shm"), Path(tempfile.gettempdir())}:  # nosec B108
        if not base.is_dir():
            continue
        for candidate in base.glob(f"{TEMP_PREFIX}*.tmp"):
            try:
                candidate.unlink(missing_ok=True)
                removed += 1
            except OSError:
                pass
    return removed


def _inspect_wav(payload: bytes) -> tuple[float | None, bool]:
    if len(payload) < 44 or payload[:4] != b"RIFF" or payload[8:12] != b"WAVE":
        return None, False
    try:
        with wave.open(io.BytesIO(payload), "rb") as wf:
            frames, rate, width = wf.getnframes(), wf.getframerate(), wf.getsampwidth()
            duration_s = (frames / float(rate)) if rate > 0 else 0.0
            if frames == 0:
                return duration_s, True
            raw = wf.readframes(min(frames, rate * 10))
            if width == 2 and len(raw) >= 2:
                samples = struct.unpack(
                    f"<{len(raw) // 2}h", raw[: (len(raw) // 2) * 2]
                )
                return duration_s, max((abs(s) for s in samples), default=0) < 32
            return duration_s, all(b == 0 for b in raw)
    except (wave.Error, EOFError, struct.error):
        return None, False


def filter_silence_hallucination(text: str, no_speech_prob: float = 0.0) -> str:
    cleaned = text.strip()
    if not cleaned or no_speech_prob >= NO_SPEECH_PROB_THRESHOLD:
        return ""
    return "" if cleaned.lower().rstrip(".!") in HALLUCINATION_PHRASES else cleaned


class VoiceTranscriber:
    """Thread-safe local Whisper transcriber with 0600 temp-file hygiene."""

    def __init__(
        self,
        model_name: str | None = None,
        backend_override: Callable[[Path], TranscriptionResult] | None = None,
    ) -> None:
        self.model_name: str = (
            model_name or os.environ.get("SIDEKICK_WHISPER_MODEL") or "base.en"
        )
        self._backend_override = backend_override
        self._lock = threading.Semaphore(1)
        self._cached_model: Any = None
        sweep_stale_voice_temp_files()

    def detect_capability(self) -> VoiceCapability:
        if self._backend_override is not None:
            return VoiceCapability(True, "override", self.model_name, "ready")
        if importlib.util.find_spec("faster_whisper") is not None:
            if shutil.which("ffmpeg") is None:
                return VoiceCapability(
                    False, "faster-whisper", self.model_name, "ffmpeg missing"
                )
            return VoiceCapability(True, "faster-whisper", self.model_name, "ready")
        for binary in ("whisper-cli", "whisper-cpp", "whisper"):
            if shutil.which(binary):
                return VoiceCapability(True, binary, self.model_name, "ready")
        return VoiceCapability(
            False, "none", self.model_name, "No local Whisper engine found."
        )

    def transcribe_bytes(
        self, payload: bytes, mime_type: str = "audio/wav"
    ) -> TranscriptionResult:
        mime = (mime_type or "").split(";", 1)[0].strip().lower()
        if mime not in ALLOWED_AUDIO_MIMES:
            raise VoiceValidationError(f"Unsupported audio MIME type: {mime!r}")
        if not payload:
            raise VoiceValidationError("Audio payload must not be empty.")
        if len(payload) > MAX_AUDIO_BYTES:
            raise VoiceValidationError(
                f"Audio payload exceeds {MAX_AUDIO_BYTES} bytes."
            )

        wav_duration_s, is_silent_wav = _inspect_wav(payload)
        if wav_duration_s is not None and wav_duration_s > MAX_AUDIO_DURATION_S:
            raise VoiceValidationError(
                f"WAV duration {wav_duration_s:.1f}s exceeds 60s limit."
            )
        if is_silent_wav:
            return TranscriptionResult(
                "", round(wav_duration_s or 0.0, 3), 1.0, "vad-silence-gate"
            )

        cap = self.detect_capability()
        if not cap.available:
            raise VoiceUnavailableError(cap.detail)

        shm = Path("/dev/shm")  # nosec B108
        temp_dir = (
            shm
            if shm.is_dir() and os.access(shm, os.W_OK)
            else Path(tempfile.gettempdir())
        )
        temp_path = temp_dir / f"{TEMP_PREFIX}{uuid.uuid4().hex}.tmp"
        try:
            fd = os.open(str(temp_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
            with self._lock:
                raw_res = self._run_engine(temp_path, cap)
        finally:
            temp_path.unlink(missing_ok=True)

        if raw_res.duration_s > MAX_AUDIO_DURATION_S:
            raise VoiceValidationError(
                f"Audio duration {raw_res.duration_s:.1f}s exceeds 60s."
            )
        filtered = filter_silence_hallucination(raw_res.text, raw_res.no_speech_prob)
        return TranscriptionResult(
            filtered,
            round(raw_res.duration_s or (wav_duration_s or 0.0), 3),
            raw_res.no_speech_prob,
            raw_res.engine,
        )

    def _run_engine(self, temp_path: Path, cap: VoiceCapability) -> TranscriptionResult:
        if self._backend_override is not None:
            return self._backend_override(temp_path)
        if cap.engine == "faster-whisper":
            fw_mod = importlib.import_module("faster_whisper")
            if self._cached_model is None:
                self._cached_model = fw_mod.WhisperModel(
                    self.model_name, device="cpu", compute_type="int8"
                )
            segments, info = self._cached_model.transcribe(
                str(temp_path), beam_size=1, vad_filter=True
            )
            seg_list = list(segments)
            text = " ".join(s.text.strip() for s in seg_list if s.text.strip())
            avg_nsp = (
                sum(getattr(s, "no_speech_prob", 0.0) for s in seg_list) / len(seg_list)
                if seg_list
                else 1.0
            )
            return TranscriptionResult(
                text, float(getattr(info, "duration", 0.0) or 0.0), avg_nsp
            )
        proc = subprocess.run(
            [cap.engine, "-f", str(temp_path), "-nt"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if proc.returncode != 0:
            raise VoiceUnavailableError(
                f"{cap.engine} exited with code {proc.returncode}"
            )
        return TranscriptionResult(proc.stdout.strip(), 0.0, 0.0, cap.engine)


def insert_transcript_into_composer(
    current_text: str, transcript: str, cursor_pos: int | None = None
) -> str:
    """Insert transcribed speech into composer text (never auto-sends)."""
    cleaned = transcript.strip()
    if not cleaned:
        return current_text
    if cursor_pos is None or cursor_pos >= len(current_text):
        sep = " " if current_text and not current_text.endswith((" ", "\n")) else ""
        return f"{current_text}{sep}{cleaned}"
    pos = max(0, cursor_pos)
    before, after = current_text[:pos], current_text[pos:]
    left_sep = " " if before and not before.endswith((" ", "\n")) else ""
    right_sep = (
        " " if after and not after.startswith((" ", "\n", ".", ",", "?", "!")) else ""
    )
    return f"{before}{left_sep}{cleaned}{right_sep}{after}"
