"""Unit tests for shared local Whisper VoiceTranscriber (Epic #2591, #5730)."""

from __future__ import annotations

import io
import wave
from pathlib import Path

import pytest

from shared.python.sidekick import voice
from shared.python.sidekick.voice import (
    TranscriptionResult,
    VoiceTranscriber,
    VoiceUnavailableError,
    VoiceValidationError,
    insert_transcript_into_composer,
)


def _make_wav(duration_s: float = 0.4, amplitude: int = 0, rate: int = 16000) -> bytes:
    frames = int(duration_s * rate)
    sample = int(amplitude).to_bytes(2, byteorder="little", signed=True)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(sample * frames)
    return buf.getvalue()


def test_silent_wav_gate_and_hallucination_suppression() -> None:
    transcriber = VoiceTranscriber(
        backend_override=lambda _p: TranscriptionResult(
            "Thank you for watching!", 0.4, 0.1
        )
    )
    silent_res = transcriber.transcribe_bytes(_make_wav(0.5, amplitude=0), "audio/wav")
    assert silent_res.text == ""
    assert silent_res.engine == "vad-silence-gate"

    hallucinated = transcriber.transcribe_bytes(
        _make_wav(0.4, amplitude=1800), "audio/wav"
    )
    assert hallucinated.text == ""


def test_transcribe_success_and_0600_temp_cleanup() -> None:
    observed_modes: list[int] = []
    observed_paths: list[Path] = []

    def _fake_engine(path: Path) -> TranscriptionResult:
        observed_paths.append(path)
        observed_modes.append(path.stat().st_mode & 0o777)
        return TranscriptionResult(
            "optimize syngas H2 CO ratio", 1.1, 0.05, "faster-whisper"
        )

    transcriber = VoiceTranscriber(backend_override=_fake_engine)
    res = transcriber.transcribe_bytes(
        _make_wav(0.4, amplitude=2200), "audio/webm;codecs=opus"
    )
    assert res.text == "optimize syngas H2 CO ratio"
    assert observed_modes == [0o600]
    assert observed_paths and not observed_paths[0].exists()


def test_validation_and_unavailable_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    transcriber = VoiceTranscriber(
        backend_override=lambda _p: TranscriptionResult("ok", 0.2, 0.0)
    )
    with pytest.raises(VoiceValidationError, match="Unsupported audio MIME"):
        transcriber.transcribe_bytes(b"abc", "application/octet-stream")
    with pytest.raises(VoiceValidationError, match="must not be empty"):
        transcriber.transcribe_bytes(b"", "audio/wav")
    with pytest.raises(VoiceValidationError, match="exceeds 60s"):
        transcriber.transcribe_bytes(
            _make_wav(61.0, amplitude=500, rate=1000), "audio/wav"
        )

    no_engine = VoiceTranscriber()
    monkeypatch.setattr("importlib.util.find_spec", lambda _n: None)
    monkeypatch.setattr("shutil.which", lambda _b: None)
    assert no_engine.detect_capability().available is False
    with pytest.raises(VoiceUnavailableError):
        no_engine.transcribe_bytes(_make_wav(0.3, amplitude=1500), "audio/wav")


def test_insert_transcript_into_composer_never_autosends() -> None:
    assert insert_transcript_into_composer("", "run simulation") == "run simulation"
    assert (
        insert_transcript_into_composer("Please", "check reactor temp")
        == "Please check reactor temp"
    )
    assert (
        insert_transcript_into_composer("Hello world", "there", cursor_pos=5)
        == "Hello there world"
    )


def test_constructing_another_transcriber_preserves_inflight_audio(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    real_path = Path

    def isolated_path(value: str | Path) -> Path:
        candidate = real_path(value)
        return tmp_path if candidate == real_path("/dev/shm") else candidate

    monkeypatch.setattr(voice, "Path", isolated_path)
    monkeypatch.setattr(voice.tempfile, "gettempdir", lambda: str(tmp_path))
    observed_paths: list[Path] = []

    def _fake_engine(path: Path) -> TranscriptionResult:
        observed_paths.append(path)
        assert path.exists()
        VoiceTranscriber(backend_override=lambda _path: TranscriptionResult("ok", 0.1))
        assert path.exists()
        return TranscriptionResult("transcribed", 0.1)

    transcriber = VoiceTranscriber(backend_override=_fake_engine)
    result = transcriber.transcribe_bytes(_make_wav(0.4, amplitude=2200), "audio/wav")

    assert result.text == "transcribed"
    assert observed_paths and not observed_paths[0].exists()
