import logging
from typing import Protocol, runtime_checkable

log = logging.getLogger("tools.ai.speech.stt")


@runtime_checkable
class STTProvider(Protocol):
    """Protocol for Speech-To-Text providers."""

    def transcribe(self, audio_data: bytes) -> str:
        """Transcribe raw audio bytes to text."""
        ...


class SpeechToTextService:
    """
    Centralized Speech-To-Text service for Sidekick.
    Supports pluggable providers (Vosk, Whisper, Web Speech API fallback).
    """

    def __init__(self, provider: STTProvider | None = None):
        self._provider = provider

    def transcribe(self, audio_data: bytes) -> str:
        """Transcribe audio data using the configured provider."""
        if not self._provider:
            log.warning("No STT provider configured. Transcription skipped.")
            return ""

        try:
            return self._provider.transcribe(audio_data)
        except Exception as e:
            log.error(f"STT Transcription failed: {e}")
            return ""


# Concrete Providers (Lazy loaded)


class VoskProvider:
    """Local offline STT using Vosk."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            try:
                from vosk import KaldiRecognizer, Model

                self._model = Model(self.model_path)
            except ImportError:
                log.error("Vosk not installed. Please install 'vosk' package.")
                raise

    def transcribe(self, audio_data: bytes) -> str:
        self._ensure_model()
        # Implementation details for Vosk recognition go here
        return "[Vosk transcription result]"


class WhisperProvider:
    """High-precision STT using OpenAI Whisper (API or local)."""

    def transcribe(self, audio_data: bytes) -> str:
        return "[Whisper transcription result]"
