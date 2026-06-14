"""Static regressions for MATLAB audio processing contracts."""

from __future__ import annotations

import re
from pathlib import Path

CORE = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "media_processing"
    / "audio_processor"
    / "matlab"
    / "audio_signal_processor"
    / "core"
)


def _source(name: str) -> str:
    return (CORE / name).read_text(encoding="utf-8")


def _function_body(source: str, name: str) -> str:
    match = re.search(
        rf"^function\s+(?:\[[^\]]+\]|\w+)\s*=\s*{name}\([^)]*\)\n"
        rf"(?P<body>.*?)(?=^function\s|\Z)",
        source,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match, f"missing MATLAB function {name}"
    return match.group("body")


def test_phase_vocoder_helper_is_exported_from_core() -> None:
    helper = _source("applyPitchShiftFrames.m")

    assert re.search(
        r"^function\s+shifted\s*=\s*applyPitchShiftFrames\(",
        helper,
        re.M,
    )
    assert "for channel = 1:size(audio, 2)" in helper
    assert "nFrames = max(1, ceil((L-N_FFT)/HOP)+1)" in helper
    assert "applyPitchShiftFrames:InvalidOutput" in helper
    assert "Phase vocoder produced invalid output" in helper


def test_advanced_audio_pitch_methods_delegate_to_phase_vocoder() -> None:
    advanced = _source("AdvancedAudioProcessor.m")

    assert (
        "sampleSemitones = expandFrameSemitones(semitones, audioSampleCount(audio))"
        in advanced
    )
    for function_name, forbidden_assignment in (
        ("correctPitch", "corrected = audio"),
        ("advancedPitchShift", "shifted = audio"),
        ("phaseVocoder", "processed = audio"),
    ):
        body = _function_body(advanced, function_name)
        assert "applyPitchShiftFrames(" in body
        assert forbidden_assignment not in body
        assert "Placeholder" not in body


def test_advanced_audio_unimplemented_spatialization_fails_loudly() -> None:
    body = _function_body(_source("AdvancedAudioProcessor.m"), "spatialize3D")

    assert "error('AdvancedAudioProcessor:Spatialize'" in body
    assert "spatialized = audio" not in body
    assert "warning(" not in body


def test_music_generation_stubs_fail_loudly_instead_of_returning_placeholders() -> None:
    music = _source("MusicProductionTools.m")

    expected_errors = {
        "generateHarmony": "MusicProductionTools:HarmonyNotImplemented",
        "generateBassline": "MusicProductionTools:BasslineNotImplemented",
        "generateDrumPattern": "MusicProductionTools:DrumsNotImplemented",
    }
    for function_name, error_id in expected_errors.items():
        body = _function_body(music, function_name)
        assert f"error('{error_id}'" in body
        assert "Placeholder" not in body
        assert "fprintf(" not in body


def test_music_tools_use_shared_phase_vocoder_file() -> None:
    music = _source("MusicProductionTools.m")

    assert (
        "applyPitchShiftFrames(audio, pitchShiftSemitones, fs, options.Speed)" in music
    )
    assert not re.search(
        r"^function\s+shifted\s*=\s*applyPitchShiftFrames\(",
        music,
        re.M,
    )
