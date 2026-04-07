"""Regression tests for the vectorized filter engine split."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from data_processor.vectorized_filter_engine import VectorizedFilterEngine


def test_vectorized_filter_engine_facade_uses_extracted_modules(
    repo_root: Path,
) -> None:
    engine_path = (
        repo_root
        / "src"
        / "data_processing"
        / "data_processor"
        / "python"
        / "data_processor"
        / "vectorized_filter_engine.py"
    )
    content = engine_path.read_text(encoding="utf-8")

    assert "vectorized_filter_time_domain" in content
    assert "vectorized_filter_frequency_domain" in content


def test_frequency_response_still_returns_arrays() -> None:
    engine = VectorizedFilterEngine(n_jobs=1)
    frequencies, magnitude = engine.calculate_frequency_response(
        "FFT Low-pass",
        {"fft_freq_low": 0.1, "fft_transition_bw": 0.05},
    )

    assert frequencies.size > 0
    assert magnitude.size == frequencies.size


def test_fft_filter_still_preserves_signal_length() -> None:
    engine = VectorizedFilterEngine(n_jobs=1)
    signal = pd.Series(np.sin(np.linspace(0, 4 * np.pi, 128)))

    filtered = engine._apply_fft_filter_vectorized(
        signal,
        {
            "filter_type": "FFT Low-pass",
            "fft_freq_low": 0.2,
            "fft_transition_bw": 0.05,
        },
    )

    assert len(filtered) == len(signal)
