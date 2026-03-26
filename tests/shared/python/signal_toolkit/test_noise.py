"""Tests for signal_toolkit.noise module.

Covers:
- NoiseType enum
- NoiseGenerator: white, pink, brown, blue, violet, uniform, impulse,
  quantization, periodic noise generation
- add_noise_to_signal
- generate_disturbance_profile
- DisturbanceSimulator chaining
"""

from __future__ import annotations

import numpy as np
import pytest
from signal_toolkit.core import Signal
from signal_toolkit.noise import (
    DisturbanceSimulator,
    NoiseGenerator,
    NoiseType,
    add_noise_to_signal,
    generate_disturbance_profile,
)


@pytest.fixture()
def time_array() -> np.ndarray:
    """1 second at 1kHz sampling."""
    return np.linspace(0, 1, 1000)


@pytest.fixture()
def constant_signal(time_array: np.ndarray) -> Signal:
    return Signal(time=time_array, values=np.ones_like(time_array), name="dc")


# ── NoiseType enum ─────────────────────────────────────────────────────


class TestNoiseType:
    def test_values(self) -> None:
        assert NoiseType.WHITE.value == "white"
        assert NoiseType.PINK.value == "pink"
        assert NoiseType.BROWN.value == "brown"
        assert NoiseType.PERIODIC.value == "periodic"


# ── NoiseGenerator ─────────────────────────────────────────────────────


class TestNoiseGenerator:
    def test_white_noise_shape(self, time_array: np.ndarray) -> None:
        gen = NoiseGenerator(seed=42)
        sig = gen.generate(time_array, NoiseType.WHITE, amplitude=1.0)
        assert isinstance(sig, Signal)
        assert len(sig.values) == len(time_array)

    def test_white_noise_mean_near_zero(self, time_array: np.ndarray) -> None:
        gen = NoiseGenerator(seed=123)
        sig = gen.generate(time_array, NoiseType.WHITE, amplitude=1.0)
        assert abs(np.mean(sig.values)) < 0.2

    def test_pink_noise(self, time_array: np.ndarray) -> None:
        gen = NoiseGenerator(seed=42)
        sig = gen.generate(time_array, NoiseType.PINK, amplitude=1.0)
        assert len(sig.values) == len(time_array)

    def test_brown_noise(self, time_array: np.ndarray) -> None:
        gen = NoiseGenerator(seed=42)
        sig = gen.generate(time_array, NoiseType.BROWN, amplitude=1.0)
        assert len(sig.values) == len(time_array)

    def test_blue_noise(self, time_array: np.ndarray) -> None:
        gen = NoiseGenerator(seed=42)
        sig = gen.generate(time_array, NoiseType.BLUE, amplitude=1.0)
        assert len(sig.values) == len(time_array)

    def test_violet_noise(self, time_array: np.ndarray) -> None:
        gen = NoiseGenerator(seed=42)
        sig = gen.generate(time_array, NoiseType.VIOLET, amplitude=1.0)
        assert len(sig.values) == len(time_array)

    def test_uniform_noise(self, time_array: np.ndarray) -> None:
        gen = NoiseGenerator(seed=42)
        sig = gen.generate(time_array, NoiseType.UNIFORM, amplitude=1.0)
        assert len(sig.values) == len(time_array)

    def test_impulse_noise_mostly_zero(self, time_array: np.ndarray) -> None:
        gen = NoiseGenerator(seed=42)
        sig = gen.generate(time_array, NoiseType.IMPULSE, amplitude=5.0, probability=0.01)
        zero_count = np.sum(np.abs(sig.values) < 1e-10)
        assert zero_count > len(time_array) * 0.9  # Most should be zero

    def test_seed_reproducibility(self, time_array: np.ndarray) -> None:
        gen1 = NoiseGenerator(seed=42)
        gen2 = NoiseGenerator(seed=42)
        sig1 = gen1.generate(time_array, NoiseType.WHITE, amplitude=1.0)
        sig2 = gen2.generate(time_array, NoiseType.WHITE, amplitude=1.0)
        np.testing.assert_array_equal(sig1.values, sig2.values)

    def test_amplitude_scaling(self, time_array: np.ndarray) -> None:
        gen = NoiseGenerator(seed=42)
        small = gen.generate(time_array, NoiseType.WHITE, amplitude=0.1)
        gen2 = NoiseGenerator(seed=42)
        large = gen2.generate(time_array, NoiseType.WHITE, amplitude=10.0)
        assert np.std(large.values) > np.std(small.values)


# ── add_noise_to_signal ────────────────────────────────────────────────


class TestAddNoiseToSignal:
    def test_noisy_signal_different(self, constant_signal: Signal) -> None:
        noisy = add_noise_to_signal(constant_signal, NoiseType.WHITE, amplitude=0.5, seed=42)
        assert not np.allclose(noisy.values, constant_signal.values)

    def test_snr_mode(self, constant_signal: Signal) -> None:
        noisy = add_noise_to_signal(constant_signal, NoiseType.WHITE, snr_db=20.0, seed=42)
        assert isinstance(noisy, Signal)
        assert len(noisy.values) == len(constant_signal.values)


# ── generate_disturbance_profile ───────────────────────────────────────


class TestDisturbanceProfile:
    def test_step_disturbance(self, time_array: np.ndarray) -> None:
        sig = generate_disturbance_profile(time_array, disturbance_type="step")
        assert isinstance(sig, Signal)
        assert len(sig.values) == len(time_array)

    def test_pulse_disturbance(self, time_array: np.ndarray) -> None:
        sig = generate_disturbance_profile(time_array, disturbance_type="pulse")
        assert isinstance(sig, Signal)


# ── DisturbanceSimulator ───────────────────────────────────────────────


class TestDisturbanceSimulator:
    def test_add_noise_chaining(self) -> None:
        sim = DisturbanceSimulator(seed=42)
        result = sim.add_noise(NoiseType.WHITE, amplitude=0.1)
        assert result is sim  # Method chaining

    def test_generate(self, time_array: np.ndarray) -> None:
        sim = DisturbanceSimulator(seed=42)
        sim.add_noise(NoiseType.WHITE, amplitude=0.1)
        sig = sim.generate(time_array)
        assert isinstance(sig, Signal)
        assert len(sig.values) == len(time_array)

    def test_apply_to_signal(self, constant_signal: Signal) -> None:
        sim = DisturbanceSimulator(seed=42)
        sim.add_noise(NoiseType.WHITE, amplitude=0.5)
        result = sim.apply_to_signal(constant_signal)
        assert isinstance(result, Signal)
        assert not np.allclose(result.values, constant_signal.values)
