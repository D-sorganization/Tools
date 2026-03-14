"""Noise injection augmentations.

Provides Gaussian, uniform, colored, and salt-pepper noise
augmentations as a mixin for DataAugmenter.
"""

from __future__ import annotations

import logging

import numpy as np

from data_processor.core.augmentation_types import AugmentationConfig

logger = logging.getLogger(__name__)


class NoiseMixin:
    """Noise injection methods for DataAugmenter.

    Expects the host class to provide:
    - self.config: AugmentationConfig
    - self._rng: numpy random generator
    - self._generate_colored_noise(shape, color): colored noise generation
    """

    config: AugmentationConfig
    _rng: np.random.Generator

    def add_gaussian_noise(
        self, data: np.ndarray, std: float | None = None
    ) -> np.ndarray:
        """Add Gaussian noise to data.

        Args:
            data: Input data
            std: Standard deviation of noise (relative to data std)

        Returns:
            Noisy data
        """
        assert data is not None, "data must be provided"
        std = std or self.config.noise_std
        data_std = np.std(data)
        if data_std == 0:
            data_std = 1.0  # Fallback for constant data

        noise = self._rng.normal(0, std * data_std, data.shape)
        return data + noise

    def add_uniform_noise(
        self, data: np.ndarray, range_: tuple[float, float] | None = None
    ) -> np.ndarray:
        """Add uniform noise to data.

        Args:
            data: Input data
            range_: (min, max) range for noise relative to data range

        Returns:
            Noisy data
        """
        assert data is not None, "data must be provided"
        range_ = range_ or self.config.noise_range
        data_range = np.ptp(data)
        noise = self._rng.uniform(
            range_[0] * data_range, range_[1] * data_range, data.shape
        )
        return data + noise

    def add_colored_noise(
        self, data: np.ndarray, color: str = "pink", amplitude: float = 0.1
    ) -> np.ndarray:
        """Add colored noise (pink, brown, blue) to data.

        Args:
            data: Input data
            color: Noise color ('white', 'pink', 'brown', 'blue')
            amplitude: Noise amplitude

        Returns:
            Noisy data
        """
        assert data is not None, "data must be provided"
        noise = self._generate_colored_noise(data.shape, color)
        noise = noise * amplitude * np.std(data)
        return data + noise

    def add_salt_pepper_noise(
        self, data: np.ndarray, prob: float | None = None
    ) -> np.ndarray:
        """Add salt and pepper noise.

        Args:
            data: Input data
            prob: Probability of noise

        Returns:
            Noisy data
        """
        assert data is not None, "data must be provided"
        prob = prob or self.config.salt_pepper_prob
        result = data.copy()

        # Salt
        salt_mask = self._rng.random(data.shape) < prob / 2
        result[salt_mask] = np.max(data)

        # Pepper
        pepper_mask = self._rng.random(data.shape) < prob / 2
        result[pepper_mask] = np.min(data)

        return result

    def _generate_colored_noise(self, shape: tuple[int, ...], color: str) -> np.ndarray:
        """Generate colored noise."""
        # Start with white noise
        assert shape is not None, "shape must be provided"
        white = self._rng.standard_normal(shape)

        if color == "white":
            return white

        # Apply frequency-dependent scaling
        if len(shape) == 1:
            n = shape[0]
            fft = np.fft.rfft(white)
            freqs = np.fft.rfftfreq(n)
            freqs[0] = 1  # Avoid division by zero

            if color == "pink":
                fft = fft / np.sqrt(freqs)
            elif color == "brown":
                fft = fft / freqs
            elif color == "blue":
                fft = fft * np.sqrt(freqs)

            return np.fft.irfft(fft, n)
        else:
            result = np.zeros(shape)
            for idx in np.ndindex(shape[:-1] if len(shape) > 1 else ()):
                result[idx] = self._generate_colored_noise((shape[-1],), color)
            return result
