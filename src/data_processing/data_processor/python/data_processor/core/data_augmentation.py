"""Data Augmentation Module.

Provides comprehensive data augmentation techniques for time series
and tabular data to improve model robustness and training.

Features:
- Noise injection (Gaussian, uniform, colored)
- Time series warping and scaling
- Synthetic data generation (SMOTE-like)
- Rotation and permutation
- Window slicing and cropping
- Magnitude warping
- Jittering and scaling
- Mix-up augmentation
- GAN-based augmentation (placeholder for neural network integration)

This module serves as a facade, composing the following submodules:
- augmentation_types: Enums and dataclasses
- augmentation_noise: Noise injection methods
- augmentation_transforms: Warping, scaling, synthetic, frequency methods
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

# Re-export all public types for backward compatibility
from data_processor.core.augmentation_noise import NoiseMixin
from data_processor.core.augmentation_transforms import TransformsMixin
from data_processor.core.augmentation_types import (
    AugmentationConfig,
    AugmentationMethod,
    AugmentationResult,
)

logger = logging.getLogger(__name__)


class DataAugmenter(NoiseMixin, TransformsMixin):
    """Comprehensive data augmentation engine.

    Provides various augmentation techniques for time series
    and tabular data.
    """

    def __init__(self, config: AugmentationConfig | None = None) -> None:
        """Initialize the augmenter.

        Args:
            config: Configuration options
        """
        self.config = config or AugmentationConfig()
        self._rng = np.random.default_rng(self.config.random_seed)

    def augment(
        self,
        data: np.ndarray,
        methods: list[AugmentationMethod] | None = None,
        n_augmentations: int = 1,
    ) -> AugmentationResult:
        """Apply augmentation methods to data.

        Args:
            data: Input data (n_samples, n_features) or (n_samples, seq_len, n_features)
            methods: List of augmentation methods to apply
            n_augmentations: Number of augmented copies per sample

        Returns:
            AugmentationResult with augmented data
        """
        if not (data is not None):
            raise ValueError("data must be provided")
        data = np.asarray(data, dtype=np.float64)
        original_shape = data.shape

        # Default methods
        if methods is None:
            methods = [AugmentationMethod.GAUSSIAN_NOISE, AugmentationMethod.SCALING]

        # Apply augmentations
        augmented_samples = []
        methods_applied = []
        params_used: dict[str, Any] = {}

        for method in methods:
            for _ in range(n_augmentations):
                aug_data = self._apply_augmentation(data, method)
                augmented_samples.append(aug_data)
                methods_applied.append(method.value)

            params_used[method.value] = self._get_method_params(method)

        # Combine original and augmented
        if len(augmented_samples) > 0:
            augmented = np.concatenate([data] + augmented_samples, axis=0)
        else:
            augmented = data.copy()

        return AugmentationResult(
            augmented_data=augmented,
            original_data=data,
            methods_applied=methods_applied,
            parameters_used=params_used,
            n_samples_original=original_shape[0],
            n_samples_augmented=augmented.shape[0],
            augmentation_factor=augmented.shape[0] / original_shape[0],
        )

    def _apply_augmentation(
        self, data: np.ndarray, method: AugmentationMethod
    ) -> np.ndarray:
        """Apply a single augmentation method."""
        if not (data is not None):
            raise ValueError("data must be provided")
        method_map = {
            AugmentationMethod.GAUSSIAN_NOISE: self.add_gaussian_noise,
            AugmentationMethod.UNIFORM_NOISE: self.add_uniform_noise,
            AugmentationMethod.COLORED_NOISE: lambda d: self.add_colored_noise(
                d, "pink"
            ),
            AugmentationMethod.SALT_PEPPER: self.add_salt_pepper_noise,
            AugmentationMethod.TIME_WARP: self.time_warp,
            AugmentationMethod.MAGNITUDE_WARP: self.magnitude_warp,
            AugmentationMethod.WINDOW_WARP: self.window_warp,
            AugmentationMethod.SCALING: self.scale,
            AugmentationMethod.ROTATION: self.rotate,
            AugmentationMethod.PERMUTATION: self.permute,
            AugmentationMethod.FLIP: self.flip,
            AugmentationMethod.WINDOW_SLICE: self.window_crop,  # Maintains size
            AugmentationMethod.WINDOW_CROP: self.window_crop,
            AugmentationMethod.SUBSAMPLE: self.subsample,
            AugmentationMethod.CUTOUT: self.cutout,
            AugmentationMethod.FREQUENCY_MASK: self.frequency_mask,
            AugmentationMethod.FREQUENCY_SHIFT: self.frequency_shift,
        }

        aug_func = method_map.get(method)
        if aug_func is None:
            logger.warning(f"Unknown augmentation method: {method}")
            return data.copy()

        return aug_func(data)

    def _get_method_params(self, method: AugmentationMethod) -> dict[str, Any]:
        """Get parameters used for a method."""
        if not (method is not None):
            raise ValueError("method must be provided")
        param_map: dict[AugmentationMethod, dict[str, Any]] = {
            AugmentationMethod.GAUSSIAN_NOISE: {"std": self.config.noise_std},
            AugmentationMethod.UNIFORM_NOISE: {"range": self.config.noise_range},
            AugmentationMethod.SALT_PEPPER: {"prob": self.config.salt_pepper_prob},
            AugmentationMethod.TIME_WARP: {
                "sigma": self.config.warp_sigma,
                "knots": self.config.warp_knots,
            },
            AugmentationMethod.MAGNITUDE_WARP: {
                "sigma": self.config.magnitude_sigma,
                "knots": self.config.warp_knots,
            },
            AugmentationMethod.SCALING: {"range": self.config.scale_range},
            AugmentationMethod.WINDOW_CROP: {"ratio": self.config.crop_ratio},
            AugmentationMethod.CUTOUT: {"ratio": self.config.cutout_ratio},
        }
        return param_map.get(method, {})


def augment_data(
    data: np.ndarray,
    methods: list[str] | None = None,
    n_augmentations: int = 1,
) -> AugmentationResult:
    """Convenience function for data augmentation.

    Args:
        data: Input data
        methods: List of method names ('gaussian_noise', 'time_warp', etc.)
        n_augmentations: Number of augmented copies

    Returns:
        AugmentationResult with augmented data

    Example:
        >>> data = np.random.randn(100, 50)  # 100 samples, 50 timesteps
        >>> result = augment_data(data, methods=['gaussian_noise', 'time_warp'])
        >>> print(f"Augmentation factor: {result.augmentation_factor:.1f}x")
    """
    if not (data is not None):
        raise ValueError("data must be provided")
    if methods is None:
        method_enums = None
    else:
        method_map = {m.value: m for m in AugmentationMethod}
        method_enums = [
            method_map.get(m, AugmentationMethod.GAUSSIAN_NOISE) for m in methods
        ]

    augmenter = DataAugmenter()
    return augmenter.augment(data, method_enums, n_augmentations)


__all__ = [
    "AugmentationMethod",
    "AugmentationConfig",
    "AugmentationResult",
    "DataAugmenter",
    "augment_data",
]
