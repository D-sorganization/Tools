"""Tests for data_processor.core.augmentation_types module."""

from __future__ import annotations

import numpy as np
import pytest
from data_processor.core.augmentation_types import (
    AugmentationConfig,
    AugmentationMethod,
    AugmentationResult,
)


class TestAugmentationMethod:
    """Tests for AugmentationMethod enum."""

    def test_noise_methods(self) -> None:
        assert AugmentationMethod.GAUSSIAN_NOISE.value == "gaussian_noise"
        assert AugmentationMethod.UNIFORM_NOISE.value == "uniform_noise"
        assert AugmentationMethod.COLORED_NOISE.value == "colored_noise"
        assert AugmentationMethod.SALT_PEPPER.value == "salt_pepper"

    def test_warp_methods(self) -> None:
        assert AugmentationMethod.TIME_WARP.value == "time_warp"
        assert AugmentationMethod.MAGNITUDE_WARP.value == "magnitude_warp"
        assert AugmentationMethod.WINDOW_WARP.value == "window_warp"

    def test_transform_methods(self) -> None:
        assert AugmentationMethod.SCALING.value == "scaling"
        assert AugmentationMethod.ROTATION.value == "rotation"
        assert AugmentationMethod.PERMUTATION.value == "permutation"
        assert AugmentationMethod.FLIP.value == "flip"

    def test_sampling_methods(self) -> None:
        assert AugmentationMethod.WINDOW_SLICE.value == "window_slice"
        assert AugmentationMethod.WINDOW_CROP.value == "window_crop"
        assert AugmentationMethod.SUBSAMPLE.value == "subsample"

    def test_synthetic_methods(self) -> None:
        assert AugmentationMethod.SMOTE.value == "smote"
        assert AugmentationMethod.MIXUP.value == "mixup"
        assert AugmentationMethod.CUTOUT.value == "cutout"
        assert AugmentationMethod.CUTMIX.value == "cutmix"

    def test_frequency_methods(self) -> None:
        assert AugmentationMethod.FREQUENCY_MASK.value == "frequency_mask"
        assert AugmentationMethod.FREQUENCY_SHIFT.value == "frequency_shift"

    def test_total_member_count(self) -> None:
        assert len(AugmentationMethod) == 20

    def test_from_value(self) -> None:
        assert AugmentationMethod("gaussian_noise") == AugmentationMethod.GAUSSIAN_NOISE


class TestAugmentationConfig:
    """Tests for AugmentationConfig dataclass."""

    def test_defaults(self) -> None:
        config = AugmentationConfig()
        assert config.random_seed is None
        assert config.noise_std == 0.05
        assert config.salt_pepper_prob == 0.02
        assert config.warp_knots == 4
        assert config.warp_sigma == 0.2
        assert config.magnitude_sigma == 0.2
        assert config.window_ratio == 0.9
        assert config.crop_ratio == 0.8
        assert config.smote_k_neighbors == 5
        assert config.mixup_alpha == 0.2
        assert config.cutout_ratio == 0.1

    def test_noise_range_default(self) -> None:
        config = AugmentationConfig()
        assert config.noise_range == (-0.1, 0.1)

    def test_scale_range_default(self) -> None:
        config = AugmentationConfig()
        assert config.scale_range == (0.8, 1.2)

    def test_custom_values(self) -> None:
        config = AugmentationConfig(
            random_seed=42,
            noise_std=0.1,
            salt_pepper_prob=0.05,
            warp_knots=8,
        )
        assert config.random_seed == 42
        assert config.noise_std == 0.1
        assert config.salt_pepper_prob == 0.05
        assert config.warp_knots == 8

    def test_noise_range_bounds(self) -> None:
        config = AugmentationConfig()
        low, high = config.noise_range
        assert low < high


class TestAugmentationResult:
    """Tests for AugmentationResult dataclass."""

    def test_construction(self) -> None:
        original = np.array([1, 2, 3])
        augmented = np.array([1.1, 2.1, 3.1, 4.0, 5.0])
        result = AugmentationResult(
            augmented_data=augmented,
            original_data=original,
            methods_applied=["gaussian_noise"],
            parameters_used={"noise_std": 0.1},
            n_samples_original=3,
            n_samples_augmented=5,
            augmentation_factor=5 / 3,
        )
        assert len(result.augmented_data) == 5
        assert len(result.original_data) == 3
        assert result.methods_applied == ["gaussian_noise"]
        assert result.n_samples_original == 3
        assert result.n_samples_augmented == 5
        assert result.augmentation_factor == pytest.approx(5 / 3)

    def test_multiple_methods(self) -> None:
        result = AugmentationResult(
            augmented_data=np.zeros(10),
            original_data=np.zeros(5),
            methods_applied=["gaussian_noise", "time_warp"],
            parameters_used={"noise_std": 0.1, "warp_sigma": 0.2},
            n_samples_original=5,
            n_samples_augmented=10,
            augmentation_factor=2.0,
        )
        assert len(result.methods_applied) == 2
