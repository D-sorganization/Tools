"""Data augmentation type definitions.

Shared enums, dataclasses, and configuration for the data augmentation module.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class AugmentationMethod(Enum):
    """Available augmentation methods."""

    # Noise-based
    GAUSSIAN_NOISE = "gaussian_noise"
    UNIFORM_NOISE = "uniform_noise"
    COLORED_NOISE = "colored_noise"
    SALT_PEPPER = "salt_pepper"

    # Time warping
    TIME_WARP = "time_warp"
    MAGNITUDE_WARP = "magnitude_warp"
    WINDOW_WARP = "window_warp"

    # Transformation
    SCALING = "scaling"
    ROTATION = "rotation"
    PERMUTATION = "permutation"
    FLIP = "flip"

    # Sampling
    WINDOW_SLICE = "window_slice"
    WINDOW_CROP = "window_crop"
    SUBSAMPLE = "subsample"

    # Synthetic
    SMOTE = "smote"
    MIXUP = "mixup"
    CUTOUT = "cutout"
    CUTMIX = "cutmix"

    # Frequency domain
    FREQUENCY_MASK = "frequency_mask"
    FREQUENCY_SHIFT = "frequency_shift"


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""

    # General
    random_seed: int | None = None

    # Noise parameters
    noise_std: float = 0.05
    noise_range: tuple[float, float] = (-0.1, 0.1)
    salt_pepper_prob: float = 0.02

    # Warping parameters
    warp_knots: int = 4
    warp_sigma: float = 0.2
    magnitude_sigma: float = 0.2

    # Scaling parameters
    scale_range: tuple[float, float] = (0.8, 1.2)

    # Window parameters
    window_ratio: float = 0.9
    crop_ratio: float = 0.8

    # SMOTE parameters
    smote_k_neighbors: int = 5

    # Mixup parameters
    mixup_alpha: float = 0.2

    # Cutout/Cutmix parameters
    cutout_ratio: float = 0.1


@dataclass
class AugmentationResult:
    """Result of augmentation operation."""

    # Augmented data
    augmented_data: np.ndarray
    original_data: np.ndarray

    # Applied augmentations
    methods_applied: list[str]
    parameters_used: dict[str, Any]

    # Statistics
    n_samples_original: int
    n_samples_augmented: int
    augmentation_factor: float
