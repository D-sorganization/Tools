"""Data transformation augmentations.

Provides warping, scaling, rotation, permutation, windowing,
synthetic data generation (SMOTE, mixup, cutout, cutmix),
and frequency domain augmentations as a mixin for DataAugmenter.
"""

from __future__ import annotations

import logging

import numpy as np

from data_processor.core.augmentation_types import AugmentationConfig

logger = logging.getLogger(__name__)


class TransformsMixin:
    """Transformation augmentation methods for DataAugmenter.

    Expects the host class to provide:
    - self.config: AugmentationConfig
    - self._rng: numpy random generator
    """

    config: AugmentationConfig
    _rng: np.random.Generator

    # =========================================================================
    # Time warping augmentations
    # =========================================================================

    def time_warp(
        self, data: np.ndarray, sigma: float | None = None, knots: int | None = None
    ) -> np.ndarray:
        """Apply time warping using smooth random curves.

        Args:
            data: Input time series (n_samples, seq_len) or
                (n_samples, seq_len, n_features)
            sigma: Warping strength
            knots: Number of warping knots

        Returns:
            Time-warped data
        """
        sigma = sigma or self.config.warp_sigma
        knots = knots or self.config.warp_knots

        if data.ndim == 1:
            return self._time_warp_1d(data, sigma, knots)
        elif data.ndim == 2:
            result = np.zeros_like(data)
            for i in range(data.shape[0]):
                result[i] = self._time_warp_1d(data[i], sigma, knots)
            return result
        else:
            result = np.zeros_like(data)
            for i in range(data.shape[0]):
                for j in range(data.shape[2]):
                    result[i, :, j] = self._time_warp_1d(data[i, :, j], sigma, knots)
            return result

    def magnitude_warp(
        self, data: np.ndarray, sigma: float | None = None, knots: int | None = None
    ) -> np.ndarray:
        """Apply magnitude warping using smooth random curves.

        Args:
            data: Input data
            sigma: Warping strength
            knots: Number of warping knots

        Returns:
            Magnitude-warped data
        """
        sigma = sigma or self.config.magnitude_sigma
        knots = knots or self.config.warp_knots

        if data.ndim == 1:
            warp_factors = self._generate_warp_curve(len(data), sigma, knots)
            return data * warp_factors
        else:
            result = np.zeros_like(data)
            for i in range(data.shape[0]):
                if data.ndim == 2:
                    warp_factors = self._generate_warp_curve(
                        data.shape[1], sigma, knots
                    )
                    result[i] = data[i] * warp_factors
                else:
                    warp_factors = self._generate_warp_curve(
                        data.shape[1], sigma, knots
                    )
                    result[i] = data[i] * warp_factors[:, np.newaxis]
            return result

    def window_warp(
        self, data: np.ndarray, ratio: float = 0.1, scales: list[float] | None = None
    ) -> np.ndarray:
        """Apply window warping (stretch/compress random windows).

        Args:
            data: Input data
            ratio: Window size ratio
            scales: Scale factors to choose from

        Returns:
            Window-warped data
        """
        if scales is None:
            scales = [0.5, 2.0]

        if data.ndim == 1:
            return self._window_warp_1d(data, ratio, scales)
        else:
            result = np.zeros_like(data)
            for i in range(data.shape[0]):
                if data.ndim == 2:
                    result[i] = self._window_warp_1d(data[i], ratio, scales)
                else:
                    for j in range(data.shape[2]):
                        result[i, :, j] = self._window_warp_1d(
                            data[i, :, j], ratio, scales
                        )
            return result

    # =========================================================================
    # Transformation augmentations
    # =========================================================================

    def scale(
        self, data: np.ndarray, range_: tuple[float, float] | None = None
    ) -> np.ndarray:
        """Apply random scaling.

        Args:
            data: Input data
            range_: (min, max) scaling factor range

        Returns:
            Scaled data
        """
        range_ = range_ or self.config.scale_range
        scale_factor = self._rng.uniform(range_[0], range_[1])
        return data * scale_factor

    def rotate(self, data: np.ndarray, max_angle: float = np.pi / 6) -> np.ndarray:
        """Apply rotation to multivariate data.

        For 2D data, applies rotation matrix.

        Args:
            data: Input data (n_samples, n_features) where n_features >= 2
            max_angle: Maximum rotation angle in radians

        Returns:
            Rotated data
        """
        if data.ndim == 1 or (data.ndim == 2 and data.shape[1] < 2):
            return data.copy()

        # Random rotation angle
        angle = self._rng.uniform(-max_angle, max_angle)

        # Apply to first two dimensions
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        result = data.copy()

        if data.ndim == 2:
            # (n_samples, n_features)
            result[:, 0] = cos_a * data[:, 0] - sin_a * data[:, 1]
            result[:, 1] = sin_a * data[:, 0] + cos_a * data[:, 1]
        else:
            # (n_samples, seq_len, n_features)
            result[:, :, 0] = cos_a * data[:, :, 0] - sin_a * data[:, :, 1]
            result[:, :, 1] = sin_a * data[:, :, 0] + cos_a * data[:, :, 1]

        return result

    def permute(self, data: np.ndarray, max_segments: int = 5) -> np.ndarray:
        """Randomly permute segments of the data.

        Args:
            data: Input data
            max_segments: Maximum number of segments

        Returns:
            Permuted data
        """
        if data.ndim == 1:
            return self._permute_1d(data, max_segments)
        else:
            result = np.zeros_like(data)
            for i in range(data.shape[0]):
                if data.ndim == 2:
                    result[i] = self._permute_1d(data[i], max_segments)
                else:
                    # Permute along time axis
                    n_segments = self._rng.integers(2, max_segments + 1)
                    segment_size = data.shape[1] // n_segments
                    perm = self._rng.permutation(n_segments)
                    for j, p in enumerate(perm):
                        start_src = p * segment_size
                        start_dst = j * segment_size
                        end = min(start_src + segment_size, data.shape[1])
                        result[i, start_dst : start_dst + (end - start_src)] = data[
                            i, start_src:end
                        ]
            return result

    def flip(self, data: np.ndarray, axis: int = -1) -> np.ndarray:
        """Flip data along specified axis.

        Args:
            data: Input data
            axis: Axis to flip along

        Returns:
            Flipped data
        """
        return np.flip(data, axis=axis)

    # =========================================================================
    # Sampling augmentations
    # =========================================================================

    def window_slice(self, data: np.ndarray, ratio: float | None = None) -> np.ndarray:
        """Extract random window slices.

        Args:
            data: Input data (seq_len,) or (n_samples, seq_len, ...)
            ratio: Window size ratio

        Returns:
            Sliced data (may have different length)
        """
        ratio = ratio or self.config.window_ratio

        if data.ndim == 1:
            length = len(data)
            window_size = int(length * ratio)
            start = self._rng.integers(0, length - window_size + 1)
            return data[start : start + window_size]
        else:
            seq_len = data.shape[1]
            window_size = int(seq_len * ratio)
            start = self._rng.integers(0, seq_len - window_size + 1)
            return data[:, start : start + window_size]

    def window_crop(self, data: np.ndarray, ratio: float | None = None) -> np.ndarray:
        """Random crop and resize to original size.

        Args:
            data: Input data
            ratio: Crop ratio

        Returns:
            Cropped and resized data
        """
        ratio = ratio or self.config.crop_ratio

        if data.ndim == 1:
            sliced = self.window_slice(data, ratio)
            # Interpolate back to original size
            return self._interpolate(sliced, len(data))
        else:
            sliced = self.window_slice(data, ratio)
            # Interpolate each sample
            result = np.zeros_like(data)
            for i in range(data.shape[0]):
                if data.ndim == 2:
                    result[i] = self._interpolate(sliced[i], data.shape[1])
                else:
                    for j in range(data.shape[2]):
                        result[i, :, j] = self._interpolate(
                            sliced[i, :, j], data.shape[1]
                        )
            return result

    def subsample(self, data: np.ndarray, keep_ratio: float = 0.5) -> np.ndarray:
        """Randomly subsample data points.

        Args:
            data: Input data
            keep_ratio: Ratio of points to keep

        Returns:
            Subsampled and interpolated data
        """
        if data.ndim == 1:
            n = len(data)
            n_keep = max(2, int(n * keep_ratio))
            indices = np.sort(self._rng.choice(n, n_keep, replace=False))
            subsampled = data[indices]
            return self._interpolate(subsampled, n)
        else:
            result = np.zeros_like(data)
            for i in range(data.shape[0]):
                if data.ndim == 2:
                    n = data.shape[1]
                    n_keep = max(2, int(n * keep_ratio))
                    indices = np.sort(self._rng.choice(n, n_keep, replace=False))
                    subsampled = data[i, indices]
                    result[i] = self._interpolate(subsampled, n)
                else:
                    for j in range(data.shape[2]):
                        n = data.shape[1]
                        n_keep = max(2, int(n * keep_ratio))
                        indices = np.sort(self._rng.choice(n, n_keep, replace=False))
                        subsampled = data[i, indices, j]
                        result[i, :, j] = self._interpolate(subsampled, n)
            return result

    # =========================================================================
    # Synthetic data augmentations
    # =========================================================================

    def smote(
        self,
        data: np.ndarray,
        labels: np.ndarray | None = None,
        k_neighbors: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Generate synthetic samples using SMOTE-like method.

        Args:
            data: Input data (n_samples, n_features)
            labels: Optional labels for stratified augmentation
            k_neighbors: Number of neighbors to use

        Returns:
            Tuple of (augmented_data, augmented_labels)
        """
        k_neighbors = k_neighbors or self.config.smote_k_neighbors
        data = np.atleast_2d(data)
        n_samples = data.shape[0]

        if n_samples < 2:
            return data, labels

        if n_samples < k_neighbors + 1:
            k_neighbors = max(1, n_samples - 1)

        # Find k nearest neighbors for each sample
        synthetic_samples = []
        synthetic_labels = []

        for i in range(n_samples):
            # Compute distances
            distances = np.linalg.norm(data - data[i], axis=1)
            neighbor_indices = np.argsort(distances)[1 : k_neighbors + 1]

            # Generate synthetic sample
            nn_idx = self._rng.choice(neighbor_indices)
            diff = data[nn_idx] - data[i]
            gap = self._rng.random()

            synthetic = data[i] + gap * diff
            synthetic_samples.append(synthetic)

            if labels is not None:
                synthetic_labels.append(labels[i])

        augmented_data = np.vstack([data, synthetic_samples])

        if labels is not None:
            augmented_labels = np.concatenate([labels, synthetic_labels])
            return augmented_data, augmented_labels

        return augmented_data, None

    def mixup(
        self,
        data: np.ndarray,
        labels: np.ndarray | None = None,
        alpha: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Apply mixup augmentation.

        Args:
            data: Input data
            labels: Optional labels (will be mixed too)
            alpha: Beta distribution parameter

        Returns:
            Tuple of (mixed_data, mixed_labels)
        """
        alpha = alpha or self.config.mixup_alpha
        n_samples = data.shape[0]

        # Generate mixing coefficients from Beta distribution
        lambdas = self._rng.beta(alpha, alpha, n_samples)

        # Random permutation for mixing pairs
        indices = self._rng.permutation(n_samples)

        # Mix data
        mixed_data = np.zeros_like(data)
        for i in range(n_samples):
            lam = lambdas[i]
            mixed_data[i] = lam * data[i] + (1 - lam) * data[indices[i]]

        # Mix labels if provided
        mixed_labels = None
        if labels is not None:
            labels = np.asarray(labels)
            if labels.ndim == 1:
                # Convert to one-hot if needed
                unique_labels = np.unique(labels)
                n_classes = len(unique_labels)
                label_map = {lbl: i for i, lbl in enumerate(unique_labels)}

                one_hot = np.zeros((n_samples, n_classes))
                for i, lbl in enumerate(labels):
                    one_hot[i, label_map[lbl]] = 1.0

                mixed_labels = np.zeros((n_samples, n_classes))
                for i in range(n_samples):
                    mixed_labels[i] = (
                        lambdas[i] * one_hot[i] + (1 - lambdas[i]) * one_hot[indices[i]]
                    )
            else:
                mixed_labels = np.zeros_like(labels)
                for i in range(n_samples):
                    mixed_labels[i] = (
                        lambdas[i] * labels[i] + (1 - lambdas[i]) * labels[indices[i]]
                    )

        return mixed_data, mixed_labels

    def cutout(self, data: np.ndarray, ratio: float | None = None) -> np.ndarray:
        """Apply cutout augmentation (mask random regions).

        Args:
            data: Input data
            ratio: Size ratio of region to mask

        Returns:
            Data with masked regions
        """
        ratio = ratio or self.config.cutout_ratio
        result = data.copy()

        if data.ndim == 1:
            length = len(data)
            mask_size = int(length * ratio)
            start = self._rng.integers(0, length - mask_size + 1)
            result[start : start + mask_size] = 0
        else:
            for i in range(data.shape[0]):
                if data.ndim == 2:
                    length = data.shape[1]
                    mask_size = int(length * ratio)
                    start = self._rng.integers(0, length - mask_size + 1)
                    result[i, start : start + mask_size] = 0
                else:
                    seq_len = data.shape[1]
                    mask_size = int(seq_len * ratio)
                    start = self._rng.integers(0, seq_len - mask_size + 1)
                    result[i, start : start + mask_size, :] = 0

        return result

    def cutmix(
        self,
        data: np.ndarray,
        labels: np.ndarray | None = None,
        ratio: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Apply CutMix augmentation.

        Args:
            data: Input data
            labels: Optional labels
            ratio: Size ratio of region to mix

        Returns:
            Tuple of (mixed_data, mixed_labels)
        """
        ratio = ratio or self.config.cutout_ratio
        n_samples = data.shape[0]

        # Random permutation for mixing pairs
        indices = self._rng.permutation(n_samples)

        result = data.copy()

        for i in range(n_samples):
            if data.ndim == 1:
                # CutMix requires at least 2 dimensions (batch, feature)
                # Fallback to no-op for 1D arrays to prevent crash
                continue
            elif data.ndim == 2:
                length = data.shape[1]
                mask_size = int(length * ratio)
                start = self._rng.integers(0, length - mask_size + 1)
                result[i, start : start + mask_size] = data[
                    indices[i], start : start + mask_size
                ]
            else:
                seq_len = data.shape[1]
                mask_size = int(seq_len * ratio)
                start = self._rng.integers(0, seq_len - mask_size + 1)
                result[i, start : start + mask_size, :] = data[
                    indices[i], start : start + mask_size, :
                ]

        # Mix labels proportionally
        mixed_labels = None
        if labels is not None:
            lam = 1 - ratio  # Proportion of original sample
            labels = np.asarray(labels)
            if labels.ndim == 1:
                unique_labels = np.unique(labels)
                n_classes = len(unique_labels)
                label_map = {lbl: i for i, lbl in enumerate(unique_labels)}

                one_hot = np.zeros((n_samples, n_classes))
                for i, lbl in enumerate(labels):
                    one_hot[i, label_map[lbl]] = 1.0

                mixed_labels = np.zeros((n_samples, n_classes))
                for i in range(n_samples):
                    mixed_labels[i] = lam * one_hot[i] + (1 - lam) * one_hot[indices[i]]
            else:
                mixed_labels = lam * labels + (1 - lam) * labels[indices]

        return result, mixed_labels

    # =========================================================================
    # Frequency domain augmentations
    # =========================================================================

    def frequency_mask(
        self, data: np.ndarray, mask_ratio: float = 0.1, num_masks: int = 1
    ) -> np.ndarray:
        """Mask random frequency bands.

        Args:
            data: Input data
            mask_ratio: Ratio of frequencies to mask
            num_masks: Number of frequency masks

        Returns:
            Frequency-masked data
        """
        if data.ndim == 1:
            return self._frequency_mask_1d(data, mask_ratio, num_masks)
        else:
            result = np.zeros_like(data)
            for i in range(data.shape[0]):
                if data.ndim == 2:
                    result[i] = self._frequency_mask_1d(data[i], mask_ratio, num_masks)
                else:
                    for j in range(data.shape[2]):
                        result[i, :, j] = self._frequency_mask_1d(
                            data[i, :, j], mask_ratio, num_masks
                        )
            return result

    def frequency_shift(
        self, data: np.ndarray, max_shift_ratio: float = 0.1
    ) -> np.ndarray:
        """Shift frequencies randomly.

        Args:
            data: Input data
            max_shift_ratio: Maximum shift as ratio of frequency range

        Returns:
            Frequency-shifted data
        """
        if data.ndim == 1:
            return self._frequency_shift_1d(data, max_shift_ratio)
        else:
            result = np.zeros_like(data)
            for i in range(data.shape[0]):
                if data.ndim == 2:
                    result[i] = self._frequency_shift_1d(data[i], max_shift_ratio)
                else:
                    for j in range(data.shape[2]):
                        result[i, :, j] = self._frequency_shift_1d(
                            data[i, :, j], max_shift_ratio
                        )
            return result

    # =========================================================================
    # Private helper methods
    # =========================================================================

    def _time_warp_1d(self, data: np.ndarray, sigma: float, knots: int) -> np.ndarray:
        """Apply time warping to 1D data."""
        n = len(data)

        # Generate random warp path
        warp_curve = self._generate_warp_curve(n, sigma, knots)

        # Cumulative sum to get time indices
        time_steps = np.cumsum(warp_curve)
        time_steps = time_steps / time_steps[-1] * (n - 1)

        # Interpolate
        original_indices = np.arange(n)
        return np.interp(original_indices, time_steps, data)

    def _generate_warp_curve(self, length: int, sigma: float, knots: int) -> np.ndarray:
        """Generate smooth random warp curve."""
        knot_positions = np.linspace(0, length - 1, knots + 2)
        knot_values = self._rng.normal(1.0, sigma, knots + 2)
        knot_values = np.maximum(knot_values, 0.1)  # Ensure positive

        # Interpolate to full length
        return np.interp(np.arange(length), knot_positions, knot_values)

    def _window_warp_1d(
        self, data: np.ndarray, ratio: float, scales: list[float]
    ) -> np.ndarray:
        """Apply window warping to 1D data."""
        n = len(data)
        window_size = int(n * ratio)
        start = self._rng.integers(0, n - window_size + 1)
        end = start + window_size

        scale = self._rng.choice(scales)

        # Extract window and scale
        window = data[start:end]
        new_size = int(window_size * scale)

        if new_size < 2:
            return data.copy()

        # Interpolate window
        scaled_window = self._interpolate(window, new_size)

        # Reconstruct
        before = data[:start]
        after = data[end:]

        # Interpolate to original length
        combined = np.concatenate([before, scaled_window, after])
        return self._interpolate(combined, n)

    def _permute_1d(self, data: np.ndarray, max_segments: int) -> np.ndarray:
        """Permute segments of 1D data."""
        n = len(data)
        n_segments = self._rng.integers(2, min(max_segments + 1, n // 2 + 1))
        segment_size = n // n_segments

        result = np.zeros_like(data)
        perm = self._rng.permutation(n_segments)

        for i, p in enumerate(perm):
            start_src = p * segment_size
            start_dst = i * segment_size
            end = min(start_src + segment_size, n)
            length = end - start_src
            result[start_dst : start_dst + length] = data[start_src:end]

        # Handle remainder
        remainder_start = n_segments * segment_size
        if remainder_start < n:
            result[remainder_start:] = data[remainder_start:]

        return result

    def _interpolate(self, data: np.ndarray, target_length: int) -> np.ndarray:
        """Interpolate data to target length."""
        n = len(data)
        if n == target_length:
            return data.copy()

        x_original = np.linspace(0, 1, n)
        x_target = np.linspace(0, 1, target_length)

        return np.interp(x_target, x_original, data)

    def _frequency_mask_1d(
        self, data: np.ndarray, mask_ratio: float, num_masks: int
    ) -> np.ndarray:
        """Apply frequency masking to 1D data."""
        n = len(data)
        fft = np.fft.rfft(data)
        n_freq = len(fft)
        mask_size = int(n_freq * mask_ratio)

        for _ in range(num_masks):
            start = self._rng.integers(0, n_freq - mask_size + 1)
            fft[start : start + mask_size] = 0

        return np.fft.irfft(fft, n)

    def _frequency_shift_1d(
        self, data: np.ndarray, max_shift_ratio: float
    ) -> np.ndarray:
        """Apply frequency shift to 1D data."""
        n = len(data)
        fft = np.fft.rfft(data)
        n_freq = len(fft)

        shift = self._rng.integers(
            -int(n_freq * max_shift_ratio), int(n_freq * max_shift_ratio) + 1
        )

        shifted_fft = np.roll(fft, shift)
        if shift > 0:
            shifted_fft[:shift] = 0
        elif shift < 0:
            shifted_fft[shift:] = 0

        return np.fft.irfft(shifted_fft, n)
