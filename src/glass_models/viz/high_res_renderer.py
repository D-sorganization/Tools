"""High-resolution rendering with anti-aliasing support for publication-quality output.

This module provides the HighResolutionRenderer class for off-screen rendering
at multiple resolutions (1080p, 2K, 4K, 8K) with anti-aliasing levels and
DPI metadata embedding in PNG files.

Design by Contract:
    - Resolution must be one of: "1080p", "2K", "4K", "8K", or "custom"
    - Custom resolution requires width and height > 0
    - AA level must be one of: 1, 2, 4, 8
    - DPI must be between 72 and 600 (printing standards)
    - Output format must be "PNG", "JPG", or "Both"

Example:
    >>> renderer = HighResolutionRenderer(aa_level=4, dpi=300)
    >>> renderer.render_to_image(
    ...     resolution="4K",
    ...     output_path="output.png",
    ...     format="PNG"
    ... )
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class HighResolutionRenderer:
    """Renders scenes at high resolution with anti-aliasing.

    Attributes:
        dpi (int): Dots per inch for printing (default: 72)
        aa_level (int): Anti-aliasing level - 1, 2, 4, or 8 (default: 1)
    """

    # Standard resolution mappings (width, height)
    STANDARD_RESOLUTIONS = {
        "1080p": (1920, 1080),
        "2K": (2560, 1440),
        "4K": (3840, 2160),
        "8K": (7680, 4320),
    }

    VALID_FORMATS = ("PNG", "JPG", "Both")
    VALID_AA_LEVELS = (1, 2, 4, 8)
    MIN_DPI = 72
    MAX_DPI = 600

    def __init__(
        self,
        aa_level: int = 1,
        dpi: int = 72,
    ) -> None:
        """Initialize the high-resolution renderer.

        Args:
            aa_level: Anti-aliasing level (1, 2, 4, or 8)
            dpi: Dots per inch for metadata (72-600)

        Raises:
            ValueError: If aa_level or dpi are invalid
            ImportError: If PyVista is not available
        """
        if aa_level not in self.VALID_AA_LEVELS:
            raise ValueError(f"AA level must be 1, 2, 4, or 8, got {aa_level}")
        if not (self.MIN_DPI <= dpi <= self.MAX_DPI):
            raise ValueError(
                f"DPI must be between {self.MIN_DPI} and {self.MAX_DPI}, got {dpi}"
            )

        self.aa_level = aa_level
        self.dpi = dpi

        # Import visualization libraries with fallback
        try:
            import pyvista as pv

            self._pv = pv
        except ImportError as e:
            logger.error("PyVista not available: %s", e)
            raise ImportError(
                "PyVista is required for high-resolution rendering. "
                "Install with: pip install pyvista"
            ) from e

        logger.debug(
            "HighResolutionRenderer initialized: aa_level=%d, dpi=%d",
            aa_level,
            dpi,
        )

    @staticmethod
    def _get_resolution_dimensions(resolution: str) -> tuple[int, int]:
        """Get (width, height) for a standard resolution name.

        Args:
            resolution: One of "1080p", "2K", "4K", "8K"

        Returns:
            Tuple of (width, height) in pixels

        Raises:
            ValueError: If resolution is not recognized
        """
        if resolution not in HighResolutionRenderer.STANDARD_RESOLUTIONS:
            valid_resolutions = list(HighResolutionRenderer.STANDARD_RESOLUTIONS.keys())
            raise ValueError(
                f"Unknown resolution: {resolution}. Must be one of: {valid_resolutions}"
            )
        return HighResolutionRenderer.STANDARD_RESOLUTIONS[resolution]

    def _render_offscreen(
        self,
        width: int,
        height: int,
    ) -> np.ndarray:
        """Render scene off-screen and return pixel array.

        This is a placeholder for actual PyVista rendering.
        In production, this would use PyVista's Plotter.screenshot()

        Args:
            width: Image width in pixels
            height: Image height in pixels

        Returns:
            RGBA pixel array with shape (height, width, 4)

        Raises:
            ValueError: If width or height are <= 0
        """
        if width <= 0 or height <= 0:
            raise ValueError(f"Width and height must be positive, got {width}x{height}")

        # Apply anti-aliasing by rendering at higher resolution and downsampling
        render_width = width * self.aa_level
        render_height = height * self.aa_level

        logger.debug(
            "Rendering off-screen: %dx%d (AA %dx)",
            render_width,
            render_height,
            self.aa_level,
        )

        # Create a plotter with off-screen rendering enabled
        plotter = self._pv.Plotter(
            off_screen=True,
            window_size=(render_width, render_height),
        )

        # Render and get screenshot
        try:
            screenshot = plotter.screenshot(
                filename=None,
                return_img=True,
            )
            # Convert to numpy array if needed
            if not isinstance(screenshot, np.ndarray):
                screenshot = np.asarray(screenshot, dtype=np.uint8)

            # Apply anti-aliasing downsampling if AA level > 1
            if self.aa_level > 1:
                screenshot_aa = self._apply_antialiasing(
                    screenshot,
                    self.aa_level,
                )
                return screenshot_aa
            else:
                return np.asarray(screenshot, dtype=np.uint8)
        finally:
            plotter.close()

    @staticmethod
    def _apply_antialiasing(
        image: np.ndarray,
        aa_level: int,
    ) -> np.ndarray:
        """Apply anti-aliasing by downsampling using local averaging.

        Args:
            image: Input image array
            aa_level: Downsampling factor (2, 4, or 8)

        Returns:
            Downsampled image array with anti-aliasing applied
        """
        if aa_level == 1:
            return image

        # Use scipy for high-quality downsampling
        try:
            from scipy import ndimage

            # Use uniform filter for local averaging (simple AA)
            # This is equivalent to box filtering
            result = ndimage.uniform_filter(image, size=aa_level)
            # Downsample
            return np.asarray(result[::aa_level, ::aa_level], dtype=np.uint8)
        except ImportError:
            # Fallback: simple strided downsampling without filtering
            logger.warning("scipy not available, using simple downsampling")
            return image[::aa_level, ::aa_level]

    def _save_png(
        self,
        image: np.ndarray,
        output_path: str,
    ) -> bool:
        """Save image as PNG with DPI metadata.

        Args:
            image: RGBA or RGB pixel array
            output_path: Output file path

        Returns:
            True if successful, False otherwise

        Raises:
            ValueError: If output_path is invalid
        """
        if not output_path:
            raise ValueError("output_path cannot be empty")

        try:
            from PIL import Image

            # Convert numpy array to PIL Image
            if image.shape[2] == 4:
                # RGBA image
                pil_image = Image.fromarray(image, mode="RGBA")
            elif image.shape[2] == 3:
                # RGB image
                pil_image = Image.fromarray(image, mode="RGB")
            else:
                msg = (
                    f"Unexpected image shape: {image.shape}. "
                    "Expected (H, W, 3) or (H, W, 4)"
                )
                raise ValueError(msg)

            # Set DPI info for printing
            pil_image.save(
                output_path,
                format="PNG",
                dpi=(self.dpi, self.dpi),
            )

            logger.info("Saved PNG: %s (DPI: %d)", output_path, self.dpi)
            return True
        except ImportError as e:
            logger.error("PIL not available: %s", e)
            return False
        except Exception as e:
            logger.error("Failed to save PNG: %s", e)
            return False

    def _save_jpg(
        self,
        image: np.ndarray,
        output_path: str,
    ) -> bool:
        """Save image as JPG with quality settings.

        Args:
            image: RGB pixel array (alpha channel discarded)
            output_path: Output file path

        Returns:
            True if successful, False otherwise

        Raises:
            ValueError: If output_path is invalid
        """
        if not output_path:
            raise ValueError("output_path cannot be empty")

        try:
            from PIL import Image

            # Convert numpy array to PIL Image
            if image.shape[2] == 4:
                # Discard alpha channel for JPG
                image_rgb = image[:, :, :3]
            else:
                image_rgb = image

            pil_image = Image.fromarray(image_rgb, mode="RGB")

            # Save with high quality
            pil_image.save(
                output_path,
                format="JPEG",
                quality=95,
                dpi=(self.dpi, self.dpi),
            )

            logger.info("Saved JPG: %s", output_path)
            return True
        except ImportError as e:
            logger.error("PIL not available: %s", e)
            return False
        except Exception as e:
            logger.error("Failed to save JPG: %s", e)
            return False

    def _validate_render_inputs(
        self,
        output_path: str,
        format: str,
    ) -> None:
        """Validate render function inputs.

        Args:
            output_path: Output file path
            format: Output format

        Raises:
            TypeError: If output_path is not a string
            ValueError: If format is invalid
        """
        if not isinstance(output_path, str):
            raise TypeError(f"output_path must be str, got {type(output_path)}")
        if format not in self.VALID_FORMATS:
            raise ValueError(
                f"Format must be {', '.join(self.VALID_FORMATS)}, got {format}"
            )

    def _get_render_dimensions(
        self,
        resolution: str,
        width: int | None,
        height: int | None,
    ) -> tuple[int, int]:
        """Get rendering dimensions for given resolution.

        Args:
            resolution: Resolution name or "custom"
            width: Custom width (for custom resolution)
            height: Custom height (for custom resolution)

        Returns:
            Tuple of (width, height) in pixels

        Raises:
            ValueError: If resolution is invalid or dimensions are invalid
        """
        if resolution == "custom":
            if width is None or height is None:
                raise ValueError(
                    "Custom resolution requires width and height parameters"
                )
            if width <= 0 or height <= 0:
                raise ValueError(
                    f"Width and height must be positive, got {width}x{height}"
                )
            return width, height

        return self._get_resolution_dimensions(resolution)

    def _save_formats(
        self,
        image_array: np.ndarray,
        output_path: str,
        format: str,
    ) -> bool:
        """Save image in requested format(s).

        Args:
            image_array: Image pixel array
            output_path: Output file path
            format: Output format ("PNG", "JPG", or "Both")

        Returns:
            True if all saves were successful, False otherwise
        """
        success = True

        if format in ("PNG", "Both"):
            output_png = (
                output_path if format == "PNG" else output_path.replace(".", "_png.")
            )
            if not self._save_png(image_array, output_png):
                success = False

        if format in ("JPG", "Both"):
            output_jpg = (
                output_path if format == "JPG" else output_path.replace(".", "_jpg.")
            )
            # Ensure .jpg extension
            if not output_jpg.endswith((".jpg", ".jpeg")):
                output_jpg = output_jpg + ".jpg"
            if not self._save_jpg(image_array, output_jpg):
                success = False

        return success

    def render_to_image(
        self,
        resolution: str,
        output_path: str,
        format: str = "PNG",
        width: int | None = None,
        height: int | None = None,
        dpi: int | None = None,
    ) -> bool:
        """Render scene to image file at specified resolution.

        Args:
            resolution: One of "1080p", "2K", "4K", "8K", or "custom"
            output_path: Output file path
            format: Output format - "PNG", "JPG", or "Both"
            width: Width in pixels (required if resolution="custom")
            height: Height in pixels (required if resolution="custom")
            dpi: Override DPI setting for this render (overrides instance dpi)

        Returns:
            True if rendering was successful, False otherwise

        Raises:
            ValueError: If resolution/format/dpi are invalid or custom
                resolution lacks width/height
            TypeError: If output_path is not a string
        """
        # Input validation
        self._validate_render_inputs(output_path, format)

        # Get resolution dimensions
        render_width, render_height = self._get_render_dimensions(
            resolution,
            width,
            height,
        )

        # Handle DPI override
        original_dpi = self.dpi
        if dpi is not None:
            if not (self.MIN_DPI <= dpi <= self.MAX_DPI):
                raise ValueError(
                    f"DPI must be between {self.MIN_DPI} and {self.MAX_DPI}, got {dpi}"
                )
            self.dpi = dpi

        try:
            # Render off-screen
            image_array = self._render_offscreen(render_width, render_height)
            # Save in requested format(s)
            return self._save_formats(image_array, output_path, format)
        finally:
            # Restore original DPI
            if dpi is not None:
                self.dpi = original_dpi

    def batch_export_views(
        self,
        views: list[str],
        output_dir: str,
        resolution: str = "1080p",
        format: str = "PNG",
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> bool:
        """Export multiple standard views in batch.

        Args:
            views: List of view names (front, back, top, bottom, left, right)
            output_dir: Output directory path
            resolution: Resolution for all views
            format: Output format for all views
            progress_callback: Optional callback(current, total, view_name)

        Returns:
            True if all exports successful, False otherwise

        Raises:
            ValueError: If views list is empty or contains invalid view names
        """
        if not views:
            raise ValueError("views list cannot be empty")

        valid_views = {
            "front",
            "back",
            "top",
            "bottom",
            "left",
            "right",
        }
        for view in views:
            if view not in valid_views:
                raise ValueError(f"Invalid view: {view}. Must be one of: {valid_views}")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        all_success = True
        for i, view in enumerate(views, 1):
            # Generate output filename
            output_file = output_path / f"{view}_{resolution}.png"

            # Render view
            try:
                success = self.render_to_image(
                    resolution=resolution,
                    output_path=str(output_file),
                    format=format,
                )
                if not success:
                    all_success = False
                    logger.warning("Failed to render view: %s", view)
            except Exception as e:
                all_success = False
                logger.error("Error rendering view %s: %s", view, e)

            # Call progress callback
            if progress_callback:
                try:
                    progress_callback(i, len(views), view)
                except Exception as e:
                    logger.warning("Progress callback failed: %s", e)

        return all_success
