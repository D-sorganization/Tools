"""
Plotting utilities for colorblind-safe visualizations and export support.

Provides colorblind-safe color palettes and export functionality for matplotlib plots.
"""

from typing import Any

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
except ImportError:
    plt = None  # type: ignore[assignment]
    ListedColormap = None  # type: ignore[assignment,misc]


# Colorblind-safe color palettes
# Based on ColorBrewer and Okabe-Ito palettes
COLORBLIND_SAFE_PALETTE = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#7f7f7f",  # Gray
    "#bcbd22",  # Olive
    "#17becf",  # Cyan
]

# Alternative palette with better contrast
COLORBLIND_SAFE_PALETTE_ALT = [
    "#006BA4",  # Blue
    "#FF800E",  # Orange
    "#ABABAB",  # Gray
    "#595959",  # Dark Gray
    "#5F9ED1",  # Light Blue
    "#C85200",  # Dark Orange
    "#898989",  # Medium Gray
    "#A2C8EC",  # Very Light Blue
    "#FFBC79",  # Light Orange
    "#CFCFCF",  # Light Gray
]


def get_colorblind_safe_colormap(name: str = "default") -> ListedColormap | None:
    """
    Get a colorblind-safe colormap.

    Args:
        name: Name of the palette ('default' or 'alt')

    Returns:
        ListedColormap instance or None if matplotlib not available
    """
    if plt is None or ListedColormap is None:
        return None

    palette = COLORBLIND_SAFE_PALETTE_ALT if name == "alt" else COLORBLIND_SAFE_PALETTE
    return ListedColormap(palette, name=f"colorblind_safe_{name}")


def apply_colorblind_safe_style(fig: Any = None, ax: Any = None) -> None:
    """
    Apply colorblind-safe styling to matplotlib figure/axes.

    Args:
        fig: Matplotlib figure (optional)
        ax: Matplotlib axes (optional)
    """
    if plt is None:
        return

    if fig is None and ax is None:
        fig = plt.gcf()
        ax = plt.gca()

    if fig is not None:
        # Set figure background to white for better contrast
        fig.patch.set_facecolor("white")

    if ax is not None:
        # Set axes background to white
        ax.set_facecolor("white")
        # Ensure grid is visible but not overwhelming
        ax.grid(True, alpha=0.3, linestyle="--")


def export_plot(
    fig: Any,
    filename: str,
    formats: list[str] | None = None,
    dpi: int = 300,
) -> list[str]:
    """
    Export matplotlib figure to multiple formats (SVG, PDF, PNG).

    Args:
        fig: Matplotlib figure to export
        filename: Base filename (without extension)
        formats: List of formats to export ['svg', 'pdf', 'png']. Defaults to all
        dpi: Resolution for raster formats (PNG). Default 300

    Returns:
        List of exported file paths

    Raises:
        ValueError: If matplotlib is not available or invalid format specified
    """
    if plt is None:
        raise ValueError("matplotlib is not available")

    if formats is None:
        formats = ["svg", "pdf", "png"]

    valid_formats = {"svg", "pdf", "png"}
    invalid = set(formats) - valid_formats
    if invalid:
        raise ValueError(f"Invalid export formats: {invalid}")

    exported_files = []
    for fmt in formats:
        filepath = f"{filename}.{fmt}"
        if fmt == "png":
            fig.savefig(filepath, format=fmt, dpi=dpi, bbox_inches="tight")
        else:
            # SVG and PDF are vector formats, no DPI needed
            fig.savefig(filepath, format=fmt, bbox_inches="tight")
        exported_files.append(filepath)

    return exported_files


def get_colorblind_safe_color(index: int, palette: str = "default") -> str:
    """
    Get a colorblind-safe color by index.

    Args:
        index: Color index (will wrap around if exceeds palette size)
        palette: Palette name ('default' or 'alt')

    Returns:
        Hex color string
    """
    color_list = (
        COLORBLIND_SAFE_PALETTE_ALT if palette == "alt" else COLORBLIND_SAFE_PALETTE
    )
    return color_list[index % len(color_list)]


# Register colorblind-safe colormaps with matplotlib if available
if plt is not None:
    try:
        plt.colormaps.register(get_colorblind_safe_colormap("default"))
        plt.colormaps.register(get_colorblind_safe_colormap("alt"))
    except Exception:
        # Colormap registration may fail in some matplotlib versions
        pass
