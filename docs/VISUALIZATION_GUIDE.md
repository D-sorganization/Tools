# Visualization and Accessibility Guide

## Colorblind-Safe Plotting

This repository provides utilities for creating colorblind-safe visualizations that are accessible to users with color vision deficiencies.

### Using Colorblind-Safe Colors

The `python/src/utils/plotting.py` module provides colorblind-safe color palettes:

```python
from python.src.utils.plotting import (
    get_colorblind_safe_color,
    get_colorblind_safe_colormap,
    apply_colorblind_safe_style,
)

# Get a single color by index
color = get_colorblind_safe_color(0)  # Returns "#1f77b4" (blue)

# Get a colormap
cmap = get_colorblind_safe_colormap("default")

# Apply colorblind-safe styling
apply_colorblind_safe_style(fig, ax)
```

### Colorblind-Safe Palettes

Two palettes are provided:

1. **Default Palette**: Based on ColorBrewer, optimized for common color vision deficiencies
2. **Alternative Palette**: Higher contrast variant for better visibility

Both palettes are tested for:
- Protanopia (red-blind)
- Deuteranopia (green-blind)
- Tritanopia (blue-blind)

### Export Formats

All plots should support export to:
- **SVG**: Vector format for web and print
- **PDF**: Vector format for documents
- **PNG**: Raster format for presentations (300 DPI default)

```python
from python.src.utils.plotting import export_plot

# Export figure to multiple formats
exported = export_plot(fig, "my_plot", formats=["svg", "pdf", "png"])
```

### Best Practices

1. **Use Colorblind-Safe Palettes**: Always use the provided palettes instead of default matplotlib colors
2. **Add Patterns/Shapes**: For critical distinctions, use patterns or shapes in addition to color
3. **High Contrast**: Ensure sufficient contrast between colors (WCAG AA minimum)
4. **Test Accessibility**: Use tools like [Color Oracle](https://colororacle.org/) to preview plots
5. **Export Support**: Always provide SVG/PDF export options for vector graphics

### Example

```python
import matplotlib.pyplot as plt
from python.src.utils.plotting import (
    get_colorblind_safe_color,
    apply_colorblind_safe_style,
    export_plot,
)

fig, ax = plt.subplots()

# Use colorblind-safe colors
colors = [get_colorblind_safe_color(i) for i in range(5)]
ax.bar(range(5), [1, 2, 3, 4, 5], color=colors)

# Apply accessibility styling
apply_colorblind_safe_style(fig, ax)

# Export to multiple formats
export_plot(fig, "accessible_plot", formats=["svg", "pdf", "png"])

plt.show()
```

### Resources

- [ColorBrewer](https://colorbrewer2.org/) - Colorblind-safe palettes
- [Okabe-Ito Palette](https://jfly.uni-koeln.de/color/) - Universal design colors
- [WCAG Guidelines](https://www.w3.org/WAI/WCAG21/quickref/) - Web accessibility standards
