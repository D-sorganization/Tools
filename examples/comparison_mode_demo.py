#!/usr/bin/env python3
"""Demo: Comparison Mode for FEA/CFD Results (GitHub issue #544).

This example demonstrates the dual viewport comparison viewer with:
- Side-by-side FEA results visualization
- Synchronized camera controls
- Difference field computation
- Agreement percentage display
- Layout toggle and position swap
"""

import sys

import numpy as np
from PyQt6.QtWidgets import QApplication

from glass_models.ui.pyqt6 import ComparisonViewController


def create_synthetic_fields(shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Create synthetic field data for demonstration.

    Args:
        shape: Shape of field (nx, ny, nz)

    Returns:
        Tuple of (field_a, field_b) for comparison
    """
    nx, ny, nz = shape

    # Create coordinate grids
    x = np.linspace(-2, 2, nx)
    y = np.linspace(-2, 2, ny)
    z = np.linspace(-2, 2, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # Field A: Gaussian blob (reference solution)
    field_a = np.exp(-((X**2 + Y**2 + Z**2) / 0.5))

    # Field B: Perturbed Gaussian (alternative solver solution)
    field_b = np.exp(-((X**2 + Y**2 + (Z - 0.1) ** 2) / 0.5)) * 0.95

    return field_a, field_b


def main() -> None:
    """Run comparison mode demo."""
    # Create Qt application
    app = QApplication(sys.argv)

    # Create comparison viewer
    viewer = ComparisonViewController()

    # Set solver labels
    viewer.set_solver_labels("COMSOL", "FEniCS")

    # Create synthetic field data
    field_a, field_b = create_synthetic_fields((20, 20, 20))

    # Load data into viewers
    viewer.load_left_field(field_a)
    viewer.load_right_field(field_b)

    # Update difference visualization
    viewer.update_difference_visualization(threshold=0.05)

    # Show the viewer
    viewer.show()
    viewer.setWindowTitle("FEA/CFD Comparison Mode - Demo")

    print("Comparison Viewer Demo")
    print("=" * 50)
    print(f"Left field (COMSOL): shape {field_a.shape}, "
          f"range [{np.nanmin(field_a):.3f}, {np.nanmax(field_a):.3f}]")
    print(f"Right field (FEniCS): shape {field_b.shape}, "
          f"range [{np.nanmin(field_b):.3f}, {np.nanmax(field_b):.3f}]")
    print(f"Agreement: {viewer.agreement_percentage:.1f}%")
    print("\nFeatures:")
    print("- Click 'Toggle Layout (H/V)' to switch between horizontal/vertical")
    print("- Click 'Swap Positions' to exchange left/right viewers")
    print("- Use iso-surface controls to adjust visualization")
    print("- Camera changes can be synchronized between viewers")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
