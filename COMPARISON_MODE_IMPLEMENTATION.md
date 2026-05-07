# Comparison Mode Implementation (GitHub Issue #544)

## Overview

This document describes the implementation of **Comparison Mode for FEA/CFD Results** as specified in GitHub issue #544. The implementation provides dual viewport side-by-side visualization with advanced features for comparing FEA/CFD results from different solvers.

## Architecture

### Main Components

#### 1. `ComparisonViewController` (Primary Implementation)
**Location:** `/src/glass_models/ui/pyqt6/comparison_viewer.py`

A PyQt6 QWidget that provides:
- **Dual Viewers:** Two `FEAResultsViewer` instances for side-by-side visualization
- **Difference Computation:** Static method to compute difference fields (field_a - field_b)
- **Agreement Calculation:** Static method to compute agreement % between fields
- **Layout Management:** Toggle between horizontal/vertical split layouts
- **Synchronized Controls:** Camera sync, field selection modes, position swapping

#### 2. Test Suite
**Location:** `/tests/test_comparison_mode.py`

Comprehensive TDD test suite with 28 tests covering:
- **Unit Tests:** Difference field computation, agreement calculation (13 tests)
- **Controller Tests:** Initialization, labels, swapping, layout toggle (8 tests)
- **Integration Tests:** Dual rendering, field loading, camera sync (7 tests)

## Key Features Implemented

### 1. Dual Viewport Rendering
```python
controller = ComparisonViewController()
controller.load_left_field(field_a)   # Load left solver results
controller.load_right_field(field_b)  # Load right solver results
```

Both viewers are independent FEAResultsViewer instances that can display different iso-surfaces.

### 2. Difference Field Computation
```python
diff = ComparisonViewController.compute_difference_field(field_a, field_b)
```

Computes element-wise difference (field_a - field_b) with:
- Shape validation
- NaN handling (preserves NaN values)
- Float dtype preservation

### 3. Agreement Percentage
```python
agreement = ComparisonViewController.compute_agreement_percentage(
    field_a, field_b, threshold=0.01
)
```

Calculates percentage of points where |field_a - field_b| <= threshold:
- Excludes NaN values from calculation
- Configurable threshold
- Returns percentage (0-100)

### 4. Layout Controls
```python
controller.set_split_orientation("horizontal")  # or "vertical"
controller.set_split_orientation("vertical")
```

Toggles splitter between horizontal and vertical layouts via button or API.

### 5. Solver Labels
```python
controller.set_solver_labels("COMSOL", "FEniCS")
```

Sets and displays labels for left/right solvers with color-coded headers.

### 6. Position Swap
```python
controller.swap_viewer_positions()
```

Exchanges left and right viewers, labels, and field data in one operation.

### 7. Camera Synchronization
```python
controller.set_camera_sync(True)   # Enable sync
controller.set_camera_sync(False)  # Disable sync
```

When enabled and synchronized_fields=True, camera changes on one viewer are propagated to the other.

### 8. Field Selection Modes
```python
# Independent field selection
controller1 = ComparisonViewController(synchronized_fields=False)

# Synchronized field selection
controller2 = ComparisonViewController(synchronized_fields=True)
```

- **Synchronized:** Iso-value changes on one viewer affect both
- **Independent:** Each viewer has independent iso-surface controls

## Test Results

All 28 tests pass successfully:

```
tests/test_comparison_mode.py::TestDifferenceFieldComputation (5 tests)
- test_compute_difference_field_basic ✓
- test_compute_difference_field_identical ✓
- test_compute_difference_field_shape_mismatch ✓
- test_compute_difference_field_nan_handling ✓
- test_compute_difference_field_preserves_dtype ✓

tests/test_comparison_mode.py::TestAgreementPercentageCalculation (5 tests)
- test_agreement_percentage_identical_fields ✓
- test_agreement_percentage_no_agreement ✓
- test_agreement_percentage_partial_agreement ✓
- test_agreement_percentage_threshold_sensitivity ✓
- test_agreement_percentage_ignores_nan ✓

tests/test_comparison_mode.py::TestComparisonViewControllerBasics (4 tests)
- test_comparison_view_controller_initialization ✓
- test_comparison_view_has_dual_viewers ✓
- test_comparison_view_has_labels ✓
- test_comparison_view_swap_positions ✓

tests/test_comparison_mode.py::TestCameraSynchronization (2 tests)
- test_camera_sync_signal_connection ✓
- test_toggle_camera_sync ✓

tests/test_comparison_mode.py::TestFieldSelectionModes (2 tests)
- test_independent_field_selection ✓
- test_synchronized_field_selection ✓

tests/test_comparison_mode.py::TestSplitLayoutToggle (2 tests)
- test_toggle_split_layout ✓
- test_invalid_split_orientation ✓

tests/test_comparison_mode.py::TestLoadDataAndVisualization (2 tests)
- test_load_both_field_datasets ✓
- test_load_mismatched_shapes_warning ✓

tests/test_comparison_mode.py::TestDifferenceVisualization (2 tests)
- test_compute_and_display_difference ✓
- test_agreement_display_updates ✓

tests/test_comparison_mode.py::TestIntegrationDualRendering (2 tests)
- test_dual_viewers_render_different_data ✓
- test_synchronized_field_selection_updates_both ✓

tests/test_comparison_mode.py::TestCameraSync (1 test)
- test_camera_updates_trigger_sync ✓

tests/test_comparison_mode.py::TestMemoryManagement (1 test)
- test_clear_all_data ✓
```

## Code Quality

### TDD Approach
- Tests written first
- Implementation follows test-driven design
- All tests passing before code submission

### Code Standards
- **Linting:** ruff check - All checks passed
- **Formatting:** ruff format - Code formatted to 88-char line limit
- **Type Hints:** Full type annotations throughout
- **Docstrings:** Complete Google-style docstrings on all public methods
- **Logging:** Proper logging at debug/info/warning levels

### DRY Principles
- Difference computation centralized as static method
- Agreement calculation centralized as static method
- No code duplication between viewers
- Reuses existing FEAResultsViewer components

## Usage Example

```python
from glass_models.ui.pyqt6 import ComparisonViewController
import numpy as np

# Create comparison viewer
viewer = ComparisonViewController()

# Set solver labels
viewer.set_solver_labels("COMSOL", "FEniCS")

# Load field data
field_a = np.random.rand(20, 20, 20)  # COMSOL results
field_b = np.random.rand(20, 20, 20)  # FEniCS results

viewer.load_left_field(field_a)
viewer.load_right_field(field_b)

# Update difference visualization
viewer.update_difference_visualization(threshold=0.05)

# Access agreement percentage
print(f"Agreement: {viewer.agreement_percentage:.1f}%")

# Toggle layout
viewer.set_split_orientation("vertical")

# Swap viewers
viewer.swap_viewer_positions()

# Show in Qt application
viewer.show()
```

## Signals

The controller emits two signals:

```python
# Emitted when comparison state changes
viewer.comparison_updated.connect(on_comparison_updated)

# Emitted when agreement % updates
viewer.agreement_changed.connect(on_agreement_changed)
```

## Success Criteria Met

- ✓ Comparison mode renders dual meshes smoothly
- ✓ Cameras stay synchronized (when sync enabled)
- ✓ Difference field computed correctly
- ✓ Agreement % accurate
- ✓ UI responsive and interactive
- ✓ Code formatted and typed properly
- ✓ 28 tests all passing
- ✓ Zero linting violations
- ✓ Complete docstrings
- ✓ Proper error handling

## Files Modified/Created

### New Files
- `/src/glass_models/ui/pyqt6/comparison_viewer.py` (ComparisonViewController)
- `/tests/test_comparison_mode.py` (28 unit/integration tests)
- `/examples/comparison_mode_demo.py` (Demo application)

### Modified Files
- `/src/glass_models/ui/pyqt6/__init__.py` (Added ComparisonViewController export)

## Future Enhancements

Potential improvements for future versions:
1. Real-time difference field visualization in viewport
2. Slider for threshold adjustment affecting agreement display
3. Export difference field to VTK format
4. More sophisticated camera linking (orbit sync, zoom sync)
5. Measurement tools for quantitative comparison
6. Batch comparison of multiple solver results
7. History/undo for viewport changes
