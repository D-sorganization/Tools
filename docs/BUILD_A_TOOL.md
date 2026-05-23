# Build a Tool: Complete Tutorial

This tutorial walks you through building a complete tool from scratch. We'll create a **Temperature Converter**—a simple but complete example that demonstrates all key concepts.

**Time estimate**: 1.5 hours  
**Difficulty**: Beginner  
**What you'll learn**: File structure, plugin discovery, writing unit tests, contract tests, GUI integration

## What We're Building

A Temperature Converter tool that:

- Converts between Celsius, Fahrenheit, and Kelvin
- Has a Python CLI
- Has a PyQt6 GUI window
- Includes unit tests
- Includes contract tests for the API surface
- Appears in the UnifiedToolsLauncher

### Final Structure

```
src/tools/temperature_converter/
├── __init__.py
├── converter.py          # Core logic
├── temperature_gui.py    # PyQt6 GUI
├── tool_manifest.json    # Auto-discovery metadata
└── README.md             # Tool documentation

tests/temperature_converter/
├── __init__.py
├── test_converter.py     # Unit tests
└── test_converter_contract.py  # API contract tests
```

## Step 1: Create the Directory Structure

```bash
# From the Tools repository root
mkdir -p src/tools/temperature_converter
mkdir -p tests/temperature_converter
touch src/tools/temperature_converter/__init__.py
touch tests/temperature_converter/__init__.py
```

## Step 2: Implement Core Logic

Create `src/tools/temperature_converter/converter.py`:

```python
"""Temperature conversion utilities."""


class TemperatureConverter:
    """Convert between Celsius, Fahrenheit, and Kelvin.

    All conversions are performed with standard physics formulas.
    Input validation raises TypeError for non-numeric values and
    ValueError for invalid Kelvin values (must be >= 0).
    """

    @staticmethod
    def celsius_to_fahrenheit(celsius: float) -> float:
        """Convert Celsius to Fahrenheit.

        Args:
            celsius: Temperature in Celsius

        Returns:
            Temperature in Fahrenheit

        Raises:
            TypeError: If celsius is not a number
        """
        if not isinstance(celsius, (int, float)):
            raise TypeError(f"Expected number, got {type(celsius).__name__}")
        return (celsius * 9/5) + 32

    @staticmethod
    def fahrenheit_to_celsius(fahrenheit: float) -> float:
        """Convert Fahrenheit to Celsius.

        Args:
            fahrenheit: Temperature in Fahrenheit

        Returns:
            Temperature in Celsius

        Raises:
            TypeError: If fahrenheit is not a number
        """
        if not isinstance(fahrenheit, (int, float)):
            raise TypeError(f"Expected number, got {type(fahrenheit).__name__}")
        return (fahrenheit - 32) * 5/9

    @staticmethod
    def celsius_to_kelvin(celsius: float) -> float:
        """Convert Celsius to Kelvin.

        Args:
            celsius: Temperature in Celsius

        Returns:
            Temperature in Kelvin (>= 0)

        Raises:
            TypeError: If celsius is not a number
            ValueError: If result would be < 0 Kelvin
        """
        if not isinstance(celsius, (int, float)):
            raise TypeError(f"Expected number, got {type(celsius).__name__}")
        kelvin = celsius + 273.15
        if kelvin < 0:
            raise ValueError(f"Invalid temperature: {kelvin}K is below absolute zero")
        return kelvin

    @staticmethod
    def kelvin_to_celsius(kelvin: float) -> float:
        """Convert Kelvin to Celsius.

        Args:
            kelvin: Temperature in Kelvin (must be >= 0)

        Returns:
            Temperature in Celsius

        Raises:
            TypeError: If kelvin is not a number
            ValueError: If kelvin < 0
        """
        if not isinstance(kelvin, (int, float)):
            raise TypeError(f"Expected number, got {type(kelvin).__name__}")
        if kelvin < 0:
            raise ValueError(f"Kelvin temperature cannot be negative: {kelvin}")
        return kelvin - 273.15

    @staticmethod
    def fahrenheit_to_kelvin(fahrenheit: float) -> float:
        """Convert Fahrenheit to Kelvin.

        Args:
            fahrenheit: Temperature in Fahrenheit

        Returns:
            Temperature in Kelvin (>= 0)

        Raises:
            TypeError: If fahrenheit is not a number
            ValueError: If result would be < 0 Kelvin
        """
        if not isinstance(fahrenheit, (int, float)):
            raise TypeError(f"Expected number, got {type(fahrenheit).__name__}")
        celsius = TemperatureConverter.fahrenheit_to_celsius(fahrenheit)
        return TemperatureConverter.celsius_to_kelvin(celsius)

    @staticmethod
    def kelvin_to_fahrenheit(kelvin: float) -> float:
        """Convert Kelvin to Fahrenheit.

        Args:
            kelvin: Temperature in Kelvin (must be >= 0)

        Returns:
            Temperature in Fahrenheit

        Raises:
            TypeError: If kelvin is not a number
            ValueError: If kelvin < 0
        """
        if not isinstance(kelvin, (int, float)):
            raise TypeError(f"Expected number, got {type(kelvin).__name__}")
        celsius = TemperatureConverter.kelvin_to_celsius(kelvin)
        return TemperatureConverter.celsius_to_fahrenheit(celsius)
```

Create `src/tools/temperature_converter/__init__.py`:

```python
"""Temperature Converter tool."""

from .converter import TemperatureConverter

__all__ = ["TemperatureConverter"]
```

## Step 3: Create the PyQt6 GUI

Create `src/tools/temperature_converter/temperature_gui.py`:

```python
"""Temperature Converter GUI using PyQt6."""

import sys
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QComboBox,
    QMessageBox,
)

from .converter import TemperatureConverter


class TemperatureConverterGUI(QWidget):
    """Temperature Converter GUI application."""

    def __init__(self):
        """Initialize the GUI."""
        super().__init__()
        self.converter = TemperatureConverter()
        self.init_ui()

    def init_ui(self):
        """Set up the user interface."""
        self.setWindowTitle("Temperature Converter")
        self.setGeometry(100, 100, 400, 300)

        main_layout = QVBoxLayout()

        # Input section
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Temperature:"))
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Enter temperature value")
        input_layout.addWidget(self.input_field)
        main_layout.addLayout(input_layout)

        # Source unit selection
        source_layout = QHBoxLayout()
        source_layout.addWidget(QLabel("From:"))
        self.source_unit = QComboBox()
        self.source_unit.addItems(["Celsius", "Fahrenheit", "Kelvin"])
        source_layout.addWidget(self.source_unit)
        main_layout.addLayout(source_layout)

        # Target unit selection
        target_layout = QHBoxLayout()
        target_layout.addWidget(QLabel("To:"))
        self.target_unit = QComboBox()
        self.target_unit.addItems(["Celsius", "Fahrenheit", "Kelvin"])
        self.target_unit.setCurrentIndex(1)  # Default to Fahrenheit
        target_layout.addWidget(self.target_unit)
        main_layout.addLayout(target_layout)

        # Convert button
        self.convert_button = QPushButton("Convert")
        self.convert_button.clicked.connect(self.convert)
        main_layout.addWidget(self.convert_button)

        # Result display
        self.result_label = QLabel("Result: (enter a value and click Convert)")
        main_layout.addWidget(self.result_label)

        self.setLayout(main_layout)

    def convert(self):
        """Perform the temperature conversion."""
        try:
            # Parse input
            value = float(self.input_field.text())
            source = self.source_unit.currentText()
            target = self.target_unit.currentText()

            # Map to converter methods
            conversions = {
                ("Celsius", "Fahrenheit"): self.converter.celsius_to_fahrenheit,
                ("Celsius", "Kelvin"): self.converter.celsius_to_kelvin,
                ("Fahrenheit", "Celsius"): self.converter.fahrenheit_to_celsius,
                ("Fahrenheit", "Kelvin"): self.converter.fahrenheit_to_kelvin,
                ("Kelvin", "Celsius"): self.converter.kelvin_to_celsius,
                ("Kelvin", "Fahrenheit"): self.converter.kelvin_to_fahrenheit,
            }

            if source == target:
                result = value
            else:
                result = conversions[(source, target)](value)

            self.result_label.setText(f"Result: {value}°{source[0]} = {result:.2f}°{target[0]}")

        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid number")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Conversion failed: {e}")


def main():
    """Entry point for the GUI application."""
    app = QApplication(sys.argv)
    window = TemperatureConverterGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
```

## Step 4: Create the Tool Manifest

Create `src/tools/temperature_converter/tool_manifest.json`:

```json
{
  "name": "Temperature Converter",
  "description": "Convert temperatures between Celsius, Fahrenheit, and Kelvin",
  "type": "python",
  "category": "Utilities",
  "path": "src/tools/temperature_converter/temperature_gui.py",
  "entry_point": "main"
}
```

**What each field means:**

- `name`: Display name in the launcher
- `description`: Short description shown in tool list
- `type`: `python`, `matlab`, or `web`
- `category`: Tab name in launcher (e.g., "Utilities", "Data Processing")
- `path`: Relative path from repo root to the tool entry point
- `entry_point`: Function name that `main()` will call

## Step 5: Write Unit Tests

Create `tests/temperature_converter/test_converter.py`:

```python
"""Unit tests for TemperatureConverter."""

import pytest

from src.tools.temperature_converter.converter import TemperatureConverter


class TestCelsiusToFahrenheit:
    """Test Celsius to Fahrenheit conversion."""

    def test_freezing_point(self):
        """Test conversion of water freezing point."""
        assert TemperatureConverter.celsius_to_fahrenheit(0) == 32

    def test_boiling_point(self):
        """Test conversion of water boiling point."""
        assert TemperatureConverter.celsius_to_fahrenheit(100) == 212

    def test_absolute_zero(self):
        """Test conversion of absolute zero."""
        result = TemperatureConverter.celsius_to_fahrenheit(-273.15)
        assert abs(result - (-459.67)) < 0.01

    def test_invalid_input_type(self):
        """Test that non-numeric input raises TypeError."""
        with pytest.raises(TypeError):
            TemperatureConverter.celsius_to_fahrenheit("100")


class TestFahrenheitToCelsius:
    """Test Fahrenheit to Celsius conversion."""

    def test_freezing_point(self):
        """Test conversion of water freezing point."""
        assert TemperatureConverter.fahrenheit_to_celsius(32) == 0

    def test_boiling_point(self):
        """Test conversion of water boiling point."""
        assert TemperatureConverter.fahrenheit_to_celsius(212) == 100

    def test_invalid_input_type(self):
        """Test that non-numeric input raises TypeError."""
        with pytest.raises(TypeError):
            TemperatureConverter.fahrenheit_to_celsius("32")


class TestCelsiusToKelvin:
    """Test Celsius to Kelvin conversion."""

    def test_absolute_zero(self):
        """Test conversion of absolute zero."""
        assert TemperatureConverter.celsius_to_kelvin(-273.15) == 0

    def test_freezing_point(self):
        """Test conversion of water freezing point."""
        assert TemperatureConverter.celsius_to_kelvin(0) == 273.15

    def test_below_absolute_zero_raises_error(self):
        """Test that below absolute zero raises ValueError."""
        with pytest.raises(ValueError, match="below absolute zero"):
            TemperatureConverter.celsius_to_kelvin(-300)

    def test_invalid_input_type(self):
        """Test that non-numeric input raises TypeError."""
        with pytest.raises(TypeError):
            TemperatureConverter.celsius_to_kelvin("0")


class TestKelvinToCelsius:
    """Test Kelvin to Celsius conversion."""

    def test_absolute_zero(self):
        """Test conversion of absolute zero."""
        assert TemperatureConverter.kelvin_to_celsius(0) == -273.15

    def test_freezing_point(self):
        """Test conversion of water freezing point."""
        assert TemperatureConverter.kelvin_to_celsius(273.15) == 0

    def test_negative_kelvin_raises_error(self):
        """Test that negative Kelvin raises ValueError."""
        with pytest.raises(ValueError, match="cannot be negative"):
            TemperatureConverter.kelvin_to_celsius(-1)

    def test_invalid_input_type(self):
        """Test that non-numeric input raises TypeError."""
        with pytest.raises(TypeError):
            TemperatureConverter.kelvin_to_celsius("273.15")
```

## Step 6: Write Contract Tests

Create `tests/temperature_converter/test_converter_contract.py`:

```python
"""Contract tests for TemperatureConverter API surface.

These tests define the public API that downstream code depends on.
Breaking these tests is a breaking change and requires coordinated PRs.
"""

import pytest

from src.tools.temperature_converter.converter import TemperatureConverter


@pytest.mark.contract
class TestTemperatureConverterAPI:
    """Contract tests for the TemperatureConverter API surface."""

    def test_class_exists(self):
        """Test that TemperatureConverter class exists."""
        assert TemperatureConverter is not None

    def test_all_conversion_methods_exist(self):
        """Test that all conversion methods are available."""
        methods = [
            "celsius_to_fahrenheit",
            "fahrenheit_to_celsius",
            "celsius_to_kelvin",
            "kelvin_to_celsius",
            "fahrenheit_to_kelvin",
            "kelvin_to_fahrenheit",
        ]
        for method in methods:
            assert hasattr(TemperatureConverter, method), f"Missing method: {method}"
            assert callable(getattr(TemperatureConverter, method))

    def test_methods_are_static(self):
        """Test that conversion methods are static (no instance required)."""
        # Should be callable without instantiation
        result = TemperatureConverter.celsius_to_fahrenheit(0)
        assert result == 32

    def test_type_error_on_invalid_input(self):
        """Test that TypeError is raised for non-numeric input."""
        with pytest.raises(TypeError):
            TemperatureConverter.celsius_to_fahrenheit("not a number")

    def test_value_error_on_invalid_kelvin(self):
        """Test that ValueError is raised for invalid Kelvin values."""
        with pytest.raises(ValueError):
            TemperatureConverter.kelvin_to_celsius(-1)

    def test_round_trip_conversion(self):
        """Test that round-trip conversions are accurate."""
        original = 25.0

        # Celsius -> Fahrenheit -> Celsius
        fahrenheit = TemperatureConverter.celsius_to_fahrenheit(original)
        back_to_celsius = TemperatureConverter.fahrenheit_to_celsius(fahrenheit)
        assert abs(back_to_celsius - original) < 0.001

        # Celsius -> Kelvin -> Celsius
        kelvin = TemperatureConverter.celsius_to_kelvin(original)
        back_to_celsius = TemperatureConverter.kelvin_to_celsius(kelvin)
        assert abs(back_to_celsius - original) < 0.001
```

## Step 7: Create Tool Documentation

Create `src/tools/temperature_converter/README.md`:

````markdown
# Temperature Converter Tool

A simple utility for converting temperatures between Celsius, Fahrenheit, and Kelvin.

## Features

- Convert between Celsius, Fahrenheit, and Kelvin
- PyQt6 GUI interface
- Input validation with clear error messages
- Full test coverage

## Usage

### GUI

```bash
python src/tools/temperature_converter/temperature_gui.py
```
````

Or launch from the UnifiedToolsLauncher:

```bash
python UnifiedToolsLauncher.py
# Click "Temperature Converter" in the Utilities tab
```

### Python API

```python
from src.tools.temperature_converter import TemperatureConverter

# Celsius to Fahrenheit
fahrenheit = TemperatureConverter.celsius_to_fahrenheit(0)  # 32

# Fahrenheit to Celsius
celsius = TemperatureConverter.fahrenheit_to_celsius(212)  # 100

# Celsius to Kelvin
kelvin = TemperatureConverter.celsius_to_kelvin(0)  # 273.15

# Any combination of conversions
kelvin = TemperatureConverter.fahrenheit_to_kelvin(32)  # 273.15
```

## Testing

Run unit tests:

```bash
python -m pytest tests/temperature_converter/test_converter.py -v
```

Run contract tests (API surface):

```bash
python -m pytest tests/temperature_converter/test_converter_contract.py -m contract -v
```

## API Reference

All methods are static and follow the pattern `{source_unit}_to_{target_unit}(value)`.

### Methods

- `celsius_to_fahrenheit(celsius: float) -> float`
- `fahrenheit_to_celsius(fahrenheit: float) -> float`
- `celsius_to_kelvin(celsius: float) -> float`
- `kelvin_to_celsius(kelvin: float) -> float`
- `fahrenheit_to_kelvin(fahrenheit: float) -> float`
- `kelvin_to_fahrenheit(kelvin: float) -> float`

### Error Handling

- **TypeError**: Raised if input is not a number
- **ValueError**: Raised if Kelvin temperature is negative (below absolute zero)

## Architecture

- `converter.py`: Core conversion logic (no dependencies)
- `temperature_gui.py`: PyQt6 GUI wrapper
- `tool_manifest.json`: Tool registration for auto-discovery

````

## Step 8: Verify and Test

From the repo root, run your tests:

```bash
# Run all tests for your tool
python -m pytest tests/temperature_converter/ -v

# Run unit tests only
python -m pytest tests/temperature_converter/test_converter.py -v

# Run contract tests
python -m pytest tests/temperature_converter/test_converter_contract.py -m contract -v

# Check coverage
python -m pytest tests/temperature_converter/ --cov=src.tools.temperature_converter
````

**Expected output:**

```
test_converter.py::TestCelsiusToFahrenheit::test_freezing_point PASSED
test_converter.py::TestCelsiusToFahrenheit::test_boiling_point PASSED
...
test_converter_contract.py::TestTemperatureConverterAPI::test_class_exists PASSED
...
==================== 20 passed in 0.15s ====================
```

## Step 9: Launch the Tool

### Via UnifiedToolsLauncher

```bash
python UnifiedToolsLauncher.py
```

Look for "Temperature Converter" under the "Utilities" tab. Click it and then click **Launch**.

### Direct Execution

```bash
python src/tools/temperature_converter/temperature_gui.py
```

## Step 10: Commit and Submit PR

```bash
# Create a feature branch
git checkout -b feature/temperature-converter

# Add your files
git add src/tools/temperature_converter/
git add tests/temperature_converter/

# Verify code quality
python -m ruff format .
python -m ruff check .

# Commit
git commit -m "Add Temperature Converter tool

- Converts between Celsius, Fahrenheit, Kelvin
- PyQt6 GUI interface
- Full unit and contract test coverage
- Supports auto-discovery via tool_manifest.json"

# Push
git push origin feature/temperature-converter
```

Then open a Pull Request on GitHub. CI will:

1. Run linting and formatting checks
2. Run all tests (including your contract tests)
3. Verify code coverage
4. Check the tool manifest is valid

## Checklist: Complete Tool

Before submitting your PR, verify:

- [ ] Core logic in `converter.py` with proper docstrings
- [ ] GUI in `temperature_gui.py` (if applicable)
- [ ] Tool manifest (`tool_manifest.json`) with correct paths
- [ ] Unit tests (`test_converter.py`) with >80% coverage
- [ ] Contract tests (`test_converter_contract.py`) marked with `@pytest.mark.contract`
- [ ] README.md with examples and API reference
- [ ] All tests passing: `python -m pytest tests/temperature_converter/`
- [ ] Code formatted: `python -m ruff format .`
- [ ] No style issues: `python -m ruff check .`
- [ ] Tool appears in launcher: `python UnifiedToolsLauncher.py`
- [ ] No `print()` statements (use logging instead)

## Troubleshooting

### Tool doesn't appear in launcher

**Cause**: Manifest not found or path incorrect

**Solution**:

```bash
# Check manifest exists
ls -la src/tools/temperature_converter/tool_manifest.json

# Check path is correct (relative to repo root)
cat src/tools/temperature_converter/tool_manifest.json | grep path

# Verify the file exists at that path
python -c "from pathlib import Path; print(Path('src/tools/temperature_converter/temperature_gui.py').exists())"
```

### GUI doesn't open

**Cause**: PyQt6 import error or wrong entry point

**Solution**:

```bash
# Test direct import
python -c "from src.tools.temperature_converter.temperature_gui import TemperatureConverterGUI; print('OK')"

# Test the main function
python src/tools/temperature_converter/temperature_gui.py
```

### Tests fail with import errors

**Cause**: Not running from repo root or venv not activated

**Solution**:

```bash
# Verify location
pwd  # Should end with /Tools

# Verify venv
which python  # Should show path with /venv/

# Activate if needed
source venv/bin/activate

# Try again
python -m pytest tests/temperature_converter/
```

## Next Steps

1. Create variations: add more tools to solidify your understanding
2. Read `ARCHITECTURE_OVERVIEW.md` to understand system design
3. Explore existing tools in `src/tools/` for patterns and best practices
4. Check `PLUGIN_SYSTEM.md` for advanced auto-discovery features
5. Review `CLAUDE.md` for shared library constraints (if applicable)

---

**Congratulations!** You've built a complete, tested, documented tool that integrates with the launcher and follows all project standards.
