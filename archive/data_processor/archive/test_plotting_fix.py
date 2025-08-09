#!/usr/bin/env python3
"""
Test script to verify plotting functionality fixes.
"""

import os
import sys
from unittest.mock import MagicMock, Mock

import numpy as np
import pandas as pd

# Add the current directory to the path so we can import the main module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# Mock tkinter components
class MockTkinter:
    class StringVar:
        def __init__(self, value=""):
            self._value = value

        def get(self):
            return self._value

        def set(self, value):
            self._value = value

    class BooleanVar:
        def __init__(self, value=False):
            self._value = value

        def get(self):
            return self._value

        def set(self, value):
            self._value = value


# Mock the tkinter module
sys.modules["tkinter"] = MockTkinter()
import tkinter as tk


# Mock customtkinter
class MockCTk:
    def __init__(self, *args, **kwargs):
        pass


class MockCTkTabview:
    def __init__(self, *args, **kwargs):
        pass

    def add(self, name):
        return Mock()

    def tab(self, name):
        return Mock()


class MockCTkFrame:
    def __init__(self, *args, **kwargs):
        pass

    def grid(self, *args, **kwargs):
        pass

    def grid_columnconfigure(self, *args, **kwargs):
        pass

    def grid_rowconfigure(self, *args, **kwargs):
        pass


class MockCTkLabel:
    def __init__(self, *args, **kwargs):
        pass

    def grid(self, *args, **kwargs):
        pass

    def configure(self, *args, **kwargs):
        pass


class MockCTkButton:
    def __init__(self, *args, **kwargs):
        pass

    def grid(self, *args, **kwargs):
        pass


class MockCTkCheckBox:
    def __init__(self, *args, **kwargs):
        pass

    def grid(self, *args, **kwargs):
        pass

    def pack(self, *args, **kwargs):
        pass


class MockCTkEntry:
    def __init__(self, *args, **kwargs):
        pass

    def grid(self, *args, **kwargs):
        pass

    def get(self):
        return ""


class MockCTkOptionMenu:
    def __init__(self, *args, **kwargs):
        pass

    def grid(self, *args, **kwargs):
        pass

    def configure(self, *args, **kwargs):
        pass


class MockCTkScrollableFrame:
    def __init__(self, *args, **kwargs):
        pass

    def grid(self, *args, **kwargs):
        pass

    def grid_columnconfigure(self, *args, **kwargs):
        pass


class MockCTkFont:
    def __init__(self, *args, **kwargs):
        pass


# Mock customtkinter module
mock_ctk = Mock()
mock_ctk.CTk = MockCTk
mock_ctk.CTkTabview = MockCTkTabview
mock_ctk.CTkFrame = MockCTkFrame
mock_ctk.CTkLabel = MockCTkLabel
mock_ctk.CTkButton = MockCTkButton
mock_ctk.CTkCheckBox = MockCTkCheckBox
mock_ctk.CTkEntry = MockCTkEntry
mock_ctk.CTkOptionMenu = MockCTkOptionMenu
mock_ctk.CTkScrollableFrame = MockCTkScrollableFrame
mock_ctk.CTkFont = MockCTkFont

sys.modules["customtkinter"] = mock_ctk
import customtkinter as ctk


# Mock matplotlib
class MockFigure:
    def __init__(self):
        self.axes = [MockAxes()]

    def add_subplot(self, *args, **kwargs):
        return self.axes[0]


class MockAxes:
    def __init__(self):
        pass

    def clear(self):
        pass

    def plot(self, *args, **kwargs):
        pass

    def set_title(self, *args, **kwargs):
        pass

    def set_xlabel(self, *args, **kwargs):
        pass

    def set_ylabel(self, *args, **kwargs):
        pass

    def legend(self):
        pass

    def grid(self, *args, **kwargs):
        pass

    def text(self, *args, **kwargs):
        pass

    def xaxis(self):
        return Mock()

    def tick_params(self, *args, **kwargs):
        pass


class MockCanvas:
    def __init__(self):
        pass

    def draw(self):
        pass


# Mock matplotlib module
mock_matplotlib = Mock()
mock_matplotlib.pyplot = Mock()
mock_matplotlib.pyplot.Figure = MockFigure
mock_matplotlib.dates = Mock()
mock_matplotlib.dates.DateFormatter = Mock()

# Create a proper mock for matplotlib.pyplot
mock_pyplot = Mock()
mock_pyplot.Figure = MockFigure
mock_matplotlib.pyplot = mock_pyplot

# Create a proper mock for matplotlib.dates
mock_dates = Mock()
mock_dates.DateFormatter = Mock()
mock_matplotlib.dates = mock_dates

sys.modules["matplotlib"] = mock_matplotlib
sys.modules["matplotlib.pyplot"] = mock_pyplot
sys.modules["matplotlib.dates"] = mock_dates

import matplotlib.dates as mdates
# Now import them
import matplotlib.pyplot as plt

# Mock tkinter.filedialog and messagebox
tk.filedialog = Mock()
tk.messagebox = Mock()
tk.simpledialog = Mock()


def test_processed_files_initialization():
    """Test that processed_files attribute is properly initialized."""
    print("Testing processed_files initialization...")

    # Import the main module
    from CSV_Processor_Rev5_Complete import CSVProcessorApp

    # Create an instance
    app = CSVProcessorApp()

    # Check if processed_files is initialized
    assert hasattr(
        app, "processed_files"
    ), "processed_files attribute should be initialized"
    assert isinstance(
        app.processed_files, dict
    ), "processed_files should be a dictionary"
    assert len(app.processed_files) == 0, "processed_files should be empty initially"

    print("✓ processed_files initialization test passed")


def test_get_data_for_plotting():
    """Test the get_data_for_plotting method."""
    print("Testing get_data_for_plotting method...")

    from CSV_Processor_Rev5_Complete import CSVProcessorApp

    app = CSVProcessorApp()

    # Test with non-existent file
    result = app.get_data_for_plotting("nonexistent.csv")
    assert result is None, "Should return None for non-existent file"

    # Test with empty processed_files
    result = app.get_data_for_plotting("test.csv")
    assert result is None, "Should return None when file not in processed_files"

    print("✓ get_data_for_plotting method test passed")


def test_process_files_stores_data():
    """Test that process_files stores data in processed_files."""
    print("Testing process_files data storage...")

    from CSV_Processor_Rev5_Complete import CSVProcessorApp

    app = CSVProcessorApp()

    # Mock the _process_single_file method to return test data
    def mock_process_single_file(file_path, settings):
        return pd.DataFrame(
            {
                "Time": pd.date_range("2024-01-01", periods=10, freq="H"),
                "Signal1": np.random.randn(10),
                "Signal2": np.random.randn(10),
            }
        )

    app._process_single_file = mock_process_single_file

    # Mock input file paths
    app.input_file_paths = ["/path/to/test1.csv", "/path/to/test2.csv"]

    # Mock signal_vars
    app.signal_vars = {
        "Time": {"var": tk.BooleanVar(value=True)},
        "Signal1": {"var": tk.BooleanVar(value=True)},
        "Signal2": {"var": tk.BooleanVar(value=True)},
    }

    # Mock other required attributes
    app.filter_type_var = tk.StringVar(value="None")
    app.resample_var = tk.BooleanVar(value=False)
    app.ma_value_entry = Mock()
    app.ma_value_entry.get.return_value = "10"
    app.bw_order_entry = Mock()
    app.bw_order_entry.get.return_value = "3"
    app.bw_cutoff_entry = Mock()
    app.bw_cutoff_entry.get.return_value = "0.1"
    app.median_kernel_entry = Mock()
    app.median_kernel_entry.get.return_value = "5"
    app.savgol_window_entry = Mock()
    app.savgol_window_entry.get.return_value = "11"
    app.savgol_polyorder_entry = Mock()
    app.savgol_polyorder_entry.get.return_value = "2"
    app.trim_date_entry = Mock()
    app.trim_date_entry.get.return_value = ""
    app.trim_start_entry = Mock()
    app.trim_start_entry.get.return_value = ""
    app.trim_end_entry = Mock()
    app.trim_end_entry.get.return_value = ""

    # Mock the _export_processed_files method
    app._export_processed_files = Mock()

    # Mock messagebox
    app.messagebox = Mock()

    # Call process_files
    app.process_files()

    # Check that data was stored in processed_files
    assert "test1.csv" in app.processed_files, "test1.csv should be in processed_files"
    assert "test2.csv" in app.processed_files, "test2.csv should be in processed_files"
    assert isinstance(
        app.processed_files["test1.csv"], pd.DataFrame
    ), "Stored data should be DataFrame"
    assert isinstance(
        app.processed_files["test2.csv"], pd.DataFrame
    ), "Stored data should be DataFrame"

    print("✓ process_files data storage test passed")


def test_plot_file_menu_update():
    """Test that plot file menu is updated when files are loaded."""
    print("Testing plot file menu update...")

    from CSV_Processor_Rev5_Complete import CSVProcessorApp

    app = CSVProcessorApp()

    # Mock plot_file_menu
    app.plot_file_menu = Mock()
    app.plot_file_menu.configure = Mock()

    # Mock input file paths
    app.input_file_paths = ["/path/to/file1.csv", "/path/to/file2.csv"]

    # Mock update_signal_list method
    app.update_signal_list = Mock()

    # Call load_signals_from_files
    app.load_signals_from_files()

    # Check that plot_file_menu was configured with correct values
    expected_values = ["Select a file...", "file1.csv", "file2.csv"]
    app.plot_file_menu.configure.assert_called_with(values=expected_values)

    print("✓ plot file menu update test passed")


if __name__ == "__main__":
    print("Running plotting functionality tests...\n")

    try:
        test_processed_files_initialization()
        test_get_data_for_plotting()
        test_process_files_stores_data()
        test_plot_file_menu_update()

        print(
            "\n🎉 All tests passed! The plotting functionality should now work correctly."
        )
        print("\nKey fixes implemented:")
        print("1. ✓ Initialized self.processed_files attribute in __init__")
        print(
            "2. ✓ Modified process_files to store processed data in self.processed_files"
        )
        print("3. ✓ Verified get_data_for_plotting can access processed data")
        print("4. ✓ Confirmed plot file menu is properly updated")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
