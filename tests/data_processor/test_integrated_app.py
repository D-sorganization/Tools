# Add the project directory to path
import os
import sys
import unittest
from typing import Any
from unittest.mock import MagicMock, patch

project_root = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../data_processing/data_processor/python/data_processor",
    )
)
sys.path.insert(0, project_root)


# Mock dependencies before importing
# Define a real class for CTkToplevel so inheritance works correctly
class MockCTkToplevel:
    """Mock for CustomTkinter Toplevel window."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize mock window."""

    def transient(self, master: Any) -> None:
        """Mock transient method."""

    def grab_set(self) -> None:
        """Mock grab_set method."""

    def geometry(self, g: Any) -> None:
        """Mock geometry method."""

    def resizable(self, w: Any, h: Any) -> None:
        """Mock resizable method."""

    def title(self, t: Any) -> None:
        """Mock title method."""

    def destroy(self) -> None:
        """Mock destroy method."""

    def wait_window(self) -> None:
        """Mock wait_window method."""

    def lift(self) -> None:
        """Mock lift method."""

    def attributes(self, *args: Any) -> None:
        """Mock attributes method."""


class MockScrollableFrame:
    """Mock for CustomTkinter ScrollableFrame."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize mock scrollable frame."""

    def __getattr__(self, name: str) -> MagicMock:
        """Return magic mock for any attribute."""
        return MagicMock()

    def grid(self, *args: Any, **kwargs: Any) -> None:
        """Mock grid geometry manager."""

    def pack(self, *args: Any, **kwargs: Any) -> None:
        """Mock pack geometry manager."""


mock_ctk = MagicMock()
mock_ctk.CTkToplevel = MockCTkToplevel
mock_ctk.CTkFrame = MagicMock(side_effect=lambda *args, **kwargs: MagicMock())
mock_ctk.CTkLabel = MagicMock(side_effect=lambda *args, **kwargs: MagicMock())
mock_ctk.CTkButton = MagicMock(side_effect=lambda *args, **kwargs: MagicMock())
mock_ctk.CTkTextbox = MagicMock(side_effect=lambda *args, **kwargs: MagicMock())
mock_ctk.CTkScrollableFrame = MagicMock(side_effect=lambda *args, **kwargs: MagicMock())

sys.modules["customtkinter"] = mock_ctk
sys.modules["tkinter"] = MagicMock()
sys.modules["matplotlib"] = MagicMock()
sys.modules["matplotlib.pyplot"] = MagicMock()
sys.modules["matplotlib.backends.backend_tkagg"] = MagicMock()
sys.modules["scipy.signal"] = MagicMock()
sys.modules["scipy.ndimage"] = MagicMock()
sys.modules["PIL"] = MagicMock()
# Detailed pyarrow mock to satisfy pandas' deep inspection
mock_pa = MagicMock()
mock_pa.__version__ = "14.0.0"


class MockDataType:
    pass


mock_pa.DataType = MockDataType

# Mock common pyarrow types to return instances of DataType
common_types = [
    "null",
    "bool_",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float32",
    "float64",
    "timestamp",
    "date32",
    "date64",
    "time32",
    "time64",
    "duration",
    "binary",
    "string",
    "large_binary",
    "large_string",
    "list_",
    "large_list",
    "map_",
    "struct",
    "dictionary",
]

for t in common_types:
    setattr(mock_pa, t, MagicMock(return_value=MockDataType()))

sys.modules["pyarrow"] = mock_pa
sys.modules["pyarrow.compute"] = MagicMock()
sys.modules["pyarrow.parquet"] = MagicMock()
sys.modules["pyarrow.csv"] = MagicMock()

sys.modules["tables"] = MagicMock()

sys.modules["tables"] = MagicMock()

# Mock Data_Processor_r0 since it's the base class
mock_r0 = MagicMock()


class MockCSVProcessorApp:
    """Mock for the base CSV Processor Application."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize mock application."""
        self.main_tab_view = MagicMock()
        self.main_tab_view.tab = MagicMock()

    def title(self, t: Any) -> None:
        """Set window title."""

    def geometry(self, g: Any) -> None:
        """Set window geometry."""

    def update_idletasks(self) -> None:
        """Update idle tasks."""

    def winfo_screenwidth(self) -> int:
        """Get screen width."""
        return 1920

    def winfo_screenheight(self) -> int:
        """Get screen height."""
        return 1080

    def bind(self, *args: Any) -> None:
        """Bind event handler."""

    def protocol(self, *args: Any) -> None:
        """Set protocol handler."""

    def create_status_bar(self) -> None:
        """Create status bar."""
        self.status_label = MagicMock()

    def create_setup_and_process_tab(self, tab: Any) -> None:
        """Create setup tab."""

    def create_plotting_tab(self, tab: Any) -> None:
        """Create plotting tab."""

    def create_plots_list_tab(self, tab: Any) -> None:
        """Create plots list tab."""

    def create_dat_import_tab(self, tab: Any) -> None:
        """Create import tab."""

    def create_help_tab(self, tab: Any) -> None:
        """Create help tab."""

    def _on_closing(self) -> None:
        """Handle window closing."""

    def _on_window_configure(self, event: Any) -> None:
        """Handle window configure event."""

    def _load_plots_from_file(self) -> None:
        """Load plots from file."""

    def grid_rowconfigure(self, *args: Any) -> None:
        """Configure grid rows."""

    def grid_columnconfigure(self, *args: Any) -> None:
        """Configure grid columns."""

    def _create_splitter(self, *args: Any) -> MagicMock:
        """Create splitter widget."""
        return MagicMock()


mock_r0.CSVProcessorApp = MockCSVProcessorApp
sys.modules["Data_Processor_r0"] = mock_r0

# Now import the class under test
from Data_Processor_Integrated import (
    FileFormatDetector,
    IntegratedCSVProcessorApp,
    SplitConfig,
    SplitMethod,
)


class TestIntegratedCSVProcessorApp(unittest.TestCase):
    """Test suite for the Integrated CSV Processor Application."""

    @patch("Data_Processor_Integrated.ctk.CTk")
    def test_initialization(self, mock_ctk: MagicMock) -> None:
        """Test application initialization."""
        # Base class is now our MockCSVProcessorApp
        app = IntegratedCSVProcessorApp()

        # Check if converter variables are initialized
        self.assertTrue(hasattr(app, "converter_split_config"))
        self.assertIsInstance(app.converter_split_config, SplitConfig)
        self.assertTrue(hasattr(app, "converter_input_files"))
        self.assertEqual(app.converter_input_files, [])

    def test_split_config_defaults(self) -> None:
        """Test default values of SplitConfig."""
        config = SplitConfig()
        self.assertFalse(config.enabled)
        self.assertEqual(config.method, SplitMethod.ROWS)
        self.assertEqual(config.rows_per_file, 100000)
        self.assertEqual(config.compression, "snappy")

    @patch("os.path.exists", return_value=True)
    def test_file_format_detector(self, mock_exists: MagicMock) -> None:
        """Test file format detection."""
        detector = FileFormatDetector()

        # Test extension based detection
        self.assertEqual(detector.detect_format("test.csv"), "csv")
        self.assertEqual(detector.detect_format("test.json"), "json")
        self.assertEqual(detector.detect_format("test.parquet"), "parquet")
        self.assertEqual(detector.detect_format("test.xlsx"), "excel")
        # txt falls back to tsv or content check in code?
        self.assertEqual(detector.detect_format("test.txt"), "tsv")

    @patch("Data_Processor_Integrated.pq")
    def test_parquet_analyzer(self, mock_pq: MagicMock) -> None:
        """Test the Parquet analyzer dialog."""
        from Data_Processor_Integrated import ParquetAnalyzerDialog

        # Setup mock parquet file
        mock_file = MagicMock()
        mock_file.metadata.num_rows = 100
        mock_file.metadata.num_columns = 5
        mock_file.metadata.num_row_groups = 1

        # Schema
        mock_field = MagicMock()
        mock_field.name = "col1"
        mock_field.type = "int64"
        mock_file.schema_arrow = [mock_field]

        # Row groups
        mock_rg = MagicMock()
        mock_rg.num_rows = 100
        mock_rg.total_byte_size = 1024
        mock_rg.num_columns = 1
        mock_col = MagicMock()
        mock_col.path_in_schema = ["col1"]
        mock_col.total_uncompressed_size = 512
        mock_col.total_compressed_size = 256
        mock_col.num_values = 100
        mock_col.statistics = (
            None  # Disable stats to avoid MagicMock in f-string if format is used
        )
        mock_rg.column_metadata = [mock_col]
        mock_file.metadata.row_group_metadata = [mock_rg]

        mock_pq.ParquetFile.return_value = mock_file

        # Instantiate dialog (mock parent)
        with patch("Data_Processor_Integrated.ctk.CTkToplevel"), patch(
            "Data_Processor_Integrated.ctk.CTkFrame"
        ), patch("Data_Processor_Integrated.ctk.CTkLabel"), patch(
            "Data_Processor_Integrated.ctk.CTkButton"
        ), patch(
            "Data_Processor_Integrated.ctk.CTkTextbox"
        ) as _, patch(
            "Data_Processor_Integrated.Path"
        ) as mock_path:

            # Configure path mock robustly
            mock_stat = MagicMock()
            mock_stat.st_size = 2048
            mock_path.return_value.stat.return_value = mock_stat
            mock_path.return_value.name = "test.parquet"

            dialog = ParquetAnalyzerDialog(parent=MagicMock())
            # format_file_size mock removed to test real logic

            # Verify results_text exists
            self.assertTrue(hasattr(dialog, "results_text"), "results_text not created")

            dialog.analyze_parquet_file("test.parquet")

            # Check if ParquetFile was called
            self.assertTrue(mock_pq.ParquetFile.called, "ParquetFile not called")

            # Check if results text was updated
            self.assertTrue(dialog.results_text.insert.called)

            # Verify insert was called with expected content
            args = dialog.results_text.insert.call_args[0]
            self.assertIn("=== Parquet File Analysis ===", args[1])


if __name__ == "__main__":
    unittest.main()
