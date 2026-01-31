# Add the project directory to path
import importlib
import os
import sys
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

project_root = os.path.abspath(
    Path(
        Path(__file__).parent,
        "../../src/data_processing/data_processor/python/data_processor",
    )
)
sys.path.insert(0, project_root)


# Define mock classes
class MockCTkToplevel:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def transient(self, master: Any) -> None:
        pass

    def grab_set(self) -> None:
        pass

    def geometry(self, g: Any) -> None:
        pass

    def resizable(self, w: Any, h: Any) -> None:
        pass

    def title(self, t: Any) -> None:
        pass

    def destroy(self) -> None:
        pass

    def wait_window(self) -> None:
        pass

    def lift(self) -> None:
        pass

    def attributes(self, *args: Any) -> None:
        pass


class MockScrollableFrame:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __getattr__(self, name: str) -> MagicMock:
        return MagicMock()

    def grid(self, *args: Any, **kwargs: Any) -> None:
        pass

    def pack(self, *args: Any, **kwargs: Any) -> None:
        pass


mock_ctk = MagicMock()
mock_ctk.CTkToplevel = MockCTkToplevel
mock_ctk.CTkFrame = MagicMock(side_effect=lambda *args, **kwargs: MagicMock())
mock_ctk.CTkLabel = MagicMock(side_effect=lambda *args, **kwargs: MagicMock())
mock_ctk.CTkButton = MagicMock(side_effect=lambda *args, **kwargs: MagicMock())
mock_ctk.CTkTextbox = MagicMock(side_effect=lambda *args, **kwargs: MagicMock())
mock_ctk.CTkScrollableFrame = MagicMock(side_effect=lambda *args, **kwargs: MagicMock())

mock_pa = MagicMock()
mock_pa.__version__ = "14.0.0"


class MockDataType:
    pass


mock_pa.DataType = MockDataType
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

mock_r0 = MagicMock()


class MockCSVProcessorApp:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.main_tab_view = MagicMock()
        self.main_tab_view.tab = MagicMock()

    def title(self, t: Any) -> None:
        pass

    def geometry(self, g: Any) -> None:
        pass

    def update_idletasks(self) -> None:
        pass

    def winfo_screenwidth(self) -> int:
        return 1920

    def winfo_screenheight(self) -> int:
        return 1080

    def bind(self, *args: Any) -> None:
        pass

    def protocol(self, *args: Any) -> None:
        pass

    def create_status_bar(self) -> None:
        self.status_label = MagicMock()

    def create_setup_and_process_tab(self, tab: Any) -> None:
        pass

    def create_plotting_tab(self, tab: Any) -> None:
        pass

    def create_plots_list_tab(self, tab: Any) -> None:
        pass

    def create_dat_import_tab(self, tab: Any) -> None:
        pass

    def create_help_tab(self, tab: Any) -> None:
        pass

    def _on_closing(self) -> None:
        pass

    def _on_window_configure(self, event: Any) -> None:
        pass

    def _load_plots_from_file(self) -> None:
        pass

    def grid_rowconfigure(self, *args: Any) -> None:
        pass

    def grid_columnconfigure(self, *args: Any) -> None:
        pass

    def _create_splitter(self, *args: Any) -> MagicMock:
        return MagicMock()


mock_r0.CSVProcessorApp = MockCSVProcessorApp


class TestIntegratedCSVProcessorApp(unittest.TestCase):
    """Test suite for the Integrated CSV Processor Application."""

    modules_patcher: Any
    IntegratedCSVProcessorApp: Any
    SplitConfig: Any
    SplitMethod: Any
    FileFormatDetector: Any
    module: Any

    @classmethod
    def setUpClass(cls) -> None:
        # Create the patcher
        cls.modules_patcher = patch.dict(
            sys.modules,
            {
                "customtkinter": mock_ctk,
                "tkinter": MagicMock(),
                "matplotlib": MagicMock(),
                "matplotlib.pyplot": MagicMock(),
                "matplotlib.backends.backend_tkagg": MagicMock(),
                "PIL": MagicMock(),
                "pyarrow": mock_pa,
                "pyarrow.compute": MagicMock(),
                "pyarrow.parquet": MagicMock(),
                "pyarrow.csv": MagicMock(),
                "tables": MagicMock(),
                "Data_Processor_r0": mock_r0,
            },
        )
        cls.modules_patcher.start()

        # Import the module under test
        import Data_Processor_Integrated

        # Reload to ensure it uses the mocked modules
        importlib.reload(Data_Processor_Integrated)

        cls.module = Data_Processor_Integrated
        cls.IntegratedCSVProcessorApp = (
            Data_Processor_Integrated.IntegratedCSVProcessorApp
        )
        cls.SplitConfig = Data_Processor_Integrated.SplitConfig
        cls.SplitMethod = Data_Processor_Integrated.SplitMethod
        cls.FileFormatDetector = Data_Processor_Integrated.FileFormatDetector

    @classmethod
    def tearDownClass(cls) -> None:
        cls.modules_patcher.stop()
        # Remove the module from sys.modules so subsequent tests don't use the mocked version
        if "Data_Processor_Integrated" in sys.modules:
            del sys.modules["Data_Processor_Integrated"]

    @patch("Data_Processor_Integrated.ctk.CTk")
    def test_initialization(self, mock_ctk_ctor: MagicMock) -> None:
        """Test application initialization."""
        # Base class is now our MockCSVProcessorApp
        app = self.IntegratedCSVProcessorApp()

        # Check if converter variables are initialized
        self.assertTrue(hasattr(app, "converter_split_config"))
        self.assertIsInstance(app.converter_split_config, self.SplitConfig)
        self.assertTrue(hasattr(app, "converter_input_files"))
        self.assertEqual(app.converter_input_files, [])

    def test_split_config_defaults(self) -> None:
        """Test default values of SplitConfig."""
        config = self.SplitConfig()
        self.assertFalse(config.enabled)
        self.assertEqual(config.method, self.SplitMethod.ROWS)
        self.assertEqual(config.rows_per_file, 100000)
        self.assertEqual(config.compression, "snappy")

    @patch("pathlib.Path.exists", return_value=True)
    def test_file_format_detector(self, mock_exists: MagicMock) -> None:
        """Test file format detection."""
        detector = self.FileFormatDetector()

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
        # Use the class from the loaded module
        ParquetAnalyzerDialog = self.module.ParquetAnalyzerDialog

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
        mock_col.statistics = None
        mock_rg.column_metadata = [mock_col]
        mock_file.metadata.row_group_metadata = [mock_rg]

        mock_pq.ParquetFile.return_value = mock_file

        # Instantiate dialog (mock parent)
        with (
            patch("Data_Processor_Integrated.ctk.CTkToplevel"),
            patch("Data_Processor_Integrated.ctk.CTkFrame"),
            patch("Data_Processor_Integrated.ctk.CTkLabel"),
            patch("Data_Processor_Integrated.ctk.CTkButton"),
            patch("Data_Processor_Integrated.ctk.CTkTextbox"),
            patch("Data_Processor_Integrated.Path") as mock_path,
        ):
            # Configure path mock robustly
            mock_stat = MagicMock()
            mock_stat.st_size = 2048
            mock_path.return_value.stat.return_value = mock_stat
            mock_path.return_value.name = "test.parquet"

            dialog = ParquetAnalyzerDialog(parent=MagicMock())

            self.assertTrue(hasattr(dialog, "results_text"), "results_text not created")

            dialog.analyze_parquet_file("test.parquet")

            self.assertTrue(mock_pq.ParquetFile.called, "ParquetFile not called")
            self.assertTrue(dialog.results_text.insert.called)

            args = dialog.results_text.insert.call_args[0]
            self.assertIn("=== Parquet File Analysis ===", args[1])


if __name__ == "__main__":
    unittest.main()
