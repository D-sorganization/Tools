import sys
import unittest
from unittest.mock import MagicMock, patch

# Add the project directory to path
project_root = r"c:\Users\diete\Repositories\Tools\data_processing\data_processor\python\data_processor"
sys.path.insert(0, project_root)


# Mock dependencies before importing
# Define a real class for CTkToplevel so inheritance works correctly
class MockCTkToplevel:
    def __init__(self, *args, **kwargs):
        pass

    def transient(self, master):
        pass

    def grab_set(self):
        pass

    def geometry(self, g):
        pass

    def resizable(self, w, h):
        pass

    def title(self, t):
        pass

    def destroy(self):
        pass

    def wait_window(self):
        pass

    def lift(self):
        pass

    def attributes(self, *args):
        pass


class MockScrollableFrame:
    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        return MagicMock()

    def grid(self, *args, **kwargs):
        pass

    def pack(self, *args, **kwargs):
        pass


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
    def __init__(self, *args, **kwargs):
        self.main_tab_view = MagicMock()
        self.main_tab_view.tab = MagicMock()

    def title(self, t):
        pass

    def geometry(self, g):
        pass

    def update_idletasks(self):
        pass

    def winfo_screenwidth(self):
        return 1920

    def winfo_screenheight(self):
        return 1080

    def bind(self, *args):
        pass

    def protocol(self, *args):
        pass

    def create_status_bar(self):
        self.status_label = MagicMock()

    def create_setup_and_process_tab(self, tab):
        pass

    def create_plotting_tab(self, tab):
        pass

    def create_plots_list_tab(self, tab):
        pass

    def create_dat_import_tab(self, tab):
        pass

    def create_help_tab(self, tab):
        pass

    def _on_closing(self):
        pass

    def _on_window_configure(self, event):
        pass

    def _load_plots_from_file(self):
        pass

    def grid_rowconfigure(self, *args):
        pass

    def grid_columnconfigure(self, *args):
        pass

    def _create_splitter(self, *args):
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
    @patch("Data_Processor_Integrated.ctk.CTk")
    def test_initialization(self, mock_ctk):
        # Base class is now our MockCSVProcessorApp
        app = IntegratedCSVProcessorApp()

        # Check if converter variables are initialized
        self.assertTrue(hasattr(app, "converter_split_config"))
        self.assertIsInstance(app.converter_split_config, SplitConfig)
        self.assertTrue(hasattr(app, "converter_input_files"))
        self.assertEqual(app.converter_input_files, [])

    def test_split_config_defaults(self):
        config = SplitConfig()
        self.assertFalse(config.enabled)
        self.assertEqual(config.method, SplitMethod.ROWS)
        self.assertEqual(config.rows_per_file, 100000)
        self.assertEqual(config.compression, "snappy")

    @patch("os.path.exists", return_value=True)
    def test_file_format_detector(self, mock_exists):
        detector = FileFormatDetector()

        # Test extension based detection
        self.assertEqual(detector.detect_format("test.csv"), "csv")
        self.assertEqual(detector.detect_format("test.json"), "json")
        self.assertEqual(detector.detect_format("test.parquet"), "parquet")
        self.assertEqual(detector.detect_format("test.xlsx"), "excel")
        # txt falls back to tsv or content check in code?
        self.assertEqual(detector.detect_format("test.txt"), "tsv")

    @patch("Data_Processor_Integrated.pq")
    def test_parquet_analyzer(self, mock_pq):
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
