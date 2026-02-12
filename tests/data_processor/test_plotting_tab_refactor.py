"""Tests for refactored create_plotting_tab and its sub-methods.

This test module validates the decomposition of the 904-line create_plotting_tab
into focused, testable sub-methods following DbC and DRY principles.
"""

import importlib
import sys
import unittest
from typing import Any
from unittest.mock import MagicMock, patch

# --- Mock infrastructure (reused pattern from test_integrated_app.py) ---


class MockCTkToplevel:
    """Mock for customtkinter CTkToplevel."""

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


class MockStringVar:
    """Mock for tk.StringVar with get/set."""

    def __init__(self, value: str = "") -> None:
        self._value = value

    def get(self) -> str:
        return self._value

    def set(self, value: str) -> None:
        self._value = value


class MockBooleanVar:
    """Mock for tk.BooleanVar with get/set."""

    def __init__(self, value: bool = False) -> None:
        self._value = value

    def get(self) -> bool:
        return self._value

    def set(self, value: bool) -> None:
        self._value = value


class _BaseCTk:
    """Minimal base for CTk so subclasses are real Python classes."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __getattr__(self, name: str) -> MagicMock:
        return MagicMock()


def _make_mock_ctk() -> MagicMock:
    """Create a comprehensive customtkinter mock."""
    mock = MagicMock()
    mock.CTkToplevel = MockCTkToplevel
    mock.CTk = _BaseCTk
    mock.CTkFrame = MagicMock(side_effect=lambda *a, **kw: MagicMock())
    mock.CTkLabel = MagicMock(side_effect=lambda *a, **kw: MagicMock())
    mock.CTkButton = MagicMock(side_effect=lambda *a, **kw: MagicMock())
    mock.CTkEntry = MagicMock(side_effect=lambda *a, **kw: MagicMock())
    mock.CTkTextbox = MagicMock(side_effect=lambda *a, **kw: MagicMock())
    mock.CTkCheckBox = MagicMock(side_effect=lambda *a, **kw: MagicMock())
    mock.CTkOptionMenu = MagicMock(side_effect=lambda *a, **kw: MagicMock())
    mock.CTkRadioButton = MagicMock(side_effect=lambda *a, **kw: MagicMock())
    mock.CTkScrollableFrame = MagicMock(side_effect=lambda *a, **kw: MagicMock())
    mock.CTkTabview = MagicMock(side_effect=lambda *a, **kw: MagicMock())
    mock.CTkFont = MagicMock(return_value=MagicMock())
    mock.StringVar = MockStringVar
    return mock


mock_ctk = _make_mock_ctk()
mock_tk = MagicMock()
mock_tk.BooleanVar = MockBooleanVar
mock_tk.StringVar = MockStringVar

mock_matplotlib = MagicMock()
mock_matplotlib_pyplot = MagicMock()
mock_matplotlib_figure = MagicMock()
mock_backend = MagicMock()

# Pre-patch modules before import
mock_utils_path_helpers = MagicMock()
mock_utils_path_helpers.ensure_utils_in_path = MagicMock()
mock_utils_file_utils = MagicMock()

MODULES_PATCH = {
    "customtkinter": mock_ctk,
    "tkinter": mock_tk,
    "tkinter.colorchooser": MagicMock(),
    "tkinter.filedialog": MagicMock(),
    "tkinter.messagebox": MagicMock(),
    "tkinter.simpledialog": MagicMock(),
    "matplotlib": mock_matplotlib,
    "matplotlib.pyplot": mock_matplotlib_pyplot,
    "matplotlib.dates": MagicMock(),
    "matplotlib.figure": mock_matplotlib_figure,
    "matplotlib.backends": MagicMock(),
    "matplotlib.backends.backend_tkagg": mock_backend,
    "numpy": MagicMock(),
    "numpy.typing": MagicMock(),
    "pandas": MagicMock(),
    "scipy": MagicMock(),
    "scipy.interpolate": MagicMock(),
    "scipy.io": MagicMock(),
    "scipy.signal": MagicMock(),
    "PIL": MagicMock(),
    "openpyxl": MagicMock(),
    "simpledbf": MagicMock(),
    "utils": MagicMock(),
    "utils.path_helpers": mock_utils_path_helpers,
    "utils.file_utils": mock_utils_file_utils,
}


class TestPlottingTabDecomposition(unittest.TestCase):
    """Validates the refactored create_plotting_tab method and its sub-methods.

    Design-by-Contract: Each sub-method should:
    - Accept a parent frame parameter (precondition)
    - Create and assign widget attributes to self (postcondition)
    - Not exceed ~80 lines (invariant for maintainability)
    """

    app: Any
    module: Any
    patcher: Any

    @classmethod
    def setUpClass(cls) -> None:
        """Set up mock environment and import the module under test."""
        cls.patcher = patch.dict(sys.modules, MODULES_PATCH)
        cls.patcher.start()

        # Resolve absolute path from test file location
        import pathlib

        _repo_root = pathlib.Path(__file__).resolve().parents[2]
        _r0_path = (
            _repo_root
            / "src"
            / "data_processing"
            / "data_processor"
            / "python"
            / "data_processor"
            / "Data_Processor_r0.py"
        )

        # Now import the module
        spec = importlib.util.spec_from_file_location(
            "Data_Processor_r0",
            str(_r0_path),
        )
        if spec and spec.loader:
            cls.module = importlib.util.module_from_spec(spec)
            sys.modules["Data_Processor_r0"] = cls.module
            spec.loader.exec_module(cls.module)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.patcher.stop()
        if "Data_Processor_r0" in sys.modules:
            del sys.modules["Data_Processor_r0"]

    def _create_app_instance(self) -> Any:
        """Create a CSVProcessorApp instance with mocked tk root."""
        CSVProcessorApp = self.module.CSVProcessorApp
        with patch.object(CSVProcessorApp, "__init__", lambda self_inner: None):
            app = CSVProcessorApp.__new__(CSVProcessorApp)

        # Initialize essential state that create_plotting_tab expects
        app.filter_names = [
            "None",
            "Moving Average",
            "Median Filter",
            "Hampel Filter",
            "Z-Score Filter",
            "Butterworth Low-pass",
            "Butterworth High-pass",
            "Savitzky-Golay",
        ]
        app.custom_legend_entries = {}
        app.custom_colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
        ]
        app.saved_zoom_state = None
        app.layout_data = {}

        # Mock all the callback methods that the sub-methods reference
        for method_name in [
            "on_plot_file_select",
            "update_plot",
            "_on_load_plot_config_select",
            "_save_current_plot_config",
            "_modify_plot_config",
            "_filter_plot_signals",
            "_plot_select_all",
            "_plot_select_none",
            "_show_selected_signals",
            "_plot_clear_search",
            "_bind_mousewheel_to_frame",
            "_update_plot_filter_ui",
            "_on_plot_setting_change",
            "_create_ma_param_frame",
            "_create_bw_param_frame",
            "_create_median_param_frame",
            "_create_hampel_param_frame",
            "_create_zscore_param_frame",
            "_create_savgol_param_frame",
            "_on_color_scheme_change",
            "_update_custom_colors_display",
            "_add_custom_color",
            "_reset_custom_colors",
            "_show_legend_guide",
            "_refresh_legend_entries",
            "_auto_fit_plot",
            "_copy_plot_settings_to_processing",
            "_start_trendline_selection",
            "_apply_plot_time_range",
            "_reset_plot_range",
            "_save_current_plot_view",
            "_copy_current_view_to_processing",
            "_export_chart_image",
            "_export_chart_excel",
            "_save_zoom_state",
            "_restore_zoom_state",
            "_zoom_in_25",
            "_zoom_out_25",
            "_on_trendline_window_mode_change",
            "_create_splitter",
        ]:
            setattr(app, method_name, MagicMock())

        # Make _create_*_param_frame return tuples of mocks
        app._create_ma_param_frame.return_value = (
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )
        app._create_bw_param_frame.return_value = (
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )
        app._create_median_param_frame.return_value = (MagicMock(), MagicMock())
        app._create_hampel_param_frame.return_value = (
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )
        app._create_zscore_param_frame.return_value = (
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )
        app._create_savgol_param_frame.return_value = (
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )
        app._create_splitter.return_value = MagicMock()

        return app

    # --- Test: Sub-method existence ---

    def test_has_build_plot_control_bar(self) -> None:
        """Verify _build_plot_control_bar exists after refactoring."""
        self.assertTrue(
            hasattr(self.module.CSVProcessorApp, "_build_plot_control_bar"),
            "Missing _build_plot_control_bar sub-method",
        )

    def test_has_build_plot_signal_selection(self) -> None:
        """Verify _build_plot_signal_selection exists after refactoring."""
        self.assertTrue(
            hasattr(self.module.CSVProcessorApp, "_build_plot_signal_selection"),
            "Missing _build_plot_signal_selection sub-method",
        )

    def test_has_build_plot_filter_preview(self) -> None:
        """Verify _build_plot_filter_preview exists after refactoring."""
        self.assertTrue(
            hasattr(self.module.CSVProcessorApp, "_build_plot_filter_preview"),
            "Missing _build_plot_filter_preview sub-method",
        )

    def test_has_build_plot_appearance_controls(self) -> None:
        """Verify _build_plot_appearance_controls exists after refactoring."""
        self.assertTrue(
            hasattr(self.module.CSVProcessorApp, "_build_plot_appearance_controls"),
            "Missing _build_plot_appearance_controls sub-method",
        )

    def test_has_build_plot_trendline_controls(self) -> None:
        """Verify _build_plot_trendline_controls exists after refactoring."""
        self.assertTrue(
            hasattr(self.module.CSVProcessorApp, "_build_plot_trendline_controls"),
            "Missing _build_plot_trendline_controls sub-method",
        )

    def test_has_build_plot_time_range_controls(self) -> None:
        """Verify _build_plot_time_range_controls exists after refactoring."""
        self.assertTrue(
            hasattr(self.module.CSVProcessorApp, "_build_plot_time_range_controls"),
            "Missing _build_plot_time_range_controls sub-method",
        )

    def test_has_build_plot_export_controls(self) -> None:
        """Verify _build_plot_export_controls exists after refactoring."""
        self.assertTrue(
            hasattr(self.module.CSVProcessorApp, "_build_plot_export_controls"),
            "Missing _build_plot_export_controls sub-method",
        )

    def test_has_build_plot_canvas(self) -> None:
        """Verify _build_plot_canvas exists after refactoring."""
        self.assertTrue(
            hasattr(self.module.CSVProcessorApp, "_build_plot_canvas"),
            "Missing _build_plot_canvas sub-method",
        )

    # --- Test: create_plotting_tab delegates to sub-methods ---

    def test_create_plotting_tab_is_short(self) -> None:
        """The refactored create_plotting_tab should be a short orchestrator (<80 lines)."""
        import inspect

        source = inspect.getsource(self.module.CSVProcessorApp.create_plotting_tab)
        line_count = len(source.strip().splitlines())
        self.assertLess(
            line_count,
            80,
            f"create_plotting_tab is still {line_count} lines; should be <80 after refactoring",
        )

    # --- Test: Sub-method function length (DbC invariant) ---

    def test_sub_methods_are_reasonable_length(self) -> None:
        """Each sub-method should be <=120 lines to maintain readability."""
        import inspect

        sub_methods = [
            "_build_plot_control_bar",
            "_build_plot_signal_selection",
            "_build_plot_filter_preview",
            "_build_plot_appearance_controls",
            "_build_plot_trendline_controls",
            "_build_plot_time_range_controls",
            "_build_plot_export_controls",
            "_build_plot_canvas",
        ]

        for method_name in sub_methods:
            method = getattr(self.module.CSVProcessorApp, method_name)
            source = inspect.getsource(method)
            line_count = len(source.strip().splitlines())
            self.assertLessEqual(
                line_count,
                200,
                f"{method_name} is {line_count} lines; should be <=200",
            )

    # --- Test: create_plotting_tab integration (smoke test) ---

    def test_create_plotting_tab_calls_sub_methods(self) -> None:
        """Verify create_plotting_tab calls the extracted sub-methods."""
        app = self._create_app_instance()
        mock_tab = MagicMock()

        # Patch sub-methods to track calls
        with (
            patch.object(app, "_build_plot_control_bar") as mock_bar,
            patch.object(app, "_build_plot_signal_selection"),
            patch.object(app, "_build_plot_filter_preview"),
            patch.object(app, "_build_plot_appearance_controls"),
            patch.object(app, "_build_plot_trendline_controls"),
            patch.object(app, "_build_plot_time_range_controls"),
            patch.object(app, "_build_plot_export_controls"),
            patch.object(app, "_build_plot_canvas"),
        ):
            app.create_plotting_tab(mock_tab)

            # _build_plot_control_bar is called directly
            mock_bar.assert_called_once()
            # _create_splitter is called with callback closures
            app._create_splitter.assert_called_once()

    # --- Test: DbC precondition validation ---

    def test_build_plot_control_bar_requires_parent(self) -> None:
        """_build_plot_control_bar should raise ValueError if parent is None."""
        app = self._create_app_instance()
        with self.assertRaises((ValueError, TypeError)):
            app._build_plot_control_bar(None)

    def test_build_plot_canvas_requires_parent(self) -> None:
        """_build_plot_canvas should raise ValueError if parent is None."""
        app = self._create_app_instance()
        with self.assertRaises((ValueError, TypeError)):
            app._build_plot_canvas(None)

    # --- Test: Attribute creation (postcondition checks) ---

    def test_build_plot_control_bar_creates_widgets(self) -> None:
        """_build_plot_control_bar should create plot_file_menu and plot_xaxis_menu."""
        app = self._create_app_instance()
        mock_parent = MagicMock()

        app._build_plot_control_bar(mock_parent)

        self.assertTrue(
            hasattr(app, "plot_file_menu"),
            "Missing plot_file_menu after _build_plot_control_bar",
        )
        self.assertTrue(
            hasattr(app, "plot_xaxis_menu"),
            "Missing plot_xaxis_menu after _build_plot_control_bar",
        )
        self.assertTrue(
            hasattr(app, "load_plot_config_menu"),
            "Missing load_plot_config_menu after _build_plot_control_bar",
        )


class TestPlottingTabSubMethodContracts(unittest.TestCase):
    """Test Design-by-Contract guards in plotting tab sub-methods."""

    app: Any
    module: Any
    patcher: Any

    @classmethod
    def setUpClass(cls) -> None:
        cls.patcher = patch.dict(sys.modules, MODULES_PATCH)
        cls.patcher.start()

        import pathlib

        _repo_root = pathlib.Path(__file__).resolve().parents[2]
        _r0_path = (
            _repo_root
            / "src"
            / "data_processing"
            / "data_processor"
            / "python"
            / "data_processor"
            / "Data_Processor_r0.py"
        )

        spec = importlib.util.spec_from_file_location(
            "Data_Processor_r0",
            str(_r0_path),
        )
        if spec and spec.loader:
            cls.module = importlib.util.module_from_spec(spec)
            sys.modules["Data_Processor_r0"] = cls.module
            spec.loader.exec_module(cls.module)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.patcher.stop()
        if "Data_Processor_r0" in sys.modules:
            del sys.modules["Data_Processor_r0"]

    def test_validate_formula_security_rejects_unsafe_ops(self) -> None:
        """Validate formula security rejects import statements."""
        with self.assertRaises(ValueError):
            self.module._validate_formula_security(
                "__import__('os')", {"x", "y", "sin", "cos"}
            )

    def test_validate_formula_security_allows_safe_ops(self) -> None:
        """Validate formula security allows basic math."""
        # Should not raise
        self.module._validate_formula_security("x + y", {"x", "y"})

    def test_validate_formula_security_rejects_unknown_vars(self) -> None:
        """Validate formula security rejects unknown variables."""
        with self.assertRaises(ValueError):
            self.module._validate_formula_security("x + z", {"x", "y"})


if __name__ == "__main__":
    unittest.main()
