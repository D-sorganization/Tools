import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def mock_tkinter():
    """
    Fixture to patch tkinter modules to avoid GUI creation during tests.
    """
    # Create mocks for tkinter components
    mock_tk = MagicMock()
    mock_tk.Tk = MagicMock()
    mock_tk.TclError = Exception

    # Widget mocks
    mock_tk.Label = MagicMock()
    mock_tk.Frame = MagicMock()
    mock_tk.Button = MagicMock()
    mock_tk.Canvas = MagicMock()
    mock_tk.Entry = MagicMock()
    mock_tk.StringVar = MagicMock()
    mock_tk.IntVar = MagicMock()
    mock_tk.BooleanVar = MagicMock()
    mock_tk.Menu = MagicMock()
    mock_tk.Scrollbar = MagicMock()
    mock_tk.Listbox = MagicMock()
    mock_tk.Text = MagicMock()
    mock_tk.Checkbutton = MagicMock()
    mock_tk.Radiobutton = MagicMock()
    mock_tk.Scale = MagicMock()
    mock_tk.Spinbox = MagicMock()
    mock_tk.LabelFrame = MagicMock()

    # Layout managers
    mock_tk.pack = MagicMock()
    mock_tk.grid = MagicMock()
    mock_tk.place = MagicMock()

    # Mock ttk
    mock_ttk = MagicMock()
    mock_ttk.Notebook = MagicMock()
    mock_ttk.Frame = MagicMock()
    mock_ttk.Scrollbar = MagicMock()
    mock_ttk.Style = MagicMock()
    mock_ttk.Treeview = MagicMock()
    mock_ttk.Combobox = MagicMock()
    mock_ttk.Progressbar = MagicMock()
    mock_ttk.Separator = MagicMock()
    mock_ttk.Sizegrip = MagicMock()
    mock_ttk.Label = MagicMock()
    mock_ttk.Button = MagicMock()
    mock_ttk.Entry = MagicMock()

    # Mock messagebox
    mock_messagebox = MagicMock()
    mock_messagebox.showinfo = MagicMock()
    mock_messagebox.showwarning = MagicMock()
    mock_messagebox.showerror = MagicMock()

    # Patch sys.modules
    with patch.dict(
        sys.modules,
        {
            "tkinter": mock_tk,
            "tkinter.ttk": mock_ttk,
            "tkinter.messagebox": mock_messagebox,
            "tkinter.filedialog": MagicMock(),
            "tkinter.colorchooser": MagicMock(),
            "tkinter.font": MagicMock(),
        },
    ):
        yield mock_tk


class TestToolsLauncher:
    """Test suite for ToolsLauncher."""

    @pytest.fixture(autouse=True)
    def setup_launcher(self, mock_tkinter):
        """Setup launcher module with mocked tkinter."""
        # Ensure fresh import to pick up mocks
        if "tools_launcher" in sys.modules:
            del sys.modules["tools_launcher"]
        import tools_launcher

        self.module = tools_launcher
        self.launcher_class = tools_launcher.ToolsLauncher
        yield

    def test_initialization(self):
        """Test that the launcher initializes correctly."""
        launcher = self.launcher_class()

        # Verify window setup
        launcher.root.title.assert_called_with("🧰 Tools Launcher")
        launcher.root.geometry.assert_called_with("900x700")

    def test_pdf_renamer_launch_logic(self):
        """Test launching the PDF Renamer."""
        launcher = self.launcher_class()

        with patch("subprocess.Popen") as mock_popen:
            # Mock Path.exists to return True
            with patch.object(Path, "exists", return_value=True):
                launcher.launch_pdf_renamer()

                # Verify subprocess was called
                mock_popen.assert_called_once()
                args = mock_popen.call_args[0][0]

                # Check command structure
                assert args[0] == sys.executable
                assert "launch_pdf_gui.py" in str(args[1])
                assert "document_processing" in str(args[1])
                assert "pdf_renamer" in str(args[1])

    def test_pdf_renamer_missing(self):
        """Test error handling when PDF Renamer is missing."""
        launcher = self.launcher_class()

        # We must patch the messagebox imported in the module under test
        with patch("subprocess.Popen") as mock_popen, patch(
            "tools_launcher.messagebox"
        ) as mock_mb:

            # Mock Path.exists to return False
            with patch.object(Path, "exists", return_value=False):
                launcher.launch_pdf_renamer()

                # Verify subprocess was NOT called
                mock_popen.assert_not_called()

                # Verify error message was shown
                mock_mb.showerror.assert_called_with("Error", "PDF Renamer not found")

    def test_launch_integrated_processor(self):
        """Test launching the Integrated Data Processor."""
        launcher = self.launcher_class()

        with patch("subprocess.Popen") as mock_popen:
            with patch.object(Path, "exists", return_value=True):
                launcher.launch_integrated_processor()
                mock_popen.assert_called_once()
                args = mock_popen.call_args[0][0]
                assert "launch_integrated.py" in str(args[1])

    def test_launch_solar_system(self):
        """Test launching the Solar System Model."""
        launcher = self.launcher_class()

        with patch("subprocess.Popen") as mock_popen:
            with patch.object(Path, "exists", return_value=True):
                launcher.launch_solar_system()
                mock_popen.assert_called_once()
                args = mock_popen.call_args[0][0]
                assert "solar_system" in str(args[1])

    def test_tab_creation(self):
        """Test that tabs are created."""
        launcher = self.launcher_class()
        # Tabs: Data, Folder, Media, Web, Document, Scientific Modeling, Utilities
        expected_tabs = [
            "Data",
            "Folder",
            "Media",
            "Web",
            "Document",
            "Scientific Modeling",
            "Utilities",
        ]
        # Check that notebook.add was called once for each expected tab.
        actual_call_count = launcher.notebook.add.call_count
        assert actual_call_count == len(expected_tabs), (
            f"Expected {len(expected_tabs)} tabs {expected_tabs}, "
            f"but notebook.add was called {actual_call_count} times."
        )

    def test_rrt_path_planner_info(self):
        """Test RRT Path Planner info dialog."""
        launcher = self.launcher_class()

        # Call show_matlab_info directly as if button was clicked
        launcher.show_matlab_info("RRT Path Planner")

        # Verify messagebox
        import tkinter.messagebox as mbox

        mbox.showinfo.assert_called()
        args = mbox.showinfo.call_args[0]
        assert "RRT Path Planner" in args[0]
        assert "MATLAB" in args[1]
