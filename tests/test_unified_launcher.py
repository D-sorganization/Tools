
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Patch PyQt6 modules BEFORE they are imported by UnifiedToolsLauncher
# We need to make sure QMainWindow, QWidget, etc., are classes that can be instantiated
# or at least mocked in a way that doesn't crash __init__.

mock_qt_widgets = MagicMock()
mock_qt_gui = MagicMock()
mock_qt_core = MagicMock()

# Create a mock base class for UnifiedLauncher to inherit from if it were using QMainWindow
# However, modifying the class inheritance at runtime is tricky.
# Instead, we will mock the QMainWindow class itself.

class MockQMainWindow:
    def __init__(self, *args, **kwargs):
        pass # Do nothing
    
    def setWindowTitle(self, title):
        pass
        
    def resize(self, w, h):
        pass
        
    def setStyleSheet(self, s):
        pass
        
    def setWindowIcon(self, icon):
        pass
        
    def setCentralWidget(self, w):
        pass
    
    def show(self):
        pass

# Inherited classes must be actual classes or MagicMock class
mock_qt_widgets.QMainWindow = MockQMainWindow
mock_qt_widgets.QFrame = MagicMock 

# Instantiated classes should be MagicMock instances (acting as factories)
# so that ClassName(args) calls the instance and returns a new Mock
mock_qt_widgets.QWidget = MagicMock()
mock_qt_widgets.QVBoxLayout = MagicMock()
mock_qt_widgets.QLabel = MagicMock()
mock_qt_widgets.QTabWidget = MagicMock()
mock_qt_widgets.QTextEdit = MagicMock()
mock_qt_widgets.QScrollArea = MagicMock()
mock_qt_widgets.QGridLayout = MagicMock()
mock_qt_widgets.QPushButton = MagicMock()
mock_qt_widgets.QApplication = MagicMock()

mock_qt_gui.QIcon = MagicMock()

sys.modules["PyQt6"] = MagicMock()
sys.modules["PyQt6.QtGui"] = mock_qt_gui
sys.modules["PyQt6.QtWidgets"] = mock_qt_widgets
sys.modules["PyQt6.QtCore"] = mock_qt_core

# Now we can safely import the module under test
import UnifiedToolsLauncher
from UnifiedToolsLauncher import TOOLS, UnifiedLauncher

class TestUnifiedLauncherConfig(unittest.TestCase):
    """Test the configuration of the Unified Launcher."""

    def test_all_tool_paths_exist(self):
        """Verify that every tool configured in TOOLS exists on disk."""
        missing = []
        for category, tools in TOOLS.items():
            for tool in tools:
                path = REPO_ROOT / tool["path"]
                if not path.exists():
                    missing.append(f"{category} -> {tool['name']}: {path}")
        
        if missing:
            self.fail(f"The following tools have invalid paths:\n" + "\n".join(missing))

class TestUnifiedLauncherLogic(unittest.TestCase):
    """Test the logic of the Unified Launcher."""

    def setUp(self):
        # Patch ToolCard to avoid widget creation during setup_category_tab
        self.tool_card_patcher = patch("UnifiedToolsLauncher.ToolCard")
        self.mock_tool_card = self.tool_card_patcher.start()
        
        # Instantiate launcher
        self.launcher = UnifiedLauncher()
        self.launcher.log_area = MagicMock()

    def tearDown(self):
        self.tool_card_patcher.stop()

    @patch("subprocess.Popen")
    def test_launch_python_tool(self, mock_popen):
        """Test launching a Python tool."""
        tool = {
            "name": "Test Tool",
            "path": "test_script.py",
            "type": "python",
            "desc": "Test"
        }
        
        self.launcher.launch_tool(tool)
             
        mock_popen.assert_called_once()
        args = mock_popen.call_args[0][0]
        self.assertEqual(args[0], sys.executable)
        self.assertTrue("test_script.py" in str(args[1]))

    @patch("webbrowser.open")
    def test_launch_web_tool(self, mock_web):
        """Test launching a Web tool."""
        tool = {
            "name": "Web Tool",
            "path": "index.html",
            "type": "browser",
            "desc": "Test"
        }
        
        self.launcher.launch_tool(tool)
        mock_web.assert_called_once()

    @patch("subprocess.Popen")
    def test_launch_matlab_tool(self, mock_popen):
        """Test launching a MATLAB tool."""
        tool = {
            "name": "Matlab Tool",
            "path": "script.m",
            "type": "matlab",
            "desc": "Test"
        }
        
        # UnifiedLauncher logic does not perform strict path.exists() checks 
        # that prevent launching if path is missing, it relies on Popen execution.
        # But if it does, we can patch Path.exists here if needed, but current code suggests it tries to run.
        
        self.launcher.launch_tool(tool)
        mock_popen.assert_called_once()
        args = mock_popen.call_args[0][0]
        self.assertIn("matlab", args)
        self.assertIn("script.m", args)

    @patch("subprocess.Popen")
    def test_launch_bat_tool(self, mock_popen):
        """Test launching a BATCH tool."""
        tool = {
            "name": "Batch Tool",
            "path": "script.bat",
            "type": "bat",
            "desc": "Test"
        }

        self.launcher.launch_tool(tool)
        
        mock_popen.assert_called_once()
        args = mock_popen.call_args[0][0]
        self.assertTrue("script.bat" in str(args[0]))

if __name__ == "__main__":
    unittest.main()
