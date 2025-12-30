
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
        self.launcher = UnifiedLauncher()
        # Mock the log method to prevent side effects
        self.launcher.log_area = MagicMock() 

    @patch("subprocess.Popen")
    def test_launch_python_tool(self, mock_popen):
        """Test launching a Python tool."""
        tool = {
            "name": "Test Tool",
            "path": "test_script.py",
            "type": "python",
            "desc": "Test"
        }
        
        # We assume the path exists logic handles itself since we are just testing the Popen call
        # UnifiedLauncher.launch_tool constructs path using REPO_ROOT / tool['path']
        # Popen is called with [sys.executable, str(path)]
        
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
        # MATLAB launching logic in UnifiedLauncher:
        # Popen(cmd, shell=True, ...)
        
        tool = {
            "name": "Matlab Tool",
            "path": "script.m",
            "type": "matlab",
            "desc": "Test"
        }
        
        self.launcher.launch_tool(tool)
        mock_popen.assert_called_once()
        args = mock_popen.call_args[0][0]
        # Since it uses shell=True, args is the command string
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
        # For bat, Popen([str(path)], shell=True)
        # arg[0] should be the list containing path string
        self.assertTrue("script.bat" in str(args[0]))

if __name__ == "__main__":
    unittest.main()
