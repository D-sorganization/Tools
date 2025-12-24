
import sys
import os
import subprocess
import webbrowser
import platform
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QTabWidget, 
                            QScrollArea, QFrame, QGridLayout, QTextEdit, QMessageBox)
from PyQt6.QtCore import Qt, QSize, QProcess
from PyQt6.QtGui import QIcon, QFont, QPixmap, QColor, QPalette

# =============================================================================
# CONFIGURATION & PATHS
# =============================================================================
REPO_ROOT = Path(__file__).parent.absolute()

# Tool Definitions
# Format: "Name": {"path": relative_path, "type": "python"|"matlab"|"web"|"browser"|"bat", "desc": "Description"}
TOOLS = {
    "Media Processing": [
        {
            "name": "Audio Processor (Main)",
            "path": "media_processing/audio_processor/matlab/audio_signal_processor/launch_audio_processor_pro.m",
            "type": "matlab",
            "desc": "Advanced Audio Signal Processing Suite (MATLAB)"
        },
        {
            "name": "Audio Processor (Replicant)",
            "path": "replicants/matlab/audio_signal_processor/launch_audio_processor_pro.m",
            "type": "matlab",
            "desc": "Legacy/Backup Audio Implementation"
        },
        {
            "name": "Video Processor Platform",
            "path": "media_processing/video_processor/apps/web/launch_platform.bat",
            "type": "bat",
            "desc": "Next.js Video Processing Web Platform"
        }
    ],
    "Data Processing": [
        {
            "name": "Data Processor Integrated",
            "path": "data_processing/data_processor/python/data_processor/launch_integrated.py",
            "type": "python",
            "desc": "Time Series CSV/Parquet Analyzer & Converter"
        },
        {
            "name": "Data Processor (Replicant)",
            "path": "data_processing/data_processor/archive/Data_Processor_r0.py",
            "type": "python",
            "desc": "Original Revision 0 Data Processor"
        }
    ],
    "Scientific Modeling": [
        {
            "name": "Solar System Model",
            "path": "scientific_modeling/solar_system_model/launch_solar_system.py",
            "type": "python",
            "desc": "Interactive 3D Solar System Simulation"
        },
        {
            "name": "RRT Path Planner",
            "path": "scientific_modeling/rrt_path_planner/matlab/src/gui/starWarsPathPlannerGUI.m",
            "type": "matlab",
            "desc": "Rapidly-exploring Random Tree (RRT) Navigator (MATLAB)"
        }
    ],
    "Web Applications": [
        {
            "name": "Calculator App",
            "path": "web_applications/calculator/webapp.py",
            "type": "python",
            "desc": "Flask-based Scientific Calculator"
        },
        {
            "name": "Unit Converter",
            "path": "web_applications/unit_converter/unit-converter-app/index.html",
            "type": "browser",
            "desc": "Browser-based Unit Conversion Tool"
        }
    ],
    "Development Tools": [
        {
            "name": "Folder Packer Pro",
            "path": "development_tools/folder_tools/folder_packer_pro/folder_packer_pro.py",
            "type": "python",
            "desc": "Project Archiving and Distribution Tool"
        },
        {
            "name": "Folder Tool (Utility)",
            "path": "development_tools/folder_tools/folder_tool/Folders_Tool_r0.py",
            "type": "python",
            "desc": "Directory Management Utility"
        }
    ]
}

# =============================================================================
# STYLING
# =============================================================================
STYLE_SHEET = """
QMainWindow {
    background-color: #1a1b26;
}
QTabWidget::pane {
    border: 1px solid #414868;
    background-color: #1a1b26;
    border-radius: 6px;
}
QTabBar::tab {
    background-color: #24283b;
    color: #c0caf5;
    padding: 10px 20px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background-color: #7aa2f7;
    color: #1a1b26;
    font-weight: bold;
}
QGroupBox {
    border: 1px solid #414868;
    border-radius: 6px;
    margin-top: 20px;
    background-color: #24283b;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
    color: #7aa2f7;
    font-weight: bold;
}
QPushButton {
    background-color: #7aa2f7;
    color: #1a1b26;
    border-radius: 4px;
    padding: 8px;
    font-weight: bold;
    text-align: left;
}
QPushButton:hover {
    background-color: #bb9af7;
}
QPushButton:pressed {
    background-color: #7dcfff;
}
QLabel {
    color: #c0caf5;
}
QLabel#DescLabel {
    color: #565f89;
    font-style: italic;
    font-size: 11px;
}
QTextEdit {
    background-color: #0f0f14;
    color: #9ece6a;
    border: 1px solid #414868;
    border-radius: 4px;
    font-family: Consolas, monospace;
}
"""

# =============================================================================
# LAUNCHER LOGIC
# =============================================================================
class ToolCard(QFrame):
    def __init__(self, tool_info, launch_callback):
        super().__init__()
        self.tool_info = tool_info
        self.launch_callback = launch_callback
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Check if exists
        full_path = REPO_ROOT / self.tool_info['path']
        exists = full_path.exists()
        
        # Button
        btn_text = f"🚀 {self.tool_info['name']}"
        self.btn = QPushButton(btn_text)
        self.btn.clicked.connect(lambda: self.launch_callback(self.tool_info))
        self.btn.setEnabled(exists)
        
        if not exists:
            self.btn.setStyleSheet("background-color: #f7768e; color: #1a1b26;")
            self.btn.setText(f"❌ {self.tool_info['name']} (Missing)")
            self.btn.setToolTip(f"File not found: {full_path}")
            
        layout.addWidget(self.btn)
        
        # Description
        desc = QLabel(self.tool_info['desc'])
        desc.setObjectName("DescLabel")
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        # Path
        path_lbl = QLabel(str(self.tool_info['path']))
        path_lbl.setStyleSheet("color: #414868; font-size: 10px;")
        layout.addWidget(path_lbl)

class UnifiedLauncher(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Antigravity Unified Tools Launcher")
        self.resize(1000, 700)
        self.setStyleSheet(STYLE_SHEET)
        
        # Icon
        icon_path = REPO_ROOT / "tools_icon.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        
        self.setup_ui()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Header
        header = QLabel("🛠️ Unified Tools Repository")
        header.setStyleSheet("font-size: 24px; font-weight: bold; color: #7aa2f7; margin-bottom: 10px;")
        main_layout.addWidget(header)
        
        # Tabs
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        for category, tools in TOOLS.items():
            tab = QWidget()
            self.setup_category_tab(tab, tools)
            self.tabs.addTab(tab, category)
            
        # Status Log
        log_group = QFrame()
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(0, 10, 0, 0)
        
        lbl = QLabel("Activity Log")
        lbl.setStyleSheet("font-weight: bold;")
        log_layout.addWidget(lbl)
        
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMaximumHeight(150)
        log_layout.addWidget(self.log_area)
        
        main_layout.addWidget(log_group)

    def setup_category_tab(self, tab, tools):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background-color: transparent; border: none;")
        
        content_widget = QWidget()
        layout = QGridLayout(content_widget)
        layout.setSpacing(15)
        
        row = 0
        col = 0
        max_cols = 3
        
        for tool in tools:
            card = ToolCard(tool, self.launch_tool)
            layout.addWidget(card, row, col)
            
            col += 1
            if col >= max_cols:
                col = 0
                row += 1
                
        # Push content up
        layout.setRowStretch(row + 1, 1)
        
        scroll.setWidget(content_widget)
        
        tab_layout = QVBoxLayout(tab)
        tab_layout.addWidget(scroll)

    def log(self, message):
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_area.append(f"[{timestamp}] {message}")
        cursor = self.log_area.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.log_area.setTextCursor(cursor)

    def launch_tool(self, tool_info):
        path = REPO_ROOT / tool_info['path']
        type_ = tool_info['type']
        
        self.log(f"Launching {tool_info['name']}...")
        self.log(f"Path: {path}")
        
        try:
            if type_ == "python":
                subprocess.Popen([sys.executable, str(path)], cwd=path.parent)
                self.log("✅ Process started (Python)")
                
            elif type_ == "matlab":
                self.log("ℹ️ Attempting to launch MATLAB...")
                # Try to find MATLAB executable or use 'matlab' from PATH
                cmd = f"matlab -nosplash -nodesktop -r \"run('{str(path)}');\""
                # For non-blocking, we might just open the file if automation fails
                try:
                    subprocess.Popen(cmd, shell=True, cwd=path.parent)
                    self.log("✅ MATLAB command sent")
                except:
                    os.startfile(path)
                    self.log("⚠️ Executable not found, opened file in default editor")
                    
            elif type_ == "web" or type_ == "browser":
                webbrowser.open(path.as_uri())
                self.log("✅ Opened in default browser")
                
            elif type_ == "bat":
                subprocess.Popen([str(path)], shell=True, cwd=path.parent)
                self.log("✅ Batch script executed")
                
            else:
                self.log(f"❌ Unknown type: {type_}")
                
        except Exception as e:
            self.log(f"❌ Error: {str(e)}")

# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set app style
    app.setStyle("Fusion")
    
    # Run
    window = UnifiedLauncher()
    window.show()
    
    sys.exit(app.exec())
