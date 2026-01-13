#!/usr/bin/env python3
"""
Comprehensive Tools Launcher - Professional tabbed interface for all tools.
This creates a complete launcher system using the new tools_icon.

⚠️  DEPRECATION WARNING ⚠️
This launcher is deprecated. Please use UnifiedToolsLauncher.py instead.
The new launcher provides better performance, modern UI, and improved functionality.
"""

import logging
import os
import subprocess
import sys
import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class ToolsLauncher:
    """Professional Tools Launcher with tabbed interface.

    ⚠️  DEPRECATED: Use UnifiedToolsLauncher.py instead.
    """

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("🧰 Tools Launcher (DEPRECATED)")
        self.root.geometry("900x700")
        self.root.configure(bg="#f0f0f0")

        # Show deprecation warning
        self._show_deprecation_warning()

        # Set the new tools_icon
        self.set_application_icon()

        self.setup_ui()

    def _show_deprecation_warning(self) -> None:
        """Show deprecation warning to users."""
        messagebox.showwarning(
            "Deprecated Launcher",
            "⚠️ This launcher is deprecated!\n\n"
            "Please use 'UnifiedToolsLauncher.py' instead.\n"
            "The new launcher provides:\n"
            "• Better performance\n"
            "• Modern PyQt6 interface\n"
            "• Improved functionality\n\n"
            "This old launcher will be removed in a future version.",
        )

        self.setup_ui()

    def set_application_icon(self) -> None:
        """Set the tools_icon for the application."""
        try:
            # Try different possible locations for the tools_icon
            icon_paths = [
                Path("assets/tools_icon.ico"),
                Path("tools_icon.ico"),
                Path("../assets/tools_icon.ico"),
                Path("../tools_icon.ico"),
                Path("../../assets/tools_icon.ico"),
                Path("../../tools_icon.ico"),
            ]

            for icon_path in icon_paths:
                if icon_path.exists():
                    self.root.iconbitmap(str(icon_path))
                    logger.info(f"✓ Applied tools_icon from: {icon_path}")
                    return

            logger.warning("tools_icon.ico not found - using default icon")

        except Exception as e:
            logger.warning(f"Could not set tools_icon: {e}")

    def setup_ui(self) -> None:
        """Setup the user interface."""
        # Title frame
        title_frame = tk.Frame(self.root, bg="#2c3e50", height=80)
        title_frame.pack(fill="x", padx=0, pady=0)
        title_frame.pack_propagate(False)

        # Title with icon reference
        title_label = tk.Label(
            title_frame,
            text="🧰 Professional Tools Launcher",
            font=("Arial", 24, "bold"),
            bg="#2c3e50",
            fg="white",
        )
        title_label.pack(expand=True)

        subtitle_label = tk.Label(
            title_frame,
            text="Unified access to all development and analysis tools",
            font=("Arial", 12),
            bg="#2c3e50",
            fg="#bdc3c7",
        )
        subtitle_label.pack()

        # Main content frame
        content_frame = tk.Frame(self.root, bg="#f0f0f0")
        content_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Create notebook for tabs
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook.Tab", padding=[20, 10])

        self.notebook = ttk.Notebook(content_frame)
        self.notebook.pack(fill="both", expand=True)

        # Create tabs
        self.create_data_processing_tab()
        self.create_folder_tools_tab()
        self.create_media_tools_tab()
        self.create_web_tools_tab()
        self.create_document_tools_tab()
        self.create_utilities_tab()

        # Status bar
        status_frame = tk.Frame(self.root, bg="#34495e", height=30)
        status_frame.pack(fill="x", side="bottom")
        status_frame.pack_propagate(False)

        self.status_var = tk.StringVar(value="Ready - Select a tool to launch")
        status_label = tk.Label(
            status_frame,
            textvariable=self.status_var,
            bg="#34495e",
            fg="white",
            font=("Arial", 10),
        )
        status_label.pack(side="left", padx=10, pady=5)

        # Version info
        version_label = tk.Label(
            status_frame,
            text="v2.0 - Tools Launcher with Professional Icon",
            bg="#34495e",
            fg="#95a5a6",
            font=("Arial", 9),
        )
        version_label.pack(side="right", padx=10, pady=5)

    def create_data_processing_tab(self) -> None:
        """Create Data Processing tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="📊 Data Processing")

        # Scrollable frame
        canvas = tk.Canvas(frame, bg="white")
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="white")

        scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Header
        header_frame = tk.Frame(scrollable_frame, bg="#3498db", height=60)
        header_frame.pack(fill="x", padx=0, pady=0)
        header_frame.pack_propagate(False)

        tk.Label(
            header_frame,
            text="📊 Data Processing & Analysis Suite",
            font=("Arial", 16, "bold"),
            bg="#3498db",
            fg="white",
        ).pack(expand=True)

        # Description
        desc_frame = tk.Frame(scrollable_frame, bg="#ecf0f1")
        desc_frame.pack(fill="x", padx=0, pady=0)

        desc_text = (
            "The Integrated Data Processor is the flagship tool featuring:\n"
            "• CSV/Excel/JSON data processing and statistical analysis\n"
            "• Multi-format file converter with batch processing\n"
            "• Advanced plotting and data visualization\n"
            "• Folder management and file organization tools\n"
            "• DAT file import with DBF tag support\n"
            "• Professional interface with the new tools icon"
        )

        tk.Label(
            desc_frame,
            text=desc_text,
            font=("Arial", 11),
            justify="left",
            bg="#ecf0f1",
            fg="#2c3e50",
            padx=20,
            pady=15,
        ).pack(fill="x")

        # Tools
        tools: list[tuple[str, str, Callable[[], None], str]] = [
            (
                "🚀 Integrated Data Processor",
                "Complete data processing suite with all features (MAIN TOOL)",
                lambda: self.launch_integrated_processor(),
                "#e74c3c",
            ),
            (
                "📈 CSV Data Processor",
                "Direct access to CSV processing and analysis",
                lambda: self.launch_csv_processor(),
                "#f39c12",
            ),
            (
                "🔄 Format Converter",
                "Batch convert between CSV, Excel, JSON, Parquet formats",
                lambda: self.show_converter_info(),
                "#9b59b6",
            ),
        ]

        for name, desc, command, color in tools:
            self.create_tool_card(
                scrollable_frame, name, desc, command, color, show_icon=True
            )

    def create_folder_tools_tab(self) -> None:
        """Create Folder Tools tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="📁 Folder Tools")

        # Similar structure as data processing
        canvas = tk.Canvas(frame, bg="white")
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="white")

        scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Header
        header_frame = tk.Frame(scrollable_frame, bg="#27ae60", height=60)
        header_frame.pack(fill="x")
        header_frame.pack_propagate(False)

        tk.Label(
            header_frame,
            text="📁 File & Folder Management",
            font=("Arial", 16, "bold"),
            bg="#27ae60",
            fg="white",
        ).pack(expand=True)

        tools: list[tuple[str, str, Callable[[], None], str]] = [
            (
                "📂 Folder Processor",
                "Organize, combine, and manage files and directories",
                lambda: self.launch_folder_tool(),
                "#2ecc71",
            ),
            (
                "🔧 Folder Fix Pro",
                "Advanced folder processing with deduplication",
                lambda: self.launch_folder_fix_pro(),
                "#16a085",
            ),
            (
                "📦 Project Packer",
                "Pack and unpack project files efficiently",
                lambda: self.launch_project_packer(),
                "#1abc9c",
            ),
        ]

        for name, desc, command, color in tools:
            self.create_tool_card(scrollable_frame, name, desc, command, color)

    def create_media_tools_tab(self) -> None:
        """Create Media Tools tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="🎬 Media Processing")

        canvas = tk.Canvas(frame, bg="white")
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="white")

        scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Header
        header_frame = tk.Frame(scrollable_frame, bg="#8e44ad", height=60)
        header_frame.pack(fill="x")
        header_frame.pack_propagate(False)

        tk.Label(
            header_frame,
            text="🎬 Audio & Video Processing",
            font=("Arial", 16, "bold"),
            bg="#8e44ad",
            fg="white",
        ).pack(expand=True)

        tools: list[tuple[str, str, Callable[[], None], str]] = [
            (
                "🎵 Audio Processor (MATLAB)",
                "Advanced audio signal processing and analysis",
                lambda: self.show_matlab_info("Audio Processor"),
                "#9b59b6",
            ),
            (
                "🎥 Video Processor",
                "Video processing, analysis, and conversion tools",
                lambda: self.launch_video_processor(),
                "#8e44ad",
            ),
            (
                "🔊 Audio Tools (Python)",
                "Python-based audio processing utilities",
                lambda: self.show_audio_tools_info(),
                "#663399",
            ),
        ]

        for name, desc, command, color in tools:
            self.create_tool_card(scrollable_frame, name, desc, command, color)

    def create_web_tools_tab(self) -> None:
        """Create Web Tools tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="🌐 Web Applications")

        canvas = tk.Canvas(frame, bg="white")
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="white")

        scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Header
        header_frame = tk.Frame(scrollable_frame, bg="#e67e22", height=60)
        header_frame.pack(fill="x")
        header_frame.pack_propagate(False)

        tk.Label(
            header_frame,
            text="🌐 Web-based Tools & Applications",
            font=("Arial", 16, "bold"),
            bg="#e67e22",
            fg="white",
        ).pack(expand=True)

        tools: list[tuple[str, str, Callable[[], None], str]] = [
            (
                "⚖️ Unit Converter",
                "Professional unit conversion web application",
                lambda: self.launch_unit_converter(),
                "#f39c12",
            ),
            (
                "📊 Web Dashboard",
                "Tool management and monitoring dashboard",
                lambda: self.show_web_dashboard_info(),
                "#e67e22",
            ),
        ]

        for name, desc, command, color in tools:
            self.create_tool_card(scrollable_frame, name, desc, command, color)

    def create_document_tools_tab(self) -> None:
        """Create Document Tools tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="📄 Document Processing")

        canvas = tk.Canvas(frame, bg="white")
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="white")

        scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Header
        header_frame = tk.Frame(scrollable_frame, bg="#e74c3c", height=60)
        header_frame.pack(fill="x")
        header_frame.pack_propagate(False)

        tk.Label(
            header_frame,
            text="📄 Document Processing Tools",
            font=("Arial", 16, "bold"),
            bg="#e74c3c",
            fg="white",
        ).pack(expand=True)

        tools: list[tuple[str, str, Callable[[], None], str]] = [
            (
                "📄 PDF Renamer",
                "Bulk rename PDF files based on metadata with duplicate detection",
                lambda: self.launch_pdf_renamer(),
                "#c0392b",
            ),
        ]

        for name, desc, command, color in tools:
            self.create_tool_card(scrollable_frame, name, desc, command, color)

    def create_utilities_tab(self) -> None:
        """Create Utilities tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="🔧 Utilities")

        canvas = tk.Canvas(frame, bg="white")
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="white")

        scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Header
        header_frame = tk.Frame(scrollable_frame, bg="#34495e", height=60)
        header_frame.pack(fill="x")
        header_frame.pack_propagate(False)

        tk.Label(
            header_frame,
            text="🔧 Development & System Tools",
            font=("Arial", 16, "bold"),
            bg="#34495e",
            fg="white",
        ).pack(expand=True)

        tools: list[tuple[str, str, Callable[[], None], str]] = [
            (
                "🌌 Solar System Model",
                "3D solar system visualization and simulation",
                lambda: self.launch_solar_system(),
                "#2c3e50",
            ),
            (
                "✅ Quality Check",
                "Code quality analysis and validation",
                lambda: self.launch_quality_check(),
                "#7f8c8d",
            ),
            (
                "🛠️ Development Tools",
                "Various development utilities and helpers",
                lambda: self.show_dev_tools(),
                "#95a5a6",
            ),
        ]

        for name, desc, command, color in tools:
            self.create_tool_card(scrollable_frame, name, desc, command, color)

    def create_tool_card(
        self,
        parent: tk.Widget,
        name: str,
        description: str,
        command: Callable[[], None],
        color: str,
        show_icon: bool = False,
    ) -> None:
        """Create a professional tool card."""
        card_frame = tk.Frame(parent, bg="white", relief="raised", bd=1)
        card_frame.pack(fill="x", padx=20, pady=10)

        # Color accent bar
        accent_bar = tk.Frame(card_frame, bg=color, height=4)
        accent_bar.pack(fill="x")

        # Content frame
        content_frame = tk.Frame(card_frame, bg="white")
        content_frame.pack(fill="both", expand=True, padx=20, pady=15)

        # Header frame with icon space
        header_frame = tk.Frame(content_frame, bg="white")
        header_frame.pack(fill="x", pady=(0, 10))

        # Icon placeholder (for tools_icon reference)
        if show_icon:
            icon_frame = tk.Frame(header_frame, bg=color, width=40, height=40)
            icon_frame.pack(side="left", padx=(0, 15))
            icon_frame.pack_propagate(False)

            tk.Label(
                icon_frame, text="🧰", font=("Arial", 20), bg=color, fg="white"
            ).pack(expand=True)

        # Text frame
        text_frame = tk.Frame(header_frame, bg="white")
        text_frame.pack(side="left", fill="both", expand=True)

        # Tool name
        name_label = tk.Label(
            text_frame,
            text=name,
            font=("Arial", 14, "bold"),
            bg="white",
            fg="#2c3e50",
            anchor="w",
        )
        name_label.pack(fill="x")

        # Description
        desc_label = tk.Label(
            text_frame,
            text=description,
            font=("Arial", 11),
            bg="white",
            fg="#7f8c8d",
            anchor="w",
            wraplength=500,
        )
        desc_label.pack(fill="x")

        # Button frame
        button_frame = tk.Frame(content_frame, bg="white")
        button_frame.pack(fill="x", pady=(10, 0))

        # Launch button
        launch_btn = tk.Button(
            button_frame,
            text="Launch Tool",
            command=command,
            bg=color,
            fg="white",
            font=("Arial", 11, "bold"),
            relief="flat",
            padx=30,
            pady=8,
            cursor="hand2",
        )
        launch_btn.pack(side="right")

        # Hover effects
        def on_enter(e: Any) -> None:
            card_frame.configure(relief="raised", bd=2)
            launch_btn.configure(bg=self.darken_color(color))

        def on_leave(e: Any) -> None:
            card_frame.configure(relief="raised", bd=1)
            launch_btn.configure(bg=color)

        # Bind hover events
        for widget in [
            card_frame,
            content_frame,
            header_frame,
            text_frame,
            name_label,
            desc_label,
        ]:
            widget.bind("<Enter>", on_enter)
            widget.bind("<Leave>", on_leave)

    def darken_color(self, color: str) -> str:
        """Darken a hex color for hover effects."""
        color_map = {
            "#e74c3c": "#c0392b",
            "#f39c12": "#e67e22",
            "#9b59b6": "#8e44ad",
            "#2ecc71": "#27ae60",
            "#16a085": "#138d75",
            "#1abc9c": "#17a2b8",
            "#e67e22": "#d35400",
            "#2c3e50": "#1a252f",
            "#7f8c8d": "#566573",
            "#95a5a6": "#7b7d7d",
        }
        return color_map.get(color, color)

    # Tool launch methods
    def launch_integrated_processor(self) -> None:
        """Launch the Integrated Data Processor."""
        try:
            self.status_var.set("Launching Integrated Data Processor...")
            self.root.update()

            # Try the integrated processor launcher
            integrated_path = Path(
                "data_processing/data_processor/python/data_processor/launch_integrated.py"
            )
            if integrated_path.exists():
                subprocess.Popen([sys.executable, str(integrated_path)])
                self.status_var.set("✓ Integrated Data Processor launched")
                logger.info("Launched Integrated Data Processor")
            else:
                self.show_error("Integrated Data Processor launcher not found")

        except Exception as e:
            self.show_error(f"Failed to launch Integrated Data Processor: {e}")

    def launch_csv_processor(self) -> None:
        """Launch CSV processor directly."""
        try:
            self.status_var.set("Launching CSV Processor...")
            self.root.update()

            csv_path = Path("replicants/python/folder_tool/Folders_Tool_r0.py")
            if csv_path.exists():
                subprocess.Popen([sys.executable, str(csv_path)])
                self.status_var.set("✓ CSV Processor launched")
            else:
                self.show_error("CSV Processor not found")

        except Exception as e:
            self.show_error(f"Failed to launch CSV Processor: {e}")

    def launch_folder_tool(self) -> None:
        """Launch folder tool."""
        try:
            self.status_var.set("Launching Folder Tool...")
            self.root.update()

            folder_path = Path("replicants/python/folder_tool/Folders_Tool_r0.py")
            if folder_path.exists():
                subprocess.Popen([sys.executable, str(folder_path)])
                self.status_var.set("✓ Folder Tool launched")
            else:
                self.show_error("Folder Tool not found")

        except Exception as e:
            self.show_error(f"Failed to launch Folder Tool: {e}")

    def launch_folder_fix_pro(self) -> None:
        """Launch Folder Fix Pro."""
        self.show_info(
            "Folder Fix Pro",
            "Advanced folder processing tool with deduplication capabilities.",
        )

    def launch_project_packer(self) -> None:
        """Launch Project Packer."""
        self.show_info(
            "Project Packer",
            "Tool for packing and unpacking project files efficiently.",
        )

    def launch_video_processor(self) -> None:
        """Launch Video Processor."""
        self.show_info("Video Processor", "Video processing and analysis tools.")

    def launch_unit_converter(self) -> None:
        """Launch Unit Converter."""
        try:
            converter_path = Path("web_applications/unit_converter")
            if converter_path.exists():
                self.show_info(
                    "Unit Converter",
                    f"To launch the Unit Converter:\n\n"
                    f"1. Open terminal in: {converter_path}\n"
                    "2. Run: npm install\n"
                    "3. Run: npm start\n"
                    "4. Open browser to displayed URL",
                )
            else:
                self.show_error("Unit Converter not found")
        except Exception as e:
            self.show_error(f"Failed to access Unit Converter: {e}")

    def launch_solar_system(self) -> None:
        """Launch Solar System Model."""
        try:
            solar_path = Path(
                "scientific_modeling/solar_system_model/solar_system/launcher.py"
            )
            if solar_path.exists():
                subprocess.Popen([sys.executable, str(solar_path)])
                self.status_var.set("✓ Solar System Model launched")
            else:
                self.show_error("Solar System Model not found")
        except Exception as e:
            self.show_error(f"Failed to launch Solar System Model: {e}")

    def launch_quality_check(self) -> None:
        """Launch Quality Check."""
        try:
            quality_path = Path("quality_check_script.py")
            if quality_path.exists():
                subprocess.Popen([sys.executable, str(quality_path)])
                self.status_var.set("✓ Quality Check launched")
            else:
                self.show_error("Quality Check script not found")
        except Exception as e:
            self.show_error(f"Failed to launch Quality Check: {e}")

    def launch_pdf_renamer(self) -> None:
        """Launch PDF Renamer."""
        try:
            self.status_var.set("Launching PDF Renamer...")
            self.root.update()

            renamer_path = Path("document_processing/pdf_renamer/launch_pdf_gui.py")
            if renamer_path.exists():
                subprocess.Popen([sys.executable, str(renamer_path)])
                self.status_var.set("✓ PDF Renamer launched")
            else:
                self.show_error("PDF Renamer not found")
        except Exception as e:
            self.show_error(f"Failed to launch PDF Renamer: {e}")

    # Info methods
    def show_converter_info(self) -> None:
        """Show format converter information."""
        self.show_info(
            "Format Converter",
            "Multi-format file conversion is available through:\n\n"
            "1. Integrated Data Processor (recommended)\n"
            "2. Standalone converter tools\n"
            "3. Python pandas library support\n\n"
            "Supported formats: CSV, Excel, JSON, Parquet, TSV",
        )

    def show_matlab_info(self, tool_name: str) -> None:
        """Show MATLAB tool information."""
        self.show_info(
            f"{tool_name} (MATLAB)",
            f"To launch {tool_name}:\n\n"
            "1. Open MATLAB\n"
            f"2. Navigate to the {tool_name.lower()} directory\n"
            "3. Run the main script\n\n"
            "Note: MATLAB must be installed and licensed.",
        )

    def show_audio_tools_info(self) -> None:
        """Show audio tools information."""
        self.show_info(
            "Audio Tools",
            "Python-based audio processing utilities for analysis and manipulation.",
        )

    def show_web_dashboard_info(self) -> None:
        """Show web dashboard information."""
        self.show_info(
            "Web Dashboard",
            "Tool management and monitoring dashboard (under development).",
        )

    def show_dev_tools(self) -> None:
        """Show development tools."""
        try:
            dev_path = Path("development_tools")
            if dev_path.exists():
                if sys.platform == "win32":
                    os.startfile(str(dev_path))  # type: ignore[attr-defined]
                    self.status_var.set("✓ Development Tools folder opened")
                else:
                    # Linux/Mac support could be added here
                    logger.warning("Opening folders is only supported on Windows")
                    self.status_var.set("⚠️ Feature only available on Windows")
            else:
                self.show_error("Development Tools folder not found")
        except Exception as e:
            self.show_error(f"Failed to open Development Tools: {e}")

    def show_info(self, title: str, message: str) -> None:
        """Show information dialog."""
        messagebox.showinfo(title, message)

    def show_error(self, message: str) -> None:
        """Show error dialog."""
        messagebox.showerror("Error", message)
        logger.error(message)
        self.status_var.set(f"❌ Error: {message}")

    def run(self) -> None:
        """Run the launcher."""
        logger.info("Starting Professional Tools Launcher with tools_icon")
        self.root.mainloop()


def main() -> None:
    """Main function."""
    try:
        launcher = ToolsLauncher()
        launcher.run()
    except Exception as e:
        logger.error(f"Failed to start Tools Launcher: {e}")
        messagebox.showerror("Launcher Error", f"Failed to start Tools Launcher:\n{e}")


if __name__ == "__main__":
    main()
