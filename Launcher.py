"""Basic Tools Launcher using Tkinter.

Note: UnifiedToolsLauncher.py (PyQt6) is the preferred launcher.
This is a simpler alternative for environments where PyQt6 is not available.
"""

import os
import subprocess
import sys
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

# Path helpers
BASE_DIR = Path(os.path.abspath(__file__).parent)


def get_path(rel_path: str) -> str:
    return os.path.normpath(Path(BASE_DIR) / rel_path)


# Tool Configuration
# Category -> List of (Name, Relative Path, Type)
# Type: 'python', 'bat', 'html', 'file' or 'matlab'
# Note: 'matlab' type simply opens the file/folder in OS as we can't reliably assume CLI
# matlab activation.
# But 'file' is usually enough. 'python' launches with current sys.executable.
TOOLS = {
    "Unit Converters": [
        ("Calculator App", "web_applications/calculator/webapp.py", "python"),
        (
            "Unit Converter (Web)",
            "web_applications/unit_converter/unit-converter-app/index.html",
            "html",
        ),
    ],
    "Data Processors": [
        (
            "Data Processor (Integrated)",
            "data_processing/data_processor/python/data_processor/launch_integrated.py",
            "python",
        ),
        (
            "Data Processor (Replicant r0)",
            "data_processing/data_processor/archive/Data_Processor_r0.py",
            "python",
        ),
        (
            "Data Processor (Archive Integrated)",
            "data_processing/data_processor/archive/Data_Processor_Integrated.py",
            "python",
        ),
    ],
    "Video Processors": [
        (
            "Web Platform (Next.js)",
            "media_processing/video_processor/apps/web/launch_platform.bat",
            "bat",
        ),
        ("MATLAB Engine", "media_processing/video_processor/matlab/run_all.m", "file"),
    ],
    "Audio Processors": [
        (
            "Audio Processor Pro",
            "media_processing/audio_processor/matlab/audio_signal_processor/launch_audio_processor_pro.m",
            "file",
        ),
    ],
    "Folder Tools": [
        (
            "Folder Packer Pro",
            "development_tools/folder_tools/folder_packer_pro/folder_packer_pro.py",
            "python",
        ),
        (
            "Folder Fix",
            "development_tools/folder_tools/folder_tool/Launch_FolderFix.bat",
            "bat",
        ),
    ],
}


class ToolsLauncher(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Tools Launcher")
        self.geometry("950x700")

        # Style
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", font=("Helvetica", 10), padding=5)
        style.configure("TLabel", font=("Helvetica", 12))
        style.configure("Header.TLabel", font=("Helvetica", 18, "bold"))
        style.configure("TNotebook.Tab", font=("Helvetica", 11), padding=[10, 5])

        # Title
        title_label = ttk.Label(
            self, text="Tools Repository Launcher", style="Header.TLabel"
        )
        title_label.pack(pady=20)

        # Notebook (Tabs)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        self.create_tabs()

    def _get_ordered_categories(self) -> list[str]:
        """Get categories in preferred display order."""
        categories = [
            "Unit Converters",
            "Data Processors",
            "Video Processors",
            "Audio Processors",
            "Folder Tools",
        ]
        for cat in TOOLS.keys():
            if cat not in categories:
                categories.append(cat)
        return categories

    def _create_tool_button(
        self, parent: ttk.Frame, name: str, rel_path: str, kind: str
    ) -> ttk.Frame:
        """Create a tool button frame with label and launch button."""
        full_path = get_path(rel_path)
        btn_frame = ttk.Frame(parent, borderwidth=1, relief="solid")

        lbl = ttk.Label(btn_frame, text=name, font=("Helvetica", 11, "bold"))
        lbl.pack(pady=(15, 5))

        exists = Path(full_path).exists()
        state = "normal" if exists else "disabled"
        btn_text = "Launch" if exists else "Not Found"
        if kind == "file" and exists:
            btn_text = "Open File"

        def make_launcher(p: str, k: str) -> Any:
            return lambda: self.launch_tool(p, k)

        btn = ttk.Button(
            btn_frame,
            text=btn_text,
            state=state,
            command=make_launcher(full_path, kind),
        )
        btn.pack(pady=10, padx=10, fill=tk.X)

        path_lbl = ttk.Label(
            btn_frame,
            text=rel_path,
            font=("Courier", 8),
            foreground="#555",
            wraplength=400,
        )
        path_lbl.pack(pady=(0, 10))

        if not exists:
            ttk.Label(
                btn_frame,
                text="File missing on disk",
                font=("Helvetica", 8),
                foreground="red",
            ).pack(pady=2)

        return btn_frame

    def _create_category_tab(self, category: str, tool_list: list[Any]) -> ttk.Frame:
        """Create a tab frame for a category of tools."""
        frame = ttk.Frame(self.notebook)
        grid_frame = ttk.Frame(frame)
        grid_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        max_cols = 2
        row, col = 0, 0

        for name, rel_path, kind in tool_list:
            btn_frame = self._create_tool_button(grid_frame, name, rel_path, kind)
            btn_frame.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")

            col += 1
            if col >= max_cols:
                col = 0
                row += 1

        for i in range(max_cols):
            grid_frame.columnconfigure(i, weight=1)

        return frame

    def create_tabs(self) -> None:
        """Create tabs for each tool category."""
        has_tools = False

        for category in self._get_ordered_categories():
            tool_list = TOOLS.get(category, [])
            if not tool_list:
                continue

            has_tools = True
            frame = self._create_category_tab(category, tool_list)
            self.notebook.add(frame, text=category)

        if not has_tools:
            ttk.Label(self, text="No tools configured.").pack()

    def launch_tool(self, path: str, kind: str) -> None:
        """Launch a tool with the appropriate method based on its type."""
        try:
            cwd = Path(path).parent
            if kind == "python":
                subprocess.Popen([sys.executable, path], cwd=cwd)
            elif kind == "bat":
                subprocess.Popen(["cmd.exe", "/c", path], cwd=cwd)
            elif kind == "html":
                webbrowser.open(f"file://{path}")
            elif kind == "exe":
                subprocess.Popen([path], cwd=cwd)
            else:
                if hasattr(os, "startfile"):
                    os.startfile(path)  # type: ignore[attr-defined]
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", path], cwd=cwd)
                else:
                    subprocess.Popen(["xdg-open", path], cwd=cwd)

        except FileNotFoundError as e:
            messagebox.showerror("Error", f"File not found: {path}\n{e}")
        except PermissionError as e:
            messagebox.showerror("Error", f"Permission denied: {path}\n{e}")
        except OSError as e:
            messagebox.showerror("Error", f"Failed to launch {path}:\n{e}")


def main() -> None:
    """Entry point for the basic Tkinter launcher."""
    app = ToolsLauncher()
    app.mainloop()


if __name__ == "__main__":
    main()
