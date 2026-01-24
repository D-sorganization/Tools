"""Basic Tools Launcher using Tkinter.

Note: UnifiedToolsLauncher.py (PyQt6) is the preferred launcher.
This is a simpler alternative for environments where PyQt6 is not available.
"""

import json
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

# Path helpers
BASE_DIR = Path(__file__).resolve().parent

# Import shared utilities
try:
    from tools.launch_utils import (
        LaunchError,
        PlatformError,
        SecurityError,
        ToolNotFoundError,
        launch_tool,
    )
    from tools.logger import get_logger
except ImportError:
    # If tools package not found, we can't function safely
    messagebox.showerror(
        "Critical Error",
        "Could not import tools.launch_utils. Please ensure the 'tools' package is available.",
    )
    sys.exit(1)

# Set up logging
logger = get_logger(__name__)


def load_tools_config() -> dict[str, list[Any]]:
    """Load tools configuration from tools.json."""
    try:
        from tools.config_loader import load_tools_config

        # Load config relative to this script's location (BASE_DIR)
        config = load_tools_config(BASE_DIR)
        return config
    except ImportError:
        # Fallback if tools package isn't fully installed
        json_path = BASE_DIR / "tools.json"

        try:
            with open(json_path, encoding="utf-8") as f:
                config = json.load(f)
            return config
        except Exception as e:
            logger.error("Error loading tools.json: %s", e)
            return {}

    except Exception as e:
        logger.error("Error loading tools.json: %s", e)
        return {}


# Load configuration dynamically
TOOLS = load_tools_config()


class ToolsLauncher(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Tools Launcher")
        self.geometry("950x700")

        # Set icon
        try:
            from tools.ui_utils import set_tk_icon

            set_tk_icon(self)
        except ImportError:
            pass

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
        # Define some preferred order, but allow others
        preferred = [
            "Unit Converters",
            "Data Processors",
            "Video Processors",
            "Audio Processors",
            "Folder Tools",
        ]
        categories = list(preferred)
        for cat in TOOLS.keys():
            if cat not in categories:
                categories.append(cat)

        # Only include categories that actually exist
        return [c for c in categories if c in TOOLS]

    def _create_tool_button(
        self, parent: ttk.Frame, tool_info: dict[str, Any]
    ) -> ttk.Frame:
        """Create a tool button frame with label and launch button."""
        name = tool_info.get("name", "Unknown")
        rel_path = tool_info.get("path", "")

        full_path = BASE_DIR / rel_path
        btn_frame = ttk.Frame(parent, borderwidth=1, relief="solid")

        lbl = ttk.Label(btn_frame, text=name, font=("Helvetica", 11, "bold"))
        lbl.pack(pady=(15, 5))

        exists = full_path.exists()
        state = "normal" if exists else "disabled"

        # Determine button text
        btn_text = "Launch"
        if not exists:
            btn_text = "Not Found"
        elif tool_info.get("type") == "file":
            btn_text = "Open File"

        def make_launcher(info: dict[str, Any]) -> Any:
            return lambda: self.launch_tool_wrapper(info)

        btn = ttk.Button(
            btn_frame,
            text=btn_text,
            state=state,
            command=make_launcher(tool_info),
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

        for tool_info in tool_list:
            btn_frame = self._create_tool_button(grid_frame, tool_info)
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
            ttk.Label(self, text="No tools configured or tools.json missing.").pack()

    def launch_tool_wrapper(self, tool_info: dict[str, Any]) -> None:
        """Launch a tool using repeated logic from launch_utils."""

        # Simple logging callback for Tkinter (print to stdout)
        def log_msg(msg: str) -> None:
            logger.info(msg)

        try:
            launch_tool(
                tool_info=tool_info,
                repo_root=BASE_DIR,
                is_debug=False,
                log_func=log_msg,
            )
        except (LaunchError, SecurityError, ToolNotFoundError, PlatformError) as e:
            messagebox.showerror("Launch Error", str(e))
        except Exception as e:
            messagebox.showerror(
                "Unexpected Error", f"An unexpected error occurred:\n{e}"
            )


def main() -> None:
    """Entry point for the basic Tkinter launcher."""
    app = ToolsLauncher()
    app.mainloop()


if __name__ == "__main__":
    main()
