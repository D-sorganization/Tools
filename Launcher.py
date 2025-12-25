import os
import subprocess
import sys
import tkinter as tk
import webbrowser
from tkinter import messagebox, ttk

# Path helpers
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_path(rel_path: str) -> str:
    return os.path.normpath(os.path.join(BASE_DIR, rel_path))


# Tool Configuration
# Category -> List of (Name, Relative Path, Type)
# Type: 'python', 'bat', 'html', 'file' or 'matlab'
# Note: 'matlab' type simply opens the file/folder in OS as we can't reliably assume CLI matlab activation.
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
            "Audio Processor Pro (Main)",
            "media_processing/audio_processor/matlab/audio_signal_processor/launch_audio_processor_pro.m",
            "file",
        ),
        (
            "Audio Processor Pro (Replicant)",
            "replicants/matlab/audio_signal_processor/launch_audio_processor_pro.m",
            "file",
        ),
    ],
    "Folder Tools": [
        (
            "Folder Packer Pro (Main)",
            "development_tools/folder_tools/folder_packer_pro/folder_packer_pro.py",
            "python",
        ),
        (
            "Folder Fix (Main)",
            "development_tools/folder_tools/folder_tool/Launch_FolderFix.bat",
            "bat",
        ),
        (
            "Folder Packer Pro (Replicant)",
            "replicants/python/folder_packer_pro/folder_packer_pro.py",
            "python",
        ),
        (
            "Project Packer (Replicant)",
            "replicants/python/project_packer/folder_packer_gui.py",
            "python",
        ),
        (
            "Folder Fix (Replicant)",
            "replicants/python/folder_tool/Launch_FolderFix.bat",
            "bat",
        ),
        (
            "Folder Fix Pro (Replicant)",
            "replicants/python/folder_tool_pro/folder_fix_pro.py",
            "python",
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

    def create_tabs(self) -> None:
        has_tools = False
        categories = [
            "Unit Converters",
            "Data Processors",
            "Video Processors",
            "Audio Processors",
            "Folder Tools",
        ]

        # Add any other categories that might exist in TOOLS but not in the ordered list
        for cat in TOOLS.keys():
            if cat not in categories:
                categories.append(cat)

        for category in categories:
            tool_list = TOOLS.get(category, [])
            if not tool_list:
                continue

            has_tools = True
            frame = ttk.Frame(self.notebook)
            self.notebook.add(frame, text=category)

            # Grid for icons
            grid_frame = ttk.Frame(frame)
            grid_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

            row = 0
            col = 0
            MAX_COLS = 2

            for name, rel_path, kind in tool_list:
                full_path = get_path(rel_path)

                # Button Frame
                btn_frame = ttk.Frame(grid_frame, borderwidth=1, relief="solid")
                btn_frame.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")

                # Label
                lbl = ttk.Label(btn_frame, text=name, font=("Helvetica", 11, "bold"))
                lbl.pack(pady=(15, 5))

                # Launch Button
                exists = os.path.exists(full_path)
                state = "normal" if exists else "disabled"
                btn_text = "Launch" if exists else "Not Found"

                # Check for "file" type to be more descriptive
                if kind == "file" and exists:
                    btn_text = "Open File"

                btn = ttk.Button(
                    btn_frame,
                    text=btn_text,
                    state=state,
                    command=lambda p=full_path, k=kind: self.launch_tool(
                        p, k
                    ),  # type: ignore[misc]
                )
                btn.pack(pady=10, padx=10, fill=tk.X)

                # Path Label (Small)
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

                col += 1
                if col >= MAX_COLS:
                    col = 0
                    row += 1

            # Configure grid weights
            for i in range(MAX_COLS):
                grid_frame.columnconfigure(i, weight=1)

        if not has_tools:
            ttk.Label(self, text="No tools configured.").pack()

    def launch_tool(self, path: str, kind: str) -> None:
        try:
            cwd = os.path.dirname(path)
            if kind == "python":
                # Use the same python executable
                subprocess.Popen([sys.executable, path], cwd=cwd)
            elif kind == "bat":
                subprocess.Popen([path], cwd=cwd, shell=True)
            elif kind == "html":
                webbrowser.open(f"file://{path}")
            elif kind == "exe":
                subprocess.Popen([path], cwd=cwd)
            else:
                if sys.platform == "win32":
                    os.startfile(path)  # type: ignore[attr-defined]
                else:
                    messagebox.showinfo(
                        "Info", f"Cannot open file on this platform: {path}"
                    )

        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch {path}:\n{e}")


if __name__ == "__main__":
    app = ToolsLauncher()
    app.mainloop()
