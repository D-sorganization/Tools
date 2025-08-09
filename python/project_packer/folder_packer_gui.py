#!/usr/bin/env python3
"""
Folder Packer / Unpacker Tool
=============================

This tool packs multiple programming files from a folder structure into a single text file,
preserving the folder hierarchy and file types. It can then unpack this file to recreate
the original folder structure.

Supported file types: .py, .html, .css, .js, .m, .txt, .json, .xml, .cpp, .java, .c, .h, 
                     .hpp, .php, .rb, .go, .rs, .swift, .kt, .r, .sql, .sh, .bat, .yml, .yaml

Features:
- Folder exclusion with pattern matching
- Automatic detection of archive/backup folders
- GUI-based folder selection for exclusions

Author: Claude Assistant
Version: 1.1
"""

import fnmatch
import os
import re
import sys
import threading
import tkinter as tk
import tkinter.simpledialog as simpledialog
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk


class FolderPackerGUI:
    """Main GUI application for packing and unpacking folder structures."""

    def __init__(self, root):
        self.root = root
        self.root.title("Folder Packer / Unpacker Tool v1.1")
        self.root.geometry("900x700")

        # Set application icon
        try:
            # Get the directory where the script/executable is located
            if getattr(sys, "frozen", False):
                # Running as compiled executable
                base_dir = sys._MEIPASS
            else:
                # Running as script
                base_dir = os.path.dirname(__file__)

            # Try ICO file first (best for Windows)
            ico_path = os.path.join(base_dir, "folder_icon.ico")
            if os.path.exists(ico_path):
                # Use iconbitmap for better Windows taskbar integration
                self.root.iconbitmap(ico_path)
                print(f"Loaded ICO icon: {ico_path}")

                # Also set iconphoto as a fallback
                try:
                    from PIL import Image, ImageTk

                    image = Image.open(ico_path)
                    # Create multiple sizes for better scaling
                    sizes = [16, 32, 48, 64]
                    photos = []
                    for size in sizes:
                        resized = image.resize((size, size), Image.Resampling.LANCZOS)
                        photo = ImageTk.PhotoImage(resized)
                        photos.append(photo)

                    # Set the first (largest) as the main icon
                    if photos:
                        self.root.iconphoto(True, *photos)
                        # Keep references to prevent garbage collection
                        self.icon_photos = photos
                except Exception as e:
                    print(f"Could not set iconphoto: {e}")

            else:
                # Fallback to JPG if ICO doesn't exist
                jpg_path = os.path.join(base_dir, "folder_icon.jpg")
                if os.path.exists(jpg_path):
                    from PIL import Image, ImageTk

                    image = Image.open(jpg_path)
                    # Create multiple sizes for better scaling
                    sizes = [16, 32, 48, 64]
                    photos = []
                    for size in sizes:
                        resized = image.resize((size, size), Image.Resampling.LANCZOS)
                        photo = ImageTk.PhotoImage(resized)
                        photos.append(photo)

                    if photos:
                        self.root.iconphoto(True, *photos)
                        # Keep references to prevent garbage collection
                        self.icon_photos = photos
                        print(f"Loaded JPG icon: {jpg_path}")
                else:
                    print("No icon files found (folder_icon.ico or folder_icon.jpg)")

            # On Windows, try to set the app ID for better taskbar behavior
            try:
                import ctypes

                # Set app user model ID for Windows taskbar grouping
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                    "FolderPacker.Tool.1.0"
                )
            except Exception as e:
                print(f"Could not set app ID: {e}")

        except Exception as e:
            print(f"Could not load icon: {e}")

        # Configure modern styling with consistent colors
        self.root.configure(bg="#FFFEF7")

        # Define color scheme - clean white and soft blues, no grey anywhere
        self.colors = {
            "primary": "#4A90E2",  # Soft blue for accents
            "secondary": "#7BB3F0",  # Light blue for selections
            "accent": "#5DADE2",  # Sky blue
            "background": "#FFFFFF",  # Pure white for text areas
            "card_bg": "#FFFEF7",  # Very light cream for cards
            "text_dark": "#000000",  # Pure black text
            "text_light": "#000000",  # Pure black text (no grey)
            "light_blue_bg": "#FFFEF7",  # Light cream background
            "legal_pad_yellow": "#FFFACD",  # Legal pad yellow for selected tab
            "border_color": "#FFFEF7",  # Remove visible border (match card_bg)
        }

        # Define supported file extensions
        self.file_extensions = {
            ".py",
            ".html",
            ".css",
            ".js",
            ".m",
            ".txt",
            ".json",
            ".xml",
            ".cpp",
            ".java",
            ".c",
            ".h",
            ".hpp",
            ".php",
            ".rb",
            ".go",
            ".rs",
            ".swift",
            ".kt",
            ".r",
            ".sql",
            ".sh",
            ".bat",
            ".yml",
            ".yaml",
            ".md",
            ".rst",
            ".tex",
            ".vue",
            ".jsx",
            ".tsx",
            ".ts",
        }

        # Define common archive/backup folder patterns
        self.common_exclude_patterns = [
            "*_archive",
            "*Archive",
            "_Archive",
            "*_backup",
            "*Backup",
            "_Backup",
            "*_old",
            "*_OLD",
            "*.old",
            ".git",
            "__pycache__",
            "node_modules",
            ".idea",
            ".vscode",
            "venv",
            "env",
            ".env",
            "build",
            "dist",
            "*.egg-info",
        ]

        # List to store excluded folders
        self.excluded_folders = set()

        # Define delimiter format (matching MATLAB version)
        self.start_delimiter = "%%%%%% START FILE: {} %%%%%%"
        self.end_delimiter = "%%%%%% END FILE: {} %%%%%%"

        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface components with a modern, clean look."""
        style = ttk.Style()
        style.theme_use("clam")
        # Modern, clean fonts and colors
        style.configure(
            "Title.TLabel",
            font=("Segoe UI", 22, "bold"),
            foreground=self.colors["text_dark"],
            background=self.colors["background"],
        )
        style.configure(
            "Section.TLabelframe", background=self.colors["background"], borderwidth=0
        )
        style.configure(
            "Section.TLabelframe.Label",
            font=("Segoe UI", 12, "bold"),
            foreground=self.colors["primary"],
            background=self.colors["background"],
        )
        style.configure("TButton", font=("Segoe UI", 12), padding=8)
        style.configure(
            "TLabel",
            font=("Segoe UI", 11),
            background=self.colors["background"],
            foreground=self.colors["text_dark"],
        )
        style.configure(
            "TCheckbutton",
            font=("Segoe UI", 11),
            background=self.colors["background"],
            foreground=self.colors["text_dark"],
        )
        style.configure(
            "TNotebook",
            background=self.colors["background"],
            borderwidth=0,
            tabmargins=[2, 2, 2, 0],
        )
        style.configure(
            "TNotebook.Tab",
            font=("Segoe UI", 11),
            padding=[12, 6],
            background=self.colors["background"],
            foreground=self.colors["primary"],
            borderwidth=0,
        )
        style.map(
            "TNotebook.Tab",
            background=[
                ("selected", self.colors["legal_pad_yellow"]),
                ("!selected", self.colors["background"]),
            ],
            foreground=[
                ("selected", self.colors["primary"]),
                ("!selected", self.colors["primary"]),
            ],
        )
        # Prevent tab height from changing on selection
        # Custom layout to prevent tab height change on selection
        style.layout(
            "TNotebook.Tab",
            [
                (
                    "Notebook.tab",
                    {
                        "sticky": "nswe",
                        "children": [
                            (
                                "Notebook.padding",
                                {
                                    "sticky": "nswe",
                                    "children": [
                                        (
                                            "Notebook.focus",
                                            {
                                                "sticky": "nswe",
                                                "children": [
                                                    (
                                                        "Notebook.label",
                                                        {"side": "left", "sticky": ""},
                                                    )
                                                ],
                                            },
                                        )
                                    ],
                                },
                            )
                        ],
                    },
                )
            ],
        )

        # Main container
        main_container = tk.Frame(self.root, bg=self.colors["background"])
        main_container.pack(fill="both", expand=True)

        # Title
        title_label = ttk.Label(
            main_container,
            text="Folder Packer / Unpacker Tool",
            style="Title.TLabel",
            anchor="center",
        )
        title_label.pack(pady=(18, 8))

        # Notebook for tabs
        notebook = ttk.Notebook(main_container, style="TNotebook")
        notebook.pack(fill="both", expand=True, padx=24, pady=8)

        # Main operations tab
        main_tab = ttk.Frame(notebook, style="TFrame")
        notebook.add(main_tab, text="Pack / Unpack")

        # Exclusions tab
        exclusions_tab = ttk.Frame(notebook, style="TFrame")
        notebook.add(exclusions_tab, text="Folder Exclusions")

        # Instructions tab (new, at the end)
        instructions_tab = ttk.Frame(notebook, style="TFrame")
        notebook.add(instructions_tab, text="Instructions")

        # Setup main tab
        self.setup_main_tab(main_tab)
        # Setup exclusions tab
        self.setup_exclusions_tab(exclusions_tab)
        # Setup instructions tab
        self.setup_instructions_tab(instructions_tab)

        # Status bar (outside notebook)
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = tk.Label(
            self.root,
            textvariable=self.status_var,
            anchor="w",
            bg=self.colors["card_bg"],
            fg=self.colors["text_dark"],
            font=("Segoe UI", 11),
            relief="flat",
            bd=1,
            padx=12,
            pady=6,
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def setup_main_tab(self, parent):
        """Set up the main operations tab with a modern, clean layout."""
        main_frame = tk.Frame(parent, bg=self.colors["background"])
        main_frame.pack(fill="both", expand=True)

        # Action buttons (always visible, centered)
        button_frame = tk.Frame(main_frame, bg=self.colors["background"])
        button_frame.pack(pady=30)
        self.pack_button = ttk.Button(
            button_frame,
            text="Select Project to Pack",
            command=self.pack_folder_clicked,
            width=22,
        )
        self.pack_button.grid(row=0, column=0, padx=24)
        self.unpack_button = ttk.Button(
            button_frame,
            text="Select Project to Unpack",
            command=self.unpack_file_clicked,
            width=22,
        )
        self.unpack_button.grid(row=0, column=1, padx=24)

        # Output/Progress area
        output_frame = ttk.LabelFrame(
            main_frame,
            text="Progress & Output",
            style="Section.TLabelframe",
            padding=12,
        )
        output_frame.pack(fill="both", expand=True, pady=(18, 0), padx=8)
        self.output_text = scrolledtext.ScrolledText(
            output_frame,
            wrap=tk.WORD,
            width=90,
            height=16,
            bg=self.colors["background"],
            fg=self.colors["text_dark"],
            font=("Consolas", 12),
            insertbackground=self.colors["primary"],
            selectbackground=self.colors["secondary"],
            borderwidth=1,
            relief="solid",
        )
        self.output_text.pack(fill="both", expand=True)

    def setup_instructions_tab(self, parent):
        """Set up the instructions tab with detailed usage info."""
        instructions = (
            "This tool helps you pack multiple programming files into a single text file while preserving folder structure.\n\n"
            "• Select a source folder containing your project files.\n"
            "• Exclude folders based on your exclusion settings (see Folder Exclusions tab).\n"
            "• Record relative paths and combine files with clear delimiters.\n"
            "• Add metadata about the packing operation.\n\n"
            "Unpacking will recreate the original folder hierarchy and restore all files.\n\n"
            "▶ How Unpacking Works:\n"
            "- When you unpack into a folder, the tool will MERGE the packed files into the selected destination folder.\n"
            "- If a file in the packed archive has the same name and path as a file in the destination, it will OVERWRITE the existing file.\n"
            "- Files and folders that do not match any in the packed archive (such as .git, .gitignore, or other project files) will be preserved and left untouched.\n\n"
            "▶ Backup Option Before Unpacking:\n"
            "- If you choose to unpack into a folder that is not empty, you will be prompted to create a BACKUP of the destination folder before unpacking.\n"
            "- If you select 'Yes', the tool will create a timestamped backup copy of the destination folder (excluding .git and common temporary/backup folders) before merging the new files.\n"
            "- This is especially useful for git-tracked projects, as it preserves your .git folder and history, and allows you to restore the previous state if needed.\n"
            "- If you select 'No', unpacking will proceed and files with matching names will be overwritten.\n\n"
            "▶ Wildcards and Exclusion Patterns:\n"
            "- You can exclude folders from packing using patterns with * as a wildcard (e.g., *_archive, node_modules, .git).\n"
            "- Examples:\n"
            "    - *_backup matches any folder ending with '_backup' (e.g., 'my_backup')\n"
            "    - *.old matches any file or folder ending with '.old'\n"
            "    - *data* matches any folder or file containing 'data'\n"
            "- Exclusions are applied only during packing.\n"
            "- See the 'Folder Exclusions' tab for more details and to configure exclusions.\n"
        )
        instructions_frame = tk.Frame(parent, bg=self.colors["background"])
        instructions_frame.pack(fill="both", expand=True, padx=8, pady=18)
        instructions_label = ttk.Label(
            instructions_frame,
            text="Instructions",
            style="Section.TLabelframe.Label",
            anchor="w",
        )
        instructions_label.pack(anchor="w", pady=(0, 12))
        instructions_text = ttk.Label(
            instructions_frame,
            text=instructions,
            wraplength=800,
            justify=tk.LEFT,
            style="TLabel",
        )
        instructions_text.pack(anchor="w", fill="both", expand=True)

    def setup_exclusions_tab(self, parent):
        """Set up the folder exclusions tab with a modern, clean layout."""
        exclusions_frame = tk.Frame(parent, bg=self.colors["background"])
        exclusions_frame.pack(fill="both", expand=True)

        # Instructions
        instructions = ttk.Label(
            exclusions_frame,
            text="Configure folders to exclude when packing. Use patterns with * for wildcards.",
            wraplength=800,
            style="TLabel",
        )
        instructions.pack(pady=(0, 12), anchor="w", padx=8)

        # Two column layout
        columns = tk.Frame(exclusions_frame, bg=self.colors["background"])
        columns.pack(fill="both", expand=True, padx=0, pady=0)

        # Left: Common patterns
        left_frame = ttk.LabelFrame(
            columns,
            text="Common Exclude Patterns",
            style="Section.TLabelframe",
            padding=14,
        )
        left_frame.pack(side=tk.LEFT, fill="both", expand=True, padx=(0, 12), pady=0)
        self.pattern_vars = {}
        checkbox_frame = tk.Frame(left_frame, bg=self.colors["background"])
        checkbox_frame.pack(fill="both", expand=True, pady=(2, 0))
        for i, pattern in enumerate(self.common_exclude_patterns):
            var = tk.BooleanVar(
                value=pattern
                in [
                    "*_archive",
                    "*Archive",
                    "_Archive",
                    "*_backup",
                    "*Backup",
                    "_Backup",
                ]
            )
            self.pattern_vars[pattern] = var
            cb = ttk.Checkbutton(
                checkbox_frame, text=pattern, variable=var, style="TCheckbutton"
            )
            cb.grid(row=i, column=0, sticky="w", padx=4, pady=2)

        # Right: Custom exclusions
        right_frame = ttk.LabelFrame(
            columns,
            text="Custom Folder Exclusions",
            style="Section.TLabelframe",
            padding=14,
        )
        right_frame.pack(side=tk.LEFT, fill="both", expand=True, padx=(12, 0), pady=0)
        listbox_frame = tk.Frame(right_frame, bg=self.colors["background"])
        listbox_frame.pack(fill="both", expand=True, pady=(2, 0))
        scrollbar = ttk.Scrollbar(listbox_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.exclusion_listbox = tk.Listbox(
            listbox_frame,
            yscrollcommand=scrollbar.set,
            height=10,
            bg=self.colors["background"],
            fg=self.colors["text_dark"],
            selectbackground=self.colors["secondary"],
            selectforeground=self.colors["text_dark"],
            font=("Segoe UI", 11),
            borderwidth=1,
            relief="solid",
        )
        self.exclusion_listbox.pack(side=tk.LEFT, fill="both", expand=True)
        scrollbar.config(command=self.exclusion_listbox.yview)
        # Buttons for custom exclusions
        button_frame = tk.Frame(right_frame, bg=self.colors["background"])
        button_frame.pack(fill="x", pady=(8, 0))
        add_folder_btn = ttk.Button(
            button_frame, text="Add Folder", command=self.add_excluded_folder
        )
        add_folder_btn.pack(side=tk.LEFT, padx=(0, 8))
        add_pattern_btn = ttk.Button(
            button_frame, text="Add Pattern", command=self.add_excluded_pattern
        )
        add_pattern_btn.pack(side=tk.LEFT, padx=(0, 8))
        remove_btn = ttk.Button(
            button_frame, text="Remove Selected", command=self.remove_excluded
        )
        remove_btn.pack(side=tk.LEFT)
        # Info label
        info_label = ttk.Label(
            exclusions_frame,
            text="Note: Exclusions are applied during packing only. Patterns use * as wildcard.",
            font=("Segoe UI", 10, "italic"),
            style="TLabel",
        )
        info_label.pack(side=tk.BOTTOM, pady=(12, 0), anchor="w", padx=8)

    def add_excluded_folder(self):
        """Add a specific folder to exclusion list via folder browser."""
        folder = filedialog.askdirectory(title="Select Folder to Exclude")
        if folder:
            # Convert to relative pattern if possible
            folder_name = os.path.basename(folder)
            self.excluded_folders.add(folder_name)
            self.update_exclusion_listbox()

    def add_excluded_pattern(self):
        """Add a pattern to exclusion list via dialog."""
        pattern = simpledialog.askstring(
            "Add Pattern",
            "Enter folder pattern (use * for wildcard):",
            parent=self.root,
        )
        if pattern:
            self.excluded_folders.add(pattern)
            self.update_exclusion_listbox()

    def remove_excluded(self):
        """Remove selected items from exclusion list."""
        selection = self.exclusion_listbox.curselection()
        if selection:
            # Get selected items
            items_to_remove = [self.exclusion_listbox.get(i) for i in selection]
            # Remove from set
            for item in items_to_remove:
                self.excluded_folders.discard(item)
            self.update_exclusion_listbox()

    def update_exclusion_listbox(self):
        """Update the exclusion listbox display."""
        self.exclusion_listbox.delete(0, tk.END)
        for item in sorted(self.excluded_folders):
            self.exclusion_listbox.insert(tk.END, item)

    def get_all_exclusions(self):
        """Get all active exclusion patterns."""
        exclusions = set(self.excluded_folders)

        # Add checked common patterns
        for pattern, var in self.pattern_vars.items():
            if var.get():
                exclusions.add(pattern)

        return exclusions

    def should_exclude_path(self, path, exclusions):
        """Check if a path should be excluded based on exclusion patterns."""
        # Check each part of the path
        parts = Path(path).parts

        for part in parts:
            for pattern in exclusions:
                if fnmatch.fnmatch(part, pattern):
                    return True

        return False

    def log_message(self, message, level="INFO"):
        """Log a message to the output text area."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {level}: {message}\n"
        self.output_text.insert(tk.END, formatted_message)
        self.output_text.see(tk.END)
        self.root.update_idletasks()

    def set_buttons_state(self, enabled):
        """Enable or disable action buttons."""
        state = tk.NORMAL if enabled else tk.DISABLED
        self.pack_button.config(state=state)
        self.unpack_button.config(state=state)

    def pack_folder_clicked(self):
        """Handle pack folder button click."""
        # Select source folder
        source_folder = filedialog.askdirectory(
            title="Select Project Folder to Pack", initialdir=os.getcwd()
        )

        if not source_folder:
            return

        # Select output file
        default_name = f"{Path(source_folder).name}_packed.txt"
        output_file = filedialog.asksaveasfilename(
            title="Save Packed File As",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialfile=default_name,
        )

        if not output_file:
            return

        # Clear output area
        self.output_text.delete(1.0, tk.END)

        # Run packing in a separate thread to keep UI responsive
        thread = threading.Thread(
            target=self.pack_folder, args=(source_folder, output_file)
        )
        thread.start()

    def pack_folder(self, source_folder, output_file):
        """Pack all supported files from source folder into output file."""
        self.set_buttons_state(False)
        self.status_var.set("Packing in progress...")

        try:
            self.log_message(f"Starting to pack folder: {source_folder}")
            self.log_message(f"Output file: {output_file}")

            # Get exclusions
            exclusions = self.get_all_exclusions()
            if exclusions:
                self.log_message(f"Active exclusions: {', '.join(sorted(exclusions))}")

            # Find all supported files
            files_to_pack = []
            excluded_count = 0

            for extension in self.file_extensions:
                pattern = f"**/*{extension}"
                for file_path in Path(source_folder).glob(pattern):
                    # Check if file should be excluded
                    relative_path = file_path.relative_to(source_folder)
                    if self.should_exclude_path(relative_path, exclusions):
                        excluded_count += 1
                        continue
                    files_to_pack.append(file_path)

            # Sort files for consistent ordering
            files_to_pack.sort()

            if excluded_count > 0:
                self.log_message(
                    f"Excluded {excluded_count} files based on exclusion patterns"
                )

            if not files_to_pack:
                self.log_message(
                    "No supported files found after applying exclusions!", "WARNING"
                )
                messagebox.showwarning(
                    "No Files Found",
                    "No supported programming files found after applying exclusions.",
                )
                return

            self.log_message(f"Found {len(files_to_pack)} files to pack")

            # Write packed file
            with open(output_file, "w", encoding="utf-8") as f:
                # Write header
                f.write("%%%%%% FOLDER PACKER / UNPACKER - v1.1 %%%%%%\n")
                f.write(f"%%%%%% Source Folder: {source_folder} %%%%%%\n")
                f.write(
                    f"%%%%%% Pack Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} %%%%%%\n"
                )
                f.write(f"%%%%%% Total Files: {len(files_to_pack)} %%%%%%\n")
                f.write(f"%%%%%% Files Excluded: {excluded_count} %%%%%%\n")
                if exclusions:
                    f.write(
                        f"%%%%%% Active Exclusions: {', '.join(sorted(exclusions))} %%%%%%\n"
                    )
                f.write(
                    "%%%%%% Supported Extensions: "
                    + ", ".join(sorted(self.file_extensions))
                    + " %%%%%%\n\n"
                )

                # Pack each file
                packed_count = 0
                error_count = 0

                for file_path in files_to_pack:
                    try:
                        # Calculate relative path
                        relative_path = file_path.relative_to(source_folder)
                        relative_path_str = str(relative_path).replace("\\", "/")

                        self.log_message(f"Packing: {relative_path_str}")

                        # Write start delimiter
                        f.write(f"{self.start_delimiter.format(relative_path_str)}\n")

                        # Read and write file content
                        try:
                            content = file_path.read_text(encoding="utf-8")
                            f.write(content)

                            # Ensure newline before end delimiter
                            if not content.endswith("\n"):
                                f.write("\n")

                        except UnicodeDecodeError:
                            # Try with different encoding
                            try:
                                content = file_path.read_text(encoding="latin-1")
                                f.write(content)
                                if not content.endswith("\n"):
                                    f.write("\n")
                                self.log_message(
                                    f"  Note: Used latin-1 encoding for {relative_path_str}",
                                    "WARNING",
                                )
                            except Exception as e:
                                f.write(
                                    f"****** ERROR: Could not read file content. Error: {str(e)} ******\n"
                                )
                                self.log_message(
                                    f"  Error reading file: {str(e)}", "ERROR"
                                )
                                error_count += 1

                        # Write end delimiter
                        f.write(f"{self.end_delimiter.format(relative_path_str)}\n\n")
                        packed_count += 1

                    except Exception as e:
                        self.log_message(
                            f"  Error processing file {file_path}: {str(e)}", "ERROR"
                        )
                        error_count += 1

                # Write footer
                f.write("%%%%%% END OF PACKED PROJECT %%%%%%\n")

            # Report results
            self.log_message(f"\nPacking complete!")
            self.log_message(f"Successfully packed: {packed_count} files")
            if excluded_count > 0:
                self.log_message(f"Excluded: {excluded_count} files")
            if error_count > 0:
                self.log_message(f"Errors encountered: {error_count} files", "WARNING")
            self.log_message(f"Output saved to: {output_file}")

            messagebox.showinfo(
                "Packing Complete",
                f"Successfully packed {packed_count} files.\n"
                f"Excluded: {excluded_count} files\n"
                f"Errors: {error_count}\n\n"
                f"Output saved to:\n{output_file}",
            )

        except Exception as e:
            self.log_message(f"Critical error during packing: {str(e)}", "ERROR")
            messagebox.showerror("Packing Error", f"An error occurred:\n{str(e)}")

        finally:
            self.set_buttons_state(True)
            self.status_var.set("Ready")

    def unpack_file_clicked(self):
        """Handle unpack file button click, with backup option."""
        # Select packed file
        input_file = filedialog.askopenfilename(
            title="Select Packed File to Unpack",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not input_file:
            return
        # Select output folder
        output_folder = filedialog.askdirectory(
            title="Select Destination Folder for Unpacking"
        )
        if not output_folder:
            return
        # Ask if user wants to create a backup before unpacking
        backup = False
        if os.path.exists(output_folder) and os.listdir(output_folder):
            backup = messagebox.askyesno(
                "Backup Before Unpacking?",
                f"The folder '{output_folder}' is not empty.\n\n"
                "Do you want to create a backup of this folder before unpacking and overwriting files?\n\n"
                "(Recommended if you want to preserve the current state, including .git and other files.)",
            )
            if not backup:
                # Confirm overwrite if not backing up
                result = messagebox.askyesno(
                    "Folder Not Empty",
                    f"Existing files with the same names WILL BE OVERWRITTEN.\n\nDo you want to continue?",
                )
                if not result:
                    return
        # Clear output area
        self.output_text.delete(1.0, tk.END)
        # Run unpacking in a separate thread
        thread = threading.Thread(
            target=self.unpack_file, args=(input_file, output_folder, backup)
        )
        thread.start()

    def unpack_file(self, input_file, output_folder, backup=False):
        """Unpack the packed file to recreate folder structure, with optional backup."""
        import shutil

        self.set_buttons_state(False)
        self.status_var.set("Unpacking in progress...")
        try:
            self.log_message(f"Starting to unpack file: {input_file}")
            self.log_message(f"Destination folder: {output_folder}")
            # Backup if requested
            if backup:
                backup_dir = os.path.join(
                    os.path.dirname(output_folder),
                    f"{os.path.basename(output_folder)}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                )
                self.log_message(f"Creating backup at: {backup_dir}")

                def ignore_patterns(dir, files):
                    # Always ignore .git and __pycache__
                    ignore = {".git", "__pycache__"}
                    # Add common exclude patterns (folder only)
                    for pat in self.common_exclude_patterns:
                        if pat.startswith(".") or pat.startswith("*"):
                            continue
                        ignore.add(pat)
                    return [f for f in files if f in ignore]

                try:
                    shutil.copytree(output_folder, backup_dir, ignore=ignore_patterns)
                    self.log_message(f"Backup created at: {backup_dir}")
                except Exception as e:
                    self.log_message(f"Backup failed: {e}", "ERROR")
            # Read packed file
            with open(input_file, "r", encoding="utf-8") as f:
                content = f.read()
            # Parse content using regex
            start_pattern = r"%%%%%% START FILE: (.*?) %%%%%%"
            end_pattern = r"%%%%%% END FILE: (.*?) %%%%%%"
            # Find all file blocks
            file_blocks = []
            current_pos = 0
            while True:
                start_match = re.search(start_pattern, content[current_pos:])
                if not start_match:
                    break
                start_pos = current_pos + start_match.end()
                file_path = start_match.group(1).strip()
                end_match = re.search(
                    end_pattern.replace("(.*?)", re.escape(file_path)),
                    content[start_pos:],
                )
                if not end_match:
                    self.log_message(
                        f"Warning: No end delimiter found for {file_path}", "WARNING"
                    )
                    current_pos = start_pos
                    continue
                end_pos = start_pos + end_match.start()
                file_content = content[start_pos:end_pos].rstrip("\n")
                file_blocks.append((file_path, file_content))
                current_pos = start_pos + end_match.end()
            if not file_blocks:
                self.log_message("No file blocks found in the packed file!", "ERROR")
                messagebox.showerror(
                    "Unpack Error", "No valid file blocks found in the packed file."
                )
                return
            self.log_message(f"Found {len(file_blocks)} files to unpack")
            unpacked_count = 0
            error_count = 0
            for file_path, file_content in file_blocks:
                try:
                    full_path = Path(output_folder) / file_path
                    self.log_message(f"Unpacking: {file_path}")
                    full_path.parent.mkdir(parents=True, exist_ok=True)
                    if "****** ERROR: Could not read file content" in file_content:
                        self.log_message(
                            f"  Skipping file with read error: {file_path}", "WARNING"
                        )
                        error_count += 1
                        continue
                    with open(full_path, "w", encoding="utf-8") as f:
                        f.write(file_content)
                    unpacked_count += 1
                except Exception as e:
                    self.log_message(
                        f"  Error unpacking {file_path}: {str(e)}", "ERROR"
                    )
                    error_count += 1
            self.log_message(f"\nUnpacking complete!")
            self.log_message(f"Successfully unpacked: {unpacked_count} files")
            if error_count > 0:
                self.log_message(f"Errors encountered: {error_count} files", "WARNING")
            messagebox.showinfo(
                "Unpacking Complete",
                f"Successfully unpacked {unpacked_count} files.\n"
                f"Errors: {error_count}\n\n"
                f"Files unpacked to:\n{output_folder}",
            )
        except Exception as e:
            self.log_message(f"Critical error during unpacking: {str(e)}", "ERROR")
            messagebox.showerror("Unpacking Error", f"An error occurred:\n{str(e)}")
        finally:
            self.set_buttons_state(True)
            self.status_var.set("Ready")


def main():
    """Main entry point for the application."""
    # Set Windows app user model ID early for better taskbar integration
    try:
        import ctypes

        # Set a unique app user model ID for proper taskbar grouping and icon display
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "FolderPacker.Tool.v1.1"
        )
    except Exception:
        pass  # Ignore errors on non-Windows systems

    root = tk.Tk()
    app = FolderPackerGUI(root)

    # Center window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f"{width}x{height}+{x}+{y}")

    root.mainloop()


if __name__ == "__main__":
    main()
