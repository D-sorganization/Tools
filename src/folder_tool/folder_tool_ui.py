"""UICreationMixin -- UI widget creation methods for FolderProcessorApp."""

from __future__ import annotations

import ctypes
import logging
import os
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from Folders_Tool_r0 import ICON_SIZES

logger = logging.getLogger(__name__)


class UICreationMixin:
    """UI widget creation methods for FolderProcessorApp."""

    def _setup_application_icon(self) -> None:
        """Sets up the application icon with fallback options."""
        try:
            # Get the directory where the script/executable is located
            if getattr(sys, "frozen", False):
                # Running as compiled executable
                base_dir = getattr(
                    sys, "_MEIPASS", Path(os.path.abspath(__file__).parent)
                )
            else:
                # Running as script
                base_dir = Path(__file__).parent

            # On Windows, set the app ID FIRST for better taskbar behavior
            self._set_windows_app_id()

            # Try ICO file first (best for Windows)
            ico_path = Path(base_dir) / "paper_plane_icon.ico"
            if Path(ico_path).exists():
                self._load_ico_icon(ico_path)
            else:
                # Fallback to PNG if ICO doesn't exist
                self._load_png_fallback(base_dir)

        except (IOError, PermissionError, OSError) as e:
            logger.error(f"Could not load icon: {e}")

    def _set_windows_app_id(self) -> None:
        """Sets the Windows app user model ID for taskbar grouping."""
        try:
            if sys.platform == "win32":
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(  # type: ignore[attr-defined]
                    "FolderFix.Tool.2.0",
                )
                logger.info("Set Windows App User Model ID for taskbar grouping")
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Could not set app ID: {e}")

    def _load_ico_icon(self, ico_path: str) -> None:
        """Loads and sets the ICO icon for the application."""
        # Use iconbitmap for Windows taskbar integration
        assert ico_path is not None, "ico_path must be provided"
        self.root.iconbitmap(ico_path)  # type: ignore[no-untyped-call]
        logger.info(f"Loaded ICO icon for taskbar: {ico_path}")

        # Also set iconphoto with multiple sizes for better display
        try:
            from PIL import Image, ImageTk

            # Load the ICO file which now has multiple sizes
            image = Image.open(ico_path)

            # Create PhotoImage objects for different sizes using constants
            photos = []

            for size in ICON_SIZES:
                try:
                    # Try to get exact size from ICO, or resize
                    resized = image.resize(
                        (size, size),
                        Image.Resampling.LANCZOS,
                    )
                    if resized.mode != "RGBA":
                        resized = resized.convert("RGBA")
                    photo = ImageTk.PhotoImage(resized)
                    photos.append(photo)
                except (OSError, ValueError) as e:
                    logger.warning(f"Could not create {size}x{size} icon: {e}")

            # Set all sizes at once for best scaling
            if photos:
                self.root.iconphoto(True, *photos)
                # Keep references to prevent garbage collection
                self.icon_photos = photos
                logger.info(f"Set iconphoto with {len(photos)} different sizes")

        except (IOError, PermissionError, OSError) as e:
            logger.warning(f"Could not set iconphoto from ICO: {e}")

    def _load_png_fallback(self, base_dir: str) -> None:
        """Loads PNG icon as fallback when ICO is not available."""
        assert base_dir is not None, "base_dir must be provided"
        png_path = Path(base_dir) / "paper_plane_icon.png"
        if Path(png_path).exists():
            from PIL import Image, ImageTk

            try:
                image = Image.open(png_path)
                if image.mode != "RGBA":
                    image = image.convert("RGBA")

                photos = []
                for size in ICON_SIZES:
                    resized = image.resize((size, size), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(resized)
                    photos.append(photo)

                if photos:
                    self.root.iconphoto(True, *photos)
                    self.icon_photos = photos
                    logger.info(f"Loaded PNG icon: {png_path}")
            except (IOError, PermissionError, OSError) as e:
                logger.warning(f"Failed to load PNG icon: {e}")

        else:
            logger.warning(
                "No icon files found (paper_plane_icon.ico or paper_plane_icon.png)",
            )

    def create_scrollable_interface(self) -> None:
        """Creates a scrollable main interface."""
        # Create canvas and scrollbar
        canvas = tk.Canvas(self.root)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Main content frame
        main_frame = ttk.Frame(scrollable_frame, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Enable mouse wheel scrolling
        def _on_mousewheel(event: tk.Event) -> None:
            """Handle mouse wheel scrolling for the canvas.

            Args:
                event: Mouse wheel event containing delta information
            """
            try:
                if not hasattr(event, "delta"):
                    # Handle different mouse wheel event formats
                    if hasattr(event, "num"):
                        # Linux/Unix mouse wheel
                        delta = 120 if event.num == 4 else -120
                    else:
                        # Unknown format, skip
                        return
                else:
                    delta = event.delta

                # Scroll the canvas
                canvas.yview_scroll(int(-1 * (delta / 120)), "units")

            except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
                logger.debug(f"Mouse wheel scroll error: {e}")
                # Silently continue - mouse wheel errors shouldn't crash the app

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # --- UI SECTIONS ---
        self.create_source_widgets(main_frame)
        self.create_destination_widgets(main_frame)
        self.create_filtering_widgets(main_frame)
        self.create_preprocessing_widgets(main_frame)
        self.create_main_operation_widgets(main_frame)
        self.create_organization_widgets(main_frame)
        self.create_postprocessing_widgets(main_frame)
        self.create_output_options_widgets(main_frame)
        self.create_advanced_options_widgets(main_frame)
        self.create_progress_widgets(main_frame)
        self.create_run_button(main_frame)

        self.on_mode_change()  # Initial UI setup

    def create_source_widgets(self, parent: tk.Widget) -> None:
        """Create source folder selection widgets.

        Args:
            parent: Parent widget to contain the source widgets
        """
        assert parent is not None, "parent must be provided"
        self.source_frame = ttk.LabelFrame(
            parent,
            text="1. Select Folder(s) to Process",
            padding="10",
        )
        self.source_frame.pack(fill=tk.X, pady=5)

        # Source folder listbox with scrollbar
        listbox_frame = ttk.Frame(self.source_frame)
        listbox_frame.pack(fill=tk.X, expand=True)

        self.source_listbox = tk.Listbox(listbox_frame, height=6)
        source_scrollbar = ttk.Scrollbar(
            listbox_frame,
            orient="vertical",
            command=self.source_listbox.yview,
        )
        self.source_listbox.configure(yscrollcommand=source_scrollbar.set)

        self.source_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        source_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        button_frame = ttk.Frame(self.source_frame)
        button_frame.pack(fill=tk.X, pady=5)
        ttk.Button(
            button_frame,
            text="Add Folder(s)",
            command=self.select_source_folders,
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
        ttk.Button(
            button_frame,
            text="Remove Selected",
            command=self.remove_selected_source,
        ).pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(5, 0))

        # Add folder info label
        self.source_info_label = ttk.Label(
            self.source_frame,
            text="",
            foreground="blue",
        )
        self.source_info_label.pack(fill=tk.X, pady=2)

    def create_destination_widgets(self, parent: tk.Widget) -> None:
        """Create destination folder selection widgets.

        Args:
            parent: Parent widget to contain the destination widgets
        """
        assert parent is not None, "parent must be provided"
        self.dest_frame = ttk.LabelFrame(
            parent,
            text="2. Select Final Destination Folder",
            padding="10",
        )
        self.dest_frame.pack(fill=tk.X, pady=5)
        self.dest_label = ttk.Label(
            self.dest_frame,
            text="No destination selected.",
            foreground="grey",
        )
        self.dest_label.pack(fill=tk.X, expand=True, side=tk.LEFT)
        ttk.Button(
            self.dest_frame,
            text="Set Destination",
            command=self.select_dest_folder,
        ).pack(side=tk.RIGHT)

    def create_filtering_widgets(self, parent: tk.Widget) -> None:
        """Create file filtering configuration widgets.

        Args:
            parent: Parent widget to contain the filtering widgets
        """
        assert parent is not None, "parent must be provided"
        filter_frame = ttk.LabelFrame(
            parent,
            text="3. File Filtering Options",
            padding="10",
        )
        filter_frame.pack(fill=tk.X, pady=5)

        # File extensions filter
        ext_frame = ttk.Frame(filter_frame)
        ext_frame.pack(fill=tk.X, pady=2)
        ttk.Label(ext_frame, text="Include only extensions (comma-separated):").pack(
            side=tk.LEFT,
        )
        ttk.Entry(ext_frame, textvariable=self.filter_extensions, width=30).pack(
            side=tk.RIGHT,
        )
        ttk.Label(
            filter_frame,
            text="Example: .jpg,.png,.pdf (leave empty for all files)",
            foreground="grey",
        ).pack(anchor=tk.W)

        # File size filters
        size_frame = ttk.Frame(filter_frame)
        size_frame.pack(fill=tk.X, pady=5)
        ttk.Label(size_frame, text="Min size (MB):").pack(side=tk.LEFT)
        ttk.Entry(size_frame, textvariable=self.min_file_size, width=10).pack(
            side=tk.LEFT,
            padx=5,
        )
        ttk.Label(size_frame, text="Max size (MB):").pack(side=tk.LEFT, padx=(10, 0))
        ttk.Entry(size_frame, textvariable=self.max_file_size, width=10).pack(
            side=tk.LEFT,
            padx=5,
        )

    def create_preprocessing_widgets(self, parent: tk.Widget) -> None:
        """Create preprocessing configuration widgets.

        Args:
            parent: Parent widget to contain the preprocessing widgets
        """
        assert parent is not None, "parent must be provided"
        self.pre_process_frame = ttk.LabelFrame(
            parent,
            text="4. Pre-processing Options (On Source)",
            padding="10",
        )
        self.pre_process_frame.pack(fill=tk.X, pady=5)

        ttk.Checkbutton(
            self.pre_process_frame,
            text="Bulk extract archives (.zip, .rar, .7z)",
            variable=self.unzip_var,
        ).pack(anchor=tk.W)
        ttk.Checkbutton(
            self.pre_process_frame,
            text="Safe extraction (verify before deleting originals)",
            variable=self.safe_extract_var,
        ).pack(anchor=tk.W, padx=(20, 0))

    def create_main_operation_widgets(self, parent: tk.Widget) -> None:
        """Create main operation selection widgets.

        Args:
            parent: Parent widget to contain the main operation widgets
        """
        assert parent is not None, "parent must be provided"
        self.mode_frame = ttk.LabelFrame(
            parent,
            text="5. Choose Main Operation",
            padding="10",
        )
        self.mode_frame.pack(fill=tk.X, pady=5)

        ttk.Radiobutton(
            self.mode_frame,
            text="Combine & Copy",
            variable=self.operation_mode,
            value="combine",
            command=self.on_mode_change,
        ).pack(anchor=tk.W)
        ttk.Radiobutton(
            self.mode_frame,
            text="Flatten & Tidy",
            variable=self.operation_mode,
            value="flatten",
            command=self.on_mode_change,
        ).pack(anchor=tk.W)
        ttk.Radiobutton(
            self.mode_frame,
            text="Copy & Prune Empty Folders",
            variable=self.operation_mode,
            value="prune",
            command=self.on_mode_change,
        ).pack(anchor=tk.W)
        ttk.Radiobutton(
            self.mode_frame,
            text="Deduplicate Files (In-Place)",
            variable=self.operation_mode,
            value="deduplicate",
            command=self.on_mode_change,
        ).pack(anchor=tk.W)
        ttk.Radiobutton(
            self.mode_frame,
            text="Analyze & Report Only",
            variable=self.operation_mode,
            value="analyze",
            command=self.on_mode_change,
        ).pack(anchor=tk.W)

        self.mode_description = ttk.Label(
            self.mode_frame,
            text="",
            wraplength=600,
            justify=tk.LEFT,
        )
        self.mode_description.pack(fill=tk.X, pady=(5, 0))

    def create_organization_widgets(self, parent: tk.Widget) -> None:
        """Create file organization configuration widgets.

        Args:
            parent: Parent widget to contain the organization widgets
        """
        assert parent is not None, "parent must be provided"
        org_frame = ttk.LabelFrame(
            parent,
            text="6. File Organization Options",
            padding="10",
        )
        org_frame.pack(fill=tk.X, pady=5)

        ttk.Checkbutton(
            org_frame,
            text="Organize files by type (create subfolders)",
            variable=self.organize_by_type_var,
        ).pack(anchor=tk.W)
        ttk.Checkbutton(
            org_frame,
            text="Organize files by date (YYYY/MM folders)",
            variable=self.organize_by_date_var,
        ).pack(anchor=tk.W)

    def create_postprocessing_widgets(self, parent: tk.Widget) -> None:
        """Create postprocessing configuration widgets.

        Args:
            parent: Parent widget to contain the postprocessing widgets
        """
        assert parent is not None, "parent must be provided"
        self.post_process_frame = ttk.LabelFrame(
            parent,
            text="7. Post-processing Options (On Destination)",
            padding="10",
        )
        self.post_process_frame.pack(fill=tk.X, pady=5)

        ttk.Checkbutton(
            self.post_process_frame,
            text="Deduplicate renamed files in destination folder after copy",
            variable=self.deduplicate_var,
        ).pack(anchor=tk.W)

    def create_output_options_widgets(self, parent: tk.Widget) -> None:
        """Create output options configuration widgets.

        Args:
            parent: Parent widget to contain the output options widgets
        """
        assert parent is not None, "parent must be provided"
        output_frame = ttk.LabelFrame(parent, text="8. Output Options", padding="10")
        output_frame.pack(fill=tk.X, pady=5)

        ttk.Checkbutton(
            output_frame,
            text="Create ZIP archive of final result",
            variable=self.zip_output_var,
        ).pack(anchor=tk.W)

    def create_advanced_options_widgets(self, parent: tk.Widget) -> None:
        """Create advanced options configuration widgets.

        Args:
            parent: Parent widget to contain the advanced options widgets
        """
        assert parent is not None, "parent must be provided"
        advanced_frame = ttk.LabelFrame(
            parent,
            text="9. Advanced Options",
            padding="10",
        )
        advanced_frame.pack(fill=tk.X, pady=5)

        ttk.Checkbutton(
            advanced_frame,
            text="Preview mode (show what would be done without executing)",
            variable=self.preview_mode_var,
        ).pack(anchor=tk.W)
        ttk.Checkbutton(
            advanced_frame,
            text="Create backup before processing",
            variable=self.backup_before_var,
        ).pack(anchor=tk.W)

    def create_progress_widgets(self, parent: tk.Widget) -> None:
        """Create progress tracking widgets.

        Args:
            parent: Parent widget to contain the progress widgets
        """
        assert parent is not None, "parent must be provided"
        progress_frame = ttk.LabelFrame(parent, text="Progress", padding="10")
        progress_frame.pack(fill=tk.X, pady=5)

        self.progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=100,
            mode="determinate",
        )
        self.progress_bar.pack(fill=tk.X, pady=2)

        self.status_label = ttk.Label(progress_frame, textvariable=self.status_var)
        self.status_label.pack(anchor=tk.W)

    def create_run_button(self, parent: tk.Widget) -> None:
        """Create the main run button widget.

        Args:
            parent: Parent widget to contain the run button
        """
        assert parent is not None, "parent must be provided"
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=(10, 5))

        self.run_button = ttk.Button(
            button_frame,
            text="Run Process",
            command=self.run_processing_threaded,
            style="Accent.TButton",
        )
        self.run_button.pack(
            side=tk.LEFT,
            expand=True,
            fill=tk.X,
            padx=(0, 5),
            ipady=10,
        )

        self.cancel_button = ttk.Button(
            button_frame,
            text="Cancel",
            command=self.cancel_processing,
            state=tk.DISABLED,
        )
        self.cancel_button.pack(side=tk.RIGHT, padx=(5, 0), ipady=10)

        style = ttk.Style()
        style.configure("Accent.TButton", font=("Helvetica", 10, "bold"))

    def on_mode_change(self) -> None:
        """Updates UI descriptions and widget states based on the selected operation
        mode."""
        mode = self.operation_mode.get()

        # Update description
        descriptions = {
            "combine": (
                "Copies all files from source folders into the single destination "
                "folder."
            ),
            "flatten": (
                "Finds deeply nested folders and copies them to the top level of "
                "the destination."
            ),
            "prune": (
                "Copies source folders to the destination, preserving structure but "
                "skipping empty sub-folders."
            ),
            "deduplicate": (
                "Deletes renamed duplicates like 'file (1).txt' within the source "
                "folder(s), keeping the newest version."
            ),
            "analyze": (
                "Analyzes folder contents and generates a detailed report without "
                "making changes."
            ),
        }
        self.mode_description.config(text=descriptions.get(mode, ""))

        # Enable/disable widgets
        is_deduplicate_or_analyze = mode in ["deduplicate", "analyze"]
        new_state = tk.DISABLED if is_deduplicate_or_analyze else tk.NORMAL

        frames_to_toggle = [
            self.dest_frame,
            self.pre_process_frame,
            self.post_process_frame,
        ]
        for frame in frames_to_toggle:
            for child in frame.winfo_children():
                if hasattr(child, "configure"):
                    child.configure(state=new_state)  # type: ignore[call-arg]

    def select_source_folders(self) -> None:
        """Open folder selection dialog to add source folders.

        This method allows users to select folders that will be processed by
        the application.
        Selected folders are added to the source_folders list and displayed in the UI.

        Args:
            None - uses filedialog.askdirectory() for user input

        Returns:
            None - updates self.source_folders and UI state

        Raises:
            OSError: If file system operations fail during folder validation
            PermissionError: If insufficient permissions to access selected folder
            Exception: If folder selection fails for other reasons
        """
        try:
            folder = filedialog.askdirectory(
                mustexist=True,
                title="Select a folder to process",
            )
            if folder:
                # Validate folder exists and is accessible
                if not Path(folder).exists():
                    messagebox.showerror("Error", "Selected folder no longer exists.")
                    return

                if not os.access(folder, os.R_OK):
                    messagebox.showerror(
                        "Error",
                        "Cannot access the selected folder. Check permissions.",
                    )
                    return

                if folder not in self.source_folders:
                    self.source_folders.append(folder)
                    self.source_listbox.insert(tk.END, folder)
                    self.update_source_info()
                    logger.info("Added source folder: %s", folder)
                else:
                    messagebox.showinfo(
                        "Info",
                        "This folder is already in the source list.",
                    )
            else:
                logger.debug("Folder selection cancelled by user")

        except (IOError, PermissionError, OSError) as e:
            logger.exception("Error selecting source folder")
            messagebox.showerror("Error", f"Failed to select source folder: {e}")

    def remove_selected_source(self) -> None:
        """Remove selected source folders from the list.

        This method removes user-selected folders from the source_folders list.
        It prompts for confirmation before removal and updates both the internal
        list and the UI display.

        Args:
            None - uses self.source_listbox.curselection() for user input

        Returns:
            None - updates self.source_folders and UI state

        Raises:
            IndexError: If selected indices are invalid
            Exception: If folder removal fails for other reasons
        """
        try:
            selected_indices = list(self.source_listbox.curselection())  # type: ignore[no-untyped-call]
            if not selected_indices:
                messagebox.showinfo("Info", "Please select folders to remove.")
                return

            # Confirm removal
            if len(selected_indices) == 1:
                folder_name = Path(self.source_folders[selected_indices[0]]).name
                confirm = messagebox.askyesno(
                    "Confirm Removal",
                    f"Remove folder '{folder_name}' from source list?",
                )
            else:
                confirm = messagebox.askyesno(
                    "Confirm Removal",
                    f"Remove {len(selected_indices)} "
                    "selected folders from source list?",
                )

            if confirm:
                # Remove in reverse order to maintain indices
                for i in sorted(selected_indices, reverse=True):
                    removed_folder = self.source_folders.pop(i)
                    self.source_listbox.delete(i)
                    logger.info("Removed source folder: %s", removed_folder)

                self.update_source_info()

        except (IOError, PermissionError, OSError) as e:
            logger.exception("Error removing source folders")
            messagebox.showerror("Error", f"Failed to remove source folders: {e}")

    def select_dest_folder(self) -> None:
        """Open folder selection dialog to select destination folder.

        This method allows users to select the destination folder where processed
        files will be placed. The selected folder is validated for write access
        and stored in self.dest_folder.

        Args:
            None - uses filedialog.askdirectory() for user input

        Returns:
            None - updates self.dest_folder and UI state

        Raises:
            OSError: If file system operations fail during folder validation
            PermissionError: If insufficient permissions to write to selected folder
            Exception: If folder selection fails for other reasons
        """
        try:
            folder = filedialog.askdirectory(
                mustexist=True,
                title="Select the destination folder",
            )
            if folder:
                # Validate folder exists and is writable
                if not Path(folder).exists():
                    messagebox.showerror("Error", "Selected folder no longer exists.")
                    return

                if not os.access(folder, os.W_OK):
                    messagebox.showerror(
                        "Error",
                        "Cannot write to the selected folder. Check permissions.",
                    )
                    return

                self.dest_folder = folder
                self.dest_label.config(text=self.dest_folder, foreground="black")
                logger.info("Set destination folder: %s", folder)
            else:
                logger.debug("Destination folder selection cancelled by user")

        except (IOError, PermissionError, OSError) as e:
            logger.exception("Error selecting destination folder")
            messagebox.showerror("Error", f"Failed to select destination folder: {e}")
