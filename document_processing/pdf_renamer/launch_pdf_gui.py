import logging
import multiprocessing
import os
import sys
import threading
import tkinter as tk
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

# Ensure src is in path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from pdf_renamer.deduper import DuplicateFinder
from pdf_renamer.worker import process_single_file

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class RedirectText:
    """Redirect stdout/stderr to a tkinter text widget."""

    def __init__(self, text_widget):
        self.output = text_widget

    def write(self, string):
        self.output.insert(tk.END, string)
        self.output.see(tk.END)

    def flush(self):
        pass


class PDFRenamerLauncher:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("PDF Renamer Tool")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")

        # Variables
        self.dir_var = tk.StringVar()
        self.style_var = tk.StringVar(value="standard")
        self.dry_run_var = tk.BooleanVar(value=True)
        self.delete_dups_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Ready")
        self.is_running = False

        self.setup_ui()

    def setup_ui(self):
        # Header
        header_frame = tk.Frame(self.root, bg="#3498db", height=60)
        header_frame.pack(fill="x")
        header_frame.pack_propagate(False)

        tk.Label(
            header_frame,
            text="📄 PDF Renamer Tool",
            font=("Arial", 16, "bold"),
            bg="#3498db",
            fg="white",
        ).pack(side="left", padx=20)

        # Version/Status
        tk.Label(
            header_frame,
            text="v2.1 (Parallel Processing)",
            font=("Arial", 9),
            bg="#3498db",
            fg="#ecf0f1",
        ).pack(side="right", padx=20)

        # Main Layout: Top (Controls) and Bottom (Logs)
        main_pane = tk.PanedWindow(self.root, orient="vertical", bg="#f0f0f0")
        main_pane.pack(fill="both", expand=True, padx=10, pady=10)

        # Controls Frame
        controls_frame = tk.Frame(main_pane, bg="#f0f0f0")
        main_pane.add(controls_frame, minsize=200)

        # Directory Selection
        tk.Label(
            controls_frame,
            text="Target Directory:",
            font=("Arial", 10, "bold"),
            bg="#f0f0f0",
        ).grid(row=0, column=0, sticky="w", pady=(10, 5), padx=5)

        dir_entry = tk.Entry(
            controls_frame, textvariable=self.dir_var, font=("Arial", 10), width=60
        )
        dir_entry.grid(row=0, column=1, sticky="ew", pady=(10, 5), padx=5)

        tk.Button(
            controls_frame,
            text="Browse...",
            command=self.browse_directory,
            bg="#ecf0f1",
        ).grid(row=0, column=2, sticky="e", pady=(10, 5), padx=5)

        controls_frame.columnconfigure(1, weight=1)

        # Settings Group
        settings_frame = tk.LabelFrame(
            controls_frame, text="Settings", bg="#f0f0f0", font=("Arial", 9, "bold")
        )
        settings_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=10, padx=5)

        # Naming Style
        tk.Label(settings_frame, text="Naming Style:", bg="#f0f0f0").grid(
            row=0, column=0, sticky="w", padx=10, pady=5
        )

        styles = [
            ("Standard (Author - Title.pdf)", "standard"),
            ("Snake Case (author_title.pdf)", "snake_case"),
            ("Kebab Case (author-title.pdf)", "kebab_case"),
        ]

        style_frame = tk.Frame(settings_frame, bg="#f0f0f0")
        style_frame.grid(row=0, column=1, sticky="w", padx=5)

        for _, (text, value) in enumerate(styles):
            tk.Radiobutton(
                style_frame,
                text=text,
                variable=self.style_var,
                value=value,
                bg="#f0f0f0",
            ).pack(side="left", padx=5)

        # Checkboxes
        tk.Checkbutton(
            settings_frame,
            text="Dry Run (Preview changes only)",
            variable=self.dry_run_var,
            bg="#f0f0f0",
        ).grid(row=1, column=0, columnspan=2, sticky="w", padx=10, pady=5)

        tk.Checkbutton(
            settings_frame,
            text="Delete Duplicates",
            variable=self.delete_dups_var,
            bg="#f0f0f0",
        ).grid(row=1, column=2, sticky="e", padx=10, pady=5)

        # Progress Section
        progress_frame = tk.Frame(controls_frame, bg="#f0f0f0")
        progress_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=10, padx=5)

        self.progress = ttk.Progressbar(
            progress_frame, orient="horizontal", mode="determinate"
        )
        self.progress.pack(fill="x", expand=True)

        self.status_label = tk.Label(
            progress_frame, textvariable=self.status_var, bg="#f0f0f0", anchor="w"
        )
        self.status_label.pack(fill="x", pady=(5, 0))

        # Run Button
        self.run_btn = tk.Button(
            controls_frame,
            text="Start Renaming",
            command=self.start_processing_thread,
            bg="#2ecc71",
            fg="white",
            font=("Arial", 12, "bold"),
            padx=20,
            pady=10,
            relief="flat",
        )
        self.run_btn.grid(row=3, column=0, columnspan=3, pady=10)

        # Log Output Frame
        log_frame = tk.LabelFrame(main_pane, text="Execution Log", bg="#f0f0f0")
        main_pane.add(log_frame, stretch="always")

        self.log_text = scrolledtext.ScrolledText(
            log_frame, font=("Consolas", 9), state="normal"
        )
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)

        # Tags for colored logs
        self.log_text.tag_config("INFO", foreground="black")
        self.log_text.tag_config("SUCCESS", foreground="green")
        self.log_text.tag_config("WARNING", foreground="#e67e22")
        self.log_text.tag_config("ERROR", foreground="red")

    def browse_directory(self):
        directory = filedialog.askdirectory()
        if directory:
            self.dir_var.set(directory)

    def log(self, message: str, level: str = "INFO"):
        """Thread-safe logging to the text widget."""

        def _log():
            self.log_text.insert(tk.END, f"[{level}] {message}\n", level)
            self.log_text.see(tk.END)

        self.root.after(0, _log)

    def update_status(self, message: str, progress: int = 0, total: int = 0):
        """Thread-safe status update."""

        def _update():
            self.status_var.set(message)
            if total > 0:
                self.progress["value"] = (progress / total) * 100
            else:
                self.progress["value"] = 0

        self.root.after(0, _update)

    def start_processing_thread(self):
        if self.is_running:
            return

        target_dir = self.dir_var.get()
        if not target_dir or not os.path.exists(target_dir):
            messagebox.showerror("Error", "Please select a valid directory.")
            return

        self.is_running = True
        self.run_btn.config(state="disabled", text="Processing...")
        self.log_text.delete(1.0, tk.END)  # Clear logs

        # Start worker thread
        threading.Thread(target=self.run_process, daemon=True).start()

    def run_process(self):
        try:
            directory = Path(self.dir_var.get())
            dry_run = self.dry_run_var.get()
            style = self.style_var.get()
            delete_dups = self.delete_dups_var.get()

            self.log(f"Starting processing in: {directory}")
            self.log(
                f"Configuration: Style={style}, DryRun={dry_run}, DeleteDups={delete_dups}"
            )

            # 1. Handle Duplicates
            self.update_status("Scanning for duplicates...", 0, 100)
            self.log("Scanning for duplicates...")

            # Note: Duplicate finding is still sequential as it requires whole-directory context
            finder = DuplicateFinder(directory)
            duplicates = finder.find_duplicates()

            if duplicates:
                self.log(f"Found {len(duplicates)} sets of duplicates.", "WARNING")
                for _, paths in duplicates.items():
                    if delete_dups:
                        sorted_paths = sorted(
                            paths, key=lambda p: (len(str(p)), p.name)
                        )
                        keep = sorted_paths[0]
                        to_delete = sorted_paths[1:]
                        self.log(f"Keeping: {keep.name}", "SUCCESS")

                        for p in to_delete:
                            if dry_run:
                                self.log(f"[DRY RUN] Would delete: {p.name}", "WARNING")
                            else:
                                try:
                                    p.unlink()
                                    self.log(f"Deleted: {p.name}", "WARNING")
                                except Exception as e:
                                    self.log(f"Failed to delete {p.name}: {e}", "ERROR")
                    else:
                        self.log(
                            f"Duplicate set: {[p.name for p in paths]} (Use 'Delete Duplicates' to fix)",
                            "WARNING",
                        )
            else:
                self.log("No duplicates found.", "SUCCESS")

            # 2. Rename Files (Parallel)
            self.update_status("Scanning files...", 10, 100)
            pdf_files = list(directory.glob("**/*.pdf"))
            total_files = len(pdf_files)

            if total_files == 0:
                self.log("No PDF files found in directory.", "WARNING")
                self.finish_processing()
                return

            self.log(f"Found {total_files} PDF files. Starting renaming...", "INFO")

            # ThreadPoolExecutor is better here because process_single_file might not be pickleable
            # if we are not careful, but PDF processing is CPU heavy.
            # Let's try ProcessPool first. If it fails on Windows due to pickling, we fallback to ThreadPool.
            # However, PDFMiner (inside pdfplumber) is pure python, so it blocks GIL. ProcessPool is needed.

            processed_count = 0

            # Using ProcessPoolExecutor for CPU bound tasks
            # Max workers = CPU count
            max_workers = min(os.cpu_count() or 4, 8)

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Map futures
                future_to_file = {
                    executor.submit(process_single_file, p, style, dry_run): p
                    for p in pdf_files
                }

                for future in as_completed(future_to_file):
                    processed_count += 1
                    file_path = future_to_file[future]

                    try:
                        result = future.result()
                        # Log based on result content
                        if "❌" in result:
                            self.log(result, "ERROR")
                        elif "⚠️" in result:
                            self.log(result, "WARNING")
                        elif "✅" in result:
                            self.log(result, "SUCCESS")
                        else:
                            self.log(result, "INFO")

                    except Exception as e:
                        self.log(f"Executor failed for {file_path.name}: {e}", "ERROR")

                    self.update_status(
                        f"Processed {processed_count}/{total_files} files",
                        processed_count,
                        total_files,
                    )

            self.log("Processing complete!", "SUCCESS")
            self.update_status("Done", 100, 100)
            messagebox.showinfo(
                "Complete", f"Processed {total_files} files successfully."
            )

        except Exception as e:
            self.log(f"Critical Error: {e}", "ERROR")
            messagebox.showerror("Error", f"An error occurred:\n{e}")
        finally:
            self.finish_processing()

    def finish_processing(self):
        self.is_running = False

        def _reset():
            self.run_btn.config(state="normal", text="Start Renaming")

        self.root.after(0, _reset)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    multiprocessing.freeze_support()  # Required for Windows PyInstaller/ProcessPool
    PDFRenamerLauncher().run()
