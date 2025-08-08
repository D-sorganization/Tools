import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import shutil
import logging
import re

# Set up logging to capture detailed information
log_filename = 'folder_processor.log'
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_filename, mode='w')])

class FolderProcessorApp:
    """
    A GUI application for folder processing tasks, with optional pre and post-processing steps.
    """

    def __init__(self, root_window):
        """
        Initializes the application's user interface.
        """
        self.root = root_window
        self.root.title("Folder Processor Tool")
        self.root.geometry("600x710")
        self.root.minsize(500, 650)

        # --- UI Variables ---
        self.source_folders = []
        self.dest_folder = ""
        self.unzip_var = tk.BooleanVar(value=False)
        self.deduplicate_var = tk.BooleanVar(value=False)
        self.operation_mode = tk.StringVar(value="combine")

        # --- UI Style ---
        style = ttk.Style()
        style.configure("TButton", padding=6, relief="flat")
        style.configure("TLabel", padding=5)

        # --- Main Frame ---
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- UI SECTIONS ---
        self.create_source_widgets(main_frame)
        self.create_destination_widgets(main_frame)
        self.create_preprocessing_widgets(main_frame)
        self.create_main_operation_widgets(main_frame)
        self.create_postprocessing_widgets(main_frame)
        self.create_run_button(main_frame)
        
        self.on_mode_change() # Initial UI setup

    # --- Widget Creation Methods ---
    def create_source_widgets(self, parent):
        self.source_frame = ttk.LabelFrame(parent, text="1. Select Folder(s) to Process", padding="10")
        self.source_frame.pack(fill=tk.X, expand=True, pady=5)
        self.source_listbox = tk.Listbox(self.source_frame, height=6)
        self.source_listbox.pack(fill=tk.X, expand=True, side=tk.TOP)
        button_frame = ttk.Frame(self.source_frame)
        button_frame.pack(fill=tk.X, pady=5)
        ttk.Button(button_frame, text="Add Folder(s)", command=self.select_source_folders).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
        ttk.Button(button_frame, text="Remove Selected", command=self.remove_selected_source).pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(5, 0))

    def create_destination_widgets(self, parent):
        self.dest_frame = ttk.LabelFrame(parent, text="2. Select Final Destination Folder", padding="10")
        self.dest_frame.pack(fill=tk.X, expand=True, pady=5)
        self.dest_label = ttk.Label(self.dest_frame, text="No destination selected.", foreground="grey")
        self.dest_label.pack(fill=tk.X, expand=True, side=tk.LEFT)
        ttk.Button(self.dest_frame, text="Set Destination", command=self.select_dest_folder).pack(side=tk.RIGHT)

    def create_preprocessing_widgets(self, parent):
        self.pre_process_frame = ttk.LabelFrame(parent, text="3. Pre-processing Options (On Source)", padding="10")
        self.pre_process_frame.pack(fill=tk.X, expand=True, pady=5)
        ttk.Checkbutton(self.pre_process_frame, text="Bulk extract archives (.zip, .rar, .7z) and delete originals", variable=self.unzip_var).pack(anchor=tk.W)

    def create_main_operation_widgets(self, parent):
        self.mode_frame = ttk.LabelFrame(parent, text="4. Choose Main Operation", padding="10")
        self.mode_frame.pack(fill=tk.X, expand=True, pady=5)
        ttk.Radiobutton(self.mode_frame, text="Combine & Copy", variable=self.operation_mode, value="combine", command=self.on_mode_change).pack(anchor=tk.W)
        ttk.Radiobutton(self.mode_frame, text="Flatten & Tidy", variable=self.operation_mode, value="flatten", command=self.on_mode_change).pack(anchor=tk.W)
        ttk.Radiobutton(self.mode_frame, text="Copy & Prune Empty Folders", variable=self.operation_mode, value="prune", command=self.on_mode_change).pack(anchor=tk.W)
        ttk.Radiobutton(self.mode_frame, text="Deduplicate Files (In-Place)", variable=self.operation_mode, value="deduplicate", command=self.on_mode_change).pack(anchor=tk.W)
        self.mode_description = ttk.Label(self.mode_frame, text="", wraplength=500, justify=tk.LEFT)
        self.mode_description.pack(fill=tk.X, pady=(5,0))

    def create_postprocessing_widgets(self, parent):
        self.post_process_frame = ttk.LabelFrame(parent, text="5. Post-processing Options (On Destination)", padding="10")
        self.post_process_frame.pack(fill=tk.X, expand=True, pady=5)
        ttk.Checkbutton(self.post_process_frame, text="Deduplicate renamed files in destination folder after copy", variable=self.deduplicate_var).pack(anchor=tk.W)

    def create_run_button(self, parent):
        ttk.Button(parent, text="Run Process", command=self.run_processing, style="Accent.TButton").pack(fill=tk.X, ipady=10, pady=(10,5))
        style = ttk.Style(); style.configure("Accent.TButton", font=("Helvetica", 10, "bold"))

    # --- UI Logic and Control ---
    def on_mode_change(self):
        """Updates UI descriptions and widget states based on the selected operation mode."""
        mode = self.operation_mode.get()
        # Update description
        if mode == "combine": self.mode_description.config(text="Copies all files from source folders into the single destination folder.")
        elif mode == "flatten": self.mode_description.config(text="Finds deeply nested folders and copies them to the top level of the destination.")
        elif mode == "prune": self.mode_description.config(text="Copies source folders to the destination, preserving structure but skipping empty sub-folders.")
        elif mode == "deduplicate": self.mode_description.config(text="Deletes renamed duplicates like 'file (1).txt' within the source folder(s), keeping the newest version.")
        
        # Enable/disable widgets
        is_deduplicate_mode = (mode == "deduplicate")
        new_state = tk.DISABLED if is_deduplicate_mode else tk.NORMAL
        
        for frame in [self.dest_frame, self.pre_process_frame, self.post_process_frame]:
            for child in frame.winfo_children():
                child.configure(state=new_state)

    # --- Core Application Logic ---
    def run_processing(self):
        """Main function to start the selected processing workflow."""
        mode = self.operation_mode.get()

        if mode == 'deduplicate':
            # Handle standalone deduplication
            if not self.validate_inputs(check_destination=False): return
            try:
                results_log = self._run_deduplicate_main_op()
                messagebox.showinfo("Operation Complete", "Deduplication complete.\n\n" + "\n".join(results_log))
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred during deduplication: {e}")
            return

        # Handle Source -> Destination workflows
        if not self.validate_inputs(check_destination=True): return
        
        # Pre-processing
        if self.unzip_var.get():
            try:
                unzip_log = self._bulk_unzip()
                if not messagebox.askyesno("Pre-processing Complete", f"Bulk Extraction Complete!\n\n" + "\n".join(unzip_log) + "\n\nDo you want to proceed?"): return
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred during bulk unzip: {e}"); return
        
        # Main Operation
        try:
            main_op_log = []
            if mode == "combine": main_op_log = self._combine_folders()
            elif mode == "flatten": main_op_log = self._flatten_folders()
            elif mode == "prune": main_op_log = self._prune_empty_folders()
            final_summary = "Main Operation Complete!\n\n" + "\n".join(main_op_log)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during the main operation: {e}"); return

        # Post-processing
        if self.deduplicate_var.get():
            try:
                dedupe_log = self._perform_deduplication(self.dest_folder)
                final_summary += "\n\n--- Deduplication Results ---\n" + "\n".join(dedupe_log)
            except Exception as e:
                final_summary += f"\n\n--- Deduplication FAILED: {e}"
        
        messagebox.showinfo("All Operations Complete", final_summary)

    def validate_inputs(self, check_destination=True):
        if not self.source_folders:
            messagebox.showerror("Error", "Please add at least one source folder."); return False
        if check_destination:
            if not self.dest_folder:
                messagebox.showerror("Error", "Please select a destination folder."); return False
            if any(src == self.dest_folder for src in self.source_folders):
                messagebox.showerror("Error", "The destination folder cannot be a source folder."); return False
        return True

    # --- Backend Processing Methods ---
    def _perform_deduplication(self, target_folder):
        """Core logic to find and delete renamed duplicates in a single target folder."""
        log = []; deleted_count = 0
        pattern = re.compile(r"(.+?)(?: \((\d+)\))?(\.\w+)$")
        
        confirm = messagebox.askyesno(
            "Confirm Deletion",
            f"This will permanently delete duplicate files in:\n{target_folder}\n\nIt keeps the newest version of files like 'file (1).txt'. This cannot be undone. Are you sure?"
        )
        if not confirm: return ["Deduplication cancelled by user."]

        log.append(f"Processing folder: {target_folder}")
        for dirpath, _, filenames in os.walk(target_folder):
            files_by_base_name = {}
            for filename in filenames:
                match = pattern.match(filename)
                if match:
                    base, _, ext = match.groups(); base_name = f"{base}{ext}"
                    files_by_base_name.setdefault(base_name, []).append(os.path.join(dirpath, filename))
            for base_name, files in files_by_base_name.items():
                if len(files) > 1:
                    try: file_to_keep = max(files, key=lambda f: os.path.getmtime(f))
                    except FileNotFoundError: continue
                    log.append(f"Duplicate set for '{base_name}': Keeping '{os.path.basename(file_to_keep)}'")
                    for file_path in files:
                        if file_path != file_to_keep:
                            try: os.remove(file_path); log.append(f"  - DELETED: '{os.path.basename(file_path)}'"); deleted_count += 1
                            except OSError as e: log.append(f"  - FAILED to delete '{os.path.basename(file_path)}': {e}")
        
        summary = [f"Deduplication complete.", f"Deleted a total of {deleted_count} files."] + log[:20]
        if len(log) > 20: summary.append("... (see log for full details)")
        return summary
    
    def _run_deduplicate_main_op(self):
        """Wrapper for running deduplication as a main, in-place operation on source folders."""
        full_log = []
        for folder in self.source_folders:
            folder_log = self._perform_deduplication(folder)
            full_log.extend(folder_log)
            full_log.append("---")
        return full_log

    def _get_unique_path(self, path):
        if not os.path.exists(path): return path
        parent, name = os.path.split(path)
        is_file = '.' in name and not os.path.isdir(path)
        filename, ext = os.path.splitext(name) if is_file else (name, '')
        counter = 1
        new_path = os.path.join(parent, f"{filename} ({counter}){ext}")
        while os.path.exists(new_path):
            counter += 1; new_path = os.path.join(parent, f"{filename} ({counter}){ext}")
        return new_path

    # ... Other backend methods (_bulk_unzip, _combine_folders, etc.) remain here ...
    def select_source_folders(self):
        folder = filedialog.askdirectory(mustexist=True, title="Select a folder to process")
        if folder and folder not in self.source_folders:
            self.source_folders.append(folder); self.source_listbox.insert(tk.END, folder)

    def remove_selected_source(self):
        for i in sorted(self.source_listbox.curselection(), reverse=True):
            self.source_folders.pop(i); self.source_listbox.delete(i)

    def select_dest_folder(self):
        folder = filedialog.askdirectory(mustexist=True, title="Select the destination folder")
        if folder: self.dest_folder = folder; self.dest_label.config(text=self.dest_folder, foreground="black")

    def _bulk_unzip(self):
        log, extracted_count = ["Starting bulk extraction..."], 0
        archives = [os.path.join(dp, f) for sf in self.source_folders for dp, dn, fn in os.walk(sf) for f in fn if f.lower().endswith(('.zip', '.rar', '.7z'))]
        if not archives: return ["No archives found to extract."]
        for archive_path in archives:
            if not os.path.exists(archive_path): continue
            extract_dir = self._get_unique_path(os.path.splitext(archive_path)[0])
            try: shutil.unpack_archive(archive_path, extract_dir); os.remove(archive_path); log.append(f"Extracted and deleted '{os.path.basename(archive_path)}'"); extracted_count += 1
            except Exception as e: log.append(f"FAILED to extract '{os.path.basename(archive_path)}': {e}")
        return [f"Processed {len(archives)} archive(s). Successfully extracted: {extracted_count}"] + log[1:]

    def _combine_folders(self):
        log, file_count, renamed_count = [], 0, 0; os.makedirs(self.dest_folder, exist_ok=True)
        for src in self.source_folders:
            for dp, _, fn in os.walk(src):
                for f in fn:
                    sp, dp = os.path.join(dp, f), os.path.join(self.dest_folder, f); fdp = self._get_unique_path(dp)
                    if fdp != dp: log.append(f"Renamed: '{f}' to '{os.path.basename(fdp)}'"); renamed_count += 1
                    try: shutil.copy2(sp, fdp); file_count += 1
                    except Exception as e: log.append(f"ERROR copying '{f}': {e}")
        return [f"Copied {file_count} files.", f"Renamed {renamed_count} files due to duplicates."] + log[:10]

    def _flatten_folders(self):
        log, moved_count = [], 0
        for src in self.source_folders:
            cp = src
            while True:
                try: children = os.listdir(cp)
                except FileNotFoundError: cp = None; break
                if any(os.path.isfile(os.path.join(cp, c)) for c in children): break
                subdirs = [d for d in children if os.path.isdir(os.path.join(cp, d))];
                if len(subdirs) == 1: cp = os.path.join(cp, subdirs[0])
                else: break
            if cp and os.path.exists(cp):
                fdp = self._get_unique_path(os.path.join(self.dest_folder, os.path.basename(cp)))
                if fdp != os.path.join(self.dest_folder, os.path.basename(cp)): log.append(f"Renamed folder: '{os.path.basename(cp)}' to '{os.path.basename(fdp)}'")
                try: shutil.copytree(cp, fdp); log.append(f"Copied '{os.path.basename(cp)}'"); moved_count += 1
                except Exception as e: log.append(f"ERROR copying '{os.path.basename(cp)}': {e}")
            else: log.append(f"Skipped '{os.path.basename(src)}': No content folder found.")
        return [f"Copied {moved_count} tidy folder structures."] + log[:10]

    def _prune_empty_folders(self):
        log, fc, pf = [], 0, 0
        for src in self.source_folders:
            dbp = self._get_unique_path(os.path.join(self.dest_folder, os.path.basename(src)))
            if dbp != os.path.join(self.dest_folder, os.path.basename(src)): log.append(f"Renamed base folder: '{os.path.basename(src)}' to '{os.path.basename(dbp)}'")
            if not any(fn for _, _, fn in os.walk(src)): log.append(f"Skipped '{os.path.basename(src)}' as it's empty."); continue
            pf += 1
            for dp, _, fn in os.walk(src):
                if not fn: continue
                for f in fn:
                    sp, rp = os.path.join(dp, f), os.path.relpath(sp, src)
                    dfp, ddff = os.path.join(dbp, rp), os.path.dirname(os.path.join(dbp, rp))
                    try: os.makedirs(ddff, exist_ok=True); shutil.copy2(sp, dfp); fc += 1
                    except Exception as e: log.append(f"ERROR copying '{f}': {e}")
        return [f"Processed {pf} non-empty source folder(s).", f"Copied a total of {fc} files."] + log[:10]

if __name__ == "__main__":
    root = tk.Tk()
    app = FolderProcessorApp(root)
    root.mainloop()