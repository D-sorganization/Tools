"""
gui/main_window.py
==================
Main application window for the DWSIM Gasification Model GUI.

Starts with:
    python -m dwsim_model.gui

or from the project root:
    python launch_gui.py

Layout
------
┌─────────────────────────────────────────────────────┐
│  Title bar + toolbar (Load Config | Save | Run)     │
├──────────┬──────────────────────────────────────────┤
│          │ Tab content area                         │
│  Sidebar │  ┌──────────────────────────────────┐   │
│  (status │  │ Feeds | Reactors | Energy | ...  │   │
│   panel) │  └──────────────────────────────────┘   │
└──────────┴──────────────────────────────────────────┘

The "Run" button validates all tabs, builds the simulation config,
runs the model in a background thread (so the GUI stays responsive),
and then displays results in the Results tab.
"""

from __future__ import annotations

import logging
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from dwsim_model.gui.widgets import apply_styles

logger = logging.getLogger(__name__)


class MainWindow(tk.Tk):
    """
    Root Tkinter window for the gasification GUI.

    Manages the shared application state (current config dict, results,
    background simulation thread) and coordinates the tab panels.
    """

    APP_TITLE = "DWSIM Gasification Model — v2.0"
    APP_WIDTH = 1100
    APP_HEIGHT = 780
    MIN_WIDTH = 900
    MIN_HEIGHT = 600

    def __init__(self):
        super().__init__()
        self.title(self.APP_TITLE)
        self.geometry(f"{self.APP_WIDTH}x{self.APP_HEIGHT}")
        self.minsize(self.MIN_WIDTH, self.MIN_HEIGHT)

        apply_styles(self)

        # Application state
        self._config: dict = {}
        self._config_path: Path | None = None
        self._sim_thread: threading.Thread | None = None
        self._last_results = None
        self._last_metrics = None

        self._build_menu()
        self._build_toolbar()
        self._build_main_area()
        self._build_status_bar()

        # Route all module logging to the Results tab log panel
        self._results_tab.log_panel.attach_to_logger("dwsim_model")

        self._load_default_config()

    # ─────────────────────────────────────────────────────────────────────
    # UI construction
    # ─────────────────────────────────────────────────────────────────────

    def _build_menu(self) -> None:
        menubar = tk.Menu(self)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Load Config…", command=self._on_load_config)
        file_menu.add_command(label="Save Config", command=self._on_save_config)
        file_menu.add_command(label="Save Config As…", command=self._on_save_config_as)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=file_menu)

        # Run menu
        run_menu = tk.Menu(menubar, tearoff=0)
        run_menu.add_command(label="Run Simulation", command=self._on_run)
        run_menu.add_command(label="Validate Config", command=self._on_validate)
        run_menu.add_command(label="Export to DWSIM…", command=self._on_export)
        menubar.add_cascade(label="Run", menu=run_menu)

        # Scenario menu
        sc_menu = tk.Menu(menubar, tearoff=0)

        def make_command(s: str):
            return lambda: self._on_load_scenario(s)

        for scenario in ("baseline", "high_steam", "air_blown"):
            sc_menu.add_command(
                label=scenario.replace("_", " ").title(),
                command=make_command(scenario),
            )
        menubar.add_cascade(label="Scenarios", menu=sc_menu)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self._on_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.config(menu=menubar)

    def _build_toolbar(self) -> None:
        toolbar = ttk.Frame(self)
        toolbar.pack(side="top", fill="x", padx=0, pady=0)

        # Separator line below toolbar
        ttk.Separator(toolbar, orient="horizontal").pack(fill="x", side="bottom")

        ttk.Button(toolbar, text="📂 Load Config", command=self._on_load_config).pack(
            side="left", padx=4, pady=4
        )
        ttk.Button(toolbar, text="💾 Save Config", command=self._on_save_config).pack(
            side="left", padx=4, pady=4
        )
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Button(toolbar, text="✓ Validate", command=self._on_validate).pack(
            side="left", padx=4, pady=4
        )

        # Run button — styled prominently
        self._run_btn = ttk.Button(
            toolbar,
            text="▶  Run Simulation",
            style="Accent.TButton",
            command=self._on_run,
        )
        self._run_btn.pack(side="right", padx=8, pady=4)

        # Scenario picker
        ttk.Label(toolbar, text="Scenario:").pack(side="right", padx=(0, 4))
        self._scenario_var = tk.StringVar(value="baseline")
        scenario_combo = ttk.Combobox(
            toolbar,
            textvariable=self._scenario_var,
            width=14,
            values=["baseline", "high_steam", "air_blown"],
            state="readonly",
        )
        scenario_combo.pack(side="right", padx=4, pady=4)
        scenario_combo.bind(
            "<<ComboboxSelected>>",
            lambda e: self._on_load_scenario(str(self._scenario_var.get())),
        )

    def _build_main_area(self) -> None:
        """Create the notebook (tabs) and the results tab."""
        from dwsim_model.gui.tabs.feeds_tab import FeedsTab
        from dwsim_model.gui.tabs.reactors_tab import ReactorsTab
        from dwsim_model.gui.tabs.results_tab import ResultsTab

        main_frame = ttk.Frame(self)
        main_frame.pack(fill="both", expand=True, padx=8, pady=4)

        self._notebook = ttk.Notebook(main_frame)
        self._notebook.pack(fill="both", expand=True)

        self._feeds_tab = FeedsTab(self._notebook, self)
        self._reactors_tab = ReactorsTab(self._notebook, self)
        self._results_tab = ResultsTab(self._notebook, self)

        self._notebook.add(self._feeds_tab, text="  Feeds  ")
        self._notebook.add(self._reactors_tab, text="  Reactors  ")
        self._notebook.add(self._results_tab, text="  Results  ")

    def _build_status_bar(self) -> None:
        self._status_var = tk.StringVar(value="Ready.")
        status_bar = ttk.Frame(self, relief="sunken")
        status_bar.pack(side="bottom", fill="x")
        ttk.Label(
            status_bar,
            textvariable=self._status_var,
            foreground="#4a5568",
            font=("TkDefaultFont", 8),
        ).pack(side="left", padx=8, pady=2)

    # ─────────────────────────────────────────────────────────────────────
    # Config loading / saving
    # ─────────────────────────────────────────────────────────────────────

    def _load_default_config(self) -> None:
        """Try to load the default master_config.yaml at startup."""
        here = Path(__file__).resolve().parent
        project_root = here.parent.parent.parent.parent  # .../DWSIM_Model/
        default = project_root / "config" / "master_config.yaml"
        if default.exists():
            self._load_config_file(default)

    def _load_config_file(self, path: Path) -> None:
        import yaml

        try:
            with path.open("r", encoding="utf-8") as fh:
                self._config = yaml.safe_load(fh) or {}
            self._config_path = path
            self._set_status(f"Config loaded: {path.name}")
            self._push_config_to_tabs()
        except Exception as exc:
            messagebox.showerror("Load Error", f"Could not load config:\n{exc}")

    def _push_config_to_tabs(self) -> None:
        """Distribute config values from the shared dict to the tab panels."""
        feeds = self._config.get("feeds", {})
        if feeds:
            self._feeds_tab.load_values(feeds)

    def _collect_config_from_tabs(self) -> dict:
        """Read current GUI values from all tabs and assemble the config dict."""
        cfg = dict(self._config)  # start from loaded base
        cfg["feeds"] = self._feeds_tab.get_values()
        cfg["reactors"] = self._reactors_tab.get_values()
        return cfg

    # ─────────────────────────────────────────────────────────────────────
    # Toolbar / menu handlers
    # ─────────────────────────────────────────────────────────────────────

    def _on_load_config(self) -> None:
        path = filedialog.askopenfilename(
            title="Open Config File",
            filetypes=[
                ("YAML files", "*.yaml *.yml"),
                ("JSON", "*.json"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._load_config_file(Path(path))

    def _on_save_config(self) -> None:
        if self._config_path:
            self._save_config_to(self._config_path)
        else:
            self._on_save_config_as()

    def _on_save_config_as(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save Config As",
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")],
        )
        if path:
            self._save_config_to(Path(path))

    def _save_config_to(self, path: Path) -> None:
        import yaml

        cfg = self._collect_config_from_tabs()
        try:
            with path.open("w", encoding="utf-8") as fh:
                yaml.dump(cfg, fh, default_flow_style=False, allow_unicode=True)
            self._config_path = path
            self._set_status(f"Config saved: {path.name}")
        except Exception as exc:
            messagebox.showerror("Save Error", f"Could not save config:\n{exc}")

    def _on_load_scenario(self, scenario_name: str) -> None:
        """Load a scenario override YAML and apply it to the GUI."""
        here = Path(__file__).resolve().parent
        project_root = here.parent.parent.parent.parent
        scenario_path = project_root / "config" / "scenarios" / f"{scenario_name}.yaml"
        if scenario_path.exists():
            self._load_config_file(scenario_path)
            self._scenario_var.set(scenario_name)
            self._set_status(f"Scenario loaded: {scenario_name}")
        else:
            messagebox.showwarning(
                "Scenario Not Found",
                f"Scenario file not found:\n{scenario_path}\n\n"
                "Check that the config/scenarios/ directory exists.",
            )

    def _on_validate(self) -> None:
        """Validate all tab inputs and show a summary."""
        errors = []
        ok_f, errs_f = self._feeds_tab.is_valid()
        ok_r, errs_r = self._reactors_tab.is_valid()
        errors.extend(errs_f + errs_r)

        # Also validate against Pydantic schema
        try:
            from dwsim_model.config.schema import validate_master_config

            cfg = self._collect_config_from_tabs()
            validate_master_config(cfg)
        except Exception as exc:
            errors.append(f"Schema validation: {exc}")

        if errors:
            messagebox.showerror(
                "Validation Failed",
                "The following issues were found:\n\n"
                + "\n".join(f"• {e}" for e in errors),
            )
        else:
            messagebox.showinfo(
                "Validation Passed",
                "✓ All configuration values are valid.\n\nReady to run.",
            )

    def _on_run(self) -> None:
        """Validate, then run the simulation in a background thread."""
        # Don't allow two runs at once
        if self._sim_thread and self._sim_thread.is_alive():
            messagebox.showwarning(
                "Already Running", "A simulation is already in progress."
            )
            return

        # Validate first
        errors = []
        _, errs_f = self._feeds_tab.is_valid()
        _, errs_r = self._reactors_tab.is_valid()
        errors.extend(errs_f + errs_r)
        if errors:
            messagebox.showerror(
                "Validation Error",
                "Please fix the following before running:\n\n"
                + "\n".join(f"• {e}" for e in errors),
            )
            return

        # Switch to Results tab and start run
        self._notebook.select(2)
        self._results_tab.log("─" * 60)
        self._results_tab.log("Starting simulation...")
        self._results_tab.log_panel.clear()

        cfg = self._collect_config_from_tabs()
        self._run_btn.configure(state="disabled", text="⏳ Running…")
        self._set_status("Simulation running…")

        self._sim_thread = threading.Thread(
            target=self._run_simulation_thread,
            args=(cfg,),
            daemon=True,
        )
        self._sim_thread.start()

    def _run_simulation_thread(self, cfg: dict) -> None:
        """Worker thread: runs the simulation and posts results to GUI."""
        try:
            from dwsim_model.gasification import GasificationFlowsheet
            from dwsim_model.results.extractor import ResultsExtractor
            from dwsim_model.results.metrics import MetricsCalculator
            from dwsim_model.results.reporter import (
                generate_html_report,
                generate_json_report,
            )

            self._results_tab.log("Building flowsheet…")
            flowsheet = GasificationFlowsheet()
            flowsheet._injected_config = cfg
            flowsheet.build()

            self._results_tab.log("Solving…")
            flowsheet.solve()
            self._results_tab.log("Solve complete.", "INFO")

            extractor = ResultsExtractor()
            results = extractor.extract(flowsheet.builder)

            calculator = MetricsCalculator()
            metrics = calculator.calculate(results)

            self._last_results = results
            self._last_metrics = metrics

            # Write reports
            out_dir = Path("results")
            out_dir.mkdir(exist_ok=True)
            scenario = self._scenario_var.get()

            html_path = generate_html_report(
                results,
                metrics,
                out_dir / f"{scenario}_report.html",
                scenario_name=scenario,
            )
            generate_json_report(
                results,
                metrics,
                out_dir / f"{scenario}_report.json",
                scenario_name=scenario,
            )

            # Schedule GUI update on the main thread
            self.after(0, self._on_run_success, results, metrics, html_path)

        except Exception as exc:
            logger.error(f"Simulation error: {exc}", exc_info=True)
            self.after(0, self._on_run_failure, str(exc))

    def _on_run_success(self, results, metrics, html_path: Path) -> None:
        """Called on main thread when simulation completes successfully."""
        self._results_tab.update_results(results, metrics)
        self._results_tab.set_html_path(html_path)
        self._results_tab.log(f"✓ Complete. HTML report: {html_path}", "INFO")
        self._run_btn.configure(state="normal", text="▶  Run Simulation")
        self._set_status("Simulation complete.")

    def _on_run_failure(self, error_msg: str) -> None:
        """Called on main thread when simulation fails."""
        self._results_tab.log(f"✗ Simulation failed: {error_msg}", "ERROR")
        self._run_btn.configure(state="normal", text="▶  Run Simulation")
        self._set_status("Simulation failed — see log.")
        messagebox.showerror(
            "Simulation Error", f"The simulation encountered an error:\n\n{error_msg}"
        )

    def _on_export(self) -> None:
        """Export the flowsheet to a DWSIM GUI file."""
        path = filedialog.asksaveasfilename(
            title="Export to DWSIM",
            defaultextension=".dwxml",
            filetypes=[("DWSIM XML", "*.dwxml"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            from dwsim_model.gasification import GasificationFlowsheet

            cfg = self._collect_config_from_tabs()
            flowsheet = GasificationFlowsheet()
            flowsheet._injected_config = cfg
            flowsheet.build()
            flowsheet.builder.sim.SaveToFile(path)
            messagebox.showinfo(
                "Export Complete",
                f"Flowsheet exported to:\n{path}\n\n"
                "Open with DWSIM → File → Open Simulation.",
            )
        except Exception as exc:
            messagebox.showerror("Export Error", f"Could not export:\n{exc}")

    def _on_about(self) -> None:
        messagebox.showinfo(
            "About",
            "DWSIM Gasification Model v2.0\n\n"
            "Three-stage plasma-assisted gasification model:\n"
            "  • Downdraft Gasifier (RCT_Conversion)\n"
            "  • PEM Plasma Entrained Melter (RCT_Equilibrium)\n"
            "  • TRC Thermal Reduction Chamber (RCT_PFR)\n\n"
            "Requires DWSIM Automation3 for full simulation.\n"
            "Standalone mode available without DWSIM installed.\n\n"
            "Config files: config/\n"
            "Results:      results/",
        )

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────

    def _set_status(self, message: str) -> None:
        self._status_var.set(message)
        self.update_idletasks()


def launch() -> None:
    """Entry point for the GUI application."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(name)s — %(message)s",
    )
    app = MainWindow()
    app.mainloop()


if __name__ == "__main__":
    launch()
