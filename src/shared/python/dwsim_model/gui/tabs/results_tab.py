"""
gui/tabs/results_tab.py
=======================
The "Results" tab — shows KPI cards, stream data, and log output
after the simulation runs.
"""

from __future__ import annotations

from tkinter import ttk
from typing import Optional

from dwsim_model.gui.widgets import KPIPanel, LogPanel


class ResultsTab(ttk.Frame):
    """Tab for displaying simulation results."""

    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self._build()

    def _build(self) -> None:
        # ── Top: KPI panel ────────────────────────────────────────────────
        kpi_frame = ttk.LabelFrame(self, text="Key Performance Indicators", padding=10)
        kpi_frame.pack(fill="x", padx=16, pady=8)

        self.kpi_panel = KPIPanel(kpi_frame)
        self.kpi_panel.pack(fill="x")

        # ── Middle: Stream table ──────────────────────────────────────────
        stream_frame = ttk.LabelFrame(self, text="Stream Summary", padding=10)
        stream_frame.pack(fill="both", expand=True, padx=16, pady=4)

        columns = (
            "stream",
            "temp_c",
            "pres_kpa",
            "flow_kgs",
            "vol_nm3h",
            "co_mol",
            "h2_mol",
            "co2_mol",
        )
        self._tree = ttk.Treeview(
            stream_frame, columns=columns, show="headings", height=8
        )
        headers = {
            "stream": ("Stream Name", 160),
            "temp_c": ("T (°C)", 75),
            "pres_kpa": ("P (kPa)", 75),
            "flow_kgs": ("Flow (kg/s)", 90),
            "vol_nm3h": ("Vol (Nm³/h)", 90),
            "co_mol": ("CO (mol)", 70),
            "h2_mol": ("H₂ (mol)", 70),
            "co2_mol": ("CO₂ (mol)", 70),
        }
        for col, (heading, width) in headers.items():
            self._tree.heading(col, text=heading)
            self._tree.column(col, width=width, anchor="center")

        tree_scroll = ttk.Scrollbar(
            stream_frame, orient="vertical", command=self._tree.yview
        )
        self._tree.configure(yscrollcommand=tree_scroll.set)
        tree_scroll.pack(side="right", fill="y")
        self._tree.pack(fill="both", expand=True)

        # ── Bottom: Log panel ─────────────────────────────────────────────
        log_frame = ttk.LabelFrame(self, text="Simulation Log", padding=6)
        log_frame.pack(fill="both", expand=True, padx=16, pady=8)

        self.log_panel = LogPanel(log_frame)
        self.log_panel.pack(fill="both", expand=True)

        # Report buttons
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", padx=16, pady=4)

        ttk.Button(
            btn_frame, text="Open HTML Report", command=self._open_html_report
        ).pack(side="left", padx=4)
        ttk.Button(
            btn_frame, text="Open Results Folder", command=self._open_results_folder
        ).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Clear Log", command=self.log_panel.clear).pack(
            side="right", padx=4
        )

        self._last_html_path = None

    # ── Public API ─────────────────────────────────────────────────────────

    def update_results(self, results, metrics, targets: Optional[dict] = None) -> None:
        """Populate the KPI panel and stream table from results objects."""
        # KPIs
        self.kpi_panel.update_kpis(metrics.to_dict(), targets=targets)

        # Stream table
        for row in self._tree.get_children():
            self._tree.delete(row)

        for name, s in sorted(results.streams.items()):
            mf = getattr(s, "mole_fractions", {}) or {}

            def fmt(v, spec=".2f"):
                return f"{v:{spec}}" if v is not None else "—"

            self._tree.insert(
                "",
                "end",
                values=(
                    name,
                    fmt(getattr(s, "temperature_C", None)),
                    fmt(getattr(s, "pressure_kPa", None)),
                    fmt(getattr(s, "mass_flow_kg_s", None), ".4f"),
                    fmt(getattr(s, "volumetric_flow_Nm3_h", None)),
                    fmt(mf.get("Carbon monoxide"), ".3f"),
                    fmt(mf.get("Hydrogen"), ".3f"),
                    fmt(mf.get("Carbon dioxide"), ".3f"),
                ),
            )

    def set_html_path(self, path) -> None:
        """Store the path to the generated HTML report."""
        self._last_html_path = path

    def log(self, message: str, level: str = "INFO") -> None:
        """Append a log message to the panel."""
        self.log_panel.append(message, level)

    def _open_html_report(self) -> None:
        """Open the HTML report in the system's default browser."""
        import webbrowser

        if self._last_html_path and self._last_html_path.exists():
            webbrowser.open(self._last_html_path.as_uri())
        else:
            self.log(
                "No HTML report available yet. Run the simulation first.", "WARNING"
            )

    def _open_results_folder(self) -> None:
        """Open the results folder in the OS file manager."""
        import subprocess  # nosec B404 # AUTO-FIXED
        import sys
        from pathlib import Path

        results_dir = Path("results").resolve()
        if not results_dir.exists():
            results_dir.mkdir(parents=True)
        if sys.platform == "win32":
            subprocess.Popen(
                ["explorer", str(results_dir)]
            )  # nosec B603 B607 B404 # AUTO-FIXED
        elif sys.platform == "darwin":
            subprocess.Popen(
                ["open", str(results_dir)]
            )  # nosec B603 B607 B404 # AUTO-FIXED
        else:
            subprocess.Popen(
                ["xdg-open", str(results_dir)]
            )  # nosec B603 B607 B404 # AUTO-FIXED
