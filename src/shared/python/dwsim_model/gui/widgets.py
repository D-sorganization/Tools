"""
gui/widgets.py
==============
Reusable Tkinter widget components for the gasification GUI.

Each widget handles its own layout, validation, and value retrieval so that
the tab panels stay clean and readable.

Design approach
---------------
For a newer Python developer: rather than building the GUI ad-hoc with raw
Tkinter calls scattered everywhere, we create small, self-contained "widget
classes" that encapsulate their own appearance and validation logic.

For example, ValidatedEntry:
    - Shows a label + text entry field
    - Turns red if the entered value is outside the allowed range
    - Exposes a .get_value() method that returns None if invalid
    - Exposes a .set_value() method for loading config values

This makes the tab code read like:
    self.biomass_flow = ValidatedEntry(frame, "Biomass flow (kg/s)", 0.1, 20.0)
    value = self.biomass_flow.get_value()   # None if invalid
"""

from __future__ import annotations

import logging
import tkinter as tk
from collections.abc import Callable
from tkinter import ttk
from typing import Any, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Colour constants (matches the HTML report palette)
# ─────────────────────────────────────────────────────────────────────────────
COLOR_GOOD = "#22c55e"
COLOR_WARN = "#f59e0b"
COLOR_BAD = "#ef4444"
COLOR_BG = "#f0f2f5"
COLOR_ACCENT = "#0f3460"
COLOR_TEXT = "#1a1a2e"
COLOR_SUBTLE = "#718096"


# ─────────────────────────────────────────────────────────────────────────────
# ValidatedEntry
# ─────────────────────────────────────────────────────────────────────────────


class ValidatedEntry:
    """
    A label + entry field pair with optional range validation.

    The entry border turns red if the user types a value outside [min_val, max_val].
    A tooltip-style label shows the allowed range.

    Parameters
    ----------
    parent:
        Parent Tkinter widget (e.g. a Frame or LabelFrame).
    label:
        Descriptive text shown to the left of the entry.
    min_val, max_val:
        Allowed range.  Either can be None to skip that bound.
    default:
        Initial value to display.
    units:
        Optional units string shown after the entry (e.g. "kg/s").
    on_change:
        Optional callback called whenever the value changes.
    row:
        If provided, the widget is placed using .grid(row=row, ...).
        Otherwise, the caller must pack/grid the returned frame.
    """

    def __init__(
        self,
        parent,
        label: str,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        default: Any = "",
        units: str = "",
        on_change: Optional[Callable] = None,
        row: Optional[int] = None,
    ):
        self.min_val = min_val
        self.max_val = max_val
        self._on_change = on_change

        # StringVar so we can trace changes
        self._var = tk.StringVar(value=str(default) if default != "" else "")
        self._var.trace_add("write", self._validate)

        # Widgets
        self._label = ttk.Label(parent, text=label, foreground=COLOR_TEXT)
        self._entry = ttk.Entry(parent, textvariable=self._var, width=14)
        self._units_label = ttk.Label(
            parent, text=units, foreground=COLOR_SUBTLE, width=8
        )

        # Range hint
        hints = []
        if min_val is not None:
            hints.append(f"min {min_val}")
        if max_val is not None:
            hints.append(f"max {max_val}")
        hint_text = " | ".join(hints)
        self._hint = ttk.Label(
            parent, text=hint_text, foreground=COLOR_SUBTLE, font=("TkDefaultFont", 8)
        )

        if row is not None:
            self._label.grid(row=row, column=0, sticky="w", padx=(0, 8), pady=3)
            self._entry.grid(row=row, column=1, sticky="ew", pady=3)
            self._units_label.grid(row=row, column=2, sticky="w", padx=(4, 0))
            self._hint.grid(row=row, column=3, sticky="w", padx=(8, 0))

        self._valid = True

    def _validate(self, *args):
        """Called on every keystroke to check the value."""
        val = self._var.get().strip()
        if val == "":
            self._entry.configure(style="TEntry")
            self._valid = True
            return
        try:
            f = float(val)
            in_range = (self.min_val is None or f >= self.min_val) and (
                self.max_val is None or f <= self.max_val
            )
            if in_range:
                self._entry.configure(style="TEntry")
                self._valid = True
            else:
                self._entry.configure(style="Error.TEntry")
                self._valid = False
        except ValueError:
            self._entry.configure(style="Error.TEntry")
            self._valid = False

        if self._on_change:
            self._on_change(self.get_value())

    def get_value(self) -> Optional[float]:
        """Return the current value as float, or None if invalid."""
        if not self._valid:
            return None
        val = self._var.get().strip()
        if val == "":
            return None
        try:
            return float(val)
        except ValueError:
            return None

    def set_value(self, value: Any) -> None:
        """Set the entry value programmatically."""
        self._var.set(str(value) if value is not None else "")

    def is_valid(self) -> bool:
        return self._valid

    @property
    def label_widget(self):
        return self._label

    @property
    def entry_widget(self):
        return self._entry


# ─────────────────────────────────────────────────────────────────────────────
# ComboField
# ─────────────────────────────────────────────────────────────────────────────


class ComboField:
    """Label + Combobox (dropdown) widget pair."""

    def __init__(
        self,
        parent,
        label: str,
        choices: list[str],
        default: Optional[str] = None,
        on_change: Optional[Callable] = None,
        row: Optional[int] = None,
    ):
        self._var = tk.StringVar(value=default or (choices[0] if choices else ""))
        if on_change:
            self._var.trace_add("write", lambda *a: on_change(self._var.get()))

        self._label = ttk.Label(parent, text=label, foreground=COLOR_TEXT)
        self._combo = ttk.Combobox(
            parent, textvariable=self._var, values=choices, state="readonly", width=20
        )

        if row is not None:
            self._label.grid(row=row, column=0, sticky="w", padx=(0, 8), pady=3)
            self._combo.grid(row=row, column=1, columnspan=3, sticky="ew", pady=3)

    def get_value(self) -> str:
        return self._var.get()

    def set_value(self, value: str) -> None:
        self._var.set(value)


# ─────────────────────────────────────────────────────────────────────────────
# SectionFrame
# ─────────────────────────────────────────────────────────────────────────────


class SectionFrame(ttk.LabelFrame):
    """
    A LabelFrame with consistent styling used as a group box for related fields.

    Automatically configures column weights so the entry column expands.
    """

    def __init__(self, parent, title: str, **kwargs):
        super().__init__(parent, text=title, padding=10, **kwargs)
        self.columnconfigure(1, weight=1)


# ─────────────────────────────────────────────────────────────────────────────
# KPIPanel  — traffic-light KPI display
# ─────────────────────────────────────────────────────────────────────────────


class KPIPanel(ttk.Frame):
    """
    Displays a grid of KPI cards with traffic-light colouring.

    Usage
    -----
        panel = KPIPanel(parent)
        panel.update_kpis({
            "cold_gas_efficiency": 0.72,
            "h2_co_ratio": 1.8,
            ...
        })
    """

    _KPI_LAYOUT = [
        ("cold_gas_efficiency", "Cold Gas Eff.", ".1%", None),
        ("carbon_conversion_efficiency", "Carbon Conv.", ".1%", None),
        ("h2_co_ratio", "H₂/CO", ".2f", None),
        ("syngas_lhv_mj_nm3", "LHV (MJ/Nm³)", ".2f", None),
        ("specific_energy_consumption_kWh_t", "Sp. Energy", ".0f", " kWh/t"),
        ("tar_loading_mg_Nm3", "Tar Loading", ".1f", " mg/Nm³"),
        ("mass_balance_closure", "Mass Balance", ".4f", None),
        ("energy_balance_closure", "Energy Balance", ".4f", None),
    ]

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self._cards: dict[str, dict] = {}
        self._build()

    def _build(self):
        cols = 4
        for i, (key, label, fmt, suffix) in enumerate(self._KPI_LAYOUT):
            row, col = divmod(i, cols)
            frame = tk.Frame(self, bg="white", relief="solid", bd=1, padx=12, pady=8)
            frame.grid(row=row, column=col, padx=6, pady=6, sticky="nsew")
            self.columnconfigure(col, weight=1)

            val_lbl = tk.Label(
                frame,
                text="—",
                bg="white",
                font=("TkDefaultFont", 16, "bold"),
                fg=COLOR_TEXT,
            )
            val_lbl.pack()
            name_lbl = tk.Label(
                frame,
                text=label,
                bg="white",
                font=("TkDefaultFont", 8),
                fg=COLOR_SUBTLE,
            )
            name_lbl.pack()

            self._cards[key] = {
                "frame": frame,
                "val_lbl": val_lbl,
                "name_lbl": name_lbl,
                "fmt": fmt,
                "suffix": suffix or "",
            }

    def update_kpis(self, kpi_dict: dict, targets: Optional[dict] = None) -> None:
        """Update displayed values; apply traffic-light colours if targets given."""
        targets = targets or {}
        for key, card in self._cards.items():
            val = kpi_dict.get(key)
            fmt = card["fmt"]
            suffix = card["suffix"]

            if val is None:
                text = "—"
                bg = "white"
            else:
                try:
                    text = f"{val:{fmt}}{suffix}"
                except (ValueError, TypeError):
                    text = str(val)
                bg = self._traffic_light_bg(key, val, targets)

            card["val_lbl"].configure(text=text, bg=bg)
            card["frame"].configure(bg=bg)
            card["name_lbl"].configure(bg=bg)

    @staticmethod
    def _traffic_light_bg(key: str, value: float, targets: dict) -> str:
        target = targets.get(key)
        if target is None:
            return "white"
        higher_better = {
            "cold_gas_efficiency",
            "carbon_conversion_efficiency",
            "mass_balance_closure",
            "energy_balance_closure",
        }
        lower_better = {"tar_loading_mg_Nm3", "specific_energy_consumption_kWh_t"}
        if key in higher_better:
            if value >= target * 0.98:
                return "#f0fdf4"  # light green
            elif value >= target * 0.90:
                return "#fffbeb"  # light amber
            else:
                return "#fef2f2"  # light red
        elif key in lower_better:
            if value <= target * 1.02:
                return "#f0fdf4"
            elif value <= target * 1.15:
                return "#fffbeb"
            else:
                return "#fef2f2"
        return "white"


# ─────────────────────────────────────────────────────────────────────────────
# LogPanel — scrollable text log
# ─────────────────────────────────────────────────────────────────────────────


class LogPanel(ttk.Frame):
    """
    A scrollable text widget that acts as an in-GUI log output.

    Attach to Python logging with LogPanel.attach_to_logger().
    """

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self._text = tk.Text(
            self,
            state="disabled",
            wrap="word",
            font=("Courier New", 9),
            bg="#1e1e1e",
            fg="#d4d4d4",
            insertbackground="white",
        )
        scroll = ttk.Scrollbar(self, command=self._text.yview)
        self._text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self._text.pack(side="left", fill="both", expand=True)

        # Colour tags for log levels
        self._text.tag_config("ERROR", foreground="#ef4444")
        self._text.tag_config("WARNING", foreground="#f59e0b")
        self._text.tag_config("INFO", foreground="#d4d4d4")
        self._text.tag_config("DEBUG", foreground="#6b7280")

    def append(self, message: str, level: str = "INFO") -> None:
        """Append a line to the log panel."""
        self._text.configure(state="normal")
        self._text.insert("end", message + "\n", level)
        self._text.see("end")
        self._text.configure(state="disabled")

    def clear(self) -> None:
        self._text.configure(state="normal")
        self._text.delete("1.0", "end")
        self._text.configure(state="disabled")

    def attach_to_logger(self, logger_name: str = "") -> None:
        """Route Python logging output to this panel."""
        import logging

        panel = self

        class _GUIHandler(logging.Handler):
            def emit(self, record):
                try:
                    panel.append(
                        self.format(record),
                        level=record.levelname,
                    )
                except Exception:
                    self.handleError(record)  # AUTO-FIXED

        handler = _GUIHandler()
        handler.setFormatter(
            logging.Formatter("[%(levelname)s] %(name)s — %(message)s")
        )
        logging.getLogger(logger_name).addHandler(handler)


# ─────────────────────────────────────────────────────────────────────────────
# Style helper
# ─────────────────────────────────────────────────────────────────────────────


def apply_styles(root: tk.Tk) -> None:
    """Apply consistent ttk styles to the entire application."""
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception as exc:
        logging.getLogger(__name__).debug(
            f"Could not apply clam theme: {exc}"
        )  # AUTO-FIXED  # Fall back to default

    style.configure("TFrame", background=COLOR_BG)
    style.configure("TLabelframe", background=COLOR_BG)
    style.configure(
        "TLabelframe.Label",
        background=COLOR_BG,
        foreground=COLOR_ACCENT,
        font=("TkDefaultFont", 9, "bold"),
    )
    style.configure("TLabel", background=COLOR_BG, foreground=COLOR_TEXT)
    style.configure(
        "TButton", foreground="white", background=COLOR_ACCENT, padding=(8, 4)
    )
    style.map("TButton", background=[("active", "#16213e"), ("pressed", "#0a1628")])
    style.configure("Accent.TButton", background=COLOR_GOOD, foreground="white")
    style.map("Accent.TButton", background=[("active", "#16a34a")])

    # Error entry style (red border)
    style.configure("Error.TEntry", fieldbackground="#fef2f2", foreground=COLOR_BAD)
    style.map("Error.TEntry", fieldbackground=[("focus", "#fef2f2")])

    style.configure("TNotebook", background=COLOR_BG)
    style.configure("TNotebook.Tab", padding=(12, 6))
    style.map(
        "TNotebook.Tab", background=[("selected", "white"), ("!selected", "#e2e8f0")]
    )
