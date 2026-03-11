"""
gui/tabs/feeds_tab.py
=====================
The "Feeds" tab of the gasification GUI.

Allows the user to set feed stream conditions for:
  • Biomass feed (flow, ultimate analysis, moisture, ash)
  • Oxygen / Air feed (flow, purity)
  • Steam feed (flow, temperature, pressure)

All entry fields validate in real time.  Values are read from and written
to the shared config dict via the controller.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from dwsim_model.gui.widgets import ComboField, SectionFrame, ValidatedEntry


class FeedsTab(ttk.Frame):
    """
    Tab panel for configuring feed stream inputs.

    Parameters
    ----------
    parent:
        The ttk.Notebook that owns this tab.
    controller:
        Reference to the MainWindow object, used to push changes back to the
        shared application state.
    """

    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self._build()

    def _build(self) -> None:
        # ── Scrollable canvas so the tab works on small screens ──────────
        canvas = tk.Canvas(self, bg="#f0f2f5", highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self._scroll_frame = ttk.Frame(canvas)
        self._scroll_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self._scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        parent = self._scroll_frame

        # ── Biomass Section ───────────────────────────────────────────────
        bio_sec = SectionFrame(parent, "Biomass Feed (Gasifier_Biomass_Feed)")
        bio_sec.pack(fill="x", padx=16, pady=8)

        self.biomass_flow = ValidatedEntry(
            bio_sec, "Mass flow rate", 0.1, 20.0, 4.0, "kg/s", row=0
        )
        self.biomass_moisture = ValidatedEntry(
            bio_sec, "Moisture (as-received)", 0.0, 0.60, 0.15, "(0–0.60)", row=1
        )
        self.biomass_ash = ValidatedEntry(
            bio_sec, "Ash (as-received)", 0.0, 0.50, 0.10, "(0–0.50)", row=2
        )

        # Ultimate analysis sub-section
        ua_lbl = ttk.Label(
            bio_sec,
            text="Ultimate analysis (daf mass fractions):",
            foreground="#4a5568",
        )
        ua_lbl.grid(row=3, column=0, columnspan=4, sticky="w", pady=(10, 2))

        self.ua_C = ValidatedEntry(bio_sec, "C (Carbon)", 0.0, 1.0, 0.501, row=4)
        self.ua_H = ValidatedEntry(bio_sec, "H (Hydrogen)", 0.0, 1.0, 0.062, row=5)
        self.ua_O = ValidatedEntry(bio_sec, "O (Oxygen)", 0.0, 1.0, 0.421, row=6)
        self.ua_N = ValidatedEntry(bio_sec, "N (Nitrogen)", 0.0, 0.10, 0.008, row=7)
        self.ua_S = ValidatedEntry(bio_sec, "S (Sulfur)", 0.0, 0.05, 0.005, row=8)
        self.ua_Cl = ValidatedEntry(bio_sec, "Cl (Chlorine)", 0.0, 0.02, 0.003, row=9)

        self.ua_sum_label = ttk.Label(bio_sec, text="Sum = 1.000", foreground="#22c55e")
        self.ua_sum_label.grid(row=10, column=0, columnspan=4, sticky="w", pady=4)

        # Wire sum-validation callback
        for field in [
            self.ua_C,
            self.ua_H,
            self.ua_O,
            self.ua_N,
            self.ua_S,
            self.ua_Cl,
        ]:
            field._var.trace_add("write", self._update_ua_sum)

        self.hhv_field = ValidatedEntry(
            bio_sec, "HHV (dry basis)", 5.0, 35.0, 18.5, "MJ/kg", row=11
        )

        # ── Gasification Agent ────────────────────────────────────────────
        ox_sec = SectionFrame(parent, "Gasification Agent (Oxygen / Air)")
        ox_sec.pack(fill="x", padx=16, pady=8)

        self.agent_type = ComboField(
            ox_sec,
            "Agent type",
            [
                "Pure Oxygen (95% O₂)",
                "Industrial Oxygen (90% O₂)",
                "Air (21% O₂)",
                "Enriched Air (40% O₂)",
            ],
            default="Pure Oxygen (95% O₂)",
            on_change=self._on_agent_change,
            row=0,
        )
        self.oxygen_flow = ValidatedEntry(
            ox_sec, "Mass flow rate", 0.1, 30.0, 3.34, "kg/s", row=1
        )
        self.oxygen_purity = ValidatedEntry(
            ox_sec, "O₂ purity", 0.21, 1.0, 0.95, "(mol frac)", row=2
        )

        # ── Steam Feed ───────────────────────────────────────────────────
        steam_sec = SectionFrame(parent, "Steam Feed (Gasifier_Steam_Feed)")
        steam_sec.pack(fill="x", padx=16, pady=8)

        self.steam_flow = ValidatedEntry(
            steam_sec, "Mass flow rate", 0.0, 20.0, 1.0, "kg/s", row=0
        )
        self.steam_temp = ValidatedEntry(
            steam_sec, "Temperature", 100.0, 600.0, 200.0, "°C", row=1
        )
        self.steam_pressure = ValidatedEntry(
            steam_sec, "Pressure", 100_000, 2_000_000, 500_000, "Pa", row=2
        )

        # ── Steam/Biomass ratio indicator ─────────────────────────────────
        sbr_frame = ttk.Frame(parent)
        sbr_frame.pack(fill="x", padx=16, pady=4)
        self.sbr_label = ttk.Label(
            sbr_frame, text="Steam/Biomass ratio (SBR): —", foreground="#4a5568"
        )
        self.sbr_label.pack(side="left")

        # Wire SBR update
        self.steam_flow._var.trace_add("write", self._update_sbr)
        self.biomass_flow._var.trace_add("write", self._update_sbr)

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _update_ua_sum(self, *args) -> None:
        """Recalculate and display the ultimate analysis sum."""
        fields = [self.ua_C, self.ua_H, self.ua_O, self.ua_N, self.ua_S, self.ua_Cl]
        vals = [f.get_value() for f in fields]
        if any(v is None for v in vals):
            self.ua_sum_label.configure(
                text="Sum = ?  (invalid entry)", foreground="#ef4444"
            )
            return
        total = sum(vals)
        ok = abs(total - 1.0) <= 0.02
        self.ua_sum_label.configure(
            text=f"Sum = {total:.4f}{'  ✓' if ok else '  ✗  (must be 1.00 ± 0.02)'}",
            foreground="#22c55e" if ok else "#ef4444",
        )

    def _on_agent_change(self, value: str) -> None:
        """Auto-fill oxygen purity when agent type changes."""
        presets = {
            "Pure Oxygen (95% O₂)": 0.95,
            "Industrial Oxygen (90% O₂)": 0.90,
            "Air (21% O₂)": 0.21,
            "Enriched Air (40% O₂)": 0.40,
        }
        purity = presets.get(value)
        if purity:
            self.oxygen_purity.set_value(purity)

    def _update_sbr(self, *args) -> None:
        """Show the Steam/Biomass mass ratio."""
        sf = self.steam_flow.get_value()
        bf = self.biomass_flow.get_value()
        if sf is not None and bf is not None and bf > 0:
            sbr = sf / bf
            color = "#22c55e" if 0.3 <= sbr <= 1.5 else "#f59e0b"
            self.sbr_label.configure(
                text=f"Steam/Biomass ratio (SBR): {sbr:.2f}",
                foreground=color,
            )
        else:
            self.sbr_label.configure(
                text="Steam/Biomass ratio (SBR): —", foreground="#718096"
            )

    # ── Public API ────────────────────────────────────────────────────────

    def get_values(self) -> dict:
        """Return all feed values as a nested dict for config application."""
        return {
            "Gasifier_Biomass_Feed": {
                "mass_flow_kg_s": self.biomass_flow.get_value(),
                "moisture_ar": self.biomass_moisture.get_value(),
                "ash_ar": self.biomass_ash.get_value(),
                "hhv_mj_kg": self.hhv_field.get_value(),
                "ultimate_analysis": {
                    "C": self.ua_C.get_value(),
                    "H": self.ua_H.get_value(),
                    "O": self.ua_O.get_value(),
                    "N": self.ua_N.get_value(),
                    "S": self.ua_S.get_value(),
                    "Cl": self.ua_Cl.get_value(),
                },
            },
            "Gasifier_Oxygen_Feed": {
                "mass_flow_kg_s": self.oxygen_flow.get_value(),
                "o2_purity": self.oxygen_purity.get_value(),
            },
            "Gasifier_Steam_Feed": {
                "mass_flow_kg_s": self.steam_flow.get_value(),
                "temperature_C": self.steam_temp.get_value(),
                "pressure_Pa": self.steam_pressure.get_value(),
            },
        }

    def load_values(self, feeds_dict: dict) -> None:
        """Load values from a feeds config dict into the GUI fields."""
        bio = feeds_dict.get("Gasifier_Biomass_Feed", {})
        self.biomass_flow.set_value(bio.get("mass_flow_kg_s", ""))
        self.biomass_moisture.set_value(bio.get("moisture_ar", ""))
        self.biomass_ash.set_value(bio.get("ash_ar", ""))
        self.hhv_field.set_value(bio.get("hhv_mj_kg", ""))
        ua = bio.get("ultimate_analysis", {})
        for attr, key in [
            ("ua_C", "C"),
            ("ua_H", "H"),
            ("ua_O", "O"),
            ("ua_N", "N"),
            ("ua_S", "S"),
            ("ua_Cl", "Cl"),
        ]:
            getattr(self, attr).set_value(ua.get(key, ""))

        ox = feeds_dict.get("Gasifier_Oxygen_Feed", {})
        self.oxygen_flow.set_value(ox.get("mass_flow_kg_s", ""))
        self.oxygen_purity.set_value(ox.get("o2_purity", ""))

        st = feeds_dict.get("Gasifier_Steam_Feed", {})
        self.steam_flow.set_value(st.get("mass_flow_kg_s", ""))
        self.steam_temp.set_value(st.get("temperature_C", ""))
        self.steam_pressure.set_value(st.get("pressure_Pa", ""))

    def is_valid(self) -> tuple[bool, list[str]]:
        """Return (all_valid, [error_messages])."""
        errors = []
        fields = {
            "Biomass flow": self.biomass_flow,
            "Biomass moisture": self.biomass_moisture,
            "Biomass ash": self.biomass_ash,
            "HHV": self.hhv_field,
            "C fraction": self.ua_C,
            "H fraction": self.ua_H,
            "O fraction": self.ua_O,
            "Oxygen flow": self.oxygen_flow,
            "Steam flow": self.steam_flow,
        }
        for name, field in fields.items():
            if not field.is_valid():
                errors.append(f"Feeds: '{name}' has an invalid value.")
            if field.get_value() is None and name in ("Biomass flow", "Oxygen flow"):
                errors.append(f"Feeds: '{name}' is required.")

        # Ultimate analysis sum
        ua_vals = [
            getattr(self, f).get_value()
            for f in ["ua_C", "ua_H", "ua_O", "ua_N", "ua_S", "ua_Cl"]
        ]
        if all(v is not None for v in ua_vals):
            if abs(sum(ua_vals) - 1.0) > 0.02:
                errors.append(
                    "Feeds: Ultimate analysis fractions must sum to 1.0 ± 0.02."
                )
        return len(errors) == 0, errors
