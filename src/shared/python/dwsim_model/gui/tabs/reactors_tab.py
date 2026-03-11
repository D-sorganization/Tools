"""
gui/tabs/reactors_tab.py
========================
The "Reactors" tab — configure conditions for Gasifier, PEM, and TRC.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from dwsim_model.gui.widgets import ComboField, SectionFrame, ValidatedEntry


class ReactorsTab(ttk.Frame):
    """Tab for reactor operating condition settings."""

    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self._build()

    def _build(self) -> None:
        canvas = tk.Canvas(self, bg="#f0f2f5", highlightthickness=0)
        sb = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        sf = ttk.Frame(canvas)
        sf.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=sf, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # ── Gasifier ─────────────────────────────────────────────────────
        gas_sec = SectionFrame(sf, "Downdraft Gasifier (RCT_Conversion)")
        gas_sec.pack(fill="x", padx=16, pady=8)

        ttk.Label(
            gas_sec,
            text="Typical range: 750–950°C, 101–500 kPa",
            foreground="#718096",
            font=("TkDefaultFont", 8),
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 6))

        self.gas_temp = ValidatedEntry(
            gas_sec, "Outlet temperature", 600.0, 1200.0, 850.0, "°C", row=1
        )
        self.gas_pressure = ValidatedEntry(
            gas_sec, "Outlet pressure", 100_000, 2_000_000, 101_325, "Pa", row=2
        )
        self.gas_mode = ComboField(
            gas_sec,
            "Operation mode",
            ["Isothermal", "Adiabatic", "Specified Duty"],
            default="Isothermal",
            row=3,
        )
        self.gas_er = ValidatedEntry(
            gas_sec,
            "Equivalence ratio (ER)",
            0.1,
            0.5,
            0.25,
            "(0.2–0.4 typical)",
            row=4,
        )

        # Reaction conversion sliders
        ttk.Label(
            gas_sec,
            text="Reaction conversions (0 = none, 1 = complete):",
            foreground="#4a5568",
        ).grid(row=5, column=0, columnspan=4, sticky="w", pady=(10, 2))

        self.conv_partial_ox = ValidatedEntry(
            gas_sec, "Partial Oxidation (C+O₂→CO₂)", 0.0, 1.0, 0.90, row=6
        )
        self.conv_water_gas = ValidatedEntry(
            gas_sec, "Water Gas (C+H₂O→CO+H₂)", 0.0, 1.0, 0.70, row=7
        )
        self.conv_boudouard = ValidatedEntry(
            gas_sec, "Boudouard (C+CO₂→2CO)", 0.0, 1.0, 0.65, row=8
        )
        self.conv_wgs = ValidatedEntry(
            gas_sec, "WGS (CO+H₂O→CO₂+H₂)", 0.0, 1.0, 0.55, row=9
        )

        # ── PEM ──────────────────────────────────────────────────────────
        pem_sec = SectionFrame(sf, "PEM Plasma Entrained Melter (RCT_Equilibrium)")
        pem_sec.pack(fill="x", padx=16, pady=8)

        ttk.Label(
            pem_sec,
            text="Plasma temperature: 1200–1600°C. Gibbs minimisation used.",
            foreground="#718096",
            font=("TkDefaultFont", 8),
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 6))

        self.pem_temp = ValidatedEntry(
            pem_sec, "Outlet temperature", 1000.0, 2000.0, 1400.0, "°C", row=1
        )
        self.pem_pressure = ValidatedEntry(
            pem_sec, "Outlet pressure", 100_000, 2_000_000, 101_325, "Pa", row=2
        )
        self.pem_ac_power = ValidatedEntry(
            pem_sec, "AC plasma input power", 0.0, 20_000_000, 5_000_000, "W", row=3
        )
        self.pem_dc_power = ValidatedEntry(
            pem_sec, "DC plasma power", 0.0, 15_000_000, 3_000_000, "W", row=4
        )
        self.pem_efficiency = ValidatedEntry(
            pem_sec, "Plasma torch efficiency", 0.50, 0.99, 0.85, "(0–1)", row=5
        )

        # ── TRC ──────────────────────────────────────────────────────────
        trc_sec = SectionFrame(sf, "TRC Thermal Reduction Chamber (RCT_PFR)")
        trc_sec.pack(fill="x", padx=16, pady=8)

        ttk.Label(
            trc_sec,
            text="PFR with Arrhenius tar cracking kinetics. Adiabatic operation.",
            foreground="#718096",
            font=("TkDefaultFont", 8),
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 6))

        self.trc_temp = ValidatedEntry(
            trc_sec, "Inlet temperature", 800.0, 1600.0, 1100.0, "°C", row=1
        )
        self.trc_pressure = ValidatedEntry(
            trc_sec, "Pressure", 100_000, 1_000_000, 101_325, "Pa", row=2
        )
        self.trc_volume = ValidatedEntry(
            trc_sec, "Reactor volume", 0.1, 50.0, 2.0, "m³", row=3
        )
        self.trc_length = ValidatedEntry(
            trc_sec, "Reactor length", 0.5, 20.0, 3.0, "m", row=4
        )
        self.trc_diameter = ValidatedEntry(
            trc_sec, "Reactor diameter", 0.1, 5.0, 0.92, "m", row=5
        )

        # Kinetics note
        ttk.Label(
            trc_sec,
            text="Arrhenius kinetics loaded from config/reactors/trc_reactions.yaml",
            foreground="#718096",
            font=("TkDefaultFont", 8),
        ).grid(row=6, column=0, columnspan=4, sticky="w", pady=(8, 0))

    # ── Public API ────────────────────────────────────────────────────────

    def get_values(self) -> dict:
        """Return reactor settings as a nested dict."""
        return {
            "Gasifier": {
                "temperature_C": self.gas_temp.get_value(),
                "pressure_Pa": self.gas_pressure.get_value(),
                "mode": self.gas_mode.get_value(),
                "equivalence_ratio": self.gas_er.get_value(),
                "conversions": {
                    "partial_oxidation": self.conv_partial_ox.get_value(),
                    "water_gas": self.conv_water_gas.get_value(),
                    "boudouard": self.conv_boudouard.get_value(),
                    "wgs": self.conv_wgs.get_value(),
                },
            },
            "PEM": {
                "temperature_C": self.pem_temp.get_value(),
                "pressure_Pa": self.pem_pressure.get_value(),
                "ac_power_W": self.pem_ac_power.get_value(),
                "dc_power_W": self.pem_dc_power.get_value(),
                "efficiency": self.pem_efficiency.get_value(),
            },
            "TRC": {
                "temperature_C": self.trc_temp.get_value(),
                "pressure_Pa": self.trc_pressure.get_value(),
                "volume_m3": self.trc_volume.get_value(),
                "length_m": self.trc_length.get_value(),
                "diameter_m": self.trc_diameter.get_value(),
            },
        }

    def load_values(self, reactors_dict: dict) -> None:
        """Load reactor config into GUI fields."""
        gas = reactors_dict.get("Gasifier", {})
        self.gas_temp.set_value(gas.get("temperature_C", ""))
        self.gas_pressure.set_value(gas.get("pressure_Pa", ""))
        self.gas_er.set_value(gas.get("equivalence_ratio", ""))

        pem = reactors_dict.get("PEM", {})
        self.pem_temp.set_value(pem.get("temperature_C", ""))
        self.pem_pressure.set_value(pem.get("pressure_Pa", ""))
        self.pem_ac_power.set_value(pem.get("ac_power_W", ""))
        self.pem_dc_power.set_value(pem.get("dc_power_W", ""))
        self.pem_efficiency.set_value(pem.get("efficiency", ""))

        trc = reactors_dict.get("TRC", {})
        self.trc_temp.set_value(trc.get("temperature_C", ""))
        self.trc_pressure.set_value(trc.get("pressure_Pa", ""))
        self.trc_volume.set_value(trc.get("volume_m3", ""))
        self.trc_length.set_value(trc.get("length_m", ""))
        self.trc_diameter.set_value(trc.get("diameter_m", ""))

    def is_valid(self) -> tuple[bool, list[str]]:
        errors = []
        for name, field in [
            ("Gasifier temperature", self.gas_temp),
            ("Gasifier pressure", self.gas_pressure),
            ("PEM temperature", self.pem_temp),
            ("PEM AC power", self.pem_ac_power),
            ("TRC volume", self.trc_volume),
        ]:
            if not field.is_valid() or field.get_value() is None:
                errors.append(f"Reactors: '{name}' is required and must be valid.")
        return len(errors) == 0, errors
