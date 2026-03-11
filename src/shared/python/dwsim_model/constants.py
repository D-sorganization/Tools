"""
Shared constants for the gasification model.

Centralising the compound list here means a single edit propagates to the
main flowsheet, all three standalone models, tests, and the GUI.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Compound registry
# ─────────────────────────────────────────────────────────────────────────────

#: Core syngas species — always included.
SYNGAS_CORE = [
    "Carbon monoxide",
    "Hydrogen",
    "Carbon dioxide",
    "Methane",
    "Water",
    "Nitrogen",
    "Oxygen",
]

#: C2 hydrocarbons — minor products, important for heating value.
C2_HYDROCARBONS = [
    "Ethylene",
    "Ethane",
    "Acetylene",
]

#: Tar surrogate species — naphthalene as primary heavy-tar indicator.
TAR_SURROGATES = [
    "Naphthalene",
    "Toluene",
]

#: Trace contaminants — H₂S and NH₃ are critical for downstream design.
TRACE_CONTAMINANTS = [
    "Hydrogen sulfide",
    "Ammonia",
]

#: Inerts used as pseudo-components (Helium proxies char/ash in DWSIM).
INERTS = [
    "Helium",  # Ash/char proxy — inert, well-characterised thermo
    "Argon",
]

#: Minimal compound set — used for fast/unit-test builds.
COMPOUNDS_MINIMAL = SYNGAS_CORE + INERTS[:1]  # Just Helium as char proxy

#: Standard compound set — default for production runs.
COMPOUNDS_STANDARD = SYNGAS_CORE + C2_HYDROCARBONS + INERTS[:1]

#: Extended compound set — includes tars and contaminants.
COMPOUNDS_EXTENDED = (
    SYNGAS_CORE + C2_HYDROCARBONS + TAR_SURROGATES + TRACE_CONTAMINANTS + INERTS[:1]
)

# ─────────────────────────────────────────────────────────────────────────────
# Default property packages
# ─────────────────────────────────────────────────────────────────────────────

#: Peng-Robinson — appropriate for high-temperature gas-phase reactions.
PP_PENG_ROBINSON = "Peng-Robinson (PR)"

#: Soave-Redlich-Kwong — alternative; often better near critical point.
PP_SRK = "Soave-Redlich-Kwong (SRK)"

DEFAULT_PROPERTY_PACKAGE = PP_PENG_ROBINSON

# ─────────────────────────────────────────────────────────────────────────────
# Physical constants
# ─────────────────────────────────────────────────────────────────────────────

KELVIN_OFFSET: float = 273.15  # Add to °C to get K
STANDARD_PRESSURE_PA: float = 101_325.0  # 1 atm in Pa
STANDARD_TEMPERATURE_C: float = 15.0  # ISO standard reference temperature

# ─────────────────────────────────────────────────────────────────────────────
# Stream name registry  (single source of truth so GUI / tests agree)
# ─────────────────────────────────────────────────────────────────────────────

STREAM_NAMES = {
    # ── Gasifier feeds ──────────────────────────────────────────────────────
    "biomass_feed": "Gasifier_Biomass_Feed",
    "gasifier_solids": "Gasifier_Solids_Feed",
    "gasifier_oxygen": "Gasifier_Oxygen_Feed",
    "gasifier_steam": "Gasifier_Steam_Feed",
    # ── PEM feeds ────────────────────────────────────────────────────────────
    "pem_solids": "PEM_Solids_Feed",
    "pem_oxygen": "PEM_Oxygen_Feed",
    "pem_steam": "PEM_Steam_Feed",
    # ── TRC feeds ────────────────────────────────────────────────────────────
    "trc_solids": "TRC_Solids_Feed",
    "trc_oxygen": "TRC_Oxygen_Feed",
    "trc_steam": "TRC_Steam_Feed",
    # ── Quench ───────────────────────────────────────────────────────────────
    "quench_water": "Quench_Water_Injection",
    "quench_nitrogen": "Quench_Nitrogen",
    "quench_steam": "Quench_Steam",
    # ── Key intermediate streams ─────────────────────────────────────────────
    "syngas_pre_pem": "Syngas_Pre_PEM",
    "syngas_pre_trc": "Syngas_Pre_TRC",
    "syngas_pre_quench": "Syngas_Pre_Quench",
    "syngas_pre_baghouse": "Syngas_Pre_Baghouse",
    "clean_syngas_pre_scrub": "Clean_Syngas_Pre_Scrub",
    "scrubbed_syngas": "Scrubbed_Syngas",
    # ── Product ──────────────────────────────────────────────────────────────
    "final_syngas": "Final_Syngas",
    # ── By-products ──────────────────────────────────────────────────────────
    "gasifier_glass": "Gasifier_Glass_Out",
    "pem_glass": "PEM_Glass_Out",
    "baghouse_solids": "Baghouse_Solids_Out",
    "scrubber_blowdown": "Scrubber_Blowdown",
}

ENERGY_STREAM_NAMES = {
    "gasifier_heat_loss": "E_Gasifier_HeatLoss",
    "gasifier_flux_cw": "E_Gasifier_Flux_to_CW",
    "pem_ac_power": "E_PEM_AC_Power",
    "pem_dc_power": "E_PEM_DC_Power",
    "pem_heat_loss": "E_PEM_HeatLoss",
    "trc_heat_loss": "E_TRC_HeatLoss",
    "blower_power": "E_Blower",
}
