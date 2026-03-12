# Gasification Model: Critical Review & Upgrade Proposal

**Date:** 2026-03-05
**Codebase:** DWSIM_Model (Python + DWSIM Automation3)
**Author:** Dieter Olson / Claude Review

---

## Part 1: Critical Review of the Current Model

### What Currently Exists

The model is a Python-driven DWSIM flowsheet builder for a three-stage gasification system: a Downdraft Gasifier → PEM (Plasma Entrained Melter) → TRC (Thermal Reduction Chamber), followed by a Quench Vessel → Baghouse → Scrubber → Blower. It uses DWSIM's Automation3 API via pythonnet/CLR to programmatically construct the flowsheet, and can export to `.dwxml` for GUI visualization.

### Strengths

1. **Good architectural skeleton.** The three-reactor-plus-gas-cleanup topology is a legitimate representation of a plasma-assisted gasification train.
2. **Reactor mode flexibility.** The `ReactorMode` enum (MIXED, KINETIC, EQUILIBRIUM, CONVERSION, CUSTOM) is a smart design pattern that lets users swap reactor fidelity without rebuilding the flowsheet.
3. **Sequential thermal staging.** Using discrete Heater/Cooler blocks instead of trying to multiplex energy streams is a pragmatic approach that avoids DWSIM energy-mixer limitations.
4. **Config file concept started.** The `ConfigLoader` + `feed_conditions.json` is the right idea for separating simulation parameters from code.
5. **Test infrastructure.** Mock-based conftest allows CI testing without DWSIM installed.

### Critical Gaps (Showstoppers)

#### 1. Reactors Are Empty Shells

This is the single biggest issue. `_configure_reactors()` is entirely stubbed out with `pass` statements:

- **Downdraft Gasifier (RCT_Conversion):** No conversion reactions defined. Without reactions, the reactor does nothing — feed passes through unchanged.
- **PEM Reactor (RCT_Equilibrium):** No equilibrium reactions (WGS, methanation, Boudouard), no temperature specification, no isothermal/adiabatic mode set.
- **TRC Reactor (RCT_PFR):** No kinetic expressions, no volume/length, no catalyst parameters.

**Impact:** The model currently produces zero chemistry. Running it produces the same composition at output as input.

#### 2. Biomass Feed Representation Is Wrong

The feed (`Gasifier_Biomass_Feed`) uses methane (90%) and water (10%) as a biomass proxy. Real biomass is a complex solid with an ultimate analysis (C, H, O, N, S, Cl, ash). DWSIM doesn't natively handle solid fuels well, but using pure methane misrepresents:

- The energy content (methane HHV ≈ 55.5 MJ/kg vs. typical biomass ≈ 15–20 MJ/kg)
- The oxygen demand (stoichiometric O₂ for CH₄ is completely different from biomass)
- The product composition (no tar precursors, no ash, no chlorine chemistry)

#### 3. No Composition Handling in ConfigLoader

The config file includes `"components"` blocks, but `apply_to_flowsheet()` never reads or applies them. Composition data is defined but silently ignored.

#### 4. Compound List Too Small

Eight compounds (CO, H₂, CO₂, CH₄, H₂O, N₂, O₂, He) miss critical species for gasification modeling: H₂S, HCl, NH₃, tar surrogates (naphthalene, toluene, phenol), C₂H₂, C₂H₄, C₂H₆, and char/ash representation.

#### 5. No Results Extraction

There's no mechanism to read back simulation results — temperatures, compositions, flow rates, energy balances. The model builds and runs, but you can't programmatically check what came out.

#### 6. No Validation or Sanity Checks

No mass balance closure check, no energy balance verification, no comparison against known performance data.

### Moderate Gaps

#### 7. Duplicated Code Across Standalone Models

`gasifier_model.py`, `pem_model.py`, and `trc_model.py` each copy the same compound list, same `setup_thermo()`, same `safe_connect()` pattern. This violates DRY and means any change (like adding a compound) must be made in 4+ places.

#### 8. Peng-Robinson May Be Wrong

PR EOS is reasonable for high-temperature syngas, but at quench/scrubber conditions (near water condensation), NRTL-PR or SRK with Wong-Sandler mixing rules may be more appropriate. The model doesn't support switching property packages per section.

#### 9. No Sensitivity Analysis or Parametric Studies

No built-in way to sweep parameters (e.g., equivalence ratio, steam-to-carbon ratio) and collect results.

#### 10. Error Handling Silences Failures

`safe_connect()` catches all exceptions and logs warnings, meaning a broken connection won't stop the build — it'll just produce a disconnected, silently-wrong flowsheet.

#### 11. Bug: `energies` vs `energy_streams` Attribute Mismatch

In `gasification.py` line 334, the config loader is called with `b.energies`, but `FlowsheetBuilder` stores energy streams in `self.energy_streams`, not `self.energies`. This means energy configuration from the JSON file is silently failing — the energy values are never applied.

#### 12. ConfigLoader Uses Fragile Relative Path

`ConfigLoader` defaults to `"config/feed_conditions.json"` as a relative path. If the script is run from any directory other than the project root, this silently falls back to defaults. Should use `Path(__file__).parent` or accept absolute paths from the master config.

---

## Part 2: Config File Architecture

The current single `feed_conditions.json` is a start, but it conflates feed conditions, energy inputs, and reactor parameters. Here's a proposed multi-file config structure:

### Proposed Config Directory Layout

```
config/
├── master_config.yaml           # Top-level: references all other configs
├── compounds.yaml               # Species list + property package selection
├── feeds/
│   ├── gasifier_feeds.yaml      # Biomass, oxygen, steam feeds
│   ├── pem_feeds.yaml           # PEM additional feeds
│   ├── trc_feeds.yaml           # TRC additional feeds
│   └── quench_feeds.yaml        # Water injection, N2, steam
├── reactors/
│   ├── gasifier_reactions.yaml  # Conversion reactions + parameters
│   ├── pem_reactions.yaml       # Equilibrium reactions + temp spec
│   └── trc_reactions.yaml       # Kinetic expressions + geometry
├── energy/
│   └── energy_inputs.yaml       # All energy stream values
├── equipment/
│   └── gas_cleanup.yaml         # Baghouse, scrubber, blower specs
├── scenarios/
│   ├── baseline.yaml            # Default operating point
│   ├── high_steam.yaml          # High S/C ratio scenario
│   └── air_blown.yaml           # Air-blown vs. oxygen-blown
└── validation/
    └── expected_results.yaml    # Known-good outputs for regression testing
```

### Why YAML Instead of JSON

- Supports comments (users need to annotate why they chose a value)
- More readable for multi-level nesting
- Widely used in engineering tools (ASPEN templates, gPROMS configs)
- Python's `pyyaml` or `ruamel.yaml` handles it cleanly

### Example: `master_config.yaml`

```yaml
# Master configuration for Gasification Model
# Edit scenario or override individual files as needed

model:
  name: "Plasma-Assisted Gasification Train"
  reactor_mode: "mixed" # mixed | kinetic | equilibrium | conversion | custom

compounds: "config/compounds.yaml"
feeds:
  gasifier: "config/feeds/gasifier_feeds.yaml"
  pem: "config/feeds/pem_feeds.yaml"
  trc: "config/feeds/trc_feeds.yaml"
  quench: "config/feeds/quench_feeds.yaml"
reactors:
  gasifier: "config/reactors/gasifier_reactions.yaml"
  pem: "config/reactors/pem_reactions.yaml"
  trc: "config/reactors/trc_reactions.yaml"
energy: "config/energy/energy_inputs.yaml"
equipment: "config/equipment/gas_cleanup.yaml"

# Active scenario (overrides base values)
scenario: "config/scenarios/baseline.yaml"
```

### Example: `gasifier_feeds.yaml`

```yaml
# Gasifier Feed Conditions
# ========================
# Biomass is represented as a pseudo-mixture derived from ultimate analysis.
# The decomposition model converts solid biomass composition into an equivalent
# gas-phase mixture that DWSIM can process.

biomass_feed:
  stream_name: "Gasifier_Biomass_Feed"
  temperature_C: 25.0
  pressure_Pa: 101325.0
  mass_flow_kg_s: 10.0

  # Ultimate Analysis (dry, ash-free basis, mass fractions)
  ultimate_analysis:
    C: 0.501
    H: 0.062
    O: 0.421
    N: 0.008
    S: 0.005
    Cl: 0.003

  # Proximate Analysis (as-received basis, mass fractions)
  proximate_analysis:
    moisture: 0.15
    volatile_matter: 0.72
    fixed_carbon: 0.18
    ash: 0.10

  # HHV in MJ/kg (for energy balance validation)
  hhv_mj_kg: 18.5

oxygen_feed:
  stream_name: "Gasifier_Oxygen_Feed"
  temperature_C: 25.0
  pressure_Pa: 101325.0
  mass_flow_kg_s: 5.0
  components:
    Oxygen: 0.95
    Nitrogen: 0.05 # Accounts for impure O2 supply

steam_feed:
  stream_name: "Gasifier_Steam_Feed"
  temperature_C: 150.0
  pressure_Pa: 1013250.0
  mass_flow_kg_s: 2.0
  components:
    Water: 1.0
```

### Example: `gasifier_reactions.yaml`

```yaml
# Downdraft Gasifier Reaction Set
# ================================
# For RCT_Conversion mode, specify fractional conversions.
# For RCT_Equilibrium or RCT_PFR, these are ignored; see pem_reactions or trc_reactions.

reactor_type: "RCT_Conversion"
temperature_C: 900.0
pressure_Pa: 101325.0
mode: "isothermal" # isothermal | adiabatic | specified_duty

reactions:
  - name: "Partial Oxidation"
    stoichiometry: "2C + O2 -> 2CO"
    base_component: "Oxygen"
    conversion: 0.98
    heat_of_reaction_kJ_mol: -221.0 # exothermic

  - name: "Water Gas Reaction"
    stoichiometry: "C + H2O -> CO + H2"
    base_component: "Water"
    conversion: 0.60
    heat_of_reaction_kJ_mol: 131.3 # endothermic

  - name: "Boudouard Reaction"
    stoichiometry: "C + CO2 -> 2CO"
    base_component: "Carbon dioxide"
    conversion: 0.30
    heat_of_reaction_kJ_mol: 172.5

  - name: "Methanation"
    stoichiometry: "C + 2H2 -> CH4"
    base_component: "Hydrogen"
    conversion: 0.10
    heat_of_reaction_kJ_mol: -74.8

  - name: "Water Gas Shift"
    stoichiometry: "CO + H2O -> CO2 + H2"
    base_component: "Carbon monoxide"
    conversion: 0.25
    heat_of_reaction_kJ_mol: -41.2
```

### Config Loader Upgrade

The new `ConfigLoader` would:

1. Read `master_config.yaml` first
2. Resolve all referenced file paths
3. Merge scenario overrides on top of base values
4. Validate every field against a schema (using `pydantic` or `cerberus`)
5. Apply composition data to streams (fixing the current bug)
6. Return a typed configuration object, not a raw dict

---

## Part 3: Input GUI

### Approach: Tkinter Desktop GUI

Since this model runs locally on Windows with DWSIM, a lightweight Tkinter GUI is the right choice — no web server needed, ships with Python, and can be frozen into an .exe with PyInstaller.

### Proposed GUI Layout

```
┌─────────────────────────────────────────────────────────┐
│  Gasification Model Configurator              [Run] [⚙]│
├────────────┬────────────────────────────────────────────┤
│            │                                            │
│ Navigation │  [Active Tab Content Area]                 │
│            │                                            │
│ ○ Feeds    │  Example: Feeds Tab                        │
│ ○ Reactors │  ┌─────────────────────────────────────┐   │
│ ○ Energy   │  │ Gasifier Biomass Feed               │   │
│ ○ Equipment│  │ Temperature (°C): [  25.0  ] ✓      │   │
│ ○ Scenario │  │ Pressure (Pa):    [101325.0] ✓      │   │
│ ○ Results  │  │ Mass Flow (kg/s): [ 10.0   ] ✓      │   │
│            │  │                                     │   │
│            │  │ Ultimate Analysis (mass frac):       │   │
│            │  │ C: [0.501] H: [0.062] O: [0.421]    │   │
│            │  │ N: [0.008] S: [0.005] Cl:[0.003]    │   │
│            │  │ Sum: 1.000 ✓                         │   │
│            │  └─────────────────────────────────────┘   │
│            │                                            │
├────────────┴────────────────────────────────────────────┤
│ Status: Ready │ Config: baseline.yaml │ Mode: MIXED     │
└─────────────────────────────────────────────────────────┘
```

### Key GUI Features

1. **Tabbed input panels** — one per config section (Feeds, Reactors, Energy, Equipment, Scenario)
2. **Real-time validation** with green ✓ / red ✗ indicators:
   - Numeric range checks (temperature > 0 K, pressure > 0, mass fractions 0–1)
   - Mass fraction sum-to-one validation with tolerance display
   - Cross-field checks (e.g., steam temp must be > 100°C at atmospheric pressure)
3. **Tooltips** on every field explaining what it does, typical ranges, and units
4. **Load/Save config** buttons to import/export YAML files
5. **Scenario dropdown** to quickly switch between saved scenarios
6. **Run button** with progress bar and log output panel
7. **Results tab** showing key outputs: syngas composition, temperatures, flow rates, energy balance, cold gas efficiency

### Error Handling Strategy

```
Input Validation Layer (GUI):
  ├── Type checking (is it a number?)
  ├── Range checking (is it physically reasonable?)
  ├── Consistency checking (do fractions sum to 1?)
  └── Dependency checking (does steam feed exist if steam_to_carbon > 0?)

Config Validation Layer (before build):
  ├── Schema validation (all required fields present?)
  ├── Reference validation (do all stream names match flowsheet?)
  └── Unit consistency (no mixing °C and K in same section?)

Runtime Validation Layer (during/after solve):
  ├── Convergence check (did the solver converge?)
  ├── Mass balance closure (within 0.1%?)
  ├── Energy balance closure (within 1%?)
  └── Physical sanity (no negative temperatures, no > 100% mole fractions?)
```

### Implementation Skeleton

```python
# gui/main_window.py — Top-level structure

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import yaml
from pathlib import Path

class GasificationGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Gasification Model Configurator")
        self.geometry("900x650")
        self.config_data = {}
        self.validators = {}

        self._build_menu()
        self._build_nav()
        self._build_tabs()
        self._build_status_bar()
        self._build_run_panel()

    def _build_tabs(self):
        self.notebook = ttk.Notebook(self.main_frame)
        self.feeds_tab = FeedsTab(self.notebook, self)
        self.reactors_tab = ReactorsTab(self.notebook, self)
        self.energy_tab = EnergyTab(self.notebook, self)
        self.equipment_tab = EquipmentTab(self.notebook, self)
        self.results_tab = ResultsTab(self.notebook, self)
        # ... add tabs to notebook

    def load_config(self, filepath):
        """Load and validate a YAML config, populate GUI fields."""
        ...

    def save_config(self, filepath):
        """Extract current GUI values, validate, write YAML."""
        ...

    def run_simulation(self):
        """Validate all inputs → build config → launch model in thread."""
        ...
```

### Alternative: Browser-Based GUI (Future)

For remote access or multi-user setups, a `Flask` or `FastAPI` + simple HTML frontend could serve the same purpose and would be accessible from any machine on the network. This is a natural Phase 2 evolution.

---

## Part 4: Headless / CLI Environment

### Why Headless Matters

- **Batch runs:** sweep 100 parameter combinations overnight
- **CI/CD:** automated regression testing on every commit
- **Remote servers:** run on HPC or cloud machines without a display
- **Reproducibility:** exact same results from config file, no manual GUI interaction

### Proposed CLI Interface

```bash
# Run a single simulation from config
python -m gasification_model run --config config/master_config.yaml

# Run with scenario override
python -m gasification_model run --config config/master_config.yaml \
    --scenario config/scenarios/high_steam.yaml

# Parameter sweep
python -m gasification_model sweep --config config/master_config.yaml \
    --param "feeds.gasifier.steam_feed.mass_flow_kg_s" \
    --range 1.0 5.0 0.5 \
    --output results/steam_sweep/

# Export flowsheet to DWSIM GUI format
python -m gasification_model export --config config/master_config.yaml \
    --output model.dwxml

# Validate config without running
python -m gasification_model validate --config config/master_config.yaml

# Generate results report
python -m gasification_model report --results results/run_001.json \
    --format html
```

### Implementation: `__main__.py`

```python
# src/dwsim_model/__main__.py
import argparse
import sys
import logging

def main():
    parser = argparse.ArgumentParser(
        description="Gasification Process Model (DWSIM-backed)"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Run command
    run_parser = subparsers.add_parser("run", help="Execute a simulation")
    run_parser.add_argument("--config", required=True, help="Path to master config YAML")
    run_parser.add_argument("--scenario", help="Optional scenario override YAML")
    run_parser.add_argument("--output-dir", default="results/", help="Results output dir")
    run_parser.add_argument("--verbose", action="store_true")

    # Sweep command
    sweep_parser = subparsers.add_parser("sweep", help="Parameter sweep")
    sweep_parser.add_argument("--config", required=True)
    sweep_parser.add_argument("--param", required=True, help="Dotted parameter path")
    sweep_parser.add_argument("--range", nargs=3, type=float,
                              metavar=("START", "END", "STEP"))
    sweep_parser.add_argument("--output", default="results/sweep/")

    # Validate command
    val_parser = subparsers.add_parser("validate", help="Validate config files")
    val_parser.add_argument("--config", required=True)

    # Export command
    exp_parser = subparsers.add_parser("export", help="Export to DWSIM GUI format")
    exp_parser.add_argument("--config", required=True)
    exp_parser.add_argument("--output", default="model.dwxml")

    args = parser.parse_args()
    # ... dispatch to appropriate handler
```

### Headless Considerations

- **No DWSIM GUI dependency:** The `Automation3` API already works headless — it doesn't launch the DWSIM window. This is already the right approach.
- **Logging to file + stdout:** Configurable via `--verbose` and `--log-file` flags.
- **JSON results output:** Machine-readable for post-processing pipelines.
- **Return codes:** 0 = success, 1 = config error, 2 = convergence failure, 3 = validation failure.

---

## Part 5: Feature Upgrades for Gasification Relevance

### Priority 1 — Make Chemistry Work (Essential)

#### 1A. Biomass Decomposition Model

Since DWSIM can't handle solid biomass directly, implement a **pre-processor** that converts ultimate/proximate analysis into an equivalent gas mixture:

```python
class BiomassDecomposer:
    """Converts biomass ultimate analysis to equivalent DWSIM-compatible gas mixture.

    Approach: Biomass is decomposed into CO, CO2, H2, H2O, CH4, N2,
    and a char/ash pseudo-component based on element balances.
    """
    def decompose(self, ultimate: dict, proximate: dict,
                  mass_flow_kg_s: float) -> dict:
        # 1. Devolatilization: volatile matter → gas species
        # 2. Char: fixed carbon → "carbon" pseudo-component
        # 3. Moisture: direct H2O addition
        # 4. Ash: tracked as inert (Helium proxy or separate solid stream)
        ...
        return {"stream_compositions": {...}, "char_stream": {...}}
```

#### 1B. Implement Gasifier Reactions

For the Conversion reactor, programmatically add DWSIM reactions:

- Partial oxidation (C + ½O₂ → CO)
- Complete combustion (C + O₂ → CO₂)
- Water-gas reaction (C + H₂O → CO + H₂)
- Boudouard reaction (C + CO₂ → 2CO)
- Methanation (C + 2H₂ → CH₄)
- Water-gas shift (CO + H₂O → CO₂ + H₂)

#### 1C. Implement PEM Equilibrium Reactions

- WGS equilibrium
- Methanation equilibrium
- Tar cracking (simplified)
- Temperature specification (typically 1200–1600°C for plasma zone)

#### 1D. Implement TRC Kinetics

- Define PFR volume/length
- Add tar destruction kinetics (Arrhenius expressions from literature)
- Residence time calculation

### Priority 2 — Expanded Chemistry (Important)

#### 2A. Tar Modeling

Add naphthalene, toluene, and/or phenol as tar surrogates. These are critical for downstream equipment sizing and syngas quality assessment. Even simplified lumped-tar models add huge value.

#### 2B. Trace Contaminants

Add H₂S, HCl, NH₃ for:

- Environmental compliance modeling
- Scrubber design and performance prediction
- Catalyst poisoning assessment in downstream processes

#### 2C. Expanded Compound List

```yaml
compounds:
  syngas_core: [CO, H2, CO2, CH4, H2O, N2, O2]
  hydrocarbons: [C2H2, C2H4, C2H6]
  tar_surrogates: [Naphthalene, Toluene, Phenol]
  contaminants: [H2S, HCl, NH3]
  inerts: [Argon, Helium] # Helium as ash proxy
```

### Priority 3 — Results & Analysis (High Value)

#### 3A. Results Extraction Module

```python
class ResultsExtractor:
    """Pulls key performance metrics from solved flowsheet."""

    def extract(self, builder: FlowsheetBuilder) -> dict:
        return {
            "syngas_composition": self._get_composition("Final_Syngas"),
            "syngas_temperature_C": self._get_temp("Final_Syngas"),
            "syngas_flow_kg_s": self._get_flow("Final_Syngas"),
            "cold_gas_efficiency": self._calc_cge(),
            "carbon_conversion": self._calc_carbon_conversion(),
            "mass_balance_closure_pct": self._check_mass_balance(),
            "energy_balance_closure_pct": self._check_energy_balance(),
            "h2_co_ratio": self._calc_ratio("Hydrogen", "Carbon monoxide"),
            "tar_loading_mg_Nm3": self._calc_tar_loading(),
        }
```

#### 3B. Performance Metrics

Key gasification KPIs that should be auto-calculated:

- **Cold Gas Efficiency (CGE):** Chemical energy in syngas / Chemical energy in feed
- **Carbon Conversion Efficiency:** Carbon in syngas / Carbon in feed
- **H₂/CO Ratio:** Critical for downstream use (Fischer-Tropsch, methanol, etc.)
- **Tar Loading:** mg/Nm³ — determines if gas cleanup is adequate
- **Specific Energy Consumption:** kWh per kg of waste processed (for PEM)
- **Equivalence Ratio:** Actual O₂ / Stoichiometric O₂

#### 3C. Automated Report Generation

Generate HTML or PDF reports with:

- Stream table (all temperatures, pressures, compositions, flows)
- Sankey diagram of energy flows
- Bar chart of syngas composition
- Mass/energy balance tables
- Comparison against target/baseline values

### Priority 4 — Parametric Studies & Optimization

#### 4A. Parameter Sweep Engine

```python
class ParameterSweep:
    """Runs the model across a parameter range, collects results."""

    def sweep_1d(self, param_path: str, values: list) -> pd.DataFrame:
        """Single-parameter sweep."""
        ...

    def sweep_2d(self, param1: str, values1: list,
                 param2: str, values2: list) -> pd.DataFrame:
        """Two-parameter grid sweep."""
        ...
```

Typical sweeps for gasification:

- Equivalence ratio: 0.2 to 0.5
- Steam-to-carbon ratio: 0 to 2.0
- PEM power: 0 to 10 MW
- Biomass moisture content: 5% to 40%

#### 4B. Sensitivity Analysis

Tornado charts showing which parameters most affect:

- Syngas H₂ content
- Cold gas efficiency
- Tar loading

### Priority 5 — Robustness & Usability

#### 5A. Property Package Selection Logic

```yaml
property_packages:
  high_temp_zone: "Peng-Robinson (PR)" # >500°C, gas phase
  quench_zone: "NRTL" # Near condensation
  scrubber: "NRTL" # Liquid-vapor equilibrium
  # Or unified: "Peng-Robinson (PR)" for simplicity
```

#### 5B. Unit Conversion Layer

Users think in different units. The config should accept any common unit and convert internally:

```yaml
temperature: 900 °C # or 1173.15 K, or 1652 °F
pressure: 1 atm # or 101325 Pa, or 14.696 psi
mass_flow: 10 kg/s # or 36000 kg/h, or 36 t/h
```

A `pint`-based unit conversion layer handles this cleanly.

#### 5C. Logging & Diagnostics Overhaul

- Structured logging (JSON format for machine parsing)
- Per-block convergence status
- Iteration counts per unit operation
- Warning system for near-constraint violations

---

## Part 6: Proposed Project Structure (After Upgrades)

```
DWSIM_Model/
├── src/
│   └── dwsim_model/
│       ├── __init__.py
│       ├── __main__.py              # CLI entry point
│       ├── core.py                  # FlowsheetBuilder (existing, enhanced)
│       ├── gasification.py          # GasificationFlowsheet (existing, enhanced)
│       ├── config/
│       │   ├── loader.py            # YAML-based multi-file config loader
│       │   ├── schema.py            # Pydantic models for config validation
│       │   └── units.py             # Unit conversion (pint-based)
│       ├── chemistry/
│       │   ├── biomass_decomposer.py  # Ultimate analysis → gas mixture
│       │   ├── reactions.py           # Reaction definitions + DWSIM API calls
│       │   └── kinetics.py            # Arrhenius expressions for PFR
│       ├── results/
│       │   ├── extractor.py         # Pull results from solved flowsheet
│       │   ├── metrics.py           # CGE, carbon conversion, etc.
│       │   └── reporter.py          # HTML/PDF report generation
│       ├── analysis/
│       │   ├── sweep.py             # Parameter sweep engine
│       │   └── sensitivity.py       # Tornado / sensitivity analysis
│       ├── gui/
│       │   ├── main_window.py       # Tkinter main application
│       │   ├── tabs/
│       │   │   ├── feeds_tab.py
│       │   │   ├── reactors_tab.py
│       │   │   ├── energy_tab.py
│       │   │   ├── equipment_tab.py
│       │   │   └── results_tab.py
│       │   ├── widgets/
│       │   │   ├── validated_entry.py  # Entry with range checking
│       │   │   ├── composition_table.py # Editable composition grid
│       │   │   └── tooltip.py
│       │   └── dialogs/
│       │       ├── scenario_picker.py
│       │       └── sweep_config.py
│       └── standalone/              # (existing, refactored to use shared config)
│           ├── gasifier_model.py
│           ├── pem_model.py
│           └── trc_model.py
├── config/                          # (expanded config structure)
│   ├── master_config.yaml
│   ├── compounds.yaml
│   ├── feeds/
│   ├── reactors/
│   ├── energy/
│   ├── equipment/
│   ├── scenarios/
│   └── validation/
├── tests/
│   ├── conftest.py
│   ├── test_config_loader.py
│   ├── test_biomass_decomposer.py
│   ├── test_reactions.py
│   ├── test_results_extractor.py
│   ├── test_gui_validation.py
│   ├── test_cli.py
│   └── test_parameter_sweep.py
├── docs/
└── requirements.txt
```

---

## Part 7: Implementation Roadmap

### Phase 1: Foundation (Weeks 1–2)

- [ ] Upgrade config system to YAML with multi-file support
- [ ] Implement config schema validation with Pydantic
- [ ] Fix ConfigLoader to actually apply compositions
- [ ] Expand compound list
- [ ] Add unit conversion layer

### Phase 2: Chemistry (Weeks 3–5)

- [ ] Implement biomass decomposition model
- [ ] Add gasifier conversion reactions via DWSIM API
- [ ] Add PEM equilibrium reactions
- [ ] Add TRC kinetics (at least simplified)
- [ ] Add tar surrogate species

### Phase 3: Results & Validation (Weeks 5–6)

- [ ] Build ResultsExtractor
- [ ] Implement CGE, carbon conversion, mass/energy balance checks
- [ ] Create automated report generation
- [ ] Add regression test data from literature/known systems

### Phase 4: CLI & Headless (Week 7)

- [ ] Implement `__main__.py` with argparse CLI
- [ ] Add parameter sweep engine
- [ ] Add JSON results output
- [ ] Verify headless operation on Windows/Linux

### Phase 5: GUI (Weeks 8–10)

- [ ] Build Tkinter main window with tab structure
- [ ] Implement validated input widgets
- [ ] Connect GUI ↔ config system (bidirectional)
- [ ] Add results visualization tab
- [ ] Add progress bar and log panel

### Phase 6: Polish & Documentation (Week 11)

- [ ] Refactor standalone models to use shared config
- [ ] Comprehensive docstrings and user guide
- [ ] Example scenarios with expected outputs
- [ ] PyInstaller packaging for standalone distribution

---

## Part 8: Quick Wins (Can Do Right Now)

These don't require deep DWSIM API knowledge and would immediately improve the model:

1. **Fix ConfigLoader to apply compositions** — the data is in the JSON, it's just not being read
2. **Add more compounds** — just append to the compound list
3. **Add results extraction** — read back `Temperature`, `Pressure`, `MassFlow` from output streams
4. **Add mass balance check** — sum all input mass flows, compare to sum of all output mass flows
5. **Convert config to YAML** — direct port of existing JSON + comments
6. **Add `__main__.py`** — wrap `export_to_gui.py` logic with argparse
7. **Deduplicate compound list** — extract to a shared module

---

## Summary

The current model has a solid topological skeleton but no working chemistry — it's essentially plumbing without any reactions happening inside the pipes. The highest-impact work is implementing actual reactions in the three reactor blocks and fixing the biomass feed representation. Everything else (GUI, CLI, config files, parametric studies) builds on top of that foundation.

The good news is that the architecture is sound and extensible. The `ReactorMode` pattern, the sequential thermal staging approach, and the separation of build/configure/run phases are all the right design patterns. They just need content.
