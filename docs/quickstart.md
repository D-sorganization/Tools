# 5-Minute Quick Start

> Get the Tools library installed and call your first calculator in under five minutes.

---

## Step 1 — Clone and install (2 min)

```bash
git clone https://github.com/D-sorganization/Tools.git
cd Tools
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e ".[all]"
```

Verify the install:

```bash
python -c "import upstream_drift_tools; print('OK')"
```

---

## Step 2 — Call a calculator (1 min)

### Unit conversion

```python
from upstream_drift_tools.calculators.conversion.service import UnitConversionService

svc = UnitConversionService()
result = svc.convert(100.0, "kg/h", "lb/hr")
print(f"{result.value:.2f} lb/hr")   # → 220.46 lb/hr
```

### Thermodynamic properties

```python
from upstream_drift_tools.calculators.thermo.thermo_properties import (
    ThermoPropertiesCalculator,
)

calc = ThermoPropertiesCalculator()
props = calc.calculate(
    temperature_c=500.0,
    pressure_kpa=101.325,
    composition={"N2": 50, "CO": 25, "H2": 25},   # syngas example
)
print(f"Density: {props.density_kg_m3:.3f} kg/m³")
print(f"MW:      {props.molecular_weight_g_mol:.2f} g/mol")
```

---

## Step 3 — Open the GUI launcher (30 sec)

```bash
python UnifiedToolsLauncher.py
```

A Qt6 window opens with all tools grouped by category (Data Processing,
Engineering Drafting, Scientific Modeling, …). Click any tile to launch
the corresponding tool.

---

## Step 4 — Run the test suite (1 min)

```bash
python -m pytest tests/ -m unit -q --tb=short
```

All unit tests should pass. Integration and e2e tests require optional
dependencies (`scipy`, `PyQt6`, etc.) already installed by `.[all]`.

---

## Next steps

| Goal                        | Where to look                                             |
| --------------------------- | --------------------------------------------------------- |
| Contribute a change         | [CONTRIBUTING.md](../CONTRIBUTING.md)                     |
| Build a new tool            | [docs/BUILD_A_TOOL.md](BUILD_A_TOOL.md)                   |
| Understand the architecture | [docs/ARCHITECTURE_OVERVIEW.md](ARCHITECTURE_OVERVIEW.md) |
| Performance notes           | [docs/performance.md](performance.md)                     |
| Security audit              | [docs/security-audit.md](security-audit.md)               |
| Full API reference          | [docs/tutorials/](tutorials/)                             |

For troubleshooting, see the [QUICKSTART.md](../QUICKSTART.md) at the repo root.
