# Chapter 8 — Web Applications

**Parent Document:** [Tools User Manual](./TOOLS_USER_MANUAL.md)

---

## 8.1 Calculator Web App

**Source:** `src/web_applications/calculator/`
**Status:** ✅ Fully Implemented

### 8.1.1 Purpose

Flask-based web calculator providing engineering calculations through a browser interface.

### 8.1.2 Architecture

- **Backend:** Flask (Python) — `calculator.py` (762 lines)
- **Frontend:** HTML/CSS templates with Jinja2
- **Security:** Input validation and sanitization
- **Testing:** Comprehensive test suite including security validation tests

### 8.1.3 Features

- Unit conversions
- Basic scientific calculator
- Engineering formula evaluator
- Input sanitization to prevent injection attacks
- Responsive web design

### 8.1.4 Security Model

The calculator implements multiple layers of input validation:

| Check               | Description                          |
| ------------------- | ------------------------------------ |
| Input length limits | Maximum expression length            |
| Character whitelist | Only allowed mathematical characters |
| Function whitelist  | Only safe mathematical functions     |
| Expression parsing  | AST-based evaluation (no `eval()`)   |

---

## 8.2 Unit Converter

**Source:** `src/web_applications/unit_converter/`
**Status:** ✅ Implemented (HTML/CSS/JS)

### 8.2.1 Purpose

Standalone web-based unit conversion tool supporting common engineering units.

### 8.2.2 Supported Categories

| Category    | Example Conversions        |
| ----------- | -------------------------- |
| Length      | m, ft, in, cm, mm, km, mi  |
| Mass        | kg, lb, g, oz, ton         |
| Temperature | °C, °F, K, °R              |
| Pressure    | Pa, bar, psi, atm, mmHg    |
| Volume      | L, gal, m³, ft³            |
| Flow Rate   | m³/s, L/min, gal/min, SCFM |
| Energy      | J, kJ, BTU, cal, kWh       |

### 8.2.3 Key Conversion Formulas

**Temperature:**

$$T_F = \frac{9}{5} T_C + 32$$

$$T_K = T_C + 273.15$$

$$T_R = T_F + 459.67$$

**Pressure:**

$$P_{atm} = P_{Pa} / 101325$$

$$P_{psi} = P_{Pa} / 6894.757$$

$$P_{bar} = P_{Pa} / 100000$$

---

## 8.3 URDF Viewer

**Source:** `src/web_applications/urdf_viewer/`
**Status:** ✅ Implemented (HTML/JS/Three.js)

### 8.3.1 Purpose

Web-based 3D viewer for URDF robot models using Three.js for WebGL rendering.

### 8.3.2 Capabilities

- Load and parse URDF XML files
- 3D visualization with Three.js
- Interactive camera controls (orbit, pan, zoom)
- Joint angle manipulation
- Link/joint highlighting
- Responsive layout

---

_[← Data & Document Processing](./07_data_document_processing.md) | [Back to Manual](./TOOLS_USER_MANUAL.md) | [Next: Media Processing →](./09_media_processing.md)_
