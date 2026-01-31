# Web Applications

Browser-based tools and Progressive Web Apps for scientific computing and utility functions.

## Overview

This directory contains web-based applications that run in the browser, providing accessible interfaces for mathematical computations and unit conversions. All applications are designed to work offline and can be installed as Progressive Web Apps.

## Components

### [Aurora CAS Calculator](calculator/README.md)

A web-based Computer Algebra System (CAS) calculator featuring:

- **Symbolic Math**: Factor, expand, simplify, solve equations
- **Calculus**: Derivatives, integrals, limits, Taylor series
- **Linear Algebra**: Matrix operations, decompositions, eigentools
- **Robotics Support**: Screw theory, SE(3) operations
- **Flask Backend**: Python-powered with SymPy engine
- **Modern UI**: Touch-friendly interface with mode-specific soft keys

### [Unit Converter](unit_converter/README.md)

A NIST-compliant unit converter PWA featuring:

- **16+ Categories**: Length, Mass, Volume, Temperature, Pressure, Energy, and more
- **100+ Units**: All with NIST-standard conversion factors
- **Offline Support**: Full functionality without internet
- **iOS Optimized**: Native-feeling interface with iOS design patterns
- **Custom Units**: Add your own units with conversion factors
- **No Backend Required**: 100% client-side JavaScript

### [URDF Viewer](urdf_viewer/README.md)

A web-based 3D model explorer featuring:

- **Three.js Rendering**: Interactive 3D visualization of URDF models
- **FastAPI Backend**: Python-powered backend for model management
- **React Frontend**: Modern UI with direct file uploads

## Quick Start

### Calculator

```bash
cd web_applications/calculator

# Development server
flask --app webapp run

# Access at http://localhost:5000
```

### Unit Converter

```bash
cd web_applications/unit_converter/unit-converter-app

# Serve locally
python -m http.server 8000

# Access at http://localhost:8000
```

Or simply open `unit-converter-app/index.html` in a browser.

### URDF Viewer

```bash
cd web_applications/urdf_viewer

# Start server
uvicorn app:app --reload

# Access at http://localhost:8000
```

## Dependencies

### Calculator
```bash
pip install flask sympy
```

### Unit Converter
No dependencies required - pure HTML/CSS/JavaScript.

### URDF Viewer
```bash
pip install -r web_applications/urdf_viewer/requirements.txt
```

## Integration

These applications integrate with:
- **Scientific Modeling** (`scientific_modeling/`) - Shared math utilities
- **Data Processing** (`data_processing/`) - Data analysis workflows

## License

Part of the Tools repository. See main repository license for details.
