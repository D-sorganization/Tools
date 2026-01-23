# Aurora CAS Calculator

A web-based Computer Algebra System (CAS) calculator inspired by the TI-89, built with Flask and SymPy.

## Overview

Aurora CAS is a powerful symbolic mathematics calculator that carries forward the TI-89's computer-algebra strengths while layering in modern touch controls, clearer UX cues, and a richer linear algebra surface. It provides a web interface for advanced mathematical computations.

## Features

### Core Mathematics
- **Constants**: pi, e, imaginary unit I
- **Numeric Tools**: Rounding, floor/ceiling, factorials, nPr, nCr
- **Trigonometry**: Full circular, inverse, and hyperbolic functions
- **Complex Arithmetic**: Magnitude, argument, conjugation, polar/rectangular

### Computer Algebra System
- **Algebra**: Factor, expand, cancel, simplify
- **Partial Fractions**: Decomposition and rational simplification
- **Equation Solving**: Single equations and simultaneous systems
- **Substitution**: Symbolic variable substitution

### Calculus
- Symbolic derivatives
- Definite and indefinite integrals
- Directional limits
- Taylor series expansion
- Ordinary differential equations

### Linear Algebra
- Matrix constructors (Matrix, eye, ones, zeros)
- Decompositions (QR, LU, SVD)
- Eigentools (eigenvals, eigenvects, charpoly)
- Vector operations (dot, cross, norm)
- Matrix exponentials and powers

### Robotics and Screw Theory
- Skew/vee utilities
- SE(3) hat/vee operations
- Screw axis builder
- Twist exponentials for rigid transforms

For a complete list of features, see [FUNCTIONALITY.md](FUNCTIONALITY.md).

## Directory Structure

```
calculator/
├── README.md              # This file
├── FUNCTIONALITY.md       # Detailed feature documentation
├── __init__.py            # Package initialization
├── calculator.py          # Core CAS engine
├── webapp.py              # Flask web application
├── limiter.py             # Rate limiting
├── static/                # CSS, JavaScript, images
├── templates/             # HTML templates
└── tests/                 # Unit tests
```

## Installation

### Dependencies

```bash
pip install flask sympy
```

### Running the Application

```bash
cd web_applications/calculator

# Development server
flask --app webapp run

# Or using Python
python -c "from webapp import create_app; create_app().run(debug=True)"
```

The application will be available at `http://localhost:5000`

## API

### Endpoint: POST `/api/calculate`

**Request Body:**
```json
{
  "operation": "simplify",
  "expression": "x^2 + 2*x + 1"
}
```

**Supported Operations:**
- `simplify` - Simplify expression
- `factor` - Factor expression
- `expand` - Expand expression
- `solve` - Solve equation for variable
- `diff` - Differentiate
- `integrate` - Integrate (definite/indefinite)
- `limit` - Calculate limit
- `series` - Taylor series expansion
- `matrix` - Matrix operations

**Response:**
```json
{
  "result": "(x + 1)**2",
  "latex": "(x + 1)^{2}"
}
```

### Rate Limiting

The API is rate-limited to 100 requests per 60 seconds per IP address to prevent abuse.

## UX Features

- Mode-specific soft keys for CAS, algebra, calculus, linear algebra
- Touch-focused editing with tap-to-place cursor
- History recall and ANS insertion
- Copy buttons for inputs and outputs
- Tablet-friendly interface

## Not Yet Implemented

- Graphing (2D/3D)
- Geometry applications
- Statistics
- Numeric solvers for regressions
- Probability distributions (beyond factorial/nCr/nPr)

## Running Tests

```bash
cd web_applications/calculator
pytest tests/
```

## Security

- Rate limiting prevents DoS attacks
- Input validation on all expressions
- Sandboxed expression evaluation via SymPy

## Integration

This calculator integrates with:
- **Unit Converter** (`web_applications/unit_converter/`) - Related web tool
- **Scientific Modeling** (`scientific_modeling/`) - Shared math utilities

## License

Part of the Tools repository. See main repository license for details.
