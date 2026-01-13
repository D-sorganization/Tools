from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import sympy as sp
from flask import (
    Flask,
    Response,
    current_app,
    jsonify,
    render_template,
    request,
    send_from_directory,
)
from sympy.parsing.sympy_parser import convert_xor, parse_expr, standard_transformations

from .calculator import CalculatorResult, TI89Calculator
from .limiter import RateLimiter

logger = logging.getLogger(__name__)


@dataclass
class CalculationPayload:
    operation: str
    expression: str
    variable: str | None = None
    variables: Mapping[str, str] | None = None
    order: int | None = None
    lower: str | None = None
    upper: str | None = None
    value: str | None = None
    direction: str | None = None
    around: str | None = None
    function: str | None = None


def create_app() -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")
    calculator = TI89Calculator()
    # Rate limit: 100 requests per 60 seconds per IP
    app.limiter = RateLimiter(limit=100, window=60)  # type: ignore[attr-defined]

    @app.after_request  # type: ignore[misc]
    def add_security_headers(response: Response) -> Response:
        """Add security headers to every response."""
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "script-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "object-src 'none'; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self';"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        # HSTS (Strict-Transport-Security) - enforce HTTPS
        # Max-age: 1 year (31536000 seconds), includeSubDomains
        # Only strict if running on HTTPS, but good to have
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )
        return response

    @app.get("/")  # type: ignore[misc]
    def index() -> str:
        return cast(str, render_template("index.html"))

    @app.post("/api/calculate")  # type: ignore[misc]
    def calculate() -> tuple[Any, int]:
        # Security: Rate limiting to prevent DoS
        if not current_app.testing:
            client_ip = request.remote_addr or "unknown"
            # Access limiter via closure over 'app'
            if not app.limiter.is_allowed(client_ip):  # type: ignore[attr-defined]
                return (
                    jsonify({"error": "Rate limit exceeded. Please try again later."}),
                    429,
                )

        payload = request.get_json(silent=True) or {}
        try:
            parsed_payload = _parse_payload(payload)
            calculation = _dispatch_calculation(calculator, parsed_payload)
            response = _serialize_result(calculation)
            return jsonify(response), 200
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        except Exception:  # pragma: no cover - fallback safety
            logger.exception("Calculation failed")
            return jsonify({"error": "An internal error occurred."}), 500

    @app.get("/manifest.webmanifest")  # type: ignore[misc]
    def manifest() -> Any:
        return send_from_directory(app.static_folder, "manifest.webmanifest")

    @app.get("/service-worker.js")  # type: ignore[misc]
    def service_worker() -> Any:
        return send_from_directory(app.static_folder, "service-worker.js")

    return app


MAX_INPUT_LENGTH = 1000


def _validate_security(value: str | None) -> None:
    """Check for potentially dangerous patterns in input."""
    if value and "__" in value:
        raise ValueError("Security violation: Restricted input pattern detected.")


def _parse_payload(raw_payload: Mapping[str, object]) -> CalculationPayload:
    operation = str(raw_payload.get("operation", "")).strip()
    if not operation:
        raise ValueError("Operation is required")

    expression = str(raw_payload.get("expression", "")).strip()
    if not expression:
        raise ValueError("Expression is required")

    if len(expression) > MAX_INPUT_LENGTH:
        raise ValueError(
            f"Expression exceeds maximum length of {MAX_INPUT_LENGTH} characters"
        )
    _validate_security(expression)

    variable = _clean_optional(raw_payload.get("variable"))
    _validate_length(variable, "Variable")
    _validate_security(variable)

    variables: Mapping[str, str] | None = None
    if isinstance(raw_payload.get("variables"), Mapping):
        variables = {}
        for key, val in raw_payload["variables"].items():  # type: ignore[attr-defined]
            k_str, v_str = str(key), str(val)
            _validate_length(k_str, "Variable name")
            _validate_length(v_str, "Variable value")
            _validate_security(k_str)
            _validate_security(v_str)
            variables[k_str] = v_str

    lower = _clean_optional(raw_payload.get("lower"))
    _validate_length(lower, "Lower bound")
    _validate_security(lower)

    upper = _clean_optional(raw_payload.get("upper"))
    _validate_length(upper, "Upper bound")
    _validate_security(upper)

    value = _clean_optional(raw_payload.get("value"))
    _validate_length(value, "Value")
    _validate_security(value)

    direction = _clean_optional(raw_payload.get("direction"))
    _validate_length(direction, "Direction")
    _validate_security(direction)

    around = _clean_optional(raw_payload.get("around"))
    _validate_length(around, "Around value")
    _validate_security(around)

    function = _clean_optional(raw_payload.get("function"))
    _validate_length(function, "Function name")

    return CalculationPayload(
        operation=operation,
        expression=expression,
        variable=variable,
        variables=variables,
        order=_parse_optional_int(raw_payload.get("order")),
        lower=lower,
        upper=upper,
        value=value,
        direction=direction,
        around=around,
        function=function,
    )


def _validate_length(value: str | None, name: str) -> None:
    if value and len(value) > MAX_INPUT_LENGTH:
        raise ValueError(
            f"{name} exceeds maximum length of {MAX_INPUT_LENGTH} characters"
        )


def _dispatch_calculation(
    calculator: TI89Calculator, payload: CalculationPayload
) -> CalculatorResult:
    if payload.operation == "evaluate":
        substitutions = _normalize_variables(payload.variables, calculator)
        return calculator.evaluate(payload.expression, substitutions)

    if payload.operation == "simplify":
        return calculator.simplify_expression(payload.expression)

    if payload.operation == "solve_equation":
        if not payload.variable:
            raise ValueError("Variable is required for solving an equation")
        return calculator.solve_equation(payload.expression, payload.variable)

    if payload.operation == "solve_system":
        if not payload.variable:
            raise ValueError(
                "Comma-separated variables are required for solving a system"
            )
        variables = [
            part.strip() for part in payload.variable.split(",") if part.strip()
        ]
        equations = [
            part.strip() for part in payload.expression.split(";") if part.strip()
        ]
        if not equations or not variables:
            raise ValueError("Equations and variables are required for system solving")
        return calculator.solve_system(equations, variables)

    if payload.operation == "derivative":
        if not payload.variable:
            raise ValueError("Variable is required for derivatives")
        order = payload.order if payload.order is not None else 1
        if order <= 0:
            raise ValueError("Derivative order must be a positive integer")
        return calculator.derivative(payload.expression, payload.variable, order=order)

    if payload.operation == "integral":
        if not payload.variable:
            raise ValueError("Variable is required for integrals")
        if payload.lower is not None or payload.upper is not None:
            if payload.lower is None or payload.upper is None:
                raise ValueError(
                    "Both lower and upper bounds are required for definite integrals"
                )
            variable_symbol = sp.Symbol(payload.variable)
            lower = _sympify_value(
                payload.lower,
                calculator=calculator,
                symbols={payload.variable: variable_symbol},
            )
            upper = _sympify_value(
                payload.upper,
                calculator=calculator,
                symbols={payload.variable: variable_symbol},
            )
            return calculator.integral(
                payload.expression, payload.variable, lower=lower, upper=upper
            )
        return calculator.integral(payload.expression, payload.variable)

    if payload.operation == "limit":
        if not payload.variable:
            raise ValueError("Variable is required for limits")
        if payload.value is None:
            raise ValueError("A limit value is required")
        direction = payload.direction or "two-sided"
        return calculator.limit(
            payload.expression,
            payload.variable,
            _sympify_value(payload.value, calculator=calculator),
            direction=direction,
        )

    if payload.operation == "taylor_series":
        if not payload.variable:
            raise ValueError("Variable is required for series expansion")
        if payload.around is None:
            raise ValueError("Expansion point is required for series expansion")
        if payload.order is None or payload.order <= 0:
            raise ValueError("Series order must be a positive integer")
        return calculator.taylor_series(
            payload.expression,
            payload.variable,
            _sympify_value(payload.around, calculator=calculator),
            payload.order,
        )

    if payload.operation == "solve_ode":
        if not payload.function:
            raise ValueError(
                "Function name is required for solving differential equations"
            )
        return calculator.solve_differential_equation(
            payload.expression, payload.function
        )

    raise ValueError("Unsupported operation requested")


def _serialize_result(calculation: CalculatorResult) -> Mapping[str, object]:
    approximation = _approximate(calculation.result)
    pretty = _pretty(calculation.result)
    return {
        "input": calculation.input_expression,
        "result": _serialize(calculation.result),
        "pretty": pretty,
        "approximation": approximation,
    }


def _approximate(result: object, precision: int = 10) -> float | None:
    if isinstance(result, sp.Basic):
        try:
            numeric = sp.N(result, precision)
            return float(numeric)
        except (TypeError, ValueError):
            return None
    return None


def _pretty(result: object) -> str | None:
    if isinstance(result, sp.Basic):
        return sp.pretty(result)
    return None


def _serialize(result: object) -> object:
    if isinstance(result, sp.Basic):
        return sp.sstr(result)
    if isinstance(result, list | tuple):
        return [_serialize(item) for item in result]
    if isinstance(result, dict):
        return {str(key): _serialize(value) for key, value in result.items()}
    return result


def _normalize_variables(
    variables: Mapping[str, str] | None, calculator: TI89Calculator
) -> Mapping[str, sp.Expr]:
    if not variables:
        return {}
    return {
        name: _sympify_value(value, calculator=calculator)
        for name, value in variables.items()
    }


def _sympify_value(
    value: str,
    *,
    calculator: TI89Calculator,
    symbols: Mapping[str, sp.Symbol | sp.Expr] | None = None,
) -> sp.Expr:
    # Optimization: fast-path for simple numbers to avoid expensive parse_expr
    # This speeds up numeric inputs by >50x
    if value:
        # Strip whitespace for accurate numeric checks
        clean_value = value.strip()

        # Check for integer
        if clean_value.isdigit():
            return sp.Integer(int(clean_value))
        if clean_value.startswith("-") and clean_value[1:].isdigit():
            return sp.Integer(int(clean_value))

        # Check for simple float, avoiding symbolic expressions that start with letters
        # This heuristic prevents overhead for inputs like "x", "sin(x)" which fail float conversion
        if clean_value and not clean_value[0].isalpha():
            try:
                # Validate if it is a float using native conversion
                float(clean_value)
                # Use string constructor for sp.Float to preserve precision/range
                # (native float has limits e.g. 1e400 -> inf)
                return sp.Float(clean_value)
            except ValueError:
                pass

    try:
        # Optimization: Use cached allowed_functions directly if no extra symbols are
        # needed
        local_dict = calculator.allowed_functions
        if symbols:
            local_dict = {**calculator.allowed_functions, **symbols}

        return parse_expr(
            value,
            local_dict=local_dict,
            global_dict=calculator.safe_globals,
            transformations=standard_transformations + (convert_xor,),
            evaluate=True,
        )
    except Exception as error:
        raise ValueError("Invalid numeric or symbolic value provided") from error


def _parse_optional_int(value: object | None) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value))
    except (TypeError, ValueError):
        raise ValueError("Integer value expected") from None


def _clean_optional(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
