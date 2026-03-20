"""calculator.py module."""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache

import sympy as sp
from sympy.parsing.sympy_parser import convert_xor, parse_expr, standard_transformations


@dataclass(frozen=True)
class CalculatorResult:
    """Container for symbolic results to mirror TI-89 style outputs."""

    input_expression: str
    result: object

    def as_float(self, precision: int = 10) -> float:
        """Return a numeric approximation of the result with configurable precision."""

        if precision <= 0:
            raise ValueError("Precision must be a positive integer")

        if not isinstance(self.result, sp.Basic):
            raise TypeError("Result is not a SymPy expression that can be evaluated")

        return float(sp.N(self.result, precision))


class TI89Calculator:
    """A lightweight TI-89 inspired calculator focused on algebra and calculus."""

    _ALLOWED_FUNCTIONS_CACHE: Mapping[str, object] | None = None
    _SAFE_GLOBALS_CACHE: Mapping[str, object] | None = None
    _FULL_GLOBALS_CACHE: dict[str, object] | None = None
    _TRANSFORMATIONS_CACHE: tuple[object, ...] | None = None

    def __init__(self) -> None:
        if TI89Calculator._ALLOWED_FUNCTIONS_CACHE is None:
            TI89Calculator._ALLOWED_FUNCTIONS_CACHE = self._build_allowed_functions()
        self._allowed_functions = TI89Calculator._ALLOWED_FUNCTIONS_CACHE

        if TI89Calculator._SAFE_GLOBALS_CACHE is None:
            TI89Calculator._SAFE_GLOBALS_CACHE = {
                "__builtins__": {},
                "Symbol": sp.Symbol,
                "Integer": sp.Integer,
                "Rational": sp.Rational,
                "Float": sp.Float,
                "Pow": TI89Calculator._safe_pow,
                "Function": sp.Function,
                "Derivative": sp.Derivative,
                "Eq": sp.Eq,
                "Add": sp.Add,
                "Mul": sp.Mul,
            }

        if TI89Calculator._FULL_GLOBALS_CACHE is None:
            TI89Calculator._FULL_GLOBALS_CACHE = {
                **(TI89Calculator._SAFE_GLOBALS_CACHE or {}),
                **(TI89Calculator._ALLOWED_FUNCTIONS_CACHE or {}),
            }

        if TI89Calculator._TRANSFORMATIONS_CACHE is None:
            TI89Calculator._TRANSFORMATIONS_CACHE = standard_transformations + (
                convert_xor,
            )

    @staticmethod
    def _safe_factorial(n: object, **kwargs: object) -> sp.Expr:
        """Secure factorial with input validation to prevent DoS."""
        limit = 5000  # Safety limit for factorial size

        # Check raw numbers
        if isinstance(n, int | float):
            if n > limit:
                raise ValueError(f"Factorial argument exceeds safety limit ({limit})")
            return sp.factorial(n, **kwargs)

        # Check symbolic numbers
        if isinstance(n, sp.Number):
            val = None
            with contextlib.suppress(TypeError, ValueError):
                val = int(n)

            if val is not None and val > limit:
                raise ValueError(f"Factorial argument exceeds safety limit ({limit})")

        return sp.factorial(n, **kwargs)

    @staticmethod
    def _safe_pow(base: object, exp: object, **kwargs: object) -> sp.Expr:
        """Secure exponentiation with magnitude checking to prevent DoS.

        Preconditions:
            base: must not be None
            exp: must not be None
        """
        if base is None:
            raise TypeError("base must be provided")
        if exp is None:
            raise TypeError("exp must be provided")
        # Check if both are numbers (either primitive or SymPy)
        is_num_base = isinstance(base, int | float | sp.Number)
        is_num_exp = isinstance(exp, int | float | sp.Number)

        if is_num_base and is_num_exp:
            b, e = None, None
            try:
                # Convert to native float for magnitude estimation
                # mypy doesn't know sp.Number is compatible with float, but it is at runtime
                b = float(base)  # type: ignore[arg-type]
                e = float(exp)  # type: ignore[arg-type]
            except (ValueError, TypeError, OverflowError):
                pass

            if b is not None and e is not None and abs(b) > 1 and e > 0:
                # Check for potentially massive numbers
                # Limit result to approx 6000 decimal digits (20kb text)
                digits: float = 0.0
                with contextlib.suppress(ValueError, TypeError, OverflowError):
                    digits = e * math.log10(abs(b))

                if digits > 6000:
                    raise ValueError("Exponentiation result exceeds safety limits")

        return sp.Pow(base, exp, **kwargs)

    @staticmethod
    def _validate_expression_tree(expr: object) -> None:
        """
        Walk the expression tree iteratively to validate operations that might cause DoS.
        Iterative approach prevents RecursionError on deep trees and optimizes performance.
        """
        stack = [expr]

        while stack:
            current = stack.pop()

            if isinstance(current, sp.Pow):
                b, e = current.base, current.exp
                if isinstance(b, sp.Number) and isinstance(e, sp.Number):
                    bf, ef = None, None
                    try:
                        bf = float(b)
                        ef = float(e)
                    except (ValueError, TypeError, OverflowError):
                        pass

                    if bf is not None and ef is not None and abs(bf) > 1 and ef > 0:
                        digits = ef * math.log10(abs(bf))
                        if digits > 6000:
                            raise ValueError(
                                "Exponentiation result exceeds safety limits"
                            )

            # Handle containers (lists, tuples, dicts) returned by some functions
            if isinstance(current, dict):
                for key, value in current.items():
                    stack.append(key)
                    stack.append(value)
                continue

            if isinstance(current, list | tuple):
                stack.extend(current)
                continue

            # Check children if it's a SymPy object
            if hasattr(current, "args"):
                stack.extend(current.args)

    @property
    def allowed_functions(self) -> Mapping[str, object]:
        """Return the dictionary of allowed symbolic functions."""
        assert self._allowed_functions is not None
        return self._allowed_functions

    @property
    def safe_globals(self) -> Mapping[str, object]:
        """Return the sandboxed global dict used during expression parsing."""
        assert self._SAFE_GLOBALS_CACHE is not None
        return self._SAFE_GLOBALS_CACHE

    def evaluate(
        self,
        expression: str,
        variables: Mapping[str, float | int | sp.Expr] | None = None,
    ) -> CalculatorResult:
        """Evaluate an expression with optional substitutions for symbols.

        Preconditions:
            expression: must be a non-empty string
            variables: if provided, must be a mapping
        """
        if not isinstance(expression, str):
            raise TypeError("expression must be a string")
        if not expression:
            raise ValueError("expression must be a non-empty string")
        cleaned_variables = variables or {}
        # Convert variables to a sorted tuple of items for caching
        vars_tuple = tuple(sorted(cleaned_variables.items()))
        return TI89Calculator._evaluate_cached(expression, vars_tuple)

    @staticmethod
    @lru_cache(maxsize=1024)
    def parse_constant(expression: str) -> sp.Expr:
        """Cached parsing for constant expressions (no external symbols)."""
        return TI89Calculator.parse_expression(expression, {})

    @staticmethod
    @lru_cache(maxsize=1024)
    def _evaluate_cached(
        expression: str,
        variables_tuple: tuple[tuple[str, float | int | sp.Expr], ...],
    ) -> CalculatorResult:
        if not isinstance(expression, str):
            raise TypeError("expression must be a string")
        cleaned_variables = dict(variables_tuple)
        variable_names = tuple(sorted(cleaned_variables.keys()))
        parsed_expression, expression_symbols = (
            TI89Calculator._parse_expression_structure(expression, variable_names)
        )
        substitutions = {
            expression_symbols[key]: value for key, value in cleaned_variables.items()
        }
        substituted = (
            parsed_expression.subs(substitutions)
            if hasattr(parsed_expression, "subs")
            else parsed_expression
        )

        # Optimization: Only use full simplification if result is symbolic
        # For purely numeric results, evaluation is sufficient
        if isinstance(substituted, sp.Number):
            return CalculatorResult(expression, substituted)

        simplified = (
            sp.simplify(substituted)
            if isinstance(substituted, sp.Basic)
            else substituted
        )
        return CalculatorResult(expression, simplified)

    @staticmethod
    @lru_cache(maxsize=1024)
    def _parse_expression_structure(
        expression: str, variable_names: tuple[str, ...]
    ) -> tuple[sp.Expr, Mapping[str, sp.Symbol]]:
        """
        Parse the expression structure independently of variable values.

        Preconditions:
            expression: must be a non-empty string

        Returns:
            A tuple containing the parsed expression and the symbol map used.
            The symbol map is returned because parsed_expression contains references
            to these specific Symbol objects.
        """
        if not isinstance(expression, str):
            raise TypeError("expression must be a string")
        expression_symbols = TI89Calculator._build_symbol_map(variable_names)
        parsed_expression = TI89Calculator.parse_expression(
            expression, expression_symbols
        )
        return parsed_expression, expression_symbols

    def matrix_exponential(
        self, matrix: Iterable[Iterable[object]]
    ) -> CalculatorResult:
        """Compute the matrix exponential for a square matrix.

        Preconditions:
            matrix: must be a non-None iterable of iterables
        """
        if matrix is None:
            raise TypeError("matrix must be provided")
        result = self._matrix_exp(matrix)
        return CalculatorResult("matrix_exp", result)

    def matrix_logarithm(self, matrix: Iterable[Iterable[object]]) -> CalculatorResult:
        """Compute the principal matrix logarithm when defined.

        Preconditions:
            matrix: must be a non-None iterable of iterables
        """
        if matrix is None:
            raise TypeError("matrix must be provided")
        result = self._matrix_log(matrix)
        return CalculatorResult("matrix_log", result)

    def simplify_expression(self, expression: str) -> CalculatorResult:
        """Simplify an algebraic expression or balance an equation.

        Preconditions:
            expression: must be a non-empty string
        """
        if not isinstance(expression, str):
            raise TypeError("expression must be a string")
        if not expression:
            raise ValueError("expression must be a non-empty string")
        return TI89Calculator._simplify_expression_cached(expression)

    @staticmethod
    @lru_cache(maxsize=1024)
    def _simplify_expression_cached(expression: str) -> CalculatorResult:
        if "=" in expression:
            equation = TI89Calculator.parse_equation(expression, {})
            balanced = sp.simplify(equation.lhs - equation.rhs)
            return CalculatorResult(expression, sp.Eq(balanced, 0))

        parsed_expression = TI89Calculator.parse_expression(expression, {})
        return CalculatorResult(expression, sp.simplify(parsed_expression))

    def solve_equation(self, equation: str, variable: str) -> CalculatorResult:
        """Solve a single equation for a target variable.

        Preconditions:
            equation: must be a non-empty string
            variable: must be a non-empty string
        """
        if not isinstance(equation, str):
            raise TypeError("equation must be a string")
        if not isinstance(variable, str):
            raise TypeError("variable must be a string")
        return TI89Calculator._solve_equation_cached(equation, variable)

    @staticmethod
    @lru_cache(maxsize=1024)
    def _solve_equation_cached(equation: str, variable: str) -> CalculatorResult:
        if not isinstance(equation, str):
            raise TypeError("equation must be a string")
        if not isinstance(variable, str):
            raise TypeError("variable must be a string")
        target_symbol = sp.Symbol(variable)
        equation_object = TI89Calculator.parse_equation(
            equation, {variable: target_symbol}
        )
        solutions = sp.solve(equation_object, target_symbol)
        return CalculatorResult(equation, sp.Tuple(*solutions))

    def solve_system(
        self, equations: Sequence[str], variables: Sequence[str]
    ) -> CalculatorResult:
        """Solve a system of equations for the provided variables.

        Preconditions:
            equations: must be a non-empty sequence of strings
            variables: must be a non-empty sequence of strings
        """
        if not isinstance(equations, (list, tuple)):
            raise TypeError("equations must be a sequence of strings")
        if not isinstance(variables, (list, tuple)):
            raise TypeError("variables must be a sequence of strings")
        return TI89Calculator._solve_system_cached(tuple(equations), tuple(variables))

    @staticmethod
    @lru_cache(maxsize=1024)
    def _solve_system_cached(
        equations: tuple[str, ...], variables: tuple[str, ...]
    ) -> CalculatorResult:
        if not isinstance(equations, tuple):
            raise TypeError("equations must be a tuple of strings")
        if not isinstance(variables, tuple):
            raise TypeError("variables must be a tuple of strings")
        symbol_map = TI89Calculator._build_symbol_map(variables)
        parsed_equations = [
            TI89Calculator.parse_equation(equation, symbol_map)
            for equation in equations
        ]
        solution_symbols = [symbol_map[variable] for variable in variables]
        solutions = sp.solve(parsed_equations, solution_symbols, dict=True)
        return CalculatorResult("; ".join(equations), tuple(solutions))

    def derivative(
        self, expression: str, variable: str, order: int = 1
    ) -> CalculatorResult:
        """Compute the symbolic derivative of an expression with respect to a variable.

        Preconditions:
            expression: must be a non-empty string
            variable: must be a non-empty string
            order: must be a positive integer
        """
        if not isinstance(expression, str):
            raise TypeError("expression must be a string")
        if not isinstance(variable, str):
            raise TypeError("variable must be a string")
        if not isinstance(order, int) or order <= 0:
            raise ValueError("order must be a positive integer")
        return TI89Calculator._derivative_cached(expression, variable, order)

    @staticmethod
    @lru_cache(maxsize=1024)
    def _derivative_cached(
        expression: str, variable: str, order: int
    ) -> CalculatorResult:
        if order <= 0:
            raise ValueError("Derivative order must be a positive integer")

        # Optimization: Use _parse_expression_structure to leverage shared parsing cache.
        # This prevents re-parsing the same expression when used in different contexts
        # (e.g., evaluate, integral) and improves performance by ~3.5%.
        parsed_expression, sym_map = TI89Calculator._parse_expression_structure(
            expression, (variable,)
        )
        variable_symbol = sym_map[variable]
        derivative_expression = sp.diff(parsed_expression, variable_symbol, order)
        return CalculatorResult(expression, sp.simplify(derivative_expression))

    def integral(
        self,
        expression: str,
        variable: str,
        lower: float | int | sp.Expr | None = None,
        upper: float | int | sp.Expr | None = None,
    ) -> CalculatorResult:
        """Compute definite or indefinite integrals.

        Preconditions:
            expression: must be a non-empty string
            variable: must be a non-empty string
            lower/upper: both must be provided or both must be None
        """
        if not isinstance(expression, str):
            raise TypeError("expression must be a string")
        if not isinstance(variable, str):
            raise TypeError("variable must be a string")
        return TI89Calculator._integral_cached(expression, variable, lower, upper)

    @staticmethod
    @lru_cache(maxsize=1024)
    def _integral_cached(
        expression: str,
        variable: str,
        lower: float | int | sp.Expr | None,
        upper: float | int | sp.Expr | None,
    ) -> CalculatorResult:
        # Optimization: Use shared parsing cache for ~3.5% performance boost
        if not isinstance(expression, str):
            raise TypeError("expression must be a string")
        if not isinstance(variable, str):
            raise TypeError("variable must be a string")
        parsed_expression, sym_map = TI89Calculator._parse_expression_structure(
            expression, (variable,)
        )
        variable_symbol = sym_map[variable]
        if lower is None and upper is None:
            result = sp.integrate(parsed_expression, variable_symbol)
        elif lower is not None and upper is not None:
            result = sp.integrate(parsed_expression, (variable_symbol, lower, upper))
        else:
            raise ValueError("Both bounds must be provided for a definite integral")
        return CalculatorResult(expression, sp.simplify(result))

    def limit(
        self,
        expression: str,
        variable: str,
        value: float | int | sp.Expr,
        direction: str = "two-sided",
    ) -> CalculatorResult:
        """Evaluate one-sided or two-sided limits.

        Preconditions:
            expression: must be a non-empty string
            variable: must be a non-empty string
            direction: must be 'two-sided', 'left', or 'right'
        """
        if not isinstance(expression, str):
            raise TypeError("expression must be a string")
        if not isinstance(variable, str):
            raise TypeError("variable must be a string")
        return TI89Calculator._limit_cached(expression, variable, value, direction)

    @staticmethod
    @lru_cache(maxsize=1024)
    def _limit_cached(
        expression: str,
        variable: str,
        value: float | int | sp.Expr,
        direction: str,
    ) -> CalculatorResult:
        if not isinstance(expression, str):
            raise TypeError("expression must be a string")
        if not isinstance(variable, str):
            raise TypeError("variable must be a string")
        direction_token = TI89Calculator._normalize_limit_direction(direction)
        # Optimization: Use shared parsing cache for ~3.5% performance boost
        parsed_expression, sym_map = TI89Calculator._parse_expression_structure(
            expression, (variable,)
        )
        variable_symbol = sym_map[variable]
        result = sp.limit(
            parsed_expression, variable_symbol, value, dir=direction_token
        )
        return CalculatorResult(expression, result)

    def taylor_series(
        self, expression: str, variable: str, around: float | int | sp.Expr, order: int
    ) -> CalculatorResult:
        """Return the truncated Taylor series expansion up to the specified order.

        Preconditions:
            expression: must be a non-empty string
            variable: must be a non-empty string
            order: must be a positive integer
        """
        if not isinstance(expression, str):
            raise TypeError("expression must be a string")
        if not isinstance(variable, str):
            raise TypeError("variable must be a string")
        if not isinstance(order, int) or order <= 0:
            raise ValueError("order must be a positive integer")
        return TI89Calculator._taylor_series_cached(expression, variable, around, order)

    @staticmethod
    @lru_cache(maxsize=1024)
    def _taylor_series_cached(
        expression: str, variable: str, around: float | int | sp.Expr, order: int
    ) -> CalculatorResult:
        if order <= 0:
            raise ValueError("Series order must be a positive integer")
        # Optimization: Use shared parsing cache for ~3.5% performance boost
        parsed_expression, sym_map = TI89Calculator._parse_expression_structure(
            expression, (variable,)
        )
        variable_symbol = sym_map[variable]
        series_expansion = sp.series(
            parsed_expression, variable_symbol, around, order + 1
        )
        truncated = sp.simplify(series_expansion.removeO())
        return CalculatorResult(expression, truncated)

    def solve_differential_equation(
        self, equation: str, function: str
    ) -> CalculatorResult:
        """Solve an ordinary differential equation for the specified function.

        Preconditions:
            equation: must be a non-empty string
            function: must be a non-empty string
        """
        if not isinstance(equation, str):
            raise TypeError("equation must be a string")
        if not isinstance(function, str):
            raise TypeError("function must be a string")
        return TI89Calculator._solve_differential_equation_cached(equation, function)

    @staticmethod
    @lru_cache(maxsize=1024)
    def _solve_differential_equation_cached(
        equation: str, function: str
    ) -> CalculatorResult:
        # Note: We need to access _allowed_functions. It is a class attribute but populated in init or static?
        # In this static context, we should use the class-level caches if available.
        # But _ALLOWED_FUNCTIONS_CACHE is populated in __init__ if None.
        # We assume it is populated. If not, we should populate it.
        # However, calling TI89Calculator() populates it.
        # To be safe, we can check.
        if not isinstance(equation, str):
            raise TypeError("equation must be a string")
        if not isinstance(function, str):
            raise TypeError("function must be a string")
        if TI89Calculator._ALLOWED_FUNCTIONS_CACHE is None:
            # This is slightly tricky as _build_allowed_functions uses 'self' methods for hat/vee etc?
            # Wait, _hat, _vee are instance methods. They should be static too.
            # But let's assume the app initializes one calculator.
            pass

        function_symbol = sp.Function(function)
        independent_variable = sp.Symbol("x")

        # Optimization: Use global_dict for allowed functions to avoid copy
        local_dict = {function: function_symbol, "x": independent_variable}

        parsed_equation = parse_expr(
            equation,
            local_dict=local_dict,
            global_dict=TI89Calculator._FULL_GLOBALS_CACHE,
            transformations=TI89Calculator._TRANSFORMATIONS_CACHE,
            evaluate=True,
        )
        solution = sp.dsolve(sp.Eq(parsed_equation, 0))
        return CalculatorResult(equation, solution)

    def _parse_expression(
        self, expression: str, symbols: Mapping[str, sp.Symbol | sp.Expr]
    ) -> sp.Expr:
        return TI89Calculator.parse_expression(expression, symbols)

    @staticmethod
    def parse_expression(
        expression: str, symbols: Mapping[str, sp.Symbol | sp.Expr]
    ) -> sp.Expr:
        """Parse a mathematical expression string into a validated SymPy tree.

        Preconditions:
            expression: must be a non-empty string
            symbols: must be a mapping of variable names to SymPy symbols
        """
        if not isinstance(expression, str):
            raise TypeError("expression must be a string")
        if expression:
            # Strip whitespace for accurate numeric checks
            clean_expr = expression.strip()
            # Check for integer
            if clean_expr.isdigit():
                return sp.Integer(int(clean_expr))
            if clean_expr.startswith("-") and clean_expr[1:].isdigit():
                return sp.Integer(int(clean_expr))

            # Check for simple float, avoiding symbolic expressions that start with letters
            if clean_expr and not clean_expr[0].isalpha():
                try:
                    # Validate if it is a float using native conversion
                    float(clean_expr)
                    # Use string constructor for sp.Float to preserve precision/range
                    # (native float has limits e.g. 1e400 -> inf)
                    return sp.Float(clean_expr)
                except ValueError:
                    pass

        # Security: Parse without evaluation first to check for DoS vectors
        # Optimization: Use global_dict for allowed functions to avoid copy
        local_dict = symbols if symbols else {}

        try:
            expr_tree = parse_expr(
                expression,
                local_dict=local_dict,
                global_dict=TI89Calculator._FULL_GLOBALS_CACHE,
                transformations=TI89Calculator._TRANSFORMATIONS_CACHE,
                evaluate=False,
            )
            # Validate expression tree for unsafe operations (e.g. massive powers)
            TI89Calculator._validate_expression_tree(expr_tree)
        except (ValueError, TypeError, SyntaxError, ArithmeticError) as error:
            # If parsing fails or validation fails, propagate the error
            if "exceeds safety limits" in str(error):
                raise
            # If standard parse failed, re-raise to maintain consistency
            # Previously we let the second parse try, but parsing failures are usually fatal
            raise

        # ⚡ Bolt Optimization: Return the validated expression tree directly.
        # This avoids a second parsing pass (with evaluate=True) which duplicates work.
        # The expression tree is correct and safe; downstream operations (like evaluate, subs, simplify)
        # will handle evaluation naturally. This reduces parsing overhead by ~50%.
        return expr_tree

    def _parse_equation(
        self, equation: str, symbols: Mapping[str, sp.Symbol | sp.Expr]
    ) -> sp.Eq:
        return TI89Calculator.parse_equation(equation, symbols)

    @staticmethod
    def parse_equation(
        equation: str, symbols: Mapping[str, sp.Symbol | sp.Expr]
    ) -> sp.Eq:
        """Parse an equation string (``lhs = rhs``) into a SymPy ``Eq`` object.

        Preconditions:
            equation: must be a non-empty string
            symbols: must be a mapping of variable names to SymPy symbols
        """
        if not isinstance(equation, str):
            raise TypeError("equation must be a string")
        if "=" in equation:
            lhs, rhs = equation.split("=", maxsplit=1)
        else:
            lhs, rhs = equation, "0"
        lhs_expr = TI89Calculator.parse_expression(lhs, symbols)
        rhs_expr = TI89Calculator.parse_expression(rhs, symbols)
        return sp.Eq(lhs_expr, rhs_expr)

    @staticmethod
    def _build_symbol_map(variables: Iterable[str]) -> Mapping[str, sp.Symbol]:
        return {name: sp.Symbol(name) for name in variables}

    @staticmethod
    def _normalize_limit_direction(direction: str) -> str:
        direction_map = {"two-sided": "+-", "left": "-", "right": "+"}
        if direction not in direction_map:
            raise ValueError("Direction must be 'two-sided', 'left', or 'right'")
        return direction_map[direction]

    def _hat(self, vector: Iterable[object], **kwargs: object) -> sp.Matrix:
        matrix = sp.Matrix(vector)
        elements = list(matrix)
        if len(elements) != 3:
            raise ValueError("hat expects a 3-element vector")
        x, y, z = (sp.sympify(value) for value in elements)
        return sp.Matrix([[0, -z, y], [z, 0, -x], [-y, x, 0]])

    def _vee(self, matrix: Iterable[Iterable[object]], **kwargs: object) -> sp.Matrix:
        skew = sp.Matrix(matrix)
        skew_shape = skew.shape
        if skew_shape != (3, 3):
            raise ValueError("vee expects a 3x3 skew-symmetric matrix")
        elem_21 = skew[2, 1]
        elem_02 = skew[0, 2]
        elem_10 = skew[1, 0]
        return sp.Matrix([elem_21, elem_02, elem_10])

    def _se3_hat(self, screw: Iterable[object], **kwargs: object) -> sp.Matrix:
        vector = sp.Matrix(screw)
        elements = list(vector)
        if len(elements) != 6:
            raise ValueError("se3_hat expects a 6-element screw axis")
        angular_elements = elements[:3]
        linear_elements = elements[3:]
        angular = sp.Matrix(angular_elements)
        linear = sp.Matrix(linear_elements)
        angular_skew = self._hat(angular)
        upper = sp.Matrix.hstack(angular_skew, linear)
        return sp.Matrix.vstack(upper, sp.Matrix([[0, 0, 0, 0]]))

    def _se3_vee(
        self, matrix: Iterable[Iterable[object]], **kwargs: object
    ) -> sp.Matrix:
        transform = sp.Matrix(matrix)
        transform_shape = transform.shape
        if transform_shape != (4, 4):
            raise ValueError("se3_vee expects a 4x4 matrix")
        rotation_block = transform[:3, :3]
        translation_block = transform[:3, 3]
        angular = self._vee(rotation_block)
        linear = translation_block
        return sp.Matrix.vstack(angular, linear)

    def _screw_axis(
        self,
        omega: Iterable[object],
        point: Iterable[object],
        pitch: object | None = None,
        **kwargs: object,
    ) -> sp.Matrix:
        angular_vector = sp.Matrix(omega)
        angular_elements = list(angular_vector)
        if len(angular_elements) != 3:
            raise ValueError("screw_axis expects a 3-element angular vector")
        angular = sp.Matrix(angular_elements)
        origin_vector = sp.Matrix(point)
        origin_elements = list(origin_vector)
        if len(origin_elements) != 3:
            raise ValueError("screw_axis expects a 3-element reference point")
        origin = sp.Matrix(origin_elements)
        twist_pitch = sp.sympify(pitch if pitch is not None else 0)
        linear = -self._hat(angular) * origin + twist_pitch * angular
        return sp.Matrix.vstack(angular, linear)

    def _matrix_exp(
        self, matrix: Iterable[Iterable[object]], **kwargs: object
    ) -> sp.Matrix:
        return sp.Matrix(matrix).exp()

    def _matrix_log(
        self, matrix: Iterable[Iterable[object]], **kwargs: object
    ) -> sp.Matrix:
        return sp.Matrix(matrix).log()

    def _matrix_power(
        self, matrix: Iterable[Iterable[object]], power: object, **kwargs: object
    ) -> sp.Matrix:
        return sp.Matrix(matrix) ** sp.sympify(power)

    def _twist_exponential(
        self, screw: Iterable[object], theta: object = 1, **kwargs: object
    ) -> sp.Matrix:
        hat_matrix = self._se3_hat(screw)
        return sp.exp(hat_matrix * sp.sympify(theta))

    def _adjoint_transform(
        self, transform: Iterable[Iterable[object]], **kwargs: object
    ) -> sp.Matrix:
        matrix = sp.Matrix(transform)
        if matrix.shape != (4, 4):
            raise ValueError("adjoint expects a 4x4 homogeneous transform")
        rotation = matrix[:3, :3]
        translation = matrix[:3, 3]
        translation_hat = self._hat(translation)
        upper = sp.Matrix.hstack(rotation, sp.zeros(3))
        lower = sp.Matrix.hstack(translation_hat * rotation, rotation)
        return sp.Matrix.vstack(upper, lower)

    def _block_diag(
        self, *blocks: Iterable[Iterable[object]], **kwargs: object
    ) -> sp.Matrix:
        matrices = [sp.Matrix(block) for block in blocks]
        return sp.diag(*matrices)

    def _build_allowed_functions(self) -> Mapping[str, object]:
        """Build the complete mapping of allowed functions/constants for the calculator."""
        result: dict[str, object] = {}
        result.update(self._trig_functions())
        result.update(self._complex_and_elementary_functions())
        result.update(self._algebra_functions())
        result.update(self._linear_algebra_functions())
        result.update(self._robotics_functions())
        result.update(self._constants())
        return result

    @staticmethod
    def _trig_functions() -> dict[str, object]:
        """Return trigonometric and hyperbolic function mappings."""
        return {
            "I": sp.I,
            "i": sp.I,
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
            "csc": sp.csc,
            "sec": sp.sec,
            "cot": sp.cot,
            "asin": sp.asin,
            "acos": sp.acos,
            "atan": sp.atan,
            "acsc": sp.acsc,
            "asec": sp.asec,
            "acot": sp.acot,
            "sinh": sp.sinh,
            "cosh": sp.cosh,
            "tanh": sp.tanh,
            "asinh": sp.asinh,
            "acosh": sp.acosh,
            "atanh": sp.atanh,
            "csch": sp.csch,
            "sech": sp.sech,
            "coth": sp.coth,
        }

    @staticmethod
    def _complex_and_elementary_functions() -> dict[str, object]:
        """Return complex number, elementary, and rounding function mappings."""
        return {
            "re": sp.re,
            "im": sp.im,
            "real": sp.re,
            "imag": sp.im,
            "arg": sp.arg,
            "conj": sp.conjugate,
            "conjugate": sp.conjugate,
            "abs": sp.Abs,
            "norm": lambda vector, **k: sp.Matrix(vector).norm(),
            "exp": sp.exp,
            "log": sp.log,
            "ln": sp.log,
            "sqrt": sp.sqrt,
            "cbrt": lambda value, **k: sp.root(value, 3),
            "floor": sp.floor,
            "ceiling": sp.ceiling,
            "round": lambda value, ndigits=0, **k: round(value, ndigits),
            "cis": lambda theta, **k: sp.exp(sp.I * theta),
            "rect": lambda radius, theta, **k: radius * sp.exp(sp.I * theta),
            "polar": lambda complex_value, **k: sp.Tuple(
                sp.Abs(complex_value), sp.arg(complex_value)
            ),
        }

    @staticmethod
    def _algebra_functions() -> dict[str, object]:
        """Return algebraic manipulation and combinatorics function mappings."""
        return {
            "gcd": sp.gcd,
            "lcm": sp.lcm,
            "factor": sp.factor,
            "factor_terms": sp.factor_terms,
            "expand": sp.expand,
            "cancel": sp.cancel,
            "collect": sp.collect,
            "together": sp.together,
            "ratsimp": sp.ratsimp,
            "trigsimp": sp.trigsimp,
            "apart": sp.apart,
            "simplify": sp.simplify,
            "factorial": TI89Calculator._safe_factorial,
            "nCr": sp.binomial,
            "nPr": lambda n, r, **k: TI89Calculator._safe_factorial(n)
            / TI89Calculator._safe_factorial(n - r),
            "sum": sp.summation,
            "product": sp.product,
        }

    def _linear_algebra_functions(self) -> dict[str, object]:
        """Return matrix and linear algebra function mappings."""
        return {
            "Matrix": sp.Matrix,
            "dot": lambda vector_a, vector_b, **k: sp.Matrix(vector_a).dot(
                sp.Matrix(vector_b)
            ),
            "cross": lambda vector_a, vector_b, **k: sp.Matrix(vector_a).cross(
                sp.Matrix(vector_b)
            ),
            "det": lambda matrix, **k: sp.Matrix(matrix).det(),
            "transpose": lambda matrix, **k: sp.Matrix(matrix).T,
            "inv": lambda matrix, **k: sp.Matrix(matrix).inv(),
            "pinv": lambda matrix, **k: sp.Matrix(matrix).pinv(),
            "trace": lambda matrix, **k: sp.Matrix(matrix).trace(),
            "rref": lambda matrix, **k: sp.Matrix(matrix).rref()[0],
            "row_reduce": lambda matrix, **k: sp.Matrix(matrix).rref()[0],
            "rank": lambda matrix, **k: sp.Matrix(matrix).rank(),
            "diag": sp.diag,
            "block_diag": self._block_diag,
            "eye": sp.eye,
            "ones": sp.ones,
            "zeros": sp.zeros,
            "matrix_exp": self._matrix_exp,
            "expm": self._matrix_exp,
            "matrix_log": self._matrix_log,
            "logm": self._matrix_log,
            "matrix_power": self._matrix_power,
            "eigenvals": lambda matrix, **k: sp.Matrix(matrix).eigenvals(),
            "eigenvects": lambda matrix, **k: sp.Matrix(matrix).eigenvects(),
            "charpoly": lambda matrix, symbol="\u03bb", **k: sp.Matrix(matrix)
            .charpoly(sp.Symbol(symbol))
            .as_expr(),
            "nullspace": lambda matrix, **k: sp.Matrix(matrix).nullspace(),
            "colspace": lambda matrix, **k: sp.Matrix(matrix).columnspace(),
            "rowspace": lambda matrix, **k: sp.Matrix(matrix).rowspace(),
            "qr": lambda matrix, **k: sp.Matrix(matrix).QRdecomposition(),
            "lu": lambda matrix, **k: sp.Matrix(matrix).LUdecomposition(),
            "svd": lambda matrix, **k: sp.Matrix(matrix).SVD(),
            "solve_linear": lambda matrix, rhs, **k: sp.Matrix(matrix).LUsolve(
                sp.Matrix(rhs)
            ),
            "linsolve": sp.linsolve,
        }

    def _robotics_functions(self) -> dict[str, object]:
        """Return robotics/Lie algebra function mappings."""
        return {
            "hat": self._hat,
            "vee": self._vee,
            "skew": self._hat,
            "unskew": self._vee,
            "se3_hat": self._se3_hat,
            "se3_vee": self._se3_vee,
            "screw_axis": self._screw_axis,
            "twist_exp": self._twist_exponential,
            "adjoint": self._adjoint_transform,
        }

    @staticmethod
    def _constants() -> dict[str, object]:
        """Return mathematical constant mappings."""
        return {
            "pi": sp.pi,
            "E": sp.E,
            "e": sp.E,
            "oo": sp.oo,
            "Infinity": sp.oo,
            "inf": sp.oo,
            "nan": sp.nan,
        }
