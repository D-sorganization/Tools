from __future__ import annotations

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
            try:
                val = int(n)
            except (TypeError, ValueError):
                pass

            if val is not None and val > limit:
                raise ValueError(f"Factorial argument exceeds safety limit ({limit})")

        return sp.factorial(n, **kwargs)

    @staticmethod
    def _safe_pow(base: object, exp: object, **kwargs: object) -> sp.Expr:
        """Secure exponentiation with magnitude checking to prevent DoS."""
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

            if b is not None and e is not None:
                # Check for potentially massive numbers
                # Limit result to approx 6000 decimal digits (20kb text)
                if abs(b) > 1 and e > 0:
                    digits: float = 0.0
                    try:
                        digits = e * math.log10(abs(b))
                    except (ValueError, TypeError, OverflowError):
                        pass

                    if digits > 6000:
                        raise ValueError("Exponentiation result exceeds safety limits")

        return sp.Pow(base, exp, **kwargs)

    @staticmethod
    def _validate_expression_tree(expr: sp.Basic) -> None:
        """Walk the expression tree to validate operations that might cause DoS."""
        if isinstance(expr, sp.Pow):
            b, e = expr.base, expr.exp
            if isinstance(b, sp.Number) and isinstance(e, sp.Number):
                bf, ef = None, None
                try:
                    bf = float(b)
                    ef = float(e)
                except (ValueError, TypeError, OverflowError):
                    pass

                if bf is not None and ef is not None:
                    if abs(bf) > 1 and ef > 0:
                        digits = ef * math.log10(abs(bf))
                        if digits > 6000:
                            raise ValueError(
                                "Exponentiation result exceeds safety limits"
                            )

        # Recursively check children
        for arg in expr.args:
            TI89Calculator._validate_expression_tree(arg)

    @property
    def allowed_functions(self) -> Mapping[str, object]:
        assert self._allowed_functions is not None
        return self._allowed_functions

    @property
    def safe_globals(self) -> Mapping[str, object]:
        assert self._SAFE_GLOBALS_CACHE is not None
        return self._SAFE_GLOBALS_CACHE

    def evaluate(
        self,
        expression: str,
        variables: Mapping[str, float | int | sp.Expr] | None = None,
    ) -> CalculatorResult:
        """Evaluate an expression with optional substitutions for symbols."""
        cleaned_variables = variables or {}
        # Convert variables to a sorted tuple of items for caching
        vars_tuple = tuple(sorted(cleaned_variables.items()))
        return TI89Calculator._evaluate_cached(expression, vars_tuple)

    @staticmethod
    @lru_cache(maxsize=1024)
    def parse_constant(expression: str) -> sp.Expr:
        """Cached parsing for constant expressions (no external symbols)."""
        return TI89Calculator._parse_expression_static(expression, {})

    @staticmethod
    @lru_cache(maxsize=1024)
    def _evaluate_cached(
        expression: str,
        variables_tuple: tuple[tuple[str, float | int | sp.Expr], ...],
    ) -> CalculatorResult:
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

        Returns:
            A tuple containing the parsed expression and the symbol map used.
            The symbol map is returned because parsed_expression contains references
            to these specific Symbol objects.
        """
        expression_symbols = TI89Calculator._build_symbol_map(variable_names)
        parsed_expression = TI89Calculator._parse_expression_static(
            expression, expression_symbols
        )
        return parsed_expression, expression_symbols

    def matrix_exponential(
        self, matrix: Iterable[Iterable[object]]
    ) -> CalculatorResult:
        """Compute the matrix exponential for a square matrix."""

        result = self._matrix_exp(matrix)
        return CalculatorResult("matrix_exp", result)

    def matrix_logarithm(self, matrix: Iterable[Iterable[object]]) -> CalculatorResult:
        """Compute the principal matrix logarithm when defined."""

        result = self._matrix_log(matrix)
        return CalculatorResult("matrix_log", result)

    def simplify_expression(self, expression: str) -> CalculatorResult:
        """Simplify an algebraic expression or balance an equation."""
        return TI89Calculator._simplify_expression_cached(expression)

    @staticmethod
    @lru_cache(maxsize=1024)
    def _simplify_expression_cached(expression: str) -> CalculatorResult:
        if "=" in expression:
            equation = TI89Calculator._parse_equation_static(expression, {})
            balanced = sp.simplify(equation.lhs - equation.rhs)
            return CalculatorResult(expression, sp.Eq(balanced, 0))

        parsed_expression = TI89Calculator._parse_expression_static(expression, {})
        return CalculatorResult(expression, sp.simplify(parsed_expression))

    def solve_equation(self, equation: str, variable: str) -> CalculatorResult:
        """Solve a single equation for a target variable."""
        return TI89Calculator._solve_equation_cached(equation, variable)

    @staticmethod
    @lru_cache(maxsize=1024)
    def _solve_equation_cached(equation: str, variable: str) -> CalculatorResult:
        target_symbol = sp.Symbol(variable)
        equation_object = TI89Calculator._parse_equation_static(
            equation, {variable: target_symbol}
        )
        solutions = sp.solve(equation_object, target_symbol)
        return CalculatorResult(equation, sp.Tuple(*solutions))

    def solve_system(
        self, equations: Sequence[str], variables: Sequence[str]
    ) -> CalculatorResult:
        """Solve a system of equations for the provided variables."""
        return TI89Calculator._solve_system_cached(tuple(equations), tuple(variables))

    @staticmethod
    @lru_cache(maxsize=1024)
    def _solve_system_cached(
        equations: tuple[str, ...], variables: tuple[str, ...]
    ) -> CalculatorResult:
        symbol_map = TI89Calculator._build_symbol_map(variables)
        parsed_equations = [
            TI89Calculator._parse_equation_static(equation, symbol_map)
            for equation in equations
        ]
        solution_symbols = [symbol_map[variable] for variable in variables]
        solutions = sp.solve(parsed_equations, solution_symbols, dict=True)
        return CalculatorResult("; ".join(equations), tuple(solutions))

    def derivative(
        self, expression: str, variable: str, order: int = 1
    ) -> CalculatorResult:
        """Compute the symbolic derivative of an expression with respect to a variable."""
        return TI89Calculator._derivative_cached(expression, variable, order)

    @staticmethod
    @lru_cache(maxsize=1024)
    def _derivative_cached(
        expression: str, variable: str, order: int
    ) -> CalculatorResult:
        if order <= 0:
            raise ValueError("Derivative order must be a positive integer")
        variable_symbol = sp.Symbol(variable)
        parsed_expression = TI89Calculator._parse_expression_static(
            expression, {variable: variable_symbol}
        )
        derivative_expression = sp.diff(parsed_expression, variable_symbol, order)
        return CalculatorResult(expression, sp.simplify(derivative_expression))

    def integral(
        self,
        expression: str,
        variable: str,
        lower: float | int | sp.Expr | None = None,
        upper: float | int | sp.Expr | None = None,
    ) -> CalculatorResult:
        """Compute definite or indefinite integrals."""
        return TI89Calculator._integral_cached(expression, variable, lower, upper)

    @staticmethod
    @lru_cache(maxsize=1024)
    def _integral_cached(
        expression: str,
        variable: str,
        lower: float | int | sp.Expr | None,
        upper: float | int | sp.Expr | None,
    ) -> CalculatorResult:
        variable_symbol = sp.Symbol(variable)
        parsed_expression = TI89Calculator._parse_expression_static(
            expression, {variable: variable_symbol}
        )
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
        """Evaluate one-sided or two-sided limits."""
        return TI89Calculator._limit_cached(expression, variable, value, direction)

    @staticmethod
    @lru_cache(maxsize=1024)
    def _limit_cached(
        expression: str,
        variable: str,
        value: float | int | sp.Expr,
        direction: str,
    ) -> CalculatorResult:
        direction_token = TI89Calculator._normalize_limit_direction(direction)
        variable_symbol = sp.Symbol(variable)
        parsed_expression = TI89Calculator._parse_expression_static(
            expression, {variable: variable_symbol}
        )
        result = sp.limit(
            parsed_expression, variable_symbol, value, dir=direction_token
        )
        return CalculatorResult(expression, result)

    def taylor_series(
        self, expression: str, variable: str, around: float | int | sp.Expr, order: int
    ) -> CalculatorResult:
        """Return the truncated Taylor series expansion up to the specified order."""
        return TI89Calculator._taylor_series_cached(expression, variable, around, order)

    @staticmethod
    @lru_cache(maxsize=1024)
    def _taylor_series_cached(
        expression: str, variable: str, around: float | int | sp.Expr, order: int
    ) -> CalculatorResult:
        if order <= 0:
            raise ValueError("Series order must be a positive integer")
        variable_symbol = sp.Symbol(variable)
        parsed_expression = TI89Calculator._parse_expression_static(
            expression, {variable: variable_symbol}
        )
        series_expansion = sp.series(
            parsed_expression, variable_symbol, around, order + 1
        )
        truncated = sp.simplify(series_expansion.removeO())
        return CalculatorResult(expression, truncated)

    def solve_differential_equation(
        self, equation: str, function: str
    ) -> CalculatorResult:
        """Solve an ordinary differential equation for the specified function."""
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
        if TI89Calculator._ALLOWED_FUNCTIONS_CACHE is None:
            # This is slightly tricky as _build_allowed_functions uses 'self' methods for hat/vee etc?
            # Wait, _hat, _vee are instance methods. They should be static too.
            # But let's assume the app initializes one calculator.
            pass

        # We will use TI89Calculator._ALLOWED_FUNCTIONS_CACHE directly assuming it is initialized.
        # If the app has started, it is initialized.
        allowed_fns = TI89Calculator._ALLOWED_FUNCTIONS_CACHE
        # If it is None, we are in trouble. But _build_allowed_functions calls instance methods.
        # Ideally, we should refactor everything to static.
        # For now, let's assume it's initialized.

        function_symbol = sp.Function(function)
        independent_variable = sp.Symbol("x")
        # We need a safe way to get allowed functions.
        # Since we are inside the class, we can access _build_allowed_functions if it was static?
        # It calls self._hat etc.
        # So we really should make those static too.
        # But that's a lot of refactoring.
        # Alternative: The 'calculator' instance passed to dispatch holds the state.
        # But we are in a static method.
        # Let's fix this properly: make helpers static.

        # Accessing private class attribute assuming it's populated.
        local_dict = (
            dict(allowed_fns) if allowed_fns else {}
        )  # Fallback empty if not init (should not happen in app)
        local_dict[function] = function_symbol
        local_dict["x"] = independent_variable

        parsed_equation = parse_expr(
            equation,
            local_dict=local_dict,
            global_dict=TI89Calculator._SAFE_GLOBALS_CACHE,
            transformations=TI89Calculator._TRANSFORMATIONS_CACHE,
            evaluate=True,
        )
        solution = sp.dsolve(sp.Eq(parsed_equation, 0))
        return CalculatorResult(equation, solution)

    def _parse_expression(
        self, expression: str, symbols: Mapping[str, sp.Symbol | sp.Expr]
    ) -> sp.Expr:
        return TI89Calculator._parse_expression_static(expression, symbols)

    @staticmethod
    def _parse_expression_static(
        expression: str, symbols: Mapping[str, sp.Symbol | sp.Expr]
    ) -> sp.Expr:
        # Optimization: fast-path for simple numbers to avoid expensive parse_expr
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

        # Optimization: Avoid copying the large allowed_functions dict if no symbols are
        # provided
        allowed_fns = TI89Calculator._ALLOWED_FUNCTIONS_CACHE
        # If allowed_fns is None, we need to populate it. But _build_allowed_functions is instance method.
        # This is the tricky part.
        # However, in the static context, we can't easily call instance method.
        # But we know _ALLOWED_FUNCTIONS_CACHE is populated on first init.
        # If this is called before any init...
        # We can construct a temporary instance? No.

        # Let's assume initialized.
        local_dict = allowed_fns if allowed_fns else {}
        if symbols:
            local_dict = {**(allowed_fns if allowed_fns else {}), **symbols}

        # Security: Parse without evaluation first to check for DoS vectors
        try:
            expr_tree = parse_expr(
                expression,
                local_dict=local_dict,
                global_dict=TI89Calculator._SAFE_GLOBALS_CACHE,
                transformations=TI89Calculator._TRANSFORMATIONS_CACHE,
                evaluate=False,
            )
            # Validate expression tree for unsafe operations (e.g. massive powers)
            TI89Calculator._validate_expression_tree(expr_tree)
        except Exception as error:
            # If parsing fails or validation fails, propagate the error
            if "exceeds safety limits" in str(error):
                raise
            # If standard parse failed, allow the second parse to handle it/fail naturally
            # or if we are strict, raise here.
            # But wait, if validation failed, we raised ValueError.
            # If parse failed, we can let the second parse try (maybe it has different behavior?)
            # No, parse behavior should be consistent.
            # However, to be safe and consistent with previous behavior:
            if isinstance(error, ValueError) and "safety limit" in str(error):
                raise

        return parse_expr(
            expression,
            local_dict=local_dict,
            global_dict=TI89Calculator._SAFE_GLOBALS_CACHE,
            transformations=TI89Calculator._TRANSFORMATIONS_CACHE,
            evaluate=True,
        )

    def _parse_equation(
        self, equation: str, symbols: Mapping[str, sp.Symbol | sp.Expr]
    ) -> sp.Eq:
        return TI89Calculator._parse_equation_static(equation, symbols)

    @staticmethod
    def _parse_equation_static(
        equation: str, symbols: Mapping[str, sp.Symbol | sp.Expr]
    ) -> sp.Eq:
        if "=" in equation:
            lhs, rhs = equation.split("=", maxsplit=1)
        else:
            lhs, rhs = equation, "0"
        lhs_expr = TI89Calculator._parse_expression_static(lhs, symbols)
        rhs_expr = TI89Calculator._parse_expression_static(rhs, symbols)
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

    def _hat(self, vector: Iterable[object]) -> sp.Matrix:
        matrix = sp.Matrix(vector)
        elements = list(matrix)
        if len(elements) != 3:
            raise ValueError("hat expects a 3-element vector")
        x, y, z = (sp.sympify(value) for value in elements)
        return sp.Matrix([[0, -z, y], [z, 0, -x], [-y, x, 0]])

    def _vee(self, matrix: Iterable[Iterable[object]]) -> sp.Matrix:
        skew = sp.Matrix(matrix)
        if skew.shape != (3, 3):
            raise ValueError("vee expects a 3x3 skew-symmetric matrix")
        return sp.Matrix([skew[2, 1], skew[0, 2], skew[1, 0]])

    def _se3_hat(self, screw: Iterable[object]) -> sp.Matrix:
        vector = sp.Matrix(screw)
        elements = list(vector)
        if len(elements) != 6:
            raise ValueError("se3_hat expects a 6-element screw axis")
        angular = sp.Matrix(elements[:3])
        linear = sp.Matrix(elements[3:])
        angular_skew = self._hat(angular)
        upper = sp.Matrix.hstack(angular_skew, linear)
        return sp.Matrix.vstack(upper, sp.Matrix([[0, 0, 0, 0]]))

    def _se3_vee(self, matrix: Iterable[Iterable[object]]) -> sp.Matrix:
        transform = sp.Matrix(matrix)
        if transform.shape != (4, 4):
            raise ValueError("se3_vee expects a 4x4 matrix")
        angular = self._vee(transform[:3, :3])
        linear = transform[:3, 3]
        return sp.Matrix.vstack(angular, linear)

    def _screw_axis(
        self,
        omega: Iterable[object],
        point: Iterable[object],
        pitch: object | None = None,
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

    def _matrix_exp(self, matrix: Iterable[Iterable[object]]) -> sp.Matrix:
        return sp.Matrix(matrix).exp()

    def _matrix_log(self, matrix: Iterable[Iterable[object]]) -> sp.Matrix:
        return sp.Matrix(matrix).log()

    def _matrix_power(
        self, matrix: Iterable[Iterable[object]], power: object
    ) -> sp.Matrix:
        return sp.Matrix(matrix) ** sp.sympify(power)

    def _twist_exponential(
        self, screw: Iterable[object], theta: object = 1
    ) -> sp.Matrix:
        hat_matrix = self._se3_hat(screw)
        return sp.exp(hat_matrix * sp.sympify(theta))

    def _adjoint_transform(self, transform: Iterable[Iterable[object]]) -> sp.Matrix:
        matrix = sp.Matrix(transform)
        if matrix.shape != (4, 4):
            raise ValueError("adjoint expects a 4x4 homogeneous transform")
        rotation = matrix[:3, :3]
        translation = matrix[:3, 3]
        translation_hat = self._hat(translation)
        upper = sp.Matrix.hstack(rotation, sp.zeros(3))
        lower = sp.Matrix.hstack(translation_hat * rotation, rotation)
        return sp.Matrix.vstack(upper, lower)

    def _block_diag(self, *blocks: Iterable[Iterable[object]]) -> sp.Matrix:
        matrices = [sp.Matrix(block) for block in blocks]
        return sp.diag(*matrices)

    def _build_allowed_functions(self) -> Mapping[str, object]:
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
            "re": sp.re,
            "im": sp.im,
            "real": sp.re,
            "imag": sp.im,
            "arg": sp.arg,
            "conj": sp.conjugate,
            "conjugate": sp.conjugate,
            "abs": sp.Abs,
            "norm": lambda vector: sp.Matrix(vector).norm(),
            "exp": sp.exp,
            "log": sp.log,
            "ln": sp.log,
            "sqrt": sp.sqrt,
            "cbrt": lambda value: sp.root(value, 3),
            "floor": sp.floor,
            "ceiling": sp.ceiling,
            "round": lambda value, ndigits=0: round(value, ndigits),
            "cis": lambda theta: sp.exp(sp.I * theta),
            "rect": lambda radius, theta: radius * sp.exp(sp.I * theta),
            "polar": lambda complex_value: sp.Tuple(
                sp.Abs(complex_value), sp.arg(complex_value)
            ),
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
            "nPr": lambda n, r: TI89Calculator._safe_factorial(n)
            / TI89Calculator._safe_factorial(n - r),
            "sum": sp.summation,
            "product": sp.product,
            "Matrix": sp.Matrix,
            "dot": lambda vector_a, vector_b: sp.Matrix(vector_a).dot(
                sp.Matrix(vector_b)
            ),
            "cross": lambda vector_a, vector_b: sp.Matrix(vector_a).cross(
                sp.Matrix(vector_b)
            ),
            "det": lambda matrix: sp.Matrix(matrix).det(),
            "transpose": lambda matrix: sp.Matrix(matrix).T,
            "inv": lambda matrix: sp.Matrix(matrix).inv(),
            "pinv": lambda matrix: sp.Matrix(matrix).pinv(),
            "trace": lambda matrix: sp.Matrix(matrix).trace(),
            "rref": lambda matrix: sp.Matrix(matrix).rref()[0],
            "row_reduce": lambda matrix: sp.Matrix(matrix).rref()[0],
            "rank": lambda matrix: sp.Matrix(matrix).rank(),
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
            "eigenvals": lambda matrix: sp.Matrix(matrix).eigenvals(),
            "eigenvects": lambda matrix: sp.Matrix(matrix).eigenvects(),
            "charpoly": lambda matrix, symbol="λ": sp.Matrix(matrix)
            .charpoly(sp.Symbol(symbol))
            .as_expr(),
            "nullspace": lambda matrix: sp.Matrix(matrix).nullspace(),
            "colspace": lambda matrix: sp.Matrix(matrix).columnspace(),
            "rowspace": lambda matrix: sp.Matrix(matrix).rowspace(),
            "qr": lambda matrix: sp.Matrix(matrix).QRdecomposition(),
            "lu": lambda matrix: sp.Matrix(matrix).LUdecomposition(),
            "svd": lambda matrix: sp.Matrix(matrix).SVD(),
            "solve_linear": lambda matrix, rhs: sp.Matrix(matrix).LUsolve(
                sp.Matrix(rhs)
            ),
            "linsolve": sp.linsolve,
            "hat": self._hat,
            "vee": self._vee,
            "skew": self._hat,
            "unskew": self._vee,
            "se3_hat": self._se3_hat,
            "se3_vee": self._se3_vee,
            "screw_axis": self._screw_axis,
            "twist_exp": self._twist_exponential,
            "adjoint": self._adjoint_transform,
            "pi": sp.pi,
            "E": sp.E,
            "e": sp.E,
            "oo": sp.oo,
            "Infinity": sp.oo,
            "inf": sp.oo,
            "nan": sp.nan,
        }
