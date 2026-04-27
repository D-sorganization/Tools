"""Tests for the web-based calculator application."""

import unittest

from web_applications.calculator.calculator import TI89Calculator


class TestCalculator(unittest.TestCase):
    """Test suite for the Calculator class."""

    def setUp(self) -> None:
        """Set up the calculator instance for testing."""
        self.calculator = TI89Calculator()

    def test_evaluate_basic(self) -> None:
        """Test basic arithmetic evaluations."""
        self.assertEqual(self.calculator.evaluate("1 + 1").result, 2)
        self.assertEqual(self.calculator.evaluate("2 * 3").result, 6)

    def test_evaluate_variables(self) -> None:
        """Test evaluation with variable substitution."""
        self.assertEqual(self.calculator.evaluate("x + 1", {"x": 2}).result, 3)

    def test_simplify(self) -> None:
        """Test algebraic simplification."""
        self.assertEqual(
            str(self.calculator.simplify_expression("2 * x + 3 * x").result), "5*x"
        )

    def test_solve_equation(self) -> None:
        """Test solving equations."""
        solutions = self.calculator.solve_equation("x^2 - 1", "x").result
        # Suppress operator errors for containment checks involving Sympy objects
        self.assertTrue(-1 in solutions)
        self.assertTrue(1 in solutions)

    def test_derivative(self) -> None:
        """Test symbolic differentiation."""
        self.assertEqual(str(self.calculator.derivative("x^2", "x").result), "2*x")

    def test_integral_indefinite(self) -> None:
        """Test indefinite integration."""
        self.assertEqual(str(self.calculator.integral("2*x", "x").result), "x**2")

    def test_integral_definite(self) -> None:
        """Test definite integration."""
        self.assertEqual(self.calculator.integral("2*x", "x", 0, 1).result, 1)

    def test_limit(self) -> None:
        """Test limit calculation."""
        self.assertEqual(self.calculator.limit("sin(x)/x", "x", 0).result, 1)

    def test_taylor_series(self) -> None:
        """Test Taylor series expansion."""
        series = str(self.calculator.taylor_series("exp(x)", "x", 0, 3).result)
        self.assertTrue("x**2/2" in series)


class TestCalculatorDbCPreconditions(unittest.TestCase):
    """Test suite verifying DbC preconditions raise correct exception types."""

    def setUp(self) -> None:
        """Set up the calculator instance for testing."""
        self.calculator = TI89Calculator()

    def test_evaluate_non_string_expression_raises_type_error(self) -> None:
        """evaluate() raises TypeError when expression is not a string."""
        with self.assertRaises(TypeError):
            self.calculator.evaluate(123)  # type: ignore[arg-type]

    def test_evaluate_empty_expression_raises_value_error(self) -> None:
        """evaluate() raises ValueError when expression is empty."""
        with self.assertRaises(ValueError):
            self.calculator.evaluate("")

    def test_simplify_non_string_raises_type_error(self) -> None:
        """simplify_expression() raises TypeError when expression is not a string."""
        with self.assertRaises(TypeError):
            self.calculator.simplify_expression(None)  # type: ignore[arg-type]

    def test_simplify_empty_expression_raises_value_error(self) -> None:
        """simplify_expression() raises ValueError when expression is empty."""
        with self.assertRaises(ValueError):
            self.calculator.simplify_expression("")

    def test_solve_equation_non_string_equation_raises_type_error(self) -> None:
        """solve_equation() raises TypeError when equation is not a string."""
        with self.assertRaises(TypeError):
            self.calculator.solve_equation(42, "x")  # type: ignore[arg-type]

    def test_solve_equation_non_string_variable_raises_type_error(self) -> None:
        """solve_equation() raises TypeError when variable is not a string."""
        with self.assertRaises(TypeError):
            self.calculator.solve_equation("x^2 - 1", 0)  # type: ignore[arg-type]

    def test_solve_system_non_sequence_raises_type_error(self) -> None:
        """solve_system() raises TypeError when equations is not a sequence."""
        with self.assertRaises(TypeError):
            self.calculator.solve_system("x+y=1", ["x", "y"])  # type: ignore[arg-type]

    def test_derivative_non_string_expression_raises_type_error(self) -> None:
        """derivative() raises TypeError when expression is not a string."""
        with self.assertRaises(TypeError):
            self.calculator.derivative(123, "x")  # type: ignore[arg-type]

    def test_derivative_invalid_order_raises_value_error(self) -> None:
        """derivative() raises ValueError when order is not a positive integer."""
        with self.assertRaises(ValueError):
            self.calculator.derivative("x^2", "x", 0)

    def test_integral_non_string_expression_raises_type_error(self) -> None:
        """integral() raises TypeError when expression is not a string."""
        with self.assertRaises(TypeError):
            self.calculator.integral(None, "x")  # type: ignore[arg-type]

    def test_limit_non_string_expression_raises_type_error(self) -> None:
        """limit() raises TypeError when expression is not a string."""
        with self.assertRaises(TypeError):
            self.calculator.limit(99, "x", 0)  # type: ignore[arg-type]

    def test_taylor_series_non_string_expression_raises_type_error(self) -> None:
        """taylor_series() raises TypeError when expression is not a string."""
        with self.assertRaises(TypeError):
            self.calculator.taylor_series(None, "x", 0, 3)  # type: ignore[arg-type]

    def test_taylor_series_invalid_order_raises_value_error(self) -> None:
        """taylor_series() raises ValueError when order is not a positive integer."""
        with self.assertRaises(ValueError):
            self.calculator.taylor_series("exp(x)", "x", 0, -1)

    def test_solve_differential_equation_non_string_raises_type_error(self) -> None:
        """solve_differential_equation() raises TypeError for non-string equation."""
        with self.assertRaises(TypeError):
            self.calculator.solve_differential_equation(42, "f")  # type: ignore[arg-type]

    def test_matrix_exponential_none_raises_type_error(self) -> None:
        """matrix_exponential() raises TypeError when matrix is None."""
        with self.assertRaises(TypeError):
            self.calculator.matrix_exponential(None)  # type: ignore[arg-type]

    def test_matrix_logarithm_none_raises_type_error(self) -> None:
        """matrix_logarithm() raises TypeError when matrix is None."""
        with self.assertRaises(TypeError):
            self.calculator.matrix_logarithm(None)  # type: ignore[arg-type]

    def test_parse_expression_non_string_raises_type_error(self) -> None:
        """parse_expression() raises TypeError when expression is not a string."""
        with self.assertRaises(TypeError):
            TI89Calculator.parse_expression(123, {})  # type: ignore[arg-type]

    def test_parse_equation_non_string_raises_type_error(self) -> None:
        """parse_equation() raises TypeError when equation is not a string."""
        with self.assertRaises(TypeError):
            TI89Calculator.parse_equation(123, {})  # type: ignore[arg-type]
