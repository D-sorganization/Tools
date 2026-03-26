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
        self.assertEqual(str(self.calculator.simplify_expression("2 * x + 3 * x").result), "5*x")

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
