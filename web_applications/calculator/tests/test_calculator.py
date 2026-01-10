import unittest

from web_applications.calculator.core.calculator import Calculator


class TestCalculator(unittest.TestCase):
    """Test suite for the Calculator class."""

    def setUp(self) -> None:
        """Set up the calculator instance for testing."""
        self.calculator = Calculator()

    def test_evaluate_basic(self) -> None:
        """Test basic arithmetic evaluations."""
        self.assertEqual(self.calculator.evaluate("1 + 1"), 2)
        self.assertEqual(self.calculator.evaluate("2 * 3"), 6)

    def test_evaluate_variables(self) -> None:
        """Test evaluation with variable substitution."""
        self.assertEqual(self.calculator.evaluate("x + 1", {"x": 2}), 3)

    def test_simplify(self) -> None:
        """Test algebraic simplification."""
        self.assertEqual(str(self.calculator.simplify("2 * x + 3 * x")), "5*x")

    def test_solve_equation(self) -> None:
        """Test solving equations."""
        solutions = self.calculator.solve("x^2 - 1")
        self.assertTrue(-1 in solutions)
        self.assertTrue(1 in solutions)

    def test_derivative(self) -> None:
        """Test symbolic differentiation."""
        self.assertEqual(str(self.calculator.derivative("x^2", "x")), "2*x")

    def test_integral_indefinite(self) -> None:
        """Test indefinite integration."""
        self.assertEqual(str(self.calculator.integrate("2*x", "x")), "x**2")

    def test_integral_definite(self) -> None:
        """Test definite integration."""
        self.assertEqual(self.calculator.integrate("2*x", "x", 0, 1), 1)

    def test_limit(self) -> None:
        """Test limit calculation."""
        self.assertEqual(self.calculator.limit("sin(x)/x", "x", 0), 1)

    def test_taylor_series(self) -> None:
        """Test Taylor series expansion."""
        series = str(self.calculator.series("exp(x)", "x", 0, 3))
        self.assertTrue("x**2/2" in series)
