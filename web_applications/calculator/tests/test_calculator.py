import unittest
import sympy as sp
import sys
import os

# Add the parent directory to sys.path to import calculator
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from calculator import TI89Calculator, CalculatorResult

class TestTI89Calculator(unittest.TestCase):
    def setUp(self):
        self.calc = TI89Calculator()

    def test_evaluate_basic(self):
        result = self.calc.evaluate("2 + 2")
        self.assertEqual(result.result, 4)

    def test_evaluate_variables(self):
        result = self.calc.evaluate("x + y", variables={"x": 1, "y": 2})
        self.assertEqual(result.result, 3)

    def test_simplify(self):
        result = self.calc.simplify_expression("x + x")
        self.assertEqual(result.result, 2 * sp.Symbol('x'))

    def test_solve_equation(self):
        result = self.calc.solve_equation("x - 5 = 0", "x")
        # solve returns a tuple of solutions
        self.assertTrue(5 in result.result)

    def test_derivative(self):
        result = self.calc.derivative("x**2", "x")
        self.assertEqual(result.result, 2 * sp.Symbol('x'))

    def test_integral_indefinite(self):
        result = self.calc.integral("2*x", "x")
        self.assertEqual(result.result, sp.Symbol('x')**2)

    def test_integral_definite(self):
        result = self.calc.integral("2*x", "x", lower=0, upper=1)
        self.assertEqual(result.result, 1)

    def test_limit(self):
        result = self.calc.limit("1/x", "x", sp.oo)
        self.assertEqual(result.result, 0)

    def test_taylor_series(self):
        # Taylor series of e^x at x=0 order 2 is 1 + x + x^2/2
        result = self.calc.taylor_series("exp(x)", "x", 0, 2)
        x = sp.Symbol('x')
        expected = 1 + x + x**2/2
        self.assertEqual(result.result, expected)

if __name__ == '__main__':
    unittest.main()
