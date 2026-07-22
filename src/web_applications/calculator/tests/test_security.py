"""Security tests for calculator web application."""

import unittest

import pytest
import sympy as sp

from web_applications.calculator.calculator import TI89Calculator
from web_applications.calculator.webapp import create_app


class TestSecurity(unittest.TestCase):
    def setUp(self) -> None:
        self.app = create_app()
        self.app.config.update({"TESTING": True})
        self.client = self.app.test_client()

    def test_input_too_large(self) -> None:
        # Create a very large expression (> 1000 characters)
        large_expression = "1+" * 1000 + "1"
        payload = {"operation": "evaluate", "expression": large_expression}
        response = self.client.post("/api/calculate", json=payload)

        if response.status_code == 200:
            self.fail("VULNERABILITY CONFIRMED: Large input accepted")

        self.assertEqual(
            response.status_code, 400, "Should reject excessively large input"
        )

    def test_security_headers(self) -> None:
        """Test that security headers are present in responses."""
        response = self.client.get("/")

        # HSTS
        self.assertIn("Strict-Transport-Security", response.headers)
        self.assertIn("max-age=31536000", response.headers["Strict-Transport-Security"])

        # CSP
        self.assertIn("Content-Security-Policy", response.headers)
        self.assertIn("default-src 'self'", response.headers["Content-Security-Policy"])

        # X-Content-Type-Options
        self.assertIn("X-Content-Type-Options", response.headers)
        self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")

        # X-Frame-Options
        self.assertIn("X-Frame-Options", response.headers)
        self.assertEqual(response.headers["X-Frame-Options"], "DENY")


class TestAstSecurityGate:
    """Structural AST allowlist gate for parse_expr (issue #3293).

    The previous boundary was a substring blocklist; these tests prove the new
    structural gate rejects code-execution surface (attribute access, lambdas,
    comprehensions, walrus) before the expression reaches sympy.parse_expr,
    while still accepting legitimate mathematical input.
    """

    @pytest.fixture
    def symbols(self) -> dict[str, sp.Symbol]:
        return {"x": sp.Symbol("x"), "y": sp.Symbol("y")}

    @pytest.mark.parametrize(
        "expr",
        [
            "2*x + 3",
            "sin(x) + cos(y)",
            "x^2 + 1",
            "sqrt(x)",
            "(x + 1)*(x - 1)",
            "2 ** 8",
        ],
    )
    def test_legitimate_expressions_pass(
        self, expr: str, symbols: dict[str, sp.Symbol]
    ) -> None:
        result = TI89Calculator.parse_expression(expr, symbols)
        assert result is not None

    @pytest.mark.parametrize(
        "expr",
        [
            "().__class__",
            "().__class__.__bases__",
            "x.__class__",
            "lambda: 1",
            "(1).__class__.__mro__",
            "[i for i in range(3)]",
            "(y := 5)",
        ],
    )
    def test_dangerous_constructs_blocked(
        self, expr: str, symbols: dict[str, sp.Symbol]
    ) -> None:
        with pytest.raises((ValueError, TypeError, SyntaxError)):
            TI89Calculator.parse_expression(expr, symbols)

    def test_gate_rejects_attribute_access_directly(self) -> None:
        with pytest.raises(ValueError):
            TI89Calculator._ast_security_gate("foo.bar")

    def test_gate_rejects_lambda_directly(self) -> None:
        with pytest.raises(ValueError):
            TI89Calculator._ast_security_gate("lambda x: x")

    def test_gate_rejects_oversized_string_constant(self) -> None:
        with pytest.raises(ValueError, match="String constant"):
            TI89Calculator._ast_security_gate('"' + "a" * 1000 + '"')

    def test_gate_allows_plain_math(self) -> None:
        # Should not raise.
        TI89Calculator._ast_security_gate("2 * x + sin(y)")


    def test_solve_ode_blocks_dangerous_constructs(self) -> None:
        with pytest.raises((ValueError, TypeError, SyntaxError)):
            TI89Calculator().solve_differential_equation("().__class__.__bases__[0].__subclasses__()", "y")

if __name__ == "__main__":
    unittest.main()
