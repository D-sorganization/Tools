import sys
import time

import pytest
import sympy as sp

from web_applications.calculator.calculator import TI89Calculator


class TestDeepRecursion:
    def test_deep_expression_validation(self) -> None:
        """
        Test that _validate_expression_tree handles deep expressions without RecursionError.
        """
        # Create a deep expression manually to bypass parse_expr recursion limits
        # if we were to parse a string.
        x = sp.Symbol("x")
        deep_expr = x
        # Python's default recursion limit is usually 1000.
        # We go deeper to ensure we rely on iteration (if implemented) or crash (if recursive).
        depth = 2000

        original_limit = sys.getrecursionlimit()

        # We temporarily increase recursion limit to build the expression
        sys.setrecursionlimit(max(original_limit, depth + 1000))
        try:
            for _ in range(depth):
                # Use evaluate=False to avoid recursive property checks during construction
                deep_expr = sp.sin(deep_expr, evaluate=False)

            # Now test validation
            # We enforce a standard limit (1000) to simulate standard environment constraint
            sys.setrecursionlimit(1000)

            start_time = time.time()
            try:
                TI89Calculator._validate_expression_tree(deep_expr)
            except RecursionError:
                pytest.fail(
                    "RecursionError raised during validation of deep expression"
                )
            except Exception as e:
                pytest.fail(f"Validation failed with error: {e}")

            duration = time.time() - start_time
            # print(f"\nValidation of depth {depth} took {duration:.4f}s")

        finally:
            # Restore original limit
            sys.setrecursionlimit(original_limit)

    def test_container_handling(self) -> None:
        """Test that validation handles nested containers correctly."""
        expr = {
            "a": [1, sp.Pow(sp.Symbol("x"), 2, evaluate=False)],
            "b": (sp.Symbol("y"),),
        }
        # Should not raise
        TI89Calculator._validate_expression_tree(expr)

    def test_pow_check(self) -> None:
        """Test that unsafe powers are still caught."""
        # massive power - must be unevaluated to be a Pow object
        expr = sp.Pow(10, 10000, evaluate=False)
        with pytest.raises(ValueError, match="exceeds safety limits"):
            TI89Calculator._validate_expression_tree(expr)
