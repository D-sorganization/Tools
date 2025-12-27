import os
import sys

import sympy as sp

# Add the parent directory to sys.path to import calculator
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from calculator import TI89Calculator


def test_evaluate_with_trigonometric_identity() -> None:
    calculator = TI89Calculator()
    result = calculator.evaluate("sin(x)^2 + cos(x)^2", {"x": sp.pi / 4})
    # Cast or ignore if result.result is typed as object/Any but known to be compatible
    assert sp.simplify(result.result - 1) == 0  # type: ignore[operator]


def test_e_constant_is_available() -> None:
    calculator = TI89Calculator()
    result = calculator.evaluate("e^x", {"x": 1}).result
    assert result == sp.E


def test_solve_quadratic_equation() -> None:
    calculator = TI89Calculator()
    solutions = calculator.solve_equation("x^2 - 5*x + 6 = 0", "x").result
    # Use explicit set creation with cast if needed or type ignore
    assert {sp.Integer(2), sp.Integer(3)} == set(solutions)  # type: ignore[call-overload]


def test_solve_linear_system() -> None:
    calculator = TI89Calculator()
    res = calculator.solve_system(["x + y = 5", "x - y = 1"], ["x", "y"]).result
    # Ensure res is a list and access first element
    assert isinstance(res, list), "Result should be a list"
    assert len(res) > 0, "Result list should not be empty"
    solution = res[0]
    assert solution == {sp.Symbol("x"): 3, sp.Symbol("y"): 2}


def test_symbolic_derivative_and_integral() -> None:
    calculator = TI89Calculator()
    derivative = calculator.derivative("sin(x) * exp(x)", "x").result
    assert (
        sp.simplify(
            derivative
            - sp.exp(sp.Symbol("x")) * (sp.sin(sp.Symbol("x")) + sp.cos(sp.Symbol("x")))
        )
        == 0
    )

    integral = calculator.integral("exp(-x)", "x").result
    assert sp.simplify(integral + sp.exp(-sp.Symbol("x"))) == 0


def test_definite_integral_and_limit() -> None:
    calculator = TI89Calculator()
    integral = calculator.integral("x", "x", 0, 3).result
    assert integral == sp.Rational(9, 2)

    limit_result = calculator.limit("sin(x)/x", "x", 0, direction="right").result
    assert limit_result == 1


def test_taylor_series_and_differential_equation_solution() -> None:
    calculator = TI89Calculator()
    taylor = calculator.taylor_series("sin(x)", "x", 0, order=5).result
    assert (
        taylor == sp.Symbol("x") - sp.Symbol("x") ** 3 / 6 + sp.Symbol("x") ** 5 / 120
    )

    ode_solution = calculator.solve_differential_equation(
        "Derivative(f(x), x) + f(x)", "f"
    ).result
    # Check if ode_solution has rhs attribute
    assert hasattr(ode_solution, "rhs"), "ODE solution should have 'rhs' attribute"
    assert (
        sp.simplify(ode_solution.rhs - sp.exp(-sp.Symbol("x")) * sp.Symbol("C1")) == 0
    )


def test_complex_and_matrix_support() -> None:
    calculator = TI89Calculator()
    complex_result = calculator.evaluate("abs(3 + 4*i)").result
    assert complex_result == 5

    arg_result = calculator.evaluate("arg(-1)").result
    assert arg_result == sp.pi

    determinant = calculator.evaluate("det([[1, 2], [3, 4]])").result
    assert determinant == -2

    inverse = calculator.evaluate("inv([[1, 2], [3, 4]])").result
    assert sp.Matrix([[1, 2], [3, 4]]).inv() == inverse


def test_symbolic_algebra_utilities() -> None:
    calculator = TI89Calculator()
    expanded = calculator.evaluate("expand((x + 1)^3)").result
    assert expanded == sp.expand((sp.Symbol("x") + 1) ** 3)

    simplified_rational = calculator.evaluate("ratsimp((x^2 - 1)/(x - 1))").result
    assert simplified_rational == sp.Symbol("x") + 1

    trig_simplified = calculator.evaluate("trigsimp(sin(x)^2 + cos(x)^2)").result
    assert trig_simplified == 1


def test_vector_and_matrix_utilities() -> None:
    calculator = TI89Calculator()
    rref_result = calculator.evaluate("rref([[1, 2], [2, 4]])").result
    assert rref_result == sp.Matrix([[1, 2], [0, 0]])

    dot_result = calculator.evaluate("dot([1, 3, -1], [2, 0, 4])").result
    assert dot_result == -2

    cross_result = calculator.evaluate("cross([1, 0, 0], [0, 1, 0])").result
    assert cross_result == sp.Matrix([0, 0, 1])

    trace_result = calculator.evaluate("trace([[1, 2], [3, 4]])").result
    assert trace_result == 5


def test_complex_coordinate_helpers_and_roots() -> None:
    calculator = TI89Calculator()
    cis_result = calculator.evaluate("cis(pi/2)").result
    assert cis_result == sp.I

    polar_result = calculator.evaluate("polar(3 + 4*i)").result
    assert polar_result == sp.Tuple(5, sp.atan(sp.Rational(4, 3)))

    rect_result = calculator.evaluate("rect(5, pi/2)").result
    assert sp.simplify(rect_result - 5 * sp.I) == 0

    cube_root = calculator.evaluate("cbrt(27)").result
    assert cube_root == 3


def test_summation_and_products() -> None:
    calculator = TI89Calculator()
    summation_result = calculator.evaluate("sum(k, (k, 1, 5))").result
    assert summation_result == 15

    product_result = calculator.evaluate("product(k, (k, 1, 4))").result
    assert product_result == 24


def test_equation_simplification_balances_terms() -> None:
    calculator = TI89Calculator()
    simplified = calculator.simplify_expression("2*x + 4 = x + 10").result
    assert simplified == sp.Eq(sp.Symbol("x") - 6, 0)


def test_eigentools_and_linear_algebra_extensions() -> None:
    calculator = TI89Calculator()

    eigenvalues = calculator.evaluate("eigenvals([[2, 1], [1, 2]])").result
    assert eigenvalues == {sp.Integer(3): 1, sp.Integer(1): 1}

    nullspace = calculator.evaluate("nullspace([[1, 2], [2, 4]])").result
    assert nullspace == [sp.Matrix([[-2], [1]])]

    characteristic = calculator.evaluate("charpoly([[2, 0], [0, 3]])").result
    lam = sp.Symbol("λ")
    assert characteristic == lam**2 - 5 * lam + 6

    qr_decomposition = calculator.evaluate("qr([[1, 0], [0, 1]])").result
    assert qr_decomposition == (sp.eye(2), sp.eye(2))

    linear_solution = calculator.evaluate(
        "solve_linear([[2, 0], [0, 3]], [4, 6])"
    ).result
    assert linear_solution == sp.Matrix([2, 2])

    linear_set = calculator.evaluate(
        "linsolve((Matrix([[2, 1], [1, 3]]), Matrix([5, 7])))"
    ).result
    assert linear_set == sp.FiniteSet((sp.Rational(8, 5), sp.Rational(9, 5)))


def test_matrix_exponential_and_logarithm() -> None:
    calculator = TI89Calculator()

    generator = sp.Matrix([[0, -sp.pi / 2], [sp.pi / 2, 0]])
    exp_result = calculator.evaluate("matrix_exp([[0, -pi/2], [pi/2, 0]])").result
    assert exp_result == generator.exp()

    matrix = sp.Matrix([[0, -1], [1, 0]])
    log_result = calculator.evaluate("matrix_log([[0, -1], [1, 0]])").result
    assert sp.simplify(matrix.log() - log_result) == sp.zeros(2, 2)


def test_screw_axis_and_twist_exponential() -> None:
    calculator = TI89Calculator()

    screw = calculator.evaluate("screw_axis([0, 0, 1], [1, 0, 0])").result
    expected = sp.Matrix([0, 0, 1, 0, -1, 0])
    assert screw == expected

    twist_transform = calculator.evaluate("twist_exp([0, 0, 1, 0, 0, 0], pi/2)").result
    expected_transform = sp.Matrix(
        [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )
    assert twist_transform == expected_transform

    adjoint = calculator.evaluate(
        "adjoint([[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]])"
    ).result
    expected_adjoint = sp.Matrix(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, -3, 2, 1, 0, 0],
            [3, 0, -1, 0, 1, 0],
            [-2, 1, 0, 0, 0, 1],
        ]
    )
    assert adjoint == expected_adjoint
