"""Tests for the Gibbs free energy minimization solver.

Tests verify:
1. ElementMatrix construction and queries
2. Gibbs energy and gradient computation
3. Initial guess generation (NNLS)
4. Solver convergence at various conditions
5. Element balance conservation
6. Design by Contract preconditions
"""

import numpy as np
import pytest

from gasification_equilibrium.python.solver import (
    ElementMatrix,
    SolverResult,
    compute_gibbs,
    compute_gibbs_gradient,
    initial_guess,
    solve_equilibrium,
)
from gasification_equilibrium.python.thermo_data import P_REF, SPECIES_DB


@pytest.fixture
def full_matrix() -> ElementMatrix:
    """ElementMatrix with all species."""
    return ElementMatrix.from_species()


@pytest.fixture
def small_matrix() -> ElementMatrix:
    """ElementMatrix with minimal species for fast tests."""
    return ElementMatrix.from_species(
        ["H2", "CO", "CO2", "H2O", "CH4", "N2", "C_solid"]
    )


class TestElementMatrix:
    """Test ElementMatrix construction and methods."""

    def test_from_species_default(self, full_matrix: ElementMatrix) -> None:
        assert full_matrix.n_species == len(SPECIES_DB)
        assert full_matrix.n_elements > 0

    def test_from_species_custom(self, small_matrix: ElementMatrix) -> None:
        assert small_matrix.n_species == 7
        assert "H2" in small_matrix.species_keys

    def test_matrix_shape(self, full_matrix: ElementMatrix) -> None:
        assert full_matrix.A.shape == (full_matrix.n_elements, full_matrix.n_species)

    def test_matrix_nonnegative(self, full_matrix: ElementMatrix) -> None:
        assert np.all(full_matrix.A >= 0)

    def test_gas_mask(self, small_matrix: ElementMatrix) -> None:
        """Gas mask should be True for gas species, False for solids."""
        for i, key in enumerate(small_matrix.species_keys):
            expected = SPECIES_DB[key]["phase"] == "gas"
            assert small_matrix.gas_mask[i] == expected, f"{key} gas_mask wrong"

    def test_build_balance_vector(self, small_matrix: ElementMatrix) -> None:
        feed = {"C": 1.0, "H": 2.0, "O": 1.0}
        b = small_matrix.build_balance_vector(feed)
        assert b.shape == (small_matrix.n_elements,)
        # Check that C, H, O are correctly placed
        for j, elem in enumerate(small_matrix.element_keys):
            assert b[j] == feed.get(elem, 0.0)

    def test_build_balance_vector_missing_elements(
        self, small_matrix: ElementMatrix
    ) -> None:
        """Elements not in feed should be zero."""
        feed = {"C": 1.0}
        b = small_matrix.build_balance_vector(feed)
        c_idx = small_matrix.element_keys.index("C")
        assert b[c_idx] == 1.0
        for j, elem in enumerate(small_matrix.element_keys):
            if elem != "C":
                assert b[j] == 0.0

    def test_species_index_found(self, small_matrix: ElementMatrix) -> None:
        idx = small_matrix.species_index("H2")
        assert idx >= 0
        assert small_matrix.species_keys[idx] == "H2"

    def test_species_index_not_found(self, small_matrix: ElementMatrix) -> None:
        idx = small_matrix.species_index("NONEXISTENT")
        assert idx == -1

    def test_element_matrix_h2_row(self, small_matrix: ElementMatrix) -> None:
        """H2 should have 2 H atoms and 0 C atoms."""
        h2_idx = small_matrix.species_index("H2")
        h_idx = small_matrix.element_keys.index("H")
        c_idx = small_matrix.element_keys.index("C")
        assert small_matrix.A[h_idx, h2_idx] == 2
        assert small_matrix.A[c_idx, h2_idx] == 0


class TestComputeGibbs:
    """Test Gibbs energy computation."""

    def test_returns_finite(self, small_matrix: ElementMatrix) -> None:
        n = np.ones(small_matrix.n_species) * 0.1
        G = compute_gibbs(n, 1000, P_REF, small_matrix)
        assert np.isfinite(G)

    def test_varies_with_temperature(self, small_matrix: ElementMatrix) -> None:
        n = np.ones(small_matrix.n_species) * 0.1
        G_low = compute_gibbs(n, 500, P_REF, small_matrix)
        G_high = compute_gibbs(n, 1500, P_REF, small_matrix)
        assert G_low != G_high

    def test_varies_with_pressure(self, small_matrix: ElementMatrix) -> None:
        n = np.ones(small_matrix.n_species) * 0.1
        G_low = compute_gibbs(n, 1000, P_REF * 0.5, small_matrix)
        G_high = compute_gibbs(n, 1000, P_REF * 10, small_matrix)
        assert G_low != G_high


class TestComputeGibbsGradient:
    """Test Gibbs gradient computation."""

    def test_gradient_shape(self, small_matrix: ElementMatrix) -> None:
        n = np.ones(small_matrix.n_species) * 0.1
        grad = compute_gibbs_gradient(n, 1000, P_REF, small_matrix)
        assert grad.shape == (small_matrix.n_species,)

    def test_gradient_finite(self, small_matrix: ElementMatrix) -> None:
        n = np.ones(small_matrix.n_species) * 0.1
        grad = compute_gibbs_gradient(n, 1000, P_REF, small_matrix)
        assert np.all(np.isfinite(grad))

    def test_gradient_numerical_consistency(self, small_matrix: ElementMatrix) -> None:
        """Gradient should approximately match finite differences."""
        n = np.ones(small_matrix.n_species) * 0.5
        grad = compute_gibbs_gradient(n, 1000, P_REF, small_matrix)
        eps = 1e-6
        for i in range(small_matrix.n_species):
            n_plus = n.copy()
            n_plus[i] += eps
            G_plus = compute_gibbs(n_plus, 1000, P_REF, small_matrix)
            G_base = compute_gibbs(n, 1000, P_REF, small_matrix)
            num_grad = (G_plus - G_base) / eps
            assert (
                abs(grad[i] - num_grad) < 0.1
            ), f"Gradient mismatch at species {i}: analytical={grad[i]:.4f}, numerical={num_grad:.4f}"


class TestInitialGuess:
    """Test NNLS-based initial guess."""

    def test_shape(self, small_matrix: ElementMatrix) -> None:
        b = np.array([1.0, 2.0, 1.0, 0.0])  # C, H, N, O (depends on element order)
        b = small_matrix.build_balance_vector({"C": 1.0, "H": 2.0, "O": 1.0})
        n0 = initial_guess(small_matrix, b)
        assert n0.shape == (small_matrix.n_species,)

    def test_all_positive(self, small_matrix: ElementMatrix) -> None:
        b = small_matrix.build_balance_vector({"C": 1.0, "H": 2.0, "O": 1.0})
        n0 = initial_guess(small_matrix, b)
        assert np.all(n0 > 0)

    def test_handles_zero_feed(self, small_matrix: ElementMatrix) -> None:
        b = small_matrix.build_balance_vector({})
        n0 = initial_guess(small_matrix, b)
        assert n0.shape == (small_matrix.n_species,)
        assert np.all(n0 > 0)


class TestSolveEquilibrium:
    """Test the full equilibrium solver."""

    def test_converges_1000k(self, small_matrix: ElementMatrix) -> None:
        result = solve_equilibrium(
            T=1000,
            P=P_REF,
            feed_elements={"C": 1.0, "H": 1.0, "O": 1.0},
            matrix=small_matrix,
        )
        assert result.converged, f"balance_error={result.balance_error}"

    def test_converges_500k(self, small_matrix: ElementMatrix) -> None:
        result = solve_equilibrium(
            T=500,
            P=P_REF,
            feed_elements={"C": 1.0, "H": 2.0, "O": 1.0},
            matrix=small_matrix,
        )
        assert result.converged

    def test_converges_1500k(self, small_matrix: ElementMatrix) -> None:
        result = solve_equilibrium(
            T=1500,
            P=P_REF,
            feed_elements={"C": 1.0, "H": 2.0, "O": 0.5},
            matrix=small_matrix,
        )
        assert result.converged

    def test_element_balance_conserved(self, small_matrix: ElementMatrix) -> None:
        feed = {"C": 1.0, "H": 2.0, "O": 1.0}
        result = solve_equilibrium(
            T=1000, P=P_REF, feed_elements=feed, matrix=small_matrix
        )
        assert result.balance_error < 1e-6

    def test_moles_nonnegative(self, small_matrix: ElementMatrix) -> None:
        result = solve_equilibrium(
            T=1000,
            P=P_REF,
            feed_elements={"C": 1.0, "H": 1.0, "O": 0.5},
            matrix=small_matrix,
        )
        assert np.all(result.moles >= 0)

    def test_result_has_fields(self, small_matrix: ElementMatrix) -> None:
        result = solve_equilibrium(
            T=1000,
            P=P_REF,
            feed_elements={"C": 1.0, "H": 1.0, "O": 0.5},
            matrix=small_matrix,
        )
        assert isinstance(result, SolverResult)
        assert isinstance(result.gibbs_energy, float)
        assert isinstance(result.iterations, int)
        assert isinstance(result.balance_error, float)

    def test_warm_start(self, small_matrix: ElementMatrix) -> None:
        r1 = solve_equilibrium(
            T=1000,
            P=P_REF,
            feed_elements={"C": 1.0, "H": 1.0, "O": 0.5},
            matrix=small_matrix,
        )
        r2 = solve_equilibrium(
            T=1010,
            P=P_REF,
            feed_elements={"C": 1.0, "H": 1.0, "O": 0.5},
            matrix=small_matrix,
            warm_start=r1.moles,
        )
        assert r2.converged

    def test_negative_temperature_raises(self, small_matrix: ElementMatrix) -> None:
        with pytest.raises(AssertionError):
            solve_equilibrium(
                T=-100, P=P_REF, feed_elements={"C": 1.0}, matrix=small_matrix
            )

    def test_zero_temperature_raises(self, small_matrix: ElementMatrix) -> None:
        with pytest.raises(AssertionError):
            solve_equilibrium(
                T=0, P=P_REF, feed_elements={"C": 1.0}, matrix=small_matrix
            )

    def test_negative_pressure_raises(self, small_matrix: ElementMatrix) -> None:
        with pytest.raises(AssertionError):
            solve_equilibrium(
                T=1000, P=-1, feed_elements={"C": 1.0}, matrix=small_matrix
            )

    def test_zero_pressure_raises(self, small_matrix: ElementMatrix) -> None:
        with pytest.raises(AssertionError):
            solve_equilibrium(
                T=1000, P=0, feed_elements={"C": 1.0}, matrix=small_matrix
            )

    def test_high_pressure_convergence(self, small_matrix: ElementMatrix) -> None:
        result = solve_equilibrium(
            T=800,
            P=P_REF * 10,
            feed_elements={"C": 1.0, "H": 2.0, "O": 1.0},
            matrix=small_matrix,
        )
        assert result.converged


class TestKnownEquilibria:
    """Test solver against known thermodynamic behavior."""

    @pytest.fixture
    def matrix(self) -> ElementMatrix:
        return ElementMatrix.from_species(
            ["H2", "CO", "CO2", "H2O", "CH4", "N2", "C_solid"]
        )

    def test_boudouard_favors_co_at_high_t(self, matrix: ElementMatrix) -> None:
        """C + CO2 -> 2CO favored above ~700C."""
        result = solve_equilibrium(
            T=1200, P=P_REF, feed_elements={"C": 2.0, "O": 2.0}, matrix=matrix
        )
        comp = dict(zip(matrix.species_keys, result.moles, strict=True))
        assert comp["CO"] > comp["CO2"]

    def test_boudouard_favors_co2_at_low_t(self, matrix: ElementMatrix) -> None:
        """CO2 more stable at low temperatures."""
        result = solve_equilibrium(
            T=400, P=P_REF, feed_elements={"C": 2.0, "O": 2.0}, matrix=matrix
        )
        comp = dict(zip(matrix.species_keys, result.moles, strict=True))
        assert comp["CO2"] > comp["CO"]

    def test_methanation_favored_at_low_t(self, matrix: ElementMatrix) -> None:
        """CH4 formation favored at lower temperatures."""
        r_low = solve_equilibrium(
            T=500, P=P_REF, feed_elements={"C": 1.0, "H": 4.0}, matrix=matrix
        )
        r_high = solve_equilibrium(
            T=1200, P=P_REF, feed_elements={"C": 1.0, "H": 4.0}, matrix=matrix
        )
        ch4_idx = matrix.species_index("CH4")
        assert r_low.moles[ch4_idx] > r_high.moles[ch4_idx]

    def test_le_chatelier_pressure(self, matrix: ElementMatrix) -> None:
        """Higher P favors fewer gas moles (more CH4)."""
        r_low = solve_equilibrium(
            T=800, P=P_REF, feed_elements={"C": 1.0, "H": 4.0}, matrix=matrix
        )
        r_high = solve_equilibrium(
            T=800, P=P_REF * 30, feed_elements={"C": 1.0, "H": 4.0}, matrix=matrix
        )
        ch4_idx = matrix.species_index("CH4")
        assert r_high.moles[ch4_idx] > r_low.moles[ch4_idx]

    def test_steam_gasification_produces_syngas(self, matrix: ElementMatrix) -> None:
        """C + H2O -> CO + H2 at high temperature."""
        result = solve_equilibrium(
            T=1200, P=P_REF, feed_elements={"C": 1.0, "H": 2.0, "O": 1.0}, matrix=matrix
        )
        h2_idx = matrix.species_index("H2")
        co_idx = matrix.species_index("CO")
        assert result.moles[h2_idx] > 0.1
        assert result.moles[co_idx] > 0.1
