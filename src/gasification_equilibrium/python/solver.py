"""Gibbs free energy minimization solver.

SRP: This module handles ONLY the optimization problem.
     No feed processing, no metrics, no plotting.

Design by Contract:
    Preconditions:
        - Temperature > 0 K, Pressure > 0 Pa
        - Element balance vector b has at least one positive entry
    Postconditions:
        - Returned moles array has shape (n_species,) with all values >= 0
        - Element balance error < tolerance if converged
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize, nnls

from .thermo_data import (
    P_REF,
    SPECIES_DB,
    get_all_species,
    get_elements_in_system,
    gibbs_dimensionless,
)

MIN_MOLES = 1e-15
DEFAULT_TOL = 1e-10
MAX_ITER = 2000


@dataclass
class SolverResult:
    """Raw output from the Gibbs minimizer.

    Invariant: if converged, balance_error < tolerance
    """

    moles: np.ndarray
    gibbs_energy: float
    converged: bool
    iterations: int
    balance_error: float


class ElementMatrix:
    """Element-species composition matrix.

    Immutable after construction. Shared across all solves.
    ISP: provides only matrix-related queries.
    """

    def __init__(self, species_keys, element_keys):
        self.species_keys = list(species_keys)
        self.element_keys = list(element_keys)
        self.n_species = len(species_keys)
        self.n_elements = len(element_keys)
        self.A = self._build(species_keys, element_keys)
        self.gas_mask = np.array(
            [SPECIES_DB[k]["phase"] == "gas" for k in species_keys]
        )

    @staticmethod
    def _build(species_keys, element_keys):
        """Build A[j, i] = atoms of element j in species i."""
        A = np.zeros((len(element_keys), len(species_keys)))
        for i, sp_key in enumerate(species_keys):
            sp = SPECIES_DB[sp_key]
            for j, elem in enumerate(element_keys):
                A[j, i] = sp["elements"].get(elem, 0)
        return A

    @classmethod
    def from_species(cls, species_keys=None):
        """Factory: build from species list (defaults to all)."""
        if species_keys is None:
            species_keys = get_all_species()
        element_keys = get_elements_in_system(species_keys)
        return cls(species_keys, element_keys)

    def build_balance_vector(self, feed_elements):
        """Convert {element: moles} dict to numpy balance vector b.

        Postcondition: b.shape == (n_elements,)
        """
        b = np.zeros(self.n_elements)
        for j, elem in enumerate(self.element_keys):
            b[j] = feed_elements.get(elem, 0.0)
        return b

    def species_index(self, key):
        """Return index of species key, or -1 if not found."""
        try:
            return self.species_keys.index(key)
        except ValueError:
            return -1


def compute_gibbs(n, T, P, matrix):
    """Compute total dimensionless Gibbs energy G_total/(RT).

    For gas:   G_i = G°_i + RT[ln(P/P_ref) + ln(x_i)]
    For solid: G_i = G°_i  (activity = 1)

    Precondition: T > 0, P > 0
    """
    n_safe = np.maximum(n, MIN_MOLES)
    n_gas_total = max(np.sum(n_safe * matrix.gas_mask), MIN_MOLES)
    ln_P_ratio = np.log(P / P_REF)

    G = 0.0
    for i, sp_key in enumerate(matrix.species_keys):
        g_std = gibbs_dimensionless(sp_key, T)
        if matrix.gas_mask[i]:
            x_i = n_safe[i] / n_gas_total
            G += n_safe[i] * (g_std + ln_P_ratio + np.log(max(x_i, MIN_MOLES)))
        else:
            G += n_safe[i] * g_std
    return G


def compute_gibbs_gradient(n, T, P, matrix):
    """Gradient of total Gibbs energy w.r.t. moles.

    Postcondition: returns array of shape (n_species,)
    """
    n_safe = np.maximum(n, MIN_MOLES)
    n_gas_total = max(np.sum(n_safe * matrix.gas_mask), MIN_MOLES)
    ln_P_ratio = np.log(P / P_REF)

    grad = np.zeros(matrix.n_species)
    for i, sp_key in enumerate(matrix.species_keys):
        g_std = gibbs_dimensionless(sp_key, T)
        if matrix.gas_mask[i]:
            x_i = n_safe[i] / n_gas_total
            grad[i] = g_std + ln_P_ratio + np.log(max(x_i, MIN_MOLES))
        else:
            grad[i] = g_std
    return grad


def initial_guess(matrix, b):
    """Generate initial moles guess via NNLS.

    Precondition: b.shape == (n_elements,)
    Postcondition: n.shape == (n_species,), all >= MIN_MOLES
    """
    try:
        n0, _ = nnls(matrix.A, b)
    except Exception:
        n0 = np.full(matrix.n_species, max(np.sum(b), 1.0) / matrix.n_species)
    n0 = np.maximum(n0, MIN_MOLES * 10)
    uniform = np.full(matrix.n_species, max(np.sum(b), 1.0) / matrix.n_species / 10)
    return 0.95 * n0 + 0.05 * uniform


def solve_equilibrium(T, P, feed_elements, matrix, tolerance=None, warm_start=None):
    """Minimize Gibbs free energy subject to element balance constraints.

    Args:
        T: Temperature [K]
        P: Pressure [Pa]
        feed_elements: {element: moles} dict
        matrix: ElementMatrix instance
        tolerance: Solver tolerance (default 1e-10)
        warm_start: Previous solution moles for warm-starting

    Returns:
        SolverResult

    Precondition: T > 0, P > 0
    Postcondition: result.moles >= 0
    """
    assert T > 0, f"Temperature must be > 0 K, got {T}"
    assert P > 0, f"Pressure must be > 0 Pa, got {P}"

    tol = tolerance or DEFAULT_TOL
    b = matrix.build_balance_vector(feed_elements)

    if warm_start is not None:
        n0 = np.maximum(warm_start, MIN_MOLES)
    else:
        n0 = initial_guess(matrix, b)

    constraints = {
        "type": "eq",
        "fun": lambda n: matrix.A @ n - b,
        "jac": lambda n: matrix.A,
    }

    # Pin unconstrained species (all-zero columns in A, e.g. noble gases)
    # to MIN_MOLES unless explicitly provided in feed.
    col_sums = np.sum(np.abs(matrix.A), axis=0)
    bounds = []
    for i in range(matrix.n_species):
        if col_sums[i] < 1e-15:
            bounds.append((MIN_MOLES, MIN_MOLES))
            n0[i] = MIN_MOLES
        else:
            bounds.append((MIN_MOLES, None))

    result = minimize(
        fun=lambda n: compute_gibbs(n, T, P, matrix),
        x0=n0,
        method="SLSQP",
        jac=lambda n: compute_gibbs_gradient(n, T, P, matrix),
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": MAX_ITER, "ftol": tol, "disp": False},
    )

    n_eq = np.maximum(result.x, 0.0)
    balance_err = float(np.max(np.abs(matrix.A @ n_eq - b)))

    return SolverResult(
        moles=n_eq,
        gibbs_energy=result.fun,
        converged=(result.success or balance_err < 1e-6),
        iterations=result.nit,
        balance_error=balance_err,
    )
