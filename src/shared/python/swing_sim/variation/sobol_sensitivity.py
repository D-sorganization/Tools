"""Sobol first-order and total-order sensitivity indices via Saltelli sampling (#4253).

Implements variance-based global sensitivity analysis using Saltelli's 2002/2010
sampling design and estimator formulas (Saltelli 2010 for S1, Jansen 1999 for ST).
"""

from __future__ import annotations

import numpy as np

from shared.python.contracts import ContractViolationError, require

from ..solver.objective import EvaluationConfig
from .pipeline import evaluate_run, outputs_for_mode
from .sobol_types import (
    _CONSTANT_TOLERANCE,
    _MIN_ADEQUATE_BASE_SAMPLES,
    SOBOL_REPORT_SCHEMA_ID,
    SOBOL_REPORT_SCHEMA_VERSION,
    SaltelliDesign,
    SobolEstimate,
    SobolFactor,
    SobolObservations,
    SobolOutput,
    SobolReport,
    SobolRunCountGuidance,
    _require_int,
    sobol_run_count_guidance,
)
from .spec import VariationPlan

__all__ = [
    "SOBOL_REPORT_SCHEMA_ID",
    "SOBOL_REPORT_SCHEMA_VERSION",
    "SaltelliDesign",
    "SobolEstimate",
    "SobolFactor",
    "SobolObservations",
    "SobolOutput",
    "SobolReport",
    "SobolRunCountGuidance",
    "analyze_sobol",
    "generate_saltelli_design",
    "run_sobol_sensitivity",
    "sobol_run_count_guidance",
]


def generate_saltelli_design(
    factors: tuple[SobolFactor, ...],
    base_samples: int = _MIN_ADEQUATE_BASE_SAMPLES,
    seed: int = 0,
) -> SaltelliDesign:
    """Construct a deterministic Saltelli sampling design.

    Produces an (N*(k+2), k) matrix comprising matrices A, B, and k cross matrices
    A_B^{(i)} where the i-th column is substituted from B into A.
    """
    require(len(factors) >= 1, "at least one factor is required")
    k = len(factors)
    n = _require_int(base_samples, "base_samples", 2)
    try:
        from scipy.stats import qmc

        sampler = qmc.Sobol(d=2 * k, seed=seed)
        uniform_ab = sampler.random(n)
    except Exception:
        rng = np.random.default_rng(seed)
        uniform_ab = rng.uniform(0.0, 1.0, size=(n, 2 * k))

    mat_a = uniform_ab[:, :k]
    mat_b = uniform_ab[:, k:]

    # Blocks: A (N, k), B (N, k), followed by k matrices A_B^(i) (each N, k)
    blocks = [mat_a, mat_b]
    for i in range(k):
        cross = mat_a.copy()
        cross[:, i] = mat_b[:, i]
        blocks.append(cross)

    stacked_unit = np.vstack(blocks)  # Shape: (N*(k+2), k)

    # Map each column to [lower, upper]
    physical = np.empty_like(stacked_unit)
    for col, factor in enumerate(factors):
        physical[:, col] = factor.lower + stacked_unit[:, col] * (
            factor.upper - factor.lower
        )

    physical.setflags(write=False)
    return SaltelliDesign(
        factors=tuple(factors),
        base_samples=n,
        seed=seed,
        physical_points=physical,
    )


def _calculate_indices(
    y_a: np.ndarray,
    y_b: np.ndarray,
    y_ab_i: np.ndarray,
    total_var: float,
) -> tuple[float, float]:
    """Compute S1 (Saltelli 2010) and ST (Jansen 1999) from sample vectors."""
    if total_var <= _CONSTANT_TOLERANCE:
        return 0.0, 0.0

    # S1 = mean(y_b * (y_ab_i - y_a)) / Var(Y)
    s1_num = float(np.mean(y_b * (y_ab_i - y_a)))
    s1 = s1_num / total_var

    # ST = 0.5 * mean((y_a - y_ab_i)^2) / Var(Y)
    st_num = 0.5 * float(np.mean((y_a - y_ab_i) ** 2))
    st = st_num / total_var

    return s1, st


def analyze_sobol(
    observations: SobolObservations,
    n_bootstrap: int = 100,
    alpha: float = 0.05,
    seed: int = 42,
) -> SobolReport:
    """Compute first-order S1 and total ST Sobol indices with bootstrap CIs."""
    design = observations.design
    k = len(design.factors)
    n = design.base_samples
    total_runs = design.total_samples
    require(
        observations.values.shape[0] == total_runs,
        "observation rows must match design total samples",
    )
    guidance = sobol_run_count_guidance(factor_count=k, base_samples=n)
    estimates: list[SobolEstimate] = []

    for out_idx, output in enumerate(observations.outputs):
        vals = observations.values[:, out_idx]
        succ = observations.success
        finite_mask = succ & np.isfinite(vals)

        # Slice blocks A and B
        succ_a = finite_mask[:n]
        succ_b = finite_mask[n : 2 * n]

        # Combined common valid mask for variance computation across A and B
        var_mask = np.concatenate([succ_a, succ_b])
        var_vals = np.concatenate([vals[:n], vals[n : 2 * n]])[var_mask]
        total_var = float(np.var(var_vals, ddof=1)) if len(var_vals) > 1 else 0.0

        for factor_idx, factor in enumerate(design.factors):
            start_i = (2 + factor_idx) * n
            end_i = start_i + n
            succ_ab_i = finite_mask[start_i:end_i]

            common = succ_a & succ_b & succ_ab_i
            n_valid = int(np.count_nonzero(common))
            total_evals = 3 * n

            if n_valid < 4 or total_var <= _CONSTANT_TOLERANCE:
                estimates.append(
                    SobolEstimate(
                        spec_id=factor.spec_id,
                        variable_key=factor.variable_key,
                        output_name=output.name,
                        s1=0.0 if total_var <= _CONSTANT_TOLERANCE else float("nan"),
                        st=0.0 if total_var <= _CONSTANT_TOLERANCE else float("nan"),
                        s1_ci=(
                            (0.0, 0.0)
                            if total_var <= _CONSTANT_TOLERANCE
                            else (float("nan"), float("nan"))
                        ),
                        st_ci=(
                            (0.0, 0.0)
                            if total_var <= _CONSTANT_TOLERANCE
                            else (float("nan"), float("nan"))
                        ),
                        availability=(
                            "constant"
                            if total_var <= _CONSTANT_TOLERANCE
                            else "insufficient_valid_runs"
                        ),
                        total_evaluations=total_evals,
                        valid_evaluations=n_valid,
                    )
                )
                continue

            y_a = vals[:n][common]
            y_b = vals[n : 2 * n][common]
            y_ab_i = vals[start_i:end_i][common]

            s1, st = _calculate_indices(y_a, y_b, y_ab_i, total_var)

            # Percentile bootstrap for confidence intervals
            rng = np.random.default_rng(seed + factor_idx)
            s1_boot: list[float] = []
            st_boot: list[float] = []
            for _ in range(n_bootstrap):
                idx = rng.integers(0, n_valid, size=n_valid)
                b_s1, b_st = _calculate_indices(
                    y_a[idx], y_b[idx], y_ab_i[idx], total_var
                )
                s1_boot.append(b_s1)
                st_boot.append(b_st)

            lower_p = 100.0 * (alpha / 2.0)
            upper_p = 100.0 * (1.0 - alpha / 2.0)
            s1_ci = (
                float(np.percentile(s1_boot, lower_p)),
                float(np.percentile(s1_boot, upper_p)),
            )
            st_ci = (
                float(np.percentile(st_boot, lower_p)),
                float(np.percentile(st_boot, upper_p)),
            )

            estimates.append(
                SobolEstimate(
                    spec_id=factor.spec_id,
                    variable_key=factor.variable_key,
                    output_name=output.name,
                    s1=s1,
                    st=st,
                    s1_ci=s1_ci,
                    st_ci=st_ci,
                    availability="available",
                    total_evaluations=total_evals,
                    valid_evaluations=n_valid,
                )
            )

    return SobolReport(
        estimates=tuple(estimates),
        guidance=guidance,
        base_samples=n,
        total_design_samples=total_runs,
        seed=design.seed,
    )


def run_sobol_sensitivity(
    plan: VariationPlan,
    factors: tuple[SobolFactor, ...] | None = None,
    base_samples: int = _MIN_ADEQUATE_BASE_SAMPLES,
    output_names: tuple[str, ...] | None = None,
    n_bootstrap: int = 100,
    alpha: float = 0.05,
) -> SobolReport:
    """Run full Sobol global sensitivity study against the seeded engine."""
    require(isinstance(plan, VariationPlan), "plan must be a VariationPlan", plan)
    if factors is None:
        derived_factors = []
        base = plan.base_variables
        for spec in plan.noise:
            lower = (
                spec.lower
                if spec.lower is not None
                else base[spec.variable_key] - 3.0 * spec.scale
            )
            upper = (
                spec.upper
                if spec.upper is not None
                else base[spec.variable_key] + 3.0 * spec.scale
            )
            derived_factors.append(
                SobolFactor.from_noise_spec(spec, lower=lower, upper=upper)
            )
        factors = tuple(derived_factors)

    design = generate_saltelli_design(
        factors, base_samples=base_samples, seed=plan.seed
    )
    total_samples = design.total_samples
    k = len(factors)

    config = EvaluationConfig(flight_model=plan.flight_model)
    target_names = (
        output_names if output_names is not None else outputs_for_mode(plan.mode)
    )
    outputs_tuple = tuple(SobolOutput(name=name) for name in target_names)

    values = np.empty((total_samples, len(target_names)), dtype=float)
    success = np.empty(total_samples, dtype=bool)

    for row_idx in range(total_samples):
        run_vars = dict(plan.base_variables)
        for col in range(k):
            run_vars[factors[col].variable_key] = float(
                design.physical_points[row_idx, col]
            )
        try:
            row_out = evaluate_run(run_vars, plan.mode, config)
            success[row_idx] = True
            for out_col, name in enumerate(target_names):
                values[row_idx, out_col] = row_out.get(name, float("nan"))
        except (
            ValueError,
            ContractViolationError,
            RuntimeError,
            FloatingPointError,
            OverflowError,
        ):
            success[row_idx] = False
            values[row_idx, :] = float("nan")

    observations = SobolObservations(
        design=design,
        outputs=outputs_tuple,
        values=values,
        success=success,
    )

    return analyze_sobol(
        observations,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        seed=plan.seed,
    )
