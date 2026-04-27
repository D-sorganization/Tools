"""Repeated-measures ANOVA helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .anova_models import ANOVATable, RepeatedMeasuresResult


def mauchly_test(data: np.ndarray) -> tuple[float, float, float, float]:
    """Perform Mauchly's test of sphericity."""
    n, k = data.shape
    if k < 3:
        return 1.0, 1.0, 1.0, 1.0

    centered = data - np.mean(data, axis=0)
    covariance = np.cov(centered.T)

    contrast = np.eye(k) - np.ones((k, k)) / k
    contrast = contrast[:-1, :]
    s_star = contrast @ covariance @ contrast.T

    determinant = float(np.linalg.det(s_star))
    trace = float(np.trace(s_star))
    p = k - 1

    w_statistic = determinant / (trace / p) ** p if trace > 0 else 1.0
    df = p * (p + 1) / 2 - 1
    chi_sq = -(n - 1 - (2 * p**2 + p + 2) / (6 * p)) * np.log(max(w_statistic, 1e-10))
    p_value = float(1 - stats.chi2.cdf(chi_sq, df))

    eigenvalues = np.linalg.eigvalsh(s_star)
    eigen_sum = float(np.sum(eigenvalues))
    eigen_sq_sum = float(np.sum(eigenvalues**2))
    gg_epsilon = eigen_sum**2 / (p * eigen_sq_sum) if eigen_sq_sum > 0 else 1.0
    hf_epsilon = (n * (p - 1) * gg_epsilon - 2) / (
        (p - 1) * (n - 1 - (p - 1) * gg_epsilon)
    )
    hf_epsilon = min(1.0, max(gg_epsilon, hf_epsilon))

    return float(w_statistic), p_value, float(gg_epsilon), float(hf_epsilon)


def perform_repeated_measures_anova(
    alpha: float,
    df: pd.DataFrame,
    dependent_vars: list[str],
    subject_id: str,
) -> RepeatedMeasuresResult:
    """Perform one-way repeated-measures ANOVA."""
    data = df[[subject_id] + dependent_vars].dropna()
    n_subjects = len(data)
    n_conditions = len(dependent_vars)

    values = data[dependent_vars].values
    grand_mean = float(np.mean(values))
    condition_means = np.mean(values, axis=0)
    subject_means = np.mean(values, axis=1)

    ss_total = float(np.sum((values - grand_mean) ** 2))
    ss_between_subjects = float(
        n_conditions * np.sum((subject_means - grand_mean) ** 2)
    )
    ss_within_subjects = ss_total - ss_between_subjects
    ss_conditions = float(n_subjects * np.sum((condition_means - grand_mean) ** 2))
    ss_error = ss_within_subjects - ss_conditions

    df_between_subjects = n_subjects - 1
    df_conditions = n_conditions - 1
    df_error = df_between_subjects * df_conditions

    ms_conditions = ss_conditions / df_conditions
    ms_error = ss_error / df_error
    f_statistic = ms_conditions / ms_error
    p_value = float(1 - stats.f.cdf(f_statistic, df_conditions, df_error))

    mauchly_w, mauchly_p, gg_epsilon, hf_epsilon = mauchly_test(values)
    corrected_p_gg = float(
        1
        - stats.f.cdf(
            f_statistic,
            gg_epsilon * df_conditions,
            gg_epsilon * df_error,
        )
    )
    corrected_p_hf = float(
        1
        - stats.f.cdf(
            f_statistic,
            hf_epsilon * df_conditions,
            hf_epsilon * df_error,
        )
    )

    eta_squared = ss_conditions / ss_total
    partial_eta_squared = ss_conditions / (ss_conditions + ss_error)

    anova_table = ANOVATable(
        source=["Between Subjects", "Conditions", "Error", "Total"],
        sum_of_squares=[ss_between_subjects, ss_conditions, ss_error, ss_total],
        df=[
            df_between_subjects,
            df_conditions,
            df_error,
            n_subjects * n_conditions - 1,
        ],
        mean_square=[
            ss_between_subjects / df_between_subjects,
            ms_conditions,
            ms_error,
            np.nan,
        ],
        f_statistic=[None, f_statistic, None, None],
        p_value=[None, p_value, None, None],
    )

    return RepeatedMeasuresResult(
        f_statistic=float(f_statistic),
        p_value=p_value,
        df_effect=df_conditions,
        df_error=df_error,
        mauchly_w=mauchly_w,
        mauchly_p=mauchly_p,
        sphericity_assumed=mauchly_p > alpha,
        greenhouse_geisser_epsilon=gg_epsilon,
        huynh_feldt_epsilon=hf_epsilon,
        corrected_p_gg=corrected_p_gg,
        corrected_p_hf=corrected_p_hf,
        eta_squared=float(eta_squared),
        partial_eta_squared=float(partial_eta_squared),
        anova_table=anova_table,
    )
