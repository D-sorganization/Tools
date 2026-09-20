"""Bivariate normality diagnostic and convex hull fallback display (#4253 item c).

Tests the 2D landing dispersion distribution (carry vs lateral) for bivariate
normality using Mardia's (1970) multivariate skewness and kurtosis tests.
When the empirical landing scatter violates normality (e.g. truncated inputs,
gear-effect curvature, non-linear aerodynamics), the parametric 2-sigma ellipse
becomes misleading; a non-parametric convex hull and KDE fallback is provided.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import chdtrc, erfc

from shared.python.contracts import require

_MIN_SAMPLES_FOR_MARDIA = 8
_SINGULAR_THRESHOLD = 1e-12


@dataclass(frozen=True)
class NormalityDiagnostic:
    """Mardia's multivariate normality test result for 2D landing dispersion."""

    is_normal: bool
    p_value: float
    skewness_stat: float
    skewness_p_value: float
    kurtosis_stat: float
    kurtosis_p_value: float
    n: int
    alpha: float = 0.05
    method: str = "mardia-bivariate"


def mardia_bivariate_normality(
    points: np.ndarray, alpha: float = 0.05
) -> NormalityDiagnostic:
    """Evaluate bivariate normality of a 2D dataset using Mardia's test.

    Computes multivariate sample skewness b_{1,2} and sample kurtosis b_{2,2}.
    Under H0 (bivariate normality):
      - (n / 6) * b_{1,2} ~ Chi^2(df=4)
      - (b_{2,2} - 8) / sqrt(64 / n) ~ N(0, 1)

    Returns:
        NormalityDiagnostic summarizing test statistics, p-values, and conclusion.
    """
    pts = np.asarray(points, dtype=float)
    require(
        pts.ndim == 2 and pts.shape[1] == 2, "points must be an (N, 2) array", pts.shape
    )
    n = pts.shape[0]

    if n < _MIN_SAMPLES_FOR_MARDIA:
        return NormalityDiagnostic(
            is_normal=True,
            p_value=1.0,
            skewness_stat=0.0,
            skewness_p_value=1.0,
            kurtosis_stat=8.0,
            kurtosis_p_value=1.0,
            n=n,
            alpha=alpha,
        )

    mean = np.mean(pts, axis=0)
    centered = pts - mean

    # Sample covariance matrix (biased MLE divisor N per Mardia 1970 convention)
    cov = (centered.T @ centered) / n
    det = np.linalg.det(cov)
    if not math.isfinite(det) or det < _SINGULAR_THRESHOLD:
        # Collinear or singular distribution
        return NormalityDiagnostic(
            is_normal=False,
            p_value=0.0,
            skewness_stat=math.inf,
            skewness_p_value=0.0,
            kurtosis_stat=math.inf,
            kurtosis_p_value=0.0,
            n=n,
            alpha=alpha,
        )

    inv_cov = np.linalg.inv(cov)

    # Mahalanobis distance matrix M_ij = (x_i - mu)^T S^{-1} (x_j - mu)
    # Shape: (n, n)
    m_matrix = centered @ inv_cov @ centered.T

    # Mardia Skewness: b_{1,2} = (1 / n^2) * sum_{i,j} M_ij^3
    b1_2 = float(np.mean(m_matrix**3))
    # Degrees of freedom for p=2: p*(p+1)*(p+2)/6 = 2*3*4/6 = 4
    t_skew = (n / 6.0) * b1_2
    p_skew = float(chdtrc(4, max(0.0, t_skew)))

    # Mardia Kurtosis: b_{2,2} = (1 / n) * sum_i M_ii^2
    diag_m = np.diag(m_matrix)
    b2_2 = float(np.mean(diag_m**2))
    # E[b_{2,2}] = p*(p+2) = 8, Var = 8*p*(p+2)/n = 64/n
    z_kurt = (b2_2 - 8.0) / math.sqrt(64.0 / n)
    # Two-sided standard normal p-value using erfc
    p_kurt = float(erfc(abs(z_kurt) / math.sqrt(2.0)))

    omnibus_p = min(p_skew, p_kurt)
    is_normal = bool(p_skew >= alpha and p_kurt >= alpha)

    return NormalityDiagnostic(
        is_normal=is_normal,
        p_value=omnibus_p,
        skewness_stat=b1_2,
        skewness_p_value=p_skew,
        kurtosis_stat=b2_2,
        kurtosis_p_value=p_kurt,
        n=n,
        alpha=alpha,
    )


def convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """Compute 2D convex hull ordered polygon vertices.

    Returns an (H, 2) array of hull vertices in counter-clockwise order.
    """
    pts = np.asarray(points, dtype=float)
    require(pts.ndim == 2 and pts.shape[1] == 2, "points must be an (N, 2) array")
    if pts.shape[0] < 3:
        return np.array(pts, copy=True)

    try:
        from scipy.spatial import ConvexHull

        hull = ConvexHull(pts)
        ordered_vertices = pts[hull.vertices]
        return np.array(ordered_vertices, dtype=float, copy=True)
    except Exception:
        # Monotone chain fallback
        sorted_indices = np.lexsort((pts[:, 1], pts[:, 0]))
        sorted_pts = pts[sorted_indices]

        def _cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
            return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

        lower: list[np.ndarray] = []
        for p in sorted_pts:
            while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        upper: list[np.ndarray] = []
        for p in reversed(sorted_pts):
            while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        hull_pts = lower[:-1] + upper[:-1]
        return np.asarray(hull_pts, dtype=float)


__all__ = [
    "NormalityDiagnostic",
    "convex_hull_2d",
    "mardia_bivariate_normality",
]
