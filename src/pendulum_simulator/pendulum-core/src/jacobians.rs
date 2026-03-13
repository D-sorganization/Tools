//! Analytical Jacobian and manipulability ellipsoid computations.
//!
//! Provides Jacobians for double and triple pendulum endpoints,
//! plus SVD-based ellipsoid analysis (mobility and force ellipsoids).
//!
//! Design by Contract:
//! - All segment lengths must be positive.
//! - All angles and lengths must be finite.
//! - Returns `EllipsoidResult` with optional force ellipsoid (None near singularities).

use nalgebra::{SMatrix, SVector};

/// Threshold: singular value ratio below which the force ellipsoid is None.
const SINGULARITY_THRESHOLD: f64 = 1e-6;

/// Result of an ellipsoid computation at one endpoint.
#[derive(Debug, Clone)]
pub struct EllipsoidResult {
    /// Principal direction vectors (columns of U from SVD), shape conceptually (2, 2).
    pub directions: SMatrix<f64, 2, 2>,
    /// Semi-axis lengths of the mobility ellipsoid (√λᵢ).
    pub mob_semi_axes: SVector<f64, 2>,
    /// Semi-axis lengths of the force ellipsoid (1/√λᵢ), or None if singular.
    pub force_semi_axes: Option<SVector<f64, 2>>,
    /// Raw singular values of J.
    pub singular_values: SVector<f64, 2>,
}

// ---------------------------------------------------------------------------
// Shared ellipsoid kernel (DRY: one implementation, many callers)
// ---------------------------------------------------------------------------

/// Compute mobility and force ellipsoid data from a 2×N task-space Jacobian.
///
/// Uses SVD: J = U Σ Vᵀ.  The left singular vectors U give the principal
/// axes.  Mobility semi-axes are the singular values; force semi-axes are
/// their reciprocals (when non-singular).
fn ellipsoid_from_jacobian_2x2(j: &SMatrix<f64, 2, 2>) -> EllipsoidResult {
    let svd = j.svd(true, false);
    let s = svd.singular_values; // 2-element vector, descending
    let u = svd.u.unwrap_or_else(SMatrix::<f64, 2, 2>::identity);

    let force = if s[0] < SINGULARITY_THRESHOLD || s[1] < SINGULARITY_THRESHOLD * s[0] {
        None
    } else {
        Some(SVector::<f64, 2>::new(1.0 / s[0], 1.0 / s[1]))
    };

    EllipsoidResult {
        directions: u,
        mob_semi_axes: s,
        force_semi_axes: force,
        singular_values: s,
    }
}

/// Ellipsoid from a 2×3 Jacobian (triple pendulum).
fn ellipsoid_from_jacobian_2x3(j: &SMatrix<f64, 2, 3>) -> EllipsoidResult {
    let svd = j.svd(true, false);
    let s_full = svd.singular_values; // up to min(2,3)=2 values
    let u = svd.u.unwrap_or_else(SMatrix::<f64, 2, 2>::identity);

    // Take only the first 2 singular values
    let s = SVector::<f64, 2>::new(s_full[0], s_full[1]);

    let force = if s[0] < SINGULARITY_THRESHOLD || s[1] < SINGULARITY_THRESHOLD * s[0] {
        None
    } else {
        Some(SVector::<f64, 2>::new(1.0 / s[0], 1.0 / s[1]))
    };

    EllipsoidResult {
        directions: u,
        mob_semi_axes: s,
        force_semi_axes: force,
        singular_values: s,
    }
}

// ---------------------------------------------------------------------------
// Double pendulum Jacobians (re-exported from double.rs but also available here)
// ---------------------------------------------------------------------------

/// Compute task-space Jacobians for both endpoints of the double pendulum.
///
/// Returns (J_wrist, J_tip) where each is a 2×2 matrix.
pub fn jacobian_double(
    theta1: f64,
    phi: f64,
    l1: f64,
    l2: f64,
) -> (SMatrix<f64, 2, 2>, SMatrix<f64, 2, 2>) {
    assert!(l1 > 0.0, "l1 must be positive");
    assert!(l2 > 0.0, "l2 must be positive");
    assert!(theta1.is_finite(), "theta1 must be finite");
    assert!(phi.is_finite(), "phi must be finite");

    let theta2 = theta1 + phi;
    let (c1, s1) = (theta1.cos(), theta1.sin());
    let (c2, s2) = (theta2.cos(), theta2.sin());

    // J_wrist: wrist depends only on theta1
    let j_wrist = SMatrix::<f64, 2, 2>::new(
        l1 * c1, 0.0,
        l1 * s1, 0.0,
    );

    // J_tip: full 2×2
    let j_tip = SMatrix::<f64, 2, 2>::new(
        l1 * c1 + l2 * c2, l2 * c2,
        l1 * s1 + l2 * s2, l2 * s2,
    );

    (j_wrist, j_tip)
}

/// Compute ellipsoid data for both double-pendulum endpoints.
pub fn ellipsoids_double(
    theta1: f64,
    phi: f64,
    l1: f64,
    l2: f64,
) -> (EllipsoidResult, EllipsoidResult) {
    let (j_wrist, j_tip) = jacobian_double(theta1, phi, l1, l2);
    (
        ellipsoid_from_jacobian_2x2(&j_wrist),
        ellipsoid_from_jacobian_2x2(&j_tip),
    )
}

// ---------------------------------------------------------------------------
// Triple pendulum Jacobians
// ---------------------------------------------------------------------------

/// Compute task-space Jacobians for all three endpoints of the triple pendulum.
///
/// Returns (J_wrist1, J_wrist2, J_tip) where each is a 2×3 matrix.
pub fn jacobian_triple(
    theta1: f64,
    phi1: f64,
    phi2: f64,
    l1: f64,
    l2: f64,
    l3: f64,
) -> (SMatrix<f64, 2, 3>, SMatrix<f64, 2, 3>, SMatrix<f64, 2, 3>) {
    assert!(l1 > 0.0 && l2 > 0.0 && l3 > 0.0, "All lengths must be positive");
    assert!(
        theta1.is_finite() && phi1.is_finite() && phi2.is_finite(),
        "All angles must be finite"
    );

    let theta2 = theta1 + phi1;
    let theta3 = theta1 + phi1 + phi2;
    let (c1, s1) = (theta1.cos(), theta1.sin());
    let (c2, s2) = (theta2.cos(), theta2.sin());
    let (c3, s3) = (theta3.cos(), theta3.sin());

    // Wrist1: only theta1 contributes
    let j_w1 = SMatrix::<f64, 2, 3>::new(
        l1 * c1, 0.0, 0.0,
        l1 * s1, 0.0, 0.0,
    );

    // Wrist2: theta1 and phi1 contribute
    let j_w2 = SMatrix::<f64, 2, 3>::new(
        l1 * c1 + l2 * c2, l2 * c2, 0.0,
        l1 * s1 + l2 * s2, l2 * s2, 0.0,
    );

    // Tip: all three DOFs contribute
    let j_tip = SMatrix::<f64, 2, 3>::new(
        l1 * c1 + l2 * c2 + l3 * c3, l2 * c2 + l3 * c3, l3 * c3,
        l1 * s1 + l2 * s2 + l3 * s3, l2 * s2 + l3 * s3, l3 * s3,
    );

    (j_w1, j_w2, j_tip)
}

/// Compute ellipsoid data for all three triple-pendulum endpoints.
pub fn ellipsoids_triple(
    theta1: f64,
    phi1: f64,
    phi2: f64,
    l1: f64,
    l2: f64,
    l3: f64,
) -> (EllipsoidResult, EllipsoidResult, EllipsoidResult) {
    let (j_w1, j_w2, j_tip) = jacobian_triple(theta1, phi1, phi2, l1, l2, l3);
    (
        ellipsoid_from_jacobian_2x3(&j_w1),
        ellipsoid_from_jacobian_2x3(&j_w2),
        ellipsoid_from_jacobian_2x3(&j_tip),
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_double_jacobian_wrist_only_depends_on_theta1() {
        let (j_wrist, _) = jacobian_double(0.5, 0.3, 1.0, 1.0);
        // Second column of J_wrist should be zero (phi doesn't affect wrist)
        assert!(approx_eq(j_wrist[(0, 1)], 0.0, 1e-12));
        assert!(approx_eq(j_wrist[(1, 1)], 0.0, 1e-12));
    }

    #[test]
    fn test_double_jacobian_tip_at_zero() {
        let (_, j_tip) = jacobian_double(0.0, 0.0, 1.0, 1.0);
        // At zero: cos(0) = 1, sin(0) = 0
        // j_tip = [[2, 1], [0, 0]]
        assert!(approx_eq(j_tip[(0, 0)], 2.0, 1e-12));
        assert!(approx_eq(j_tip[(0, 1)], 1.0, 1e-12));
        assert!(approx_eq(j_tip[(1, 0)], 0.0, 1e-12));
        assert!(approx_eq(j_tip[(1, 1)], 0.0, 1e-12));
    }

    #[test]
    fn test_double_jacobian_matches_python() {
        // Match the Python jacobian_double exactly
        let theta1 = 0.3;
        let phi = 0.5;
        let l1 = 0.7;
        let l2 = 1.1;
        let (j_wrist, j_tip) = jacobian_double(theta1, phi, l1, l2);

        let theta2 = theta1 + phi;
        let (c1, s1) = (theta1.cos(), theta1.sin());
        let (c2, s2) = (theta2.cos(), theta2.sin());

        assert!(approx_eq(j_wrist[(0, 0)], l1 * c1, 1e-12));
        assert!(approx_eq(j_wrist[(1, 0)], l1 * s1, 1e-12));
        assert!(approx_eq(j_tip[(0, 0)], l1 * c1 + l2 * c2, 1e-12));
        assert!(approx_eq(j_tip[(0, 1)], l2 * c2, 1e-12));
        assert!(approx_eq(j_tip[(1, 0)], l1 * s1 + l2 * s2, 1e-12));
        assert!(approx_eq(j_tip[(1, 1)], l2 * s2, 1e-12));
    }

    #[test]
    fn test_triple_jacobian_wrist1_only_theta1() {
        let (j_w1, _, _) = jacobian_triple(0.5, 0.3, 0.2, 1.0, 1.0, 1.0);
        // Columns 1 and 2 should be zero
        assert!(approx_eq(j_w1[(0, 1)], 0.0, 1e-12));
        assert!(approx_eq(j_w1[(1, 1)], 0.0, 1e-12));
        assert!(approx_eq(j_w1[(0, 2)], 0.0, 1e-12));
        assert!(approx_eq(j_w1[(1, 2)], 0.0, 1e-12));
    }

    #[test]
    fn test_triple_jacobian_wrist2_no_phi2() {
        let (_, j_w2, _) = jacobian_triple(0.5, 0.3, 0.2, 1.0, 1.0, 1.0);
        // Column 2 should be zero (phi2 doesn't affect wrist2)
        assert!(approx_eq(j_w2[(0, 2)], 0.0, 1e-12));
        assert!(approx_eq(j_w2[(1, 2)], 0.0, 1e-12));
    }

    #[test]
    fn test_ellipsoid_double_mob_semi_axes_positive() {
        let (e_wrist, e_tip) = ellipsoids_double(FRAC_PI_4, 0.3, 0.7, 1.1);
        assert!(e_wrist.mob_semi_axes[0] >= 0.0);
        assert!(e_wrist.mob_semi_axes[1] >= 0.0);
        assert!(e_tip.mob_semi_axes[0] >= 0.0);
        assert!(e_tip.mob_semi_axes[1] >= 0.0);
    }

    #[test]
    fn test_ellipsoid_double_force_none_at_singularity() {
        // Wrist Jacobian has zero second column -> singular
        let (e_wrist, _) = ellipsoids_double(0.0, 0.0, 1.0, 1.0);
        assert!(
            e_wrist.force_semi_axes.is_none(),
            "Force ellipsoid should be None for rank-1 wrist Jacobian"
        );
    }

    #[test]
    fn test_ellipsoid_triple_tip_non_singular() {
        // At a general configuration, tip Jacobian should be full rank
        let (_, _, e_tip) = ellipsoids_triple(0.3, 0.5, 0.2, 0.7, 0.8, 0.9);
        assert!(e_tip.force_semi_axes.is_some());
    }

    #[test]
    fn test_ellipsoid_directions_orthogonal() {
        let (_, e_tip) = ellipsoids_double(FRAC_PI_4, FRAC_PI_2, 0.7, 1.1);
        let d = e_tip.directions;
        // Columns should be orthogonal
        let dot = d[(0, 0)] * d[(0, 1)] + d[(1, 0)] * d[(1, 1)];
        assert!(
            approx_eq(dot, 0.0, 1e-10),
            "Direction columns must be orthogonal, got dot={dot}"
        );
    }
}
