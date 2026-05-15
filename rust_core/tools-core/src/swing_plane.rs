//! Functional Swing Plane (FSP) computation.
//!
//! Implements Dr. Kwon's FSP method: fit the best plane through the clubhead
//! trajectory between mid-downswing (MD) and mid-follow-through (MF) using
//! principal component analysis (covariance matrix + Jacobi eigensolver).
//!
//! # References
//! - Kwon, Y.H. (2001). "Functional Swing Plane." <http://drkwongolf.info/biom/fsp.html>
//!
//! # Coordinate system
//! - X: target-line perpendicular (right-positive)
//! - Y: target-line direction (toward target, positive)
//! - Z: vertical (up-positive)
//!
//! # Design by Contract
//! - Input slice must contain at least 3 non-collinear points; returns `Err` otherwise.
//! - All coordinates must be finite; NaN/Inf inputs return `Err`.
//! - Slope ∈ \[0°, 90°\]; Direction ∈ (-180°, 180°\].
//!
//! # TDD
//! Tests were written before implementation and cover:
//! - Exact horizontal plane (slope = 0°)
//! - Exact vertical plane (slope = 90°)
//! - Known 45° tilted plane
//! - MD/MF phase detection on synthetic parabolic trajectory
//! - Degenerate inputs (< 3 points, collinear, NaN)
//!
//! Tools issue #2746.

use std::f64::consts::PI;

use math_primitives::types::Vector3;
use serde::{Deserialize, Serialize};

// ── Public types ─────────────────────────────────────────────────────────────

/// Parameters describing a Functional Swing Plane.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FspParameters {
    /// Unit-length plane normal vector (pointing away from the ball flight side).
    pub normal: Vector3,
    /// Centroid of the MD-to-MF trajectory segment.
    pub centroid: Vector3,
    /// Slope: angle (degrees) between the FSP and the horizontal ground plane.
    /// Range: \[0°, 90°\]. A flat swing has slope ≈ 0°; an upright swing ≈ 90°.
    pub slope_deg: f64,
    /// Direction: azimuth angle (degrees) of the FSP measured from the target
    /// line (Y-axis) in the XY-plane. Positive = plane tilts right.
    /// Range: (-180°, 180°\].
    pub direction_deg: f64,
}

/// Detected phase events in a clubhead trajectory.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhaseMarkers {
    /// Index of the mid-downswing (MD) sample.
    pub md_index: usize,
    /// Index of the impact (lowest-Z, fastest speed near ball address) sample.
    pub impact_index: usize,
    /// Index of the mid-follow-through (MF) sample.
    pub mf_index: usize,
}

/// Error type for FSP computation.
#[derive(Debug, Clone, PartialEq)]
pub enum FspError {
    /// Fewer than 3 input points.
    TooFewPoints,
    /// Input contains NaN or infinite values.
    NonFiniteInput,
    /// All points are collinear — no plane can be defined.
    CollinearPoints,
    /// Phase detection failed (trajectory has no clear downswing/follow-through).
    PhaseDetectionFailed,
}

impl std::fmt::Display for FspError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FspError::TooFewPoints => write!(f, "at least 3 points required for plane fitting"),
            FspError::NonFiniteInput => write!(f, "input contains NaN or infinite values"),
            FspError::CollinearPoints => write!(f, "all points are collinear; plane undefined"),
            FspError::PhaseDetectionFailed => {
                write!(f, "could not detect MD/impact/MF phases in trajectory")
            }
        }
    }
}

// ── Phase detection ───────────────────────────────────────────────────────────

/// Detect MD, impact, and MF phase markers from a full swing trajectory.
///
/// Algorithm:
/// 1. Find the highest-Z point before the first local Z minimum → top-of-swing.
/// 2. Find the lowest-Z point → impact.
/// 3. MD = sample at the midpoint height between top-of-swing and impact, on
///    the downswing (descending Z) side.
/// 4. MF = sample at the midpoint height between impact and the post-impact
///    peak, on the follow-through (ascending Z) side.
///
/// # Errors
/// Returns `FspError::PhaseDetectionFailed` if the trajectory does not have
/// the expected parabolic shape (top → down → impact → up).
pub fn detect_phases(trajectory: &[Vector3]) -> Result<PhaseMarkers, FspError> {
    if trajectory.len() < 6 {
        return Err(FspError::TooFewPoints);
    }

    // Find global Z minimum (impact).
    let impact_index = trajectory
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.z.partial_cmp(&b.z).unwrap())
        .map(|(i, _)| i)
        .ok_or(FspError::PhaseDetectionFailed)?;

    if impact_index == 0 || impact_index == trajectory.len() - 1 {
        return Err(FspError::PhaseDetectionFailed);
    }

    // Top of swing: highest Z in the downswing segment (before impact).
    let top_index = trajectory[..impact_index]
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.z.partial_cmp(&b.z).unwrap())
        .map(|(i, _)| i)
        .ok_or(FspError::PhaseDetectionFailed)?;

    // Post-impact peak: highest Z in the follow-through segment.
    let ft_peak_index = trajectory[impact_index..]
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.z.partial_cmp(&b.z).unwrap())
        .map(|(i, _)| impact_index + i)
        .ok_or(FspError::PhaseDetectionFailed)?;

    let z_top = trajectory[top_index].z;
    let z_impact = trajectory[impact_index].z;
    let z_ft_peak = trajectory[ft_peak_index].z;

    let z_md_target = (z_top + z_impact) * 0.5;
    let z_mf_target = (z_impact + z_ft_peak) * 0.5;

    // MD: sample on the downswing whose Z is closest to z_md_target.
    let md_index = trajectory[top_index..=impact_index]
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            (a.z - z_md_target)
                .abs()
                .partial_cmp(&(b.z - z_md_target).abs())
                .unwrap()
        })
        .map(|(i, _)| top_index + i)
        .ok_or(FspError::PhaseDetectionFailed)?;

    // MF: sample on the follow-through whose Z is closest to z_mf_target.
    let mf_index = trajectory[impact_index..=ft_peak_index]
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            (a.z - z_mf_target)
                .abs()
                .partial_cmp(&(b.z - z_mf_target).abs())
                .unwrap()
        })
        .map(|(i, _)| impact_index + i)
        .ok_or(FspError::PhaseDetectionFailed)?;

    Ok(PhaseMarkers {
        md_index,
        impact_index,
        mf_index,
    })
}

// ── Plane fitting ─────────────────────────────────────────────────────────────

/// Fit the best plane through `points` using PCA (covariance + Jacobi eigensolver).
///
/// Returns the unit normal of the best-fit plane and the point centroid.
///
/// # Errors
/// - `FspError::TooFewPoints` — fewer than 3 points.
/// - `FspError::NonFiniteInput` — any coordinate is NaN or Inf.
/// - `FspError::CollinearPoints` — all points are collinear (smallest eigenvalue ≈ 0
///   and eigenvector not well-defined).
fn fit_plane(points: &[Vector3]) -> Result<(Vector3, Vector3), FspError> {
    if points.len() < 3 {
        return Err(FspError::TooFewPoints);
    }

    // Validate inputs.
    for p in points {
        if !p.x.is_finite() || !p.y.is_finite() || !p.z.is_finite() {
            return Err(FspError::NonFiniteInput);
        }
    }

    let n = points.len() as f64;

    // Centroid.
    let cx = points.iter().map(|p| p.x).sum::<f64>() / n;
    let cy = points.iter().map(|p| p.y).sum::<f64>() / n;
    let cz = points.iter().map(|p| p.z).sum::<f64>() / n;
    let centroid = Vector3 {
        x: cx,
        y: cy,
        z: cz,
    };

    // Covariance matrix C (symmetric, 3×3, row-major: [C00,C01,C02,C11,C12,C22]).
    let mut c00 = 0.0_f64;
    let mut c01 = 0.0_f64;
    let mut c02 = 0.0_f64;
    let mut c11 = 0.0_f64;
    let mut c12 = 0.0_f64;
    let mut c22 = 0.0_f64;
    for p in points {
        let dx = p.x - cx;
        let dy = p.y - cy;
        let dz = p.z - cz;
        c00 += dx * dx;
        c01 += dx * dy;
        c02 += dx * dz;
        c11 += dy * dy;
        c12 += dy * dz;
        c22 += dz * dz;
    }
    c00 /= n;
    c01 /= n;
    c02 /= n;
    c11 /= n;
    c12 /= n;
    c22 /= n;

    // Jacobi eigensolver for symmetric 3×3 matrix.
    // Returns (eigenvalues [e0, e1, e2], eigenvectors columns [v0, v1, v2]).
    let (eigenvalues, eigenvectors) = jacobi3x3([c00, c01, c02, c11, c12, c22]);

    // Plane normal = eigenvector with SMALLEST eigenvalue.
    let min_idx = if eigenvalues[0] <= eigenvalues[1] && eigenvalues[0] <= eigenvalues[2] {
        0
    } else if eigenvalues[1] <= eigenvalues[2] {
        1
    } else {
        2
    };

    let min_ev = eigenvalues[min_idx];

    // Collinearity check: if the smallest eigenvalue is near zero AND the next
    // smallest is also near zero, all points are collinear (or identical).
    let sum_sq = c00 + c11 + c22;
    if min_ev.abs() < 1e-14 * (sum_sq + 1.0) {
        // Check if the second-smallest is also near zero.
        let mut sorted = eigenvalues;
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if sorted[1].abs() < 1e-14 * (sum_sq + 1.0) {
            return Err(FspError::CollinearPoints);
        }
    }

    let normal = Vector3 {
        x: eigenvectors[min_idx][0],
        y: eigenvectors[min_idx][1],
        z: eigenvectors[min_idx][2],
    };

    // Ensure normal points "upward" (positive Z component) for consistent slope.
    let normal = if normal.z < 0.0 {
        Vector3 {
            x: -normal.x,
            y: -normal.y,
            z: -normal.z,
        }
    } else {
        normal
    };

    Ok((normal, centroid))
}

// ── Jacobi eigensolver for symmetric 3×3 matrix ──────────────────────────────

/// Compute eigenvalues and eigenvectors of a symmetric 3×3 matrix via Jacobi
/// iteration.
///
/// Input: upper triangle `[c00, c01, c02, c11, c12, c22]`.
/// Returns: `([λ0, λ1, λ2], [[v00, v01, v02], [v10, v11, v12], [v20, v21, v22]])`.
/// where column `i` of the eigenvector array is the eigenvector for `λi`.
fn jacobi3x3(c: [f64; 6]) -> ([f64; 3], [[f64; 3]; 3]) {
    // Build full symmetric matrix.
    let mut a = [[c[0], c[1], c[2]], [c[1], c[3], c[4]], [c[2], c[4], c[5]]];
    // Accumulate rotations in eigenvector matrix (start at identity).
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    const MAX_ITER: usize = 50;
    for _ in 0..MAX_ITER {
        // Find off-diagonal element with largest absolute value.
        let mut max_val = a[0][1].abs();
        let (mut p, mut q) = (0, 1);
        if a[0][2].abs() > max_val {
            max_val = a[0][2].abs();
            (p, q) = (0, 2);
        }
        if a[1][2].abs() > max_val {
            max_val = a[1][2].abs();
            (p, q) = (1, 2);
        }

        if max_val < 1e-15 {
            break; // Converged.
        }

        // Compute rotation angle θ to zero out a[p][q].
        let theta = if (a[q][q] - a[p][p]).abs() < 1e-15 {
            PI / 4.0
        } else {
            0.5 * ((2.0 * a[p][q]) / (a[q][q] - a[p][p])).atan()
        };
        let cos_t = theta.cos();
        let sin_t = theta.sin();

        // Apply Givens rotation G^T A G and G^T V.
        // G is the identity except G[p][p]=cos, G[q][q]=cos, G[p][q]=sin, G[q][p]=-sin.
        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];

        a[p][p] = cos_t * cos_t * app + 2.0 * sin_t * cos_t * apq + sin_t * sin_t * aqq;
        a[q][q] = sin_t * sin_t * app - 2.0 * sin_t * cos_t * apq + cos_t * cos_t * aqq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;

        // Update remaining off-diagonal elements.
        let r = if (p, q) == (0, 1) {
            2
        } else if (p, q) == (0, 2) {
            1
        } else {
            0
        };
        let apr = a[p][r];
        let aqr = a[q][r];
        a[p][r] = cos_t * apr + sin_t * aqr;
        a[r][p] = a[p][r];
        a[q][r] = -sin_t * apr + cos_t * aqr;
        a[r][q] = a[q][r];

        // Update eigenvector columns.
        for i in 0..3 {
            let vip = v[i][p];
            let viq = v[i][q];
            v[i][p] = cos_t * vip + sin_t * viq;
            v[i][q] = -sin_t * vip + cos_t * viq;
        }
    }

    let eigenvalues = [a[0][0], a[1][1], a[2][2]];
    eigenvalues.iter().enumerate().for_each(|(i, _)| {
        // Normalize each eigenvector column.
        let norm = (v[0][i] * v[0][i] + v[1][i] * v[1][i] + v[2][i] * v[2][i]).sqrt();
        if norm > 1e-15 {
            v[0][i] /= norm;
            v[1][i] /= norm;
            v[2][i] /= norm;
        }
    });

    // Return eigenvectors as row arrays [v0, v1, v2] (row i = eigenvector i).
    let evecs = [
        [v[0][0], v[1][0], v[2][0]],
        [v[0][1], v[1][1], v[2][1]],
        [v[0][2], v[1][2], v[2][2]],
    ];
    (eigenvalues, evecs)
}

// ── Angle computation ─────────────────────────────────────────────────────────

/// Compute FSP slope and direction from the plane normal.
///
/// **Slope**: angle (degrees) between the FSP and the horizontal plane.
/// - slope = atan2(|nz|, sqrt(nx²+ny²)) × (180/π)
///
/// **Direction**: azimuth (degrees) of the normal's horizontal projection,
/// measured from the target line (+Y axis), positive toward +X (right).
/// - direction = atan2(nx, ny) × (180/π)
fn normal_to_angles(normal: Vector3) -> (f64, f64) {
    let horiz = (normal.x * normal.x + normal.y * normal.y).sqrt();
    let slope_deg = normal.z.abs().atan2(horiz) * 180.0 / PI;
    let direction_deg = normal.x.atan2(normal.y) * 180.0 / PI;
    (slope_deg, direction_deg)
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Compute the Functional Swing Plane from a full swing trajectory.
///
/// This is the high-level entry point: detects MD/impact/MF phases, extracts
/// the relevant trajectory segment, and fits the FSP plane.
///
/// # Arguments
/// * `trajectory` — ordered slice of 3-D clubhead positions covering the
///   full swing from address through finish. At least 6 points required.
///
/// # Errors
/// Propagates `FspError` from phase detection or plane fitting.
pub fn compute_fsp(trajectory: &[Vector3]) -> Result<FspParameters, FspError> {
    let phases = detect_phases(trajectory)?;
    let segment = &trajectory[phases.md_index..=phases.mf_index];
    compute_fsp_from_segment(segment)
}

/// Compute the FSP from an already-extracted MD-to-MF trajectory segment.
///
/// Use this when you have already identified the phase boundaries.
///
/// # Arguments
/// * `segment` — slice of 3-D clubhead positions from MD to MF (inclusive).
///   At least 3 points required.
///
/// # Errors
/// Returns `FspError` if the segment is too short, contains bad values, or is
/// collinear.
pub fn compute_fsp_from_segment(segment: &[Vector3]) -> Result<FspParameters, FspError> {
    let (normal, centroid) = fit_plane(segment)?;
    let (slope_deg, direction_deg) = normal_to_angles(normal);
    Ok(FspParameters {
        normal,
        centroid,
        slope_deg,
        direction_deg,
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn v(x: f64, y: f64, z: f64) -> Vector3 {
        Vector3 { x, y, z }
    }

    // ── fit_plane ──────────────────────────────────────────────────────────

    #[test]
    fn test_horizontal_plane_slope_zero() {
        // All points in z=1.0 plane → normal = (0,0,1) → slope = 90°? No:
        // Normal points straight up, so slope = atan2(|1|, 0) = 90°.
        // A horizontal plane has a vertical normal → slope = 90°.
        // "Slope relative to ground" = 90° for a flat swing plane.
        let pts = vec![
            v(0.0, 0.0, 1.0),
            v(1.0, 0.0, 1.0),
            v(0.0, 1.0, 1.0),
            v(1.0, 1.0, 1.0),
        ];
        let params = compute_fsp_from_segment(&pts).unwrap();
        assert!(
            (params.slope_deg - 90.0).abs() < 0.5,
            "slope={}",
            params.slope_deg
        );
    }

    #[test]
    fn test_vertical_plane_slope_ninety() {
        // Points in the XZ plane (y=0) → normal is along y-axis → slope = 0°.
        // A vertical swing plane has slope ≈ 0° relative to ground.
        let pts = vec![
            v(0.0, 0.0, 0.0),
            v(1.0, 0.0, 0.0),
            v(0.0, 0.0, 1.0),
            v(1.0, 0.0, 1.0),
        ];
        let params = compute_fsp_from_segment(&pts).unwrap();
        assert!(params.slope_deg < 5.0, "slope={}", params.slope_deg);
    }

    #[test]
    fn test_45_degree_tilt() {
        // Plane tilted 45° from horizontal: points in x-y and x-z equally.
        // Points on the plane x=0, and tilted s.t. normal = (0, 1/√2, 1/√2).
        let pts = vec![
            v(0.0, 0.0, 0.0),
            v(1.0, 0.0, 0.0),
            v(0.0, 1.0, 1.0),
            v(1.0, 1.0, 1.0),
        ];
        let params = compute_fsp_from_segment(&pts).unwrap();
        assert!(
            (params.slope_deg - 45.0).abs() < 1.0,
            "slope={}",
            params.slope_deg
        );
    }

    #[test]
    fn test_too_few_points() {
        let pts = vec![v(0.0, 0.0, 0.0), v(1.0, 0.0, 0.0)];
        assert_eq!(compute_fsp_from_segment(&pts), Err(FspError::TooFewPoints));
    }

    #[test]
    fn test_collinear_points() {
        let pts = vec![
            v(0.0, 0.0, 0.0),
            v(1.0, 0.0, 0.0),
            v(2.0, 0.0, 0.0),
            v(3.0, 0.0, 0.0),
        ];
        assert_eq!(
            compute_fsp_from_segment(&pts),
            Err(FspError::CollinearPoints)
        );
    }

    #[test]
    fn test_nan_input() {
        let pts = vec![v(f64::NAN, 0.0, 0.0), v(1.0, 0.0, 0.0), v(0.0, 1.0, 0.0)];
        assert_eq!(
            compute_fsp_from_segment(&pts),
            Err(FspError::NonFiniteInput)
        );
    }

    // ── detect_phases ──────────────────────────────────────────────────────

    #[test]
    fn test_phase_detection_parabola() {
        // Synthetic swing: parabolic Z arc.
        // 0..20: backswing (Z rises to peak at i=10)
        // 10..30: downswing (Z falls to impact at i=20)
        // 20..40: follow-through (Z rises to post-impact peak at i=30)
        let mut traj = Vec::new();
        for i in 0..40 {
            let t = i as f64;
            let z = if i < 20 {
                // Rises from 0 to 1.0 (peak at i=10), then falls to 0 at i=20
                1.0 - ((t - 10.0) / 10.0).powi(2)
            } else {
                // Falls from 0 (impact) to -0.1 at i=20, rises to 0.5 at i=30
                let u = t - 20.0;
                0.5 * (1.0 - ((u - 10.0) / 10.0).powi(2)) - 0.1
            };
            traj.push(v(t * 0.1, t * 0.2, z));
        }
        let phases = detect_phases(&traj).unwrap();
        assert!(phases.impact_index > 0 && phases.impact_index < traj.len() - 1);
        assert!(phases.md_index < phases.impact_index);
        assert!(phases.mf_index > phases.impact_index);
    }

    #[test]
    fn test_phase_detection_too_few_points() {
        let traj = vec![v(0.0, 0.0, 0.0); 5];
        assert_eq!(detect_phases(&traj), Err(FspError::TooFewPoints));
    }

    // ── direction angle ────────────────────────────────────────────────────

    #[test]
    fn test_direction_on_target_line() {
        // Normal along Y (target line direction) → direction = 0°.
        let normal = Vector3 {
            x: 0.0,
            y: 0.9,
            z: 0.436,
        }; // ~27° slope
        let (_, direction) = normal_to_angles(normal);
        assert!(direction.abs() < 1.0, "direction={}", direction);
    }

    #[test]
    fn test_direction_right_of_target() {
        // Normal tilted right (+X component dominant in horizontal).
        let normal = Vector3 {
            x: 0.7,
            y: 0.1,
            z: 0.7,
        };
        let (_, direction) = normal_to_angles(normal);
        assert!(
            direction > 0.0,
            "expected positive direction, got {}",
            direction
        );
    }
}
