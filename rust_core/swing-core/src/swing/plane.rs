//! Swing-plane orientation and in-plane gravity projection.
//!
//! # Convention
//! World frame: x forward (toward target line reference), y left, z up.
//! The identity ("flat") plane pose spans world x (in-plane horizontal axis)
//! and world z (in-plane up axis); its normal is world y. A pendulum swinging
//! in the identity plane therefore swings in a vertical plane and feels the
//! full gravitational acceleration in-plane.
//!
//! The plane pose is built from three sequential intrinsic tilts
//! (documented order, epic #4103 P1 spec):
//! 1. **yaw** about the world-up axis (z),
//! 2. **side tilt** about the rotated in-plane horizontal axis,
//! 3. **forward/back tilt** about the resulting in-plane up axis.
//!
//! As intrinsic rotations these compose by right-multiplication:
//! `R = Rz(yaw) · Rx(side_tilt) · Ry(fwd_tilt)`.
//!
//! # Design by Contract
//! - All angles are radians and must be finite (`debug_assert!`).
//! - `plane_rotation` returns a proper orthonormal rotation (unit tested).
//! - `in_plane_gravity` limits: identity pose ⇒ `(0, -g)` (full g "down"
//!   in-plane); pure yaw is invariant (gravity is symmetric about world-up).

use nalgebra::{Matrix3, Vector3};

/// Build the world-from-plane rotation matrix from the three tilt angles.
///
/// Columns of the returned matrix are the plane's local axes expressed in
/// world coordinates: column 0 = in-plane horizontal axis, column 1 = plane
/// normal, column 2 = in-plane up axis (see module docs for the identity
/// pose).
///
/// # Contracts (DbC)
/// - Precondition: all angles are finite radians.
/// - Postcondition: result is orthonormal with determinant +1.
#[must_use]
pub fn plane_rotation(yaw: f64, side_tilt: f64, fwd_tilt: f64) -> Matrix3<f64> {
    debug_assert!(yaw.is_finite(), "plane_rotation: yaw must be finite");
    debug_assert!(
        side_tilt.is_finite(),
        "plane_rotation: side_tilt must be finite"
    );
    debug_assert!(
        fwd_tilt.is_finite(),
        "plane_rotation: fwd_tilt must be finite"
    );

    let rz = Matrix3::new(
        yaw.cos(),
        -yaw.sin(),
        0.0,
        yaw.sin(),
        yaw.cos(),
        0.0,
        0.0,
        0.0,
        1.0,
    );
    let rx = Matrix3::new(
        1.0,
        0.0,
        0.0,
        0.0,
        side_tilt.cos(),
        -side_tilt.sin(),
        0.0,
        side_tilt.sin(),
        side_tilt.cos(),
    );
    let ry = Matrix3::new(
        fwd_tilt.cos(),
        0.0,
        fwd_tilt.sin(),
        0.0,
        1.0,
        0.0,
        -fwd_tilt.sin(),
        0.0,
        fwd_tilt.cos(),
    );
    rz * rx * ry
}

/// Project world gravity into the swing plane.
///
/// Given the world-from-plane rotation `plane_r` and the gravitational
/// acceleration magnitude `g` (m/s², non-negative), returns the in-plane
/// gravity components `(g_x_inplane, g_y_inplane)` along the plane's local
/// horizontal axis and local up axis respectively. The world gravity vector
/// is `(0, 0, -g)` (z-up world).
///
/// The double-pendulum EOM consumes this 2-vector directly — never a scalar.
///
/// # Contracts (DbC)
/// - Precondition: `g` is finite and `>= 0`.
/// - Postcondition: `g_x² + g_y² <= g²` (projection never amplifies).
#[must_use]
pub fn in_plane_gravity(plane_r: &Matrix3<f64>, g: f64) -> (f64, f64) {
    debug_assert!(
        g.is_finite() && g >= 0.0,
        "in_plane_gravity: g must be finite and >= 0"
    );

    let g_world = Vector3::new(0.0, 0.0, -g);
    // Local axes in world coordinates: columns 0 (in-plane horizontal) and
    // 2 (in-plane up) of the world-from-plane rotation.
    let x_axis = plane_r.column(0);
    let y_up_axis = plane_r.column(2);
    let gx = g_world.dot(&x_axis);
    let gy = g_world.dot(&y_up_axis);
    debug_assert!(
        gx * gx + gy * gy <= g * g + 1e-9,
        "in_plane_gravity: projection must not amplify gravity"
    );
    (gx, gy)
}

/// Convenience: in-plane gravity straight from the three tilt angles.
#[must_use]
pub fn in_plane_gravity_from_tilts(yaw: f64, side_tilt: f64, fwd_tilt: f64, g: f64) -> (f64, f64) {
    in_plane_gravity(&plane_rotation(yaw, side_tilt, fwd_tilt), g)
}

#[cfg(test)]
mod tests {
    use super::*;

    const G: f64 = 9.80665;
    const TOL: f64 = 1e-12;

    #[test]
    fn plane_rotation_is_orthonormal() {
        let cases = [
            (0.0, 0.0, 0.0),
            (0.3, 0.0, 0.0),
            (0.0, 0.7, 0.0),
            (0.0, 0.0, -0.4),
            (1.2, -0.6, 0.35),
            (-2.0, 1.1, 2.9),
        ];
        for (yaw, side, fwd) in cases {
            let r = plane_rotation(yaw, side, fwd);
            let should_be_identity = r.transpose() * r;
            let err = (should_be_identity - Matrix3::identity()).norm();
            assert!(
                err < 1e-12,
                "R^T R != I for ({yaw}, {side}, {fwd}): err {err}"
            );
            let det = r.determinant();
            assert!((det - 1.0).abs() < 1e-12, "det != +1: {det}");
        }
    }

    #[test]
    fn flat_plane_gives_full_gravity_down() {
        let r = plane_rotation(0.0, 0.0, 0.0);
        let (gx, gy) = in_plane_gravity(&r, G);
        assert!(
            gx.abs() < TOL,
            "flat plane must have zero horizontal gravity: {gx}"
        );
        assert!(
            (gy + G).abs() < TOL,
            "flat plane must feel full g down: {gy}"
        );
    }

    #[test]
    fn pure_yaw_leaves_gravity_projection_invariant() {
        let (gx0, gy0) = in_plane_gravity_from_tilts(0.0, 0.0, 0.0, G);
        for yaw in [-3.0, -1.2, 0.4, 1.7, 3.1] {
            let (gx, gy) = in_plane_gravity_from_tilts(yaw, 0.0, 0.0, G);
            assert!((gx - gx0).abs() < TOL, "yaw {yaw} changed gx: {gx}");
            assert!((gy - gy0).abs() < TOL, "yaw {yaw} changed gy: {gy}");
        }
    }

    #[test]
    fn side_tilt_scales_in_plane_up_component_by_cosine() {
        // Matches UpstreamDrift's scalar projected_gravity = g·cos(inclination).
        for side in [0.0, 0.2, 0.6, 1.0, 1.5] {
            let (_, gy) = in_plane_gravity_from_tilts(0.0, side, 0.0, G);
            assert!(
                (gy + G * side.cos()).abs() < 1e-12,
                "side tilt {side}: expected {}, got {gy}",
                -G * side.cos()
            );
        }
    }

    #[test]
    fn projection_never_amplifies_gravity() {
        for yaw in [-2.0, 0.0, 1.3] {
            for side in [-1.4, 0.0, 0.9] {
                for fwd in [-0.8, 0.0, 1.1] {
                    let (gx, gy) = in_plane_gravity_from_tilts(yaw, side, fwd, G);
                    assert!(gx * gx + gy * gy <= G * G + 1e-9);
                }
            }
        }
    }
}
