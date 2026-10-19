//! Double pendulum (2-DOF) physics implementation.
//!
//! Model:
//! - q = [θ₁, φ] where θ₁ is arm angle from vertical, φ is club angle relative to arm
//! - Absolute club angle: θ₂ = θ₁ + φ
//! - Segment 1: shoulder to wrist (arm), length L1, mass m1
//! - Segment 1: shoulder to wrist (arm), length L1, point mass m1 at the wrist
//! - Segment 2: wrist to tip (club), length L2. Canonical distributed convention
//!   (Tools #5431): uniform shaft m2 (COM L2/2, Icm = m2 L2^2/12) plus a head point
//!   mass `m_clubhead` at the tip. All formulas read the club's mass `Mc`, first
//!   moment `s` and wrist inertia `Jw` from [`DoublePendulumParams::distal_inertia`].

use crate::types::{DoubleFKResult, DoublePendulumParams};
use nalgebra::{SMatrix, SVector};

/// Compute the 2x2 mass matrix M(q).
///
/// For the desktop double-pendulum golf model with relative coordinates:
/// M[0,0] = (m1 + Mc) * L1² + Jw + 2 * s * L1 * cos(φ)
/// M[0,1] = Jw + s * L1 * cos(φ)
/// M[1,0] = M[0,1]
/// M[1,1] = Jw
/// where Mc, s, Jw are the club mass, first moment and wrist inertia.
pub fn mass_matrix(q: &[f64; 2], params: &DoublePendulumParams) -> SMatrix<f64, 2, 2> {
    let phi = q[1];
    let cos_phi = phi.cos();
    let club = params.distal_inertia();

    let l1_sq = params.l1 * params.l1;
    let coupling = club.first_moment * params.l1 * cos_phi;

    let m00 = (params.m1 + club.mass) * l1_sq + club.pivot_inertia + 2.0 * coupling;
    let m01 = club.pivot_inertia + coupling;
    let m11 = club.pivot_inertia;

    SMatrix::<f64, 2, 2>::new(m00, m01, m01, m11)
}

/// Compute the Coriolis vector C(q, qdot).
///
/// For the double pendulum:
/// C[0] = h * (2 * qdot[0] * qdot[1] + qdot[1]²)
/// C[1] = -h * qdot[0]²
/// where h = -s * L1 * sin(φ) and s is the club's first moment about the wrist
pub fn coriolis(q: &[f64; 2], qdot: &[f64; 2], params: &DoublePendulumParams) -> SVector<f64, 2> {
    let phi = q[1];
    let h = -params.distal_inertia().first_moment * params.l1 * phi.sin();
    let dtheta1 = qdot[0];
    let dphi = qdot[1];

    let c0 = h * (2.0 * dtheta1 * dphi + dphi * dphi);
    let c1 = -h * dtheta1 * dtheta1;

    SVector::<f64, 2>::new(c0, c1)
}

/// Compute the gravity vector G(q).
///
/// For the double pendulum:
/// G[0] = (m1 + Mc) * g * L1 * sin(θ₁) + s * g * sin(θ₁ + φ)
/// G[1] = s * g * sin(θ₁ + φ)
/// where Mc is the club mass and s its first moment about the wrist.
pub fn gravity_vector(q: &[f64; 2], params: &DoublePendulumParams) -> SVector<f64, 2> {
    let theta1 = q[0];
    let phi = q[1];
    let theta2 = theta1 + phi;
    let club = params.distal_inertia();

    let g1 = club.first_moment * params.g * theta2.sin();
    let g0 = (params.m1 + club.mass) * params.g * params.l1 * theta1.sin() + g1;

    SVector::<f64, 2>::new(g0, g1)
}

/// Compute viscous friction torque vector.
///
/// τ_friction[i] = -friction_i * qdot[i]
///
/// This is the linear damping (viscous friction) at each joint.
pub fn friction_torque(qdot: &[f64; 2], params: &DoublePendulumParams) -> SVector<f64, 2> {
    SVector::<f64, 2>::new(-params.friction1 * qdot[0], -params.friction2 * qdot[1])
}

/// Compute joint accelerations: qddot = M⁻¹(tau + tau_friction - C - G).
///
/// This is the full unconstrained equations of motion including friction.
/// `tau` is the applied (driving) torque vector.
pub fn equations_of_motion(
    q: &[f64; 2],
    qdot: &[f64; 2],
    tau: &[f64; 2],
    params: &DoublePendulumParams,
) -> [f64; 2] {
    let m = mass_matrix(q, params);
    let c = coriolis(q, qdot, params);
    let g = gravity_vector(q, params);
    let f = friction_torque(qdot, params);

    // RHS = tau + friction - coriolis - gravity
    let rhs = SVector::<f64, 2>::new(tau[0], tau[1]) + f - c - g;

    // Solve M * qddot = rhs
    let decomp = m.lu();
    let qddot = decomp.solve(&rhs).unwrap_or_else(|| {
        // Fallback: use pseudo-inverse if singular (shouldn't happen)
        SVector::<f64, 2>::zeros()
    });

    [qddot[0], qddot[1]]
}

/// Compute forward kinematics.
///
/// Given configuration q = [θ₁, φ], compute the positions of:
/// - Wrist (end of segment 1)
/// - Club tip (end of segment 2)
pub fn forward_kinematics(q: &[f64; 2], params: &DoublePendulumParams) -> DoubleFKResult {
    let theta1 = q[0];
    let phi = q[1];
    let theta2 = theta1 + phi;

    // Wrist position: shoulder + L1*(sin(θ₁), -cos(θ₁))
    let wrist = (params.l1 * theta1.sin(), -params.l1 * theta1.cos());

    // Club tip: wrist + L2*(sin(θ₂), -cos(θ₂))
    let club_tip = (
        wrist.0 + params.l2 * theta2.sin(),
        wrist.1 - params.l2 * theta2.cos(),
    );

    DoubleFKResult {
        wrist,
        club_tip,
        theta1,
        theta2,
    }
}

/// Compute the Jacobian of the wrist position with respect to q.
///
/// wrist(q) = [L1*sin(θ₁), -L1*cos(θ₁)]
/// J_wrist = [ [L1*cos(θ₁),     0    ],
///             [-L1*sin(θ₁),     0    ] ]
pub fn jacobian_wrist(q: &[f64; 2], params: &DoublePendulumParams) -> SMatrix<f64, 2, 2> {
    let theta1 = q[0];
    let cos_theta1 = theta1.cos();
    let sin_theta1 = theta1.sin();

    SMatrix::<f64, 2, 2>::new(params.l1 * cos_theta1, 0.0, -params.l1 * sin_theta1, 0.0)
}

/// Compute the Jacobian of the club tip position with respect to q.
///
/// club_tip(q) = [L1*sin(θ₁) + L2*sin(θ₁+φ), -L1*cos(θ₁) - L2*cos(θ₁+φ)]
/// ∂(club_tip)/∂θ₁ = [L1*cos(θ₁) + L2*cos(θ₁+φ), L1*sin(θ₁) + L2*sin(θ₁+φ)]
/// ∂(club_tip)/∂φ = [L2*cos(θ₁+φ), L2*sin(θ₁+φ)]
pub fn jacobian_club_tip(q: &[f64; 2], params: &DoublePendulumParams) -> SMatrix<f64, 2, 2> {
    let theta1 = q[0];
    let phi = q[1];
    let theta2 = theta1 + phi;

    let cos_theta1 = theta1.cos();
    let sin_theta1 = theta1.sin();
    let cos_theta2 = theta2.cos();
    let sin_theta2 = theta2.sin();

    let j00 = params.l1 * cos_theta1 + params.l2 * cos_theta2;
    let j10 = params.l1 * sin_theta1 + params.l2 * sin_theta2;
    let j01 = params.l2 * cos_theta2;
    let j11 = params.l2 * sin_theta2;

    SMatrix::<f64, 2, 2>::new(j00, j01, j10, j11)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mass_matrix_validity() {
        let params = DoublePendulumParams {
            m1: 1.0,
            m2: 1.0,
            m_clubhead: 0.0,
            l1: 1.0,
            l2: 1.0,
            g: crate::types::STANDARD_GRAVITY,
            friction1: 0.0,
            friction2: 0.0,
        };

        let q = [0.0, 0.0];
        let m = mass_matrix(&q, &params);

        // At φ=0: M should be symmetric and positive definite
        assert!((m.m12 - m.m21).abs() < 1e-12);
        assert!(m.m11 > 0.0);
        assert!(m.m22 > 0.0);
    }

    #[test]
    fn test_forward_kinematics() {
        let params = DoublePendulumParams {
            m1: 1.0,
            m2: 1.0,
            m_clubhead: 0.0,
            l1: 1.0,
            l2: 1.0,
            g: crate::types::STANDARD_GRAVITY,
            friction1: 0.0,
            friction2: 0.0,
        };

        let q = [0.0, 0.0]; // Hanging down
        let fk = forward_kinematics(&q, &params);

        // Both segments hanging straight down
        assert!(fk.wrist.0.abs() < 1e-12);
        assert!((fk.wrist.1 + 1.0).abs() < 1e-12);
        assert!(fk.club_tip.0.abs() < 1e-12);
        assert!((fk.club_tip.1 + 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_mass_matrix_matches_python_model_with_clubhead() {
        let params = DoublePendulumParams {
            m1: 1.0,
            m2: 0.5,
            m_clubhead: 0.5,
            l1: 1.0,
            l2: 1.0,
            g: 1.0,
            friction1: 0.0,
            friction2: 0.0,
        };

        let q = [0.0, 0.0];
        let m = mass_matrix(&q, &params);

        // Mc = 1, s = 0.75, Jw = 2/3 (uniform 0.5 kg shaft + 0.5 kg head, L2 = 1).
        assert!((m[(0, 0)] - (2.0 + 2.0 / 3.0 + 1.5)).abs() < 1e-12);
        assert!((m[(0, 1)] - (2.0 / 3.0 + 0.75)).abs() < 1e-12);
        assert!((m[(1, 1)] - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_coriolis_matches_python_cross_term() {
        let params = DoublePendulumParams {
            m1: 1.0,
            m2: 0.5,
            m_clubhead: 0.5,
            l1: 1.0,
            l2: 1.0,
            g: 1.0,
            friction1: 0.0,
            friction2: 0.0,
        };

        let q = [0.0, std::f64::consts::FRAC_PI_2];
        let qdot = [2.0, 3.0];
        let c = coriolis(&q, &qdot, &params);

        // h = -s * L1 * sin(phi) = -0.75
        assert!((c[0] + 15.75).abs() < 1e-12);
        assert!((c[1] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_gravity_vector_includes_clubhead_load() {
        let params = DoublePendulumParams {
            m1: 1.0,
            m2: 0.5,
            m_clubhead: 0.5,
            l1: 1.0,
            l2: 1.0,
            g: 1.0,
            friction1: 0.0,
            friction2: 0.0,
        };

        let q = [std::f64::consts::FRAC_PI_2, 0.0];
        let g = gravity_vector(&q, &params);

        assert!((g[0] - 2.75).abs() < 1e-12);
        assert!((g[1] - 0.75).abs() < 1e-12);
    }

    fn params(m2: f64, m_clubhead: f64, l2: f64) -> DoublePendulumParams {
        DoublePendulumParams {
            m1: 2.0,
            m2,
            m_clubhead,
            l1: 0.7,
            l2,
            g: crate::types::STANDARD_GRAVITY,
            friction1: 0.0,
            friction2: 0.0,
        }
    }

    #[test]
    fn test_issue_5431_reproducer_kinetic_energy() {
        // m1=2, m2=3, mClub=1, L1=0.7, L2=1, qdot=[0, 2] -> 0.5 qd.M.qd = 4 J (was 8 J lumped).
        let m = mass_matrix(&[0.0, 0.0], &params(3.0, 1.0, 1.0));
        let ke = 0.5 * 2.0 * m[(1, 1)] * 2.0;
        assert!((ke - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_head_only_limit_is_point_mass_and_shaft_only_is_rod() {
        let lumped = mass_matrix(&[0.0, 0.4], &params(1e-12, 1.5, 1.1));
        assert!((lumped[(1, 1)] - 1.5 * 1.1 * 1.1).abs() < 1e-9);
        let rod = mass_matrix(&[0.0, 0.4], &params(3.0, 0.0, 1.2));
        assert!((rod[(1, 1)] - 3.0 * 1.2 * 1.2 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_passive_swing_conserves_energy() {
        let p = params(0.108, 0.202, 1.1);
        let energy = |q: &[f64; 2], qd: &[f64; 2]| {
            let m = mass_matrix(q, &p);
            let v = SVector::<f64, 2>::new(qd[0], qd[1]);
            let club = p.distal_inertia();
            0.5 * v.dot(&(m * v))
                - (p.m1 + club.mass) * p.g * p.l1 * q[0].cos()
                - club.first_moment * p.g * (q[0] + q[1]).cos()
        };
        let (mut q, mut qd) = ([1.0, 0.5], [0.3, -0.2]);
        let e0 = energy(&q, &qd);
        let dt = 1e-4;
        let f = |q: &[f64; 2], qd: &[f64; 2]| {
            let a = equations_of_motion(q, qd, &[0.0, 0.0], &p);
            ([qd[0], qd[1]], a)
        };
        let add = |x: &[f64; 2], k: &[f64; 2], h: f64| [x[0] + h * k[0], x[1] + h * k[1]];
        for _ in 0..20_000 {
            let (k1q, k1v) = f(&q, &qd);
            let (k2q, k2v) = f(&add(&q, &k1q, dt / 2.0), &add(&qd, &k1v, dt / 2.0));
            let (k3q, k3v) = f(&add(&q, &k2q, dt / 2.0), &add(&qd, &k2v, dt / 2.0));
            let (k4q, k4v) = f(&add(&q, &k3q, dt), &add(&qd, &k3v, dt));
            for i in 0..2 {
                q[i] += dt / 6.0 * (k1q[i] + 2.0 * k2q[i] + 2.0 * k3q[i] + k4q[i]);
                qd[i] += dt / 6.0 * (k1v[i] + 2.0 * k2v[i] + 2.0 * k3v[i] + k4v[i]);
            }
        }
        assert!((energy(&q, &qd) - e0).abs() < 1e-8);
    }

    /// Extract `"key": <number>` from one JSON object block (fixture is pretty-printed).
    fn field(block: &str, key: &str) -> f64 {
        let pattern = format!("\"{key}\":");
        let start = block.find(&pattern).expect("key present") + pattern.len();
        let tail = &block[start..];
        let end = tail.find([',', '\n', '}']).unwrap();
        tail[..end].trim().parse().expect("number")
    }

    #[test]
    fn test_distal_inertia_matches_shared_fixture() {
        use crate::types::DistalInertia;
        let fixture = include_str!("../../tests/fixtures/distal_inertia_cases.json");
        let cases = &fixture[fixture.find("\"cases\"").unwrap()..];
        let mut checked = 0;
        for block in cases.split('{').skip(1) {
            let club = DistalInertia::new(
                field(block, "shaft_mass_kg"),
                field(block, "head_mass_kg"),
                field(block, "length_m"),
            )
            .unwrap();
            assert!(
                (club.centroidal_inertia - field(block, "centroidal_inertia_kg_m2")).abs()
                    <= 1e-14 * field(block, "centroidal_inertia_kg_m2").abs() + 1e-300
            );
            assert!(club.centroidal_inertia >= 0.0);
            assert!((club.mass - field(block, "mass_kg")).abs() < 1e-14);
            assert!((club.first_moment - field(block, "first_moment_kg_m")).abs() < 1e-14);
            assert!((club.pivot_inertia - field(block, "inertia_about_pivot_kg_m2")).abs() < 1e-14);
            checked += 1;
        }
        assert!(checked >= 5);
    }

    #[test]
    fn test_distal_inertia_rejects_invalid_inputs() {
        use crate::types::DistalInertia;
        assert!(DistalInertia::new(-1.0, 1.0, 1.0).is_err());
        assert!(DistalInertia::new(1.0, -0.1, 1.0).is_err());
        assert!(DistalInertia::new(0.0, 0.0, 1.0).is_err());
        assert!(DistalInertia::new(1.0, 1.0, 0.0).is_err());
        assert!(DistalInertia::new(f64::NAN, 1.0, 1.0).is_err());
        assert!(DistalInertia::new(1.0, f64::INFINITY, 1.0).is_err());
    }

    #[test]
    fn test_centroidal_inertia_never_negative_for_vanishing_shaft() {
        use crate::types::DistalInertia;
        for i in 0..200 {
            let length = 0.5 + 1.0 * (i as f64) / 199.0;
            let club = DistalInertia::new(1e-15, 0.5, length).unwrap();
            assert!(club.centroidal_inertia >= 0.0);
        }
    }

    #[test]
    fn test_dynamics_match_shared_golden_cases() {
        let fixture = include_str!("../../tests/fixtures/distal_dynamics_cases.json");
        let cases = &fixture[fixture.find("\"cases\"").unwrap()..];
        let mut checked = 0;
        for block in cases.split('{').skip(1) {
            let p = DoublePendulumParams {
                m1: field(block, "m1"),
                m2: field(block, "shaft_mass_kg"),
                m_clubhead: field(block, "head_mass_kg"),
                l1: field(block, "l1"),
                l2: field(block, "l2"),
                g: field(block, "g"),
                friction1: 0.0,
                friction2: 0.0,
            };
            let q = [field(block, "theta1"), field(block, "phi")];
            let qd = [field(block, "dtheta1"), field(block, "dphi")];
            let close = |got: f64, want: f64| (got - want).abs() <= 1e-12 * want.abs().max(1.0);
            let m = mass_matrix(&q, &p);
            assert!(close(m[(0, 0)], field(block, "m11")));
            assert!(close(m[(0, 1)], field(block, "m12")));
            assert!(close(m[(1, 1)], field(block, "m22")));
            let c = coriolis(&q, &qd, &p);
            assert!(close(c[0], field(block, "c1")));
            assert!(close(c[1], field(block, "c2")));
            let g = gravity_vector(&q, &p);
            assert!(close(g[0], field(block, "g1")));
            assert!(close(g[1], field(block, "g2")));
            let club = p.distal_inertia();
            let potential = -(p.m1 + club.mass) * p.g * p.l1 * q[0].cos()
                - club.first_moment * p.g * (q[0] + q[1]).cos();
            assert!(close(potential, field(block, "potential_energy_j")));
            checked += 1;
        }
        assert!(checked >= 5);
    }
}
