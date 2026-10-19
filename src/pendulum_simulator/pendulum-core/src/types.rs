//! Shared types and parameter structures for physics models.

/// Default gravitational acceleration (m/s^2): standard gravity.
///
/// Single source of truth for every Rust default `g` (PyO3 signature
/// defaults included).  Matches UpstreamDrift and the Python
/// `GRAVITY_STANDARD` (Tools #5635; previously 9.81).
pub const STANDARD_GRAVITY: f64 = 9.80665;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Double pendulum parameters (2-DOF model).
///
/// Represents a two-segment pendulum with:
/// - Segment 1: arm from shoulder to wrist
/// - Segment 2: club from wrist to tip
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DoublePendulumParams {
    /// Mass of arm segment (kg)
    pub m1: f64,
    /// Mass of the uniform shaft (kg)
    pub m2: f64,
    /// Head point mass at the tip (kg)
    pub m_clubhead: f64,
    /// Length of arm segment (m)
    pub l1: f64,
    /// Length of club segment (m)
    pub l2: f64,
    /// Gravitational acceleration (m/s²)
    pub g: f64,
    /// Friction coefficient for first joint
    pub friction1: f64,
    /// Friction coefficient for second joint
    pub friction2: f64,
}

/// Inertial properties of the club (uniform shaft plus head point mass) about the wrist.
///
/// Canonical convention (Tools #5431): shaft mass `m2` uniform over `l2` (COM `l2/2`,
/// centroidal inertia `m2*l2^2/12`) plus the head point mass at the tip. This is the
/// single Rust source of the three numbers every double-pendulum formula needs, and
/// it mirrors Python `club_inertia.distal_inertia` (pinned by a shared JSON fixture).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DistalInertia {
    /// Total club mass `Mc = m2 + mh` (kg)
    pub mass: f64,
    /// First moment about the wrist `s = (m2/2 + mh) * l2` (kg m)
    pub first_moment: f64,
    /// Inertia about the wrist `Jw = (m2/3 + mh) * l2^2` (kg m^2)
    pub pivot_inertia: f64,
    /// Centroidal inertia `Icm = m2 l2^2 (m2 + 4 mh) / (12 Mc)` (kg m^2).
    ///
    /// Closed form: `Jw - s^2/Mc` cancels catastrophically for a vanishing shaft.
    pub centroidal_inertia: f64,
}

impl DistalInertia {
    /// Build from shaft mass, head mass and length.
    ///
    /// Preconditions (same as Python `distal_inertia`): all inputs finite, masses
    /// non-negative with a positive total, length positive.
    pub fn new(shaft_mass: f64, head_mass: f64, length: f64) -> Result<Self, String> {
        for (name, value) in [
            ("shaft_mass", shaft_mass),
            ("head_mass", head_mass),
            ("length", length),
        ] {
            if !value.is_finite() {
                return Err(format!("{name} must be finite, got {value}"));
            }
        }
        if shaft_mass < 0.0 || head_mass < 0.0 {
            return Err("shaft and head masses must be non-negative".to_string());
        }
        let mass = shaft_mass + head_mass;
        if mass <= 0.0 {
            return Err("total club mass must be positive".to_string());
        }
        if length <= 0.0 {
            return Err(format!("length must be positive, got {length}"));
        }
        let length_sq = length * length;
        Ok(Self {
            mass,
            first_moment: (0.5 * shaft_mass + head_mass) * length,
            pivot_inertia: (shaft_mass / 3.0 + head_mass) * length_sq,
            centroidal_inertia: shaft_mass * length_sq * (shaft_mass + 4.0 * head_mass)
                / (12.0 * mass),
        })
    }
}

impl DoublePendulumParams {
    /// Canonical club inertia (uniform shaft `m2` plus head `m_clubhead`) about the wrist.
    ///
    /// # Panics
    /// Panics when the club parameters violate the `DistalInertia::new`
    /// preconditions; call [`DoublePendulumParams::validate`] first.
    pub fn distal_inertia(&self) -> DistalInertia {
        DistalInertia::new(self.m2, self.m_clubhead, self.l2)
            .unwrap_or_else(|error| panic!("invalid double pendulum club: {error}"))
    }

    /// Validate parameters are physically reasonable
    pub fn validate(&self) -> Result<(), String> {
        if self.m1 <= 0.0 {
            return Err("m1 must be positive".to_string());
        }
        if self.m2 <= 0.0 {
            return Err("m2 must be positive".to_string());
        }
        if self.m_clubhead < 0.0 {
            return Err("m_clubhead must be non-negative".to_string());
        }
        if self.l1 <= 0.0 {
            return Err("l1 must be positive".to_string());
        }
        if self.l2 <= 0.0 {
            return Err("l2 must be positive".to_string());
        }
        if self.g < 0.0 {
            return Err("g must be non-negative".to_string());
        }
        if self.friction1 < 0.0 {
            return Err("friction1 must be non-negative".to_string());
        }
        if self.friction2 < 0.0 {
            return Err("friction2 must be non-negative".to_string());
        }
        Ok(())
    }
}

/// Triple pendulum parameters (3-DOF model).
///
/// Represents a three-segment pendulum with relative angles at each joint.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TriplePendulumParams {
    /// Masses of three segments (kg)
    pub masses: [f64; 3],
    /// Lengths of three segments (m)
    pub lengths: [f64; 3],
    /// Gravitational acceleration (m/s²)
    pub g: f64,
    /// Friction coefficients for three joints
    pub friction: [f64; 3],
}

impl TriplePendulumParams {
    /// Validate parameters are physically reasonable
    pub fn validate(&self) -> Result<(), String> {
        for (i, &mass) in self.masses.iter().enumerate() {
            if mass <= 0.0 {
                return Err(format!("masses[{}] must be positive", i));
            }
        }
        for (i, &length) in self.lengths.iter().enumerate() {
            if length <= 0.0 {
                return Err(format!("lengths[{}] must be positive", i));
            }
        }
        if self.g < 0.0 {
            return Err("g must be non-negative".to_string());
        }
        for (i, &fric) in self.friction.iter().enumerate() {
            if fric < 0.0 {
                return Err(format!("friction[{}] must be non-negative", i));
            }
        }
        Ok(())
    }
}

/// Golfer body model parameters (8-DOF with 4 constraints).
///
/// Represents a simplified upper body with hub, two arms, and club.
/// q = [θ_hub, α_rs, α_re, α_rh, α_ls, α_le, α_lh, θ_club]
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GolferParams {
    // Hub parameters
    /// Hub length from ground (m)
    pub l_hub: f64,
    /// Hub mass (kg)
    pub m_hub: f64,

    // Shoulder offsets from hub
    /// Right shoulder offset from hub center, along rotation axis (m)
    pub d_rs: f64,
    /// Left shoulder offset from hub center, along rotation axis (m)
    pub d_ls: f64,

    // Right arm
    /// Right upper arm (shoulder to elbow) length (m)
    pub l_r_upper: f64,
    /// Right upper arm mass (kg)
    pub m_r_upper: f64,
    /// Right forearm (elbow to wrist) length (m)
    pub l_r_fore: f64,
    /// Right forearm mass (kg)
    pub m_r_fore: f64,

    // Left arm
    /// Left upper arm length (m)
    pub l_l_upper: f64,
    /// Left upper arm mass (kg)
    pub m_l_upper: f64,
    /// Left forearm length (m)
    pub l_l_fore: f64,
    /// Left forearm mass (kg)
    pub m_l_fore: f64,

    // Club
    /// Club length (m)
    pub l_club: f64,
    /// Club center-of-mass mass (kg)
    pub m_club: f64,
    /// Club head mass (kg)
    pub m_clubhead: f64,
    /// Grip offset: right hand relative to club base, along shaft (m)
    pub grip_right: f64,
    /// Grip offset: left hand relative to club base, along shaft (m)
    pub grip_left: f64,

    // Physics
    /// Gravitational acceleration (m/s²)
    pub g: f64,

    // Friction (viscous damping at each actuated joint)
    /// Friction coefficients for 7 actuated joints
    /// [hub, r_shoulder, r_elbow, r_wrist, l_shoulder, l_elbow, l_wrist]
    pub friction: [f64; 7],
}

impl GolferParams {
    /// Validate parameters are physically reasonable
    pub fn validate(&self) -> Result<(), String> {
        let checks = [
            (self.l_hub > 0.0, "l_hub must be positive"),
            (self.m_hub > 0.0, "m_hub must be positive"),
            (self.d_rs >= 0.0, "d_rs must be non-negative"),
            (self.d_ls >= 0.0, "d_ls must be non-negative"),
            (self.l_r_upper > 0.0, "l_r_upper must be positive"),
            (self.m_r_upper > 0.0, "m_r_upper must be positive"),
            (self.l_r_fore > 0.0, "l_r_fore must be positive"),
            (self.m_r_fore > 0.0, "m_r_fore must be positive"),
            (self.l_l_upper > 0.0, "l_l_upper must be positive"),
            (self.m_l_upper > 0.0, "m_l_upper must be positive"),
            (self.l_l_fore > 0.0, "l_l_fore must be positive"),
            (self.m_l_fore > 0.0, "m_l_fore must be positive"),
            (self.l_club > 0.0, "l_club must be positive"),
            (self.m_club > 0.0, "m_club must be positive"),
            (self.m_clubhead > 0.0, "m_clubhead must be positive"),
            (self.grip_right >= 0.0, "grip_right must be non-negative"),
            (self.grip_left >= 0.0, "grip_left must be non-negative"),
            (self.g >= 0.0, "g must be non-negative"),
        ];

        for (valid, msg) in &checks {
            if !valid {
                return Err(msg.to_string());
            }
        }
        for (i, &fric) in self.friction.iter().enumerate() {
            if fric < 0.0 {
                return Err(format!("friction[{}] must be non-negative", i));
            }
        }
        Ok(())
    }
}

/// Result of forward kinematics for double pendulum.
#[derive(Debug, Clone, Copy)]
pub struct DoubleFKResult {
    /// Position of wrist (m)
    pub wrist: (f64, f64),
    /// Position of club tip (m)
    pub club_tip: (f64, f64),
    /// Absolute angle of arm (rad)
    pub theta1: f64,
    /// Absolute angle of club (rad)
    pub theta2: f64,
}

/// Result of forward kinematics for triple pendulum.
#[derive(Debug, Clone, Copy)]
pub struct TripleFKResult {
    /// Position of first joint (m)
    pub joint1: (f64, f64),
    /// Position of second joint (m)
    pub joint2: (f64, f64),
    /// Position of third joint/tip (m)
    pub joint3: (f64, f64),
    /// Absolute angles of three segments (rad)
    pub angles: [f64; 3],
}

/// Result of forward kinematics for golfer model.
///
/// Contains all 7 mass point positions.
#[derive(Debug, Clone, Copy)]
pub struct GolferFKResult {
    /// Hub center position (m)
    pub hub: (f64, f64),

    // Right arm endpoints
    /// Right shoulder position (m)
    pub r_shoulder: (f64, f64),
    /// Right elbow position (m)
    pub r_elbow: (f64, f64),
    /// Right wrist (hand) position (m)
    pub r_wrist: (f64, f64),

    // Left arm endpoints
    /// Left shoulder position (m)
    pub l_shoulder: (f64, f64),
    /// Left elbow position (m)
    pub l_elbow: (f64, f64),
    /// Left wrist (hand) position (m)
    pub l_wrist: (f64, f64),

    // Club
    /// Club base position (m)
    pub club_base: (f64, f64),
    /// Club center-of-mass position (m)
    pub club_com: (f64, f64),
    /// Club tip position (m)
    pub club_tip: (f64, f64),
}

/// 2D vector for simple operations
#[derive(Debug, Clone, Copy)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    pub fn new(x: f64, y: f64) -> Self {
        Vec2 { x, y }
    }

    pub fn from_polar(r: f64, theta: f64) -> Self {
        Vec2 {
            x: r * theta.sin(),
            y: -r * theta.cos(),
        }
    }

    pub fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y
    }

    pub fn cross(self, other: Self) -> f64 {
        self.x * other.y - self.y * other.x
    }

    #[allow(clippy::should_implement_trait)]
    pub fn add(self, other: Self) -> Self {
        Vec2 {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn sub(self, other: Self) -> Self {
        Vec2 {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }

    pub fn scale(self, s: f64) -> Self {
        Vec2 {
            x: self.x * s,
            y: self.y * s,
        }
    }
}

#[cfg(test)]
mod gravity_default_tests {
    use super::STANDARD_GRAVITY;

    /// Owner decision (Tools #5635): the default gravity is standard gravity.
    #[test]
    fn standard_gravity_is_9_80665() {
        assert_eq!(STANDARD_GRAVITY, 9.80665);
    }

    /// The PyO3 constructors default `g` to the one constant, not a literal.
    #[test]
    fn pyo3_signature_defaults_use_standard_gravity() {
        let src = include_str!("bindings/state.rs");
        assert!(src.contains("g=STANDARD_GRAVITY"));
        assert!(!src.contains("g=9.81"));
    }
}
