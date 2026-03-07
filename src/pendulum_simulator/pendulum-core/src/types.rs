//! Shared types and parameter structures for physics models.

use nalgebra::{SMatrix, SVector};

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
    /// Mass of club segment (kg)
    pub m2: f64,
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

impl DoublePendulumParams {
    /// Validate parameters are physically reasonable
    pub fn validate(&self) -> Result<(), String> {
        if self.m1 <= 0.0 {
            return Err("m1 must be positive".to_string());
        }
        if self.m2 <= 0.0 {
            return Err("m2 must be positive".to_string());
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

    pub fn add(self, other: Self) -> Self {
        Vec2 {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }

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
