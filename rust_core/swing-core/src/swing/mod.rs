//! Swing dynamics domain module.
//!
//! - [`pendulum`] — double-pendulum equations of motion (mass matrix,
//!   Coriolis/centripetal, gravity, damping) and RK4 stepping.
//! - [`plane`] — swing-plane orientation (three sequential intrinsic tilts)
//!   and projection of world gravity into the swing plane.
//!
//! The pendulum EOM is planar; plane orientation enters exclusively through
//! the projected in-plane gravity 2-vector produced by
//! [`plane::in_plane_gravity`]. This keeps the dynamics core independent of
//! the world-frame convention (LoD: the EOM never sees the 3-D pose).

pub mod pendulum;
pub mod plane;
