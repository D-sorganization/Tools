//! Math Primitives: SE3/SO3 math kernel + geometric collision primitives.
//!
//! Core rotation/orientation algebra used across the fleet:
//! - Quaternion operations (multiply, inverse, slerp, normalize)
//! - Euler angle ↔ rotation matrix ↔ quaternion conversions
//! - Axis-angle to rotation matrix (Rodrigues formula)
//! - Pose6DOF and Transform6DOF types
//! - Geometric primitives (Sphere, Box, Capsule, Cylinder) with collision detection
//!
//! Design by Contract:
//! - Quaternions are stored as [w, x, y, z].
//! - Euler angles use ZYX convention (yaw-pitch-roll).
//! - All rotation matrices are SO(3) (orthonormal, det=+1).
//! - All primitive dimensions must be positive.

pub mod geometry;
pub mod quaternion;
pub mod rotation;
pub mod transform;

pub use geometry::{
    capsule_capsule_distance, check_collision_spheres, closest_points_segments,
    sphere_capsule_distance, sphere_sphere_distance, Capsule, Cylinder, DistanceResult,
    GeometricPrimitive, OrientedBox, Sphere,
};
pub use quaternion::{
    quaternion_inverse, quaternion_multiply, quaternion_normalize, slerp, Quaternion,
};
pub use rotation::{
    axis_angle_to_rotation_matrix, euler_to_quaternion, euler_to_rotation_matrix,
    quaternion_to_euler, quaternion_to_rotation_matrix, rotation_matrix_to_euler,
    rotation_matrix_to_quaternion,
};
pub use transform::{Pose6DOF, Transform6DOF};
