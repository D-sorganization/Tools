# math-primitives

SE3/SO3 math kernel and geometric collision primitives for the fleet.

## Modules

| Module           | Purpose                                                                                            |
| ---------------- | -------------------------------------------------------------------------------------------------- |
| `quaternion.rs`  | Unit quaternion type, Hamilton product, SLERP, inverse, normalize                                  |
| `rotation.rs`    | Euler ↔ rotation matrix ↔ quaternion ↔ axis-angle conversions                                      |
| `transform.rs`   | `Pose6DOF` and `Transform6DOF` structs with compose, inverse, interpolate                          |
| `geometry.rs`    | Sphere, OrientedBox, Capsule, Cylinder with AABB, containment, support mapping, distance/collision |
| `py_bindings.rs` | PyO3 Python bindings (feature-gated behind `python`)                                               |

## Quick Start

```rust
use math_primitives::{euler_to_rotation_matrix, quaternion_multiply, Pose6DOF, Sphere};

// Rotation conversions
let rotmat = euler_to_rotation_matrix(&[0.1, 0.2, 0.3]);

// Quaternion operations
let q1 = math_primitives::quaternion::quat(1.0, 0.0, 0.0, 0.0);
let q2 = math_primitives::quaternion::quat(0.707, 0.707, 0.0, 0.0);
let q3 = quaternion_multiply(&q1, &q2);

// Pose composition
let p1 = Pose6DOF::new([1.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
let p2 = Pose6DOF::new([0.0, 1.0, 0.0], [0.0, 0.0, std::f64::consts::FRAC_PI_2]);
let p3 = p1.compose(&p2);

// Collision detection
let s1 = Sphere::new([0.0, 0.0, 0.0], 1.0);
let s2 = Sphere::new([3.0, 0.0, 0.0], 1.0);
let result = math_primitives::sphere_sphere_distance(&s1, &s2);
println!("Distance: {}", result.distance);
```

## Feature Flags

| Flag     | Description                              |
| -------- | ---------------------------------------- |
| `python` | Enables PyO3 bindings for Python interop |

## Design by Contract

- Quaternions are stored as `[w, x, y, z]` (scalar-first).
- Euler angles use ZYX convention (yaw-pitch-roll).
- All rotation matrices are SO(3) members (orthonormal, det=+1).
- All geometric primitive dimensions must be positive.

## Testing

```bash
cargo test                          # Run all 39 tests
cargo clippy                        # Lint check
cargo fmt -- --check                # Format check
```
