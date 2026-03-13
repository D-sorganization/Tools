//! Geometric primitives for collision detection.
//!
//! Primitive types: Sphere, OBB (Oriented Bounding Box), Capsule, Cylinder.
//! Distance computation and collision detection between primitives.
//!
//! Design by Contract:
//! - All dimensions must be positive.
//! - All positions must be finite.
//! - AABB is always valid (min <= max per component).

use nalgebra::{Matrix3, Vector3};

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Common interface for all geometric primitives.
pub trait GeometricPrimitive {
    /// Axis-aligned bounding box (min_corner, max_corner).
    fn get_aabb(&self) -> (Vector3<f64>, Vector3<f64>);

    /// Point membership test.
    fn contains_point(&self, point: &Vector3<f64>) -> bool;

    /// Support mapping for GJK: furthest point in `direction`.
    fn compute_support(&self, direction: &Vector3<f64>) -> Vector3<f64>;
}

// ---------------------------------------------------------------------------
// Sphere
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Sphere {
    pub center: Vector3<f64>,
    pub radius: f64,
}

impl Sphere {
    pub fn new(center: [f64; 3], radius: f64) -> Self {
        debug_assert!(radius > 0.0, "radius must be positive");
        Self {
            center: Vector3::new(center[0], center[1], center[2]),
            radius,
        }
    }
}

impl GeometricPrimitive for Sphere {
    fn get_aabb(&self) -> (Vector3<f64>, Vector3<f64>) {
        let r = Vector3::new(self.radius, self.radius, self.radius);
        (self.center - r, self.center + r)
    }

    fn contains_point(&self, point: &Vector3<f64>) -> bool {
        (point - self.center).norm() <= self.radius
    }

    fn compute_support(&self, direction: &Vector3<f64>) -> Vector3<f64> {
        let norm = direction.norm();
        if norm < 1e-10 {
            return self.center;
        }
        self.center + self.radius * direction / norm
    }
}

// ---------------------------------------------------------------------------
// Oriented Bounding Box
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct OrientedBox {
    pub center: Vector3<f64>,
    pub half_extents: Vector3<f64>,
    pub rotation: Matrix3<f64>,
}

impl OrientedBox {
    pub fn new(center: [f64; 3], half_extents: [f64; 3], rotation: Matrix3<f64>) -> Self {
        debug_assert!(
            half_extents.iter().all(|&h| h > 0.0),
            "half_extents must be positive"
        );
        Self {
            center: Vector3::new(center[0], center[1], center[2]),
            half_extents: Vector3::new(half_extents[0], half_extents[1], half_extents[2]),
            rotation,
        }
    }

    pub fn axis_aligned(center: [f64; 3], half_extents: [f64; 3]) -> Self {
        Self::new(center, half_extents, Matrix3::identity())
    }

    fn get_corners(&self) -> [Vector3<f64>; 8] {
        let h = &self.half_extents;
        let signs: [[f64; 3]; 8] = [
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
        ];
        let mut corners = [Vector3::zeros(); 8];
        for (i, s) in signs.iter().enumerate() {
            let local = Vector3::new(s[0] * h[0], s[1] * h[1], s[2] * h[2]);
            corners[i] = self.rotation * local + self.center;
        }
        corners
    }
}

impl GeometricPrimitive for OrientedBox {
    fn get_aabb(&self) -> (Vector3<f64>, Vector3<f64>) {
        let corners = self.get_corners();
        let mut min_c = corners[0];
        let mut max_c = corners[0];
        for c in &corners[1..] {
            min_c = min_c.inf(c);
            max_c = max_c.sup(c);
        }
        (min_c, max_c)
    }

    fn contains_point(&self, point: &Vector3<f64>) -> bool {
        let local = self.rotation.transpose() * (point - self.center);
        local[0].abs() <= self.half_extents[0]
            && local[1].abs() <= self.half_extents[1]
            && local[2].abs() <= self.half_extents[2]
    }

    fn compute_support(&self, direction: &Vector3<f64>) -> Vector3<f64> {
        let local_dir = self.rotation.transpose() * direction;
        let mut local_support = Vector3::zeros();
        for i in 0..3 {
            local_support[i] = if local_dir[i].abs() < 1e-10 {
                self.half_extents[i]
            } else {
                local_dir[i].signum() * self.half_extents[i]
            };
        }
        self.rotation * local_support + self.center
    }
}

// ---------------------------------------------------------------------------
// Capsule
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Capsule {
    pub point_a: Vector3<f64>,
    pub point_b: Vector3<f64>,
    pub radius: f64,
}

impl Capsule {
    pub fn new(point_a: [f64; 3], point_b: [f64; 3], radius: f64) -> Self {
        debug_assert!(radius > 0.0, "radius must be positive");
        Self {
            point_a: Vector3::new(point_a[0], point_a[1], point_a[2]),
            point_b: Vector3::new(point_b[0], point_b[1], point_b[2]),
            radius,
        }
    }

    pub fn length(&self) -> f64 {
        (self.point_b - self.point_a).norm()
    }

    pub fn center(&self) -> Vector3<f64> {
        (self.point_a + self.point_b) / 2.0
    }

    fn closest_point_on_segment(&self, point: &Vector3<f64>) -> Vector3<f64> {
        let ab = self.point_b - self.point_a;
        let t = (point - self.point_a).dot(&ab) / (ab.dot(&ab) + 1e-10);
        let t = t.clamp(0.0, 1.0);
        self.point_a + t * ab
    }
}

impl GeometricPrimitive for Capsule {
    fn get_aabb(&self) -> (Vector3<f64>, Vector3<f64>) {
        let r = Vector3::new(self.radius, self.radius, self.radius);
        let min_c = self.point_a.inf(&self.point_b) - r;
        let max_c = self.point_a.sup(&self.point_b) + r;
        (min_c, max_c)
    }

    fn contains_point(&self, point: &Vector3<f64>) -> bool {
        let closest = self.closest_point_on_segment(point);
        (point - closest).norm() <= self.radius
    }

    fn compute_support(&self, direction: &Vector3<f64>) -> Vector3<f64> {
        let norm = direction.norm();
        if norm < 1e-10 {
            return self.point_a;
        }
        let d = direction / norm;
        if d.dot(&(self.point_b - self.point_a)) >= 0.0 {
            self.point_b + self.radius * d
        } else {
            self.point_a + self.radius * d
        }
    }
}

// ---------------------------------------------------------------------------
// Cylinder
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Cylinder {
    pub center: Vector3<f64>,
    pub radius: f64,
    pub height: f64,
    pub axis: Vector3<f64>,
}

impl Cylinder {
    pub fn new(center: [f64; 3], radius: f64, height: f64, axis: [f64; 3]) -> Self {
        debug_assert!(radius > 0.0, "radius must be positive");
        debug_assert!(height > 0.0, "height must be positive");
        let a = Vector3::new(axis[0], axis[1], axis[2]);
        let norm = a.norm();
        debug_assert!(norm > 1e-10, "axis must be non-zero");
        Self {
            center: Vector3::new(center[0], center[1], center[2]),
            radius,
            height,
            axis: a / norm,
        }
    }

    pub fn half_height(&self) -> f64 {
        self.height / 2.0
    }
}

impl GeometricPrimitive for Cylinder {
    fn get_aabb(&self) -> (Vector3<f64>, Vector3<f64>) {
        let hh = self.half_height();
        let top = self.center + hh * self.axis;
        let bottom = self.center - hh * self.axis;
        let r = Vector3::new(self.radius, self.radius, self.radius);
        let min_c = top.inf(&bottom) - r;
        let max_c = top.sup(&bottom) + r;
        (min_c, max_c)
    }

    fn contains_point(&self, point: &Vector3<f64>) -> bool {
        let to_point = point - self.center;
        let along_axis = to_point.dot(&self.axis);
        if along_axis.abs() > self.half_height() {
            return false;
        }
        let perp = to_point - along_axis * self.axis;
        perp.norm() <= self.radius
    }

    fn compute_support(&self, direction: &Vector3<f64>) -> Vector3<f64> {
        let norm = direction.norm();
        if norm < 1e-10 {
            return self.center;
        }
        let d = direction / norm;
        let d_along = d.dot(&self.axis) * self.axis;
        let d_perp = d - d_along;

        let axis_support = if d.dot(&self.axis) >= 0.0 {
            self.center + self.half_height() * self.axis
        } else {
            self.center - self.half_height() * self.axis
        };

        let perp_norm = d_perp.norm();
        if perp_norm > 1e-10 {
            axis_support + self.radius * d_perp / perp_norm
        } else {
            axis_support
        }
    }
}

// ---------------------------------------------------------------------------
// Distance computation
// ---------------------------------------------------------------------------

/// Result of a distance query between two primitives.
pub struct DistanceResult {
    pub distance: f64,
    pub point_a: Vector3<f64>,
    pub point_b: Vector3<f64>,
}

/// Sphere-sphere distance.
pub fn sphere_sphere_distance(a: &Sphere, b: &Sphere) -> DistanceResult {
    let diff = b.center - a.center;
    let center_dist = diff.norm();

    if center_dist < 1e-10 {
        return DistanceResult {
            distance: -(a.radius + b.radius),
            point_a: a.center,
            point_b: b.center,
        };
    }

    let direction = diff / center_dist;
    DistanceResult {
        distance: center_dist - a.radius - b.radius,
        point_a: a.center + a.radius * direction,
        point_b: b.center - b.radius * direction,
    }
}

/// Sphere-capsule distance.
pub fn sphere_capsule_distance(sphere: &Sphere, capsule: &Capsule) -> DistanceResult {
    let closest = capsule.closest_point_on_segment(&sphere.center);
    let diff = sphere.center - closest;
    let center_dist = diff.norm();

    if center_dist < 1e-10 {
        return DistanceResult {
            distance: -(sphere.radius + capsule.radius),
            point_a: sphere.center,
            point_b: closest,
        };
    }

    let direction = diff / center_dist;
    DistanceResult {
        distance: center_dist - sphere.radius - capsule.radius,
        point_a: sphere.center - sphere.radius * direction,
        point_b: closest + capsule.radius * direction,
    }
}

/// Closest points between two line segments.
pub fn closest_points_segments(
    a0: &Vector3<f64>,
    a1: &Vector3<f64>,
    b0: &Vector3<f64>,
    b1: &Vector3<f64>,
) -> (Vector3<f64>, Vector3<f64>) {
    let d1 = a1 - a0;
    let d2 = b1 - b0;
    let r = a0 - b0;

    let a = d1.dot(&d1);
    let e = d2.dot(&d2);
    let f = d2.dot(&r);

    if a < 1e-10 && e < 1e-10 {
        return (*a0, *b0);
    }

    let (s, t);
    if a < 1e-10 {
        s = 0.0;
        t = (f / e).clamp(0.0, 1.0);
    } else {
        let c = d1.dot(&r);
        if e < 1e-10 {
            t = 0.0;
            s = (-c / a).clamp(0.0, 1.0);
        } else {
            let b_coef = d1.dot(&d2);
            let denom = a * e - b_coef * b_coef;

            let s_raw = if denom.abs() > 1e-10 {
                ((b_coef * f - c * e) / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };

            let t_raw = (b_coef * s_raw + f) / e;

            if t_raw < 0.0 {
                t = 0.0;
                s = (-c / a).clamp(0.0, 1.0);
            } else if t_raw > 1.0 {
                t = 1.0;
                s = ((b_coef - c) / a).clamp(0.0, 1.0);
            } else {
                s = s_raw;
                t = t_raw;
            }
        }
    }

    (a0 + s * d1, b0 + t * d2)
}

/// Capsule-capsule distance.
pub fn capsule_capsule_distance(a: &Capsule, b: &Capsule) -> DistanceResult {
    let (closest_a, closest_b) =
        closest_points_segments(&a.point_a, &a.point_b, &b.point_a, &b.point_b);

    let diff = closest_b - closest_a;
    let center_dist = diff.norm();

    if center_dist < 1e-10 {
        return DistanceResult {
            distance: -(a.radius + b.radius),
            point_a: closest_a,
            point_b: closest_b,
        };
    }

    let direction = diff / center_dist;
    DistanceResult {
        distance: center_dist - a.radius - b.radius,
        point_a: closest_a + a.radius * direction,
        point_b: closest_b - b.radius * direction,
    }
}

/// Check if two spheres collide (with optional margin).
pub fn check_collision_spheres(a: &Sphere, b: &Sphere, margin: f64) -> bool {
    debug_assert!(margin >= 0.0, "margin must be non-negative");
    sphere_sphere_distance(a, b).distance <= margin
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Matrix3;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // --- Sphere ---

    #[test]
    fn test_sphere_aabb() {
        let s = Sphere::new([1.0, 2.0, 3.0], 0.5);
        let (min_c, max_c) = s.get_aabb();
        assert!(approx_eq(min_c[0], 0.5, 1e-12));
        assert!(approx_eq(max_c[2], 3.5, 1e-12));
    }

    #[test]
    fn test_sphere_contains_center() {
        let s = Sphere::new([0.0, 0.0, 0.0], 1.0);
        assert!(s.contains_point(&Vector3::zeros()));
        assert!(s.contains_point(&Vector3::new(0.5, 0.0, 0.0)));
        assert!(!s.contains_point(&Vector3::new(1.5, 0.0, 0.0)));
    }

    #[test]
    fn test_sphere_support() {
        let s = Sphere::new([0.0, 0.0, 0.0], 1.0);
        let sup = s.compute_support(&Vector3::new(1.0, 0.0, 0.0));
        assert!(approx_eq(sup[0], 1.0, 1e-10));
    }

    // --- Box ---

    #[test]
    fn test_box_contains_center() {
        let b = OrientedBox::axis_aligned([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert!(b.contains_point(&Vector3::zeros()));
        assert!(b.contains_point(&Vector3::new(0.5, 0.5, 0.5)));
        assert!(!b.contains_point(&Vector3::new(1.5, 0.0, 0.0)));
    }

    #[test]
    fn test_box_aabb_rotated() {
        // 45° rotation about Z should expand the AABB
        let angle = std::f64::consts::FRAC_PI_4;
        let r = Matrix3::new(
            angle.cos(),
            -angle.sin(),
            0.0,
            angle.sin(),
            angle.cos(),
            0.0,
            0.0,
            0.0,
            1.0,
        );
        let b = OrientedBox::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], r);
        let (min_c, max_c) = b.get_aabb();
        // Rotated box should have larger AABB than axis-aligned
        assert!(max_c[0] > 1.0);
        assert!(min_c[0] < -1.0);
    }

    // --- Capsule ---

    #[test]
    fn test_capsule_contains() {
        let c = Capsule::new([0.0, 0.0, -1.0], [0.0, 0.0, 1.0], 0.5);
        assert!(c.contains_point(&Vector3::zeros()));
        assert!(c.contains_point(&Vector3::new(0.3, 0.0, 0.0)));
        assert!(!c.contains_point(&Vector3::new(1.0, 0.0, 0.0)));
    }

    #[test]
    fn test_capsule_length() {
        let c = Capsule::new([0.0, 0.0, 0.0], [0.0, 0.0, 2.0], 0.1);
        assert!(approx_eq(c.length(), 2.0, 1e-12));
    }

    // --- Cylinder ---

    #[test]
    fn test_cylinder_contains() {
        let cyl = Cylinder::new([0.0, 0.0, 0.0], 1.0, 2.0, [0.0, 0.0, 1.0]);
        assert!(cyl.contains_point(&Vector3::zeros()));
        assert!(cyl.contains_point(&Vector3::new(0.5, 0.0, 0.5)));
        assert!(!cyl.contains_point(&Vector3::new(0.0, 0.0, 2.0))); // outside height
        assert!(!cyl.contains_point(&Vector3::new(1.5, 0.0, 0.0))); // outside radius
    }

    // --- Distance ---

    #[test]
    fn test_sphere_sphere_separated() {
        let a = Sphere::new([0.0, 0.0, 0.0], 1.0);
        let b = Sphere::new([3.0, 0.0, 0.0], 1.0);
        let result = sphere_sphere_distance(&a, &b);
        assert!(approx_eq(result.distance, 1.0, 1e-10));
    }

    #[test]
    fn test_sphere_sphere_overlapping() {
        let a = Sphere::new([0.0, 0.0, 0.0], 1.0);
        let b = Sphere::new([1.0, 0.0, 0.0], 1.0);
        let result = sphere_sphere_distance(&a, &b);
        assert!(result.distance < 0.0);
    }

    #[test]
    fn test_sphere_capsule_distance() {
        let s = Sphere::new([2.0, 0.0, 0.0], 0.5);
        let c = Capsule::new([0.0, 0.0, -1.0], [0.0, 0.0, 1.0], 0.3);
        let result = sphere_capsule_distance(&s, &c);
        // Distance should be ~2.0 - 0.5 - 0.3 = 1.2
        assert!(approx_eq(result.distance, 1.2, 1e-10));
    }

    #[test]
    fn test_capsule_capsule_parallel() {
        let a = Capsule::new([0.0, 0.0, -1.0], [0.0, 0.0, 1.0], 0.1);
        let b = Capsule::new([1.0, 0.0, -1.0], [1.0, 0.0, 1.0], 0.1);
        let result = capsule_capsule_distance(&a, &b);
        assert!(approx_eq(result.distance, 0.8, 1e-10));
    }

    #[test]
    fn test_collision_check() {
        let a = Sphere::new([0.0, 0.0, 0.0], 1.0);
        let b = Sphere::new([1.5, 0.0, 0.0], 1.0);
        assert!(check_collision_spheres(&a, &b, 0.0)); // overlapping
        assert!(!check_collision_spheres(
            &Sphere::new([0.0, 0.0, 0.0], 1.0),
            &Sphere::new([3.0, 0.0, 0.0], 1.0),
            0.0
        )); // separated
    }

    #[test]
    fn test_closest_points_parallel_segments() {
        let a0 = Vector3::new(0.0, 0.0, 0.0);
        let a1 = Vector3::new(1.0, 0.0, 0.0);
        let b0 = Vector3::new(0.0, 1.0, 0.0);
        let b1 = Vector3::new(1.0, 1.0, 0.0);
        let (pa, pb) = closest_points_segments(&a0, &a1, &b0, &b1);
        assert!(approx_eq((pb - pa).norm(), 1.0, 1e-10));
    }
}
