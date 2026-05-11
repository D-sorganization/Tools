use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::PyType;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::f64;

#[derive(Clone, Debug)]
#[pyclass]
pub struct Obstacle {
    pub obs_type: i32, // 0=sphere, 1=cube
    pub position: [f64; 3],
    pub size: f64,
}

#[pymethods]
impl Obstacle {
    #[new]
    pub fn new(obs_type: i32, position: [f64; 3], size: f64) -> Self {
        Self {
            obs_type,
            position,
            size,
        }
    }
}

impl Obstacle {
    pub fn distance_to_surface(&self, point: &[f64; 3]) -> f64 {
        if self.obs_type == 0 {
            let dx = point[0] - self.position[0];
            let dy = point[1] - self.position[1];
            let dz = point[2] - self.position[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            dist - self.size
        } else {
            let half_size = self.size / 2.0;
            let mut outside_dist_sq = 0.0;
            let mut inside_max = f64::NEG_INFINITY;

            for i in 0..3 {
                let delta = (point[i] - self.position[i]).abs() - half_size;
                if delta > 0.0 {
                    outside_dist_sq += delta * delta;
                }
                if delta > inside_max {
                    inside_max = delta;
                }
            }

            if outside_dist_sq > 0.0 {
                outside_dist_sq.sqrt()
            } else {
                inside_max
            }
        }
    }
}

#[pyclass]
pub struct RRTPlanner {
    bounds: [f64; 6],
    max_iterations: usize,
    step_size: f64,
    goal_radius: f64,
    goal_bias: f64,
    rng: StdRng,
}

#[pymethods]
impl RRTPlanner {
    #[new]
    #[pyo3(signature = (bounds, max_iterations=5000, seed=None))]
    pub fn new(bounds: [f64; 6], max_iterations: usize, seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        Self {
            bounds,
            max_iterations,
            step_size: 0.05,
            goal_radius: 0.1,
            goal_bias: 0.2,
            rng,
        }
    }

    pub fn plan_path<'py>(
        &mut self,
        py: Python<'py>,
        start: [f64; 3],
        goal: [f64; 3],
        obstacles: Vec<Obstacle>,
    ) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
        if self.check_collision(&start, &obstacles) || self.check_collision(&goal, &obstacles) {
            return Ok(None);
        }

        // Tree structure: each node is [x, y, z, parent_index]
        let mut nodes: Vec<[f64; 4]> = Vec::with_capacity(self.max_iterations);
        nodes.push([start[0], start[1], start[2], -1.0]);

        for _ in 0..self.max_iterations {
            let sample = self.sample_point(&goal);
            let nearest_idx = self.nearest_node_index(&nodes, &sample);
            let nearest_point = [
                nodes[nearest_idx][0],
                nodes[nearest_idx][1],
                nodes[nearest_idx][2],
            ];

            let new_point = self.steer(&nearest_point, &sample);

            if self.check_collision(&new_point, &obstacles) {
                continue;
            }

            if !self.segment_is_collision_free(&nearest_point, &new_point, &obstacles) {
                continue;
            }

            nodes.push([new_point[0], new_point[1], new_point[2], nearest_idx as f64]);

            let dist_to_goal = Self::distance(&new_point, &goal);
            if dist_to_goal <= self.goal_radius {
                let path = self.extract_path(&nodes, nodes.len() - 1);
                let flat_path: Vec<f64> = path.into_iter().flatten().collect();
                let rows = flat_path.len() / 3;
                let py_array = PyArray2::from_vec_bound(py, flat_path).reshape([rows, 3])?;
                return Ok(Some(py_array));
            }
        }

        Ok(None)
    }
}

impl RRTPlanner {
    fn distance(a: &[f64; 3], b: &[f64; 3]) -> f64 {
        let dx = a[0] - b[0];
        let dy = a[1] - b[1];
        let dz = a[2] - b[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    fn sample_point(&mut self, goal: &[f64; 3]) -> [f64; 3] {
        if self.rng.gen::<f64>() < self.goal_bias {
            *goal
        } else {
            [
                self.rng.gen_range(self.bounds[0]..self.bounds[1]),
                self.rng.gen_range(self.bounds[2]..self.bounds[3]),
                self.rng.gen_range(self.bounds[4]..self.bounds[5]),
            ]
        }
    }

    fn nearest_node_index(&self, nodes: &[[f64; 4]], sample: &[f64; 3]) -> usize {
        let mut min_dist = f64::MAX;
        let mut min_idx = 0;
        for (i, node) in nodes.iter().enumerate() {
            let pt = [node[0], node[1], node[2]];
            let dist = Self::distance(&pt, sample);
            if dist < min_dist {
                min_dist = dist;
                min_idx = i;
            }
        }
        min_idx
    }

    fn steer(&self, origin: &[f64; 3], target: &[f64; 3]) -> [f64; 3] {
        let dir = [
            target[0] - origin[0],
            target[1] - origin[1],
            target[2] - origin[2],
        ];
        let dist = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
        if dist == 0.0 {
            *origin
        } else {
            let ratio = self.step_size / dist;
            [
                origin[0] + dir[0] * ratio,
                origin[1] + dir[1] * ratio,
                origin[2] + dir[2] * ratio,
            ]
        }
    }

    fn check_collision(&self, point: &[f64; 3], obstacles: &[Obstacle]) -> bool {
        for obs in obstacles {
            if obs.distance_to_surface(point) <= 0.0 {
                return true;
            }
        }
        false
    }

    fn segment_is_collision_free(&self, start: &[f64; 3], end: &[f64; 3], obstacles: &[Obstacle]) -> bool {
        let dist = Self::distance(start, end);
        let step = (self.step_size / 2.0).max(1e-6);
        let samples = (dist / step).ceil() as usize;
        let samples = samples.max(2);

        let dir = [
            end[0] - start[0],
            end[1] - start[1],
            end[2] - start[2],
        ];

        for i in 0..samples {
            let fraction = i as f64 / (samples - 1) as f64;
            let probe = [
                start[0] + fraction * dir[0],
                start[1] + fraction * dir[1],
                start[2] + fraction * dir[2],
            ];
            if self.check_collision(&probe, obstacles) {
                return false;
            }
        }
        true
    }

    fn extract_path(&self, nodes: &[[f64; 4]], goal_idx: usize) -> Vec<[f64; 3]> {
        let mut path = Vec::new();
        let mut current_idx = goal_idx as i32;

        while current_idx != -1 {
            let node = nodes[current_idx as usize];
            path.push([node[0], node[1], node[2]]);
            current_idx = node[3] as i32;
        }

        path.reverse();
        path
    }
}
