//! CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer.
//!
//! Implements the (μ/μ_w, λ)-CMA-ES algorithm for optimizing torque
//! polynomial coefficients. Designed for parallel evaluation via rayon.
//!
//! # Design by Contract
//! - `population_size` must be >= 4 (minimum for stable covariance updates).
//! - `dimension` must be >= 1.
//! - `max_iterations` must be >= 1.
//! - Objective function values must be finite.
//!
//! # References
//! Hansen, N. (2016). The CMA Evolution Strategy: A Tutorial.

#![allow(clippy::needless_range_loop)]

use std::f64;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Configuration for the CMA-ES optimizer.
#[derive(Debug, Clone)]
pub struct CmaEsConfig {
    /// Number of candidate solutions per generation (λ).
    pub population_size: usize,
    /// Maximum number of generations.
    pub max_iterations: usize,
    /// Initial step size (σ₀).
    pub initial_sigma: f64,
    /// Target fitness (stop early if reached).
    pub target_fitness: Option<f64>,
    /// Tolerance for fitness stagnation.
    pub fitness_tolerance: f64,
}

impl Default for CmaEsConfig {
    fn default() -> Self {
        Self {
            population_size: 0, // 0 = auto-select based on dimension
            max_iterations: 500,
            initial_sigma: 0.3,
            target_fitness: None,
            fitness_tolerance: 1e-12,
        }
    }
}

/// Result of a CMA-ES optimization run.
#[derive(Debug, Clone)]
pub struct CmaEsResult {
    /// Best solution found.
    pub best_solution: Vec<f64>,
    /// Best fitness value (lower is better).
    pub best_fitness: f64,
    /// Fitness history (best per generation).
    pub fitness_history: Vec<f64>,
    /// Number of generations completed.
    pub generations: usize,
    /// Total function evaluations.
    pub evaluations: usize,
}

/// CMA-ES optimizer state.
pub struct CmaEsState {
    dim: usize,
    lambda: usize,     // population size
    mu: usize,         // parent count (λ/2)
    weights: Vec<f64>, // recombination weights
    mu_eff: f64,       // effective μ
    // Step size control
    sigma: f64,
    // Distribution parameters
    mean: Vec<f64>,
    // Covariance matrix (stored as flat row-major)
    cov: Vec<f64>,
    // Evolution paths
    p_sigma: Vec<f64>,
    p_c: Vec<f64>,
    // Adaptation parameters
    c_sigma: f64,
    d_sigma: f64,
    c_c: f64,
    c_1: f64,
    c_mu: f64,
    // Expected length of N(0,I)
    chi_n: f64,
    // Generation counter
    generation: usize,
}

impl CmaEsState {
    /// Initialize a new CMA-ES state.
    ///
    /// # Preconditions
    /// - `dim >= 1`
    /// - `initial_mean.len() == dim`
    /// - `config.initial_sigma > 0`
    pub fn new(dim: usize, initial_mean: &[f64], config: &CmaEsConfig) -> Self {
        assert!(dim >= 1, "Dimension must be >= 1, got {}", dim);
        assert_eq!(initial_mean.len(), dim, "Mean length must match dimension");
        assert!(config.initial_sigma > 0.0, "Initial sigma must be positive");

        // Population size: default = 4 + floor(3 * ln(dim))
        let lambda = if config.population_size == 0 {
            (4.0 + (3.0 * (dim as f64).ln()).floor()) as usize
        } else {
            config.population_size
        };
        assert!(lambda >= 4, "Population size must be >= 4, got {}", lambda);

        let mu = lambda / 2;

        // Recombination weights (log-scale)
        let raw_weights: Vec<f64> = (0..mu)
            .map(|i| (mu as f64 + 0.5).ln() - ((i + 1) as f64).ln())
            .collect();
        let w_sum: f64 = raw_weights.iter().sum();
        let weights: Vec<f64> = raw_weights.iter().map(|w| w / w_sum).collect();

        let mu_eff = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();

        // Adaptation parameters (Hansen defaults)
        let c_sigma = (mu_eff + 2.0) / (dim as f64 + mu_eff + 5.0);
        let d_sigma =
            1.0 + 2.0 * f64::max(0.0, ((mu_eff - 1.0) / (dim as f64 + 1.0)).sqrt() - 1.0) + c_sigma;
        let c_c = (4.0 + mu_eff / dim as f64) / (dim as f64 + 4.0 + 2.0 * mu_eff / dim as f64);
        let c_1 = 2.0 / ((dim as f64 + 1.3).powi(2) + mu_eff);
        let c_mu_val = f64::min(
            1.0 - c_1,
            2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((dim as f64 + 2.0).powi(2) + mu_eff),
        );

        let chi_n = (dim as f64).sqrt()
            * (1.0 - 1.0 / (4.0 * dim as f64) + 1.0 / (21.0 * (dim as f64).powi(2)));

        // Initialize covariance as identity
        let mut cov = vec![0.0; dim * dim];
        for i in 0..dim {
            cov[i * dim + i] = 1.0;
        }

        Self {
            dim,
            lambda,
            mu,
            weights,
            mu_eff,
            sigma: config.initial_sigma,
            mean: initial_mean.to_vec(),
            cov,
            p_sigma: vec![0.0; dim],
            p_c: vec![0.0; dim],
            c_sigma,
            d_sigma,
            c_c,
            c_1,
            c_mu: c_mu_val,
            chi_n,
            generation: 0,
        }
    }

    /// Sample `lambda` candidate solutions from the current distribution.
    ///
    /// # Postconditions
    /// - Returns `lambda` vectors, each of length `dim`.
    pub fn sample_population(&self) -> Vec<Vec<f64>> {
        use rand::Rng;
        use rand_distr::StandardNormal;

        let mut rng = rand::rng();
        let sqrt_cov = self.cholesky_decompose();

        let mut population = Vec::with_capacity(self.lambda);
        for _ in 0..self.lambda {
            // z ~ N(0, I)
            let z: Vec<f64> = (0..self.dim).map(|_| rng.sample(StandardNormal)).collect();
            // x = mean + sigma * L * z
            let mut x = self.mean.clone();
            for i in 0..self.dim {
                let mut lz = 0.0;
                for j in 0..=i {
                    lz += sqrt_cov[i * self.dim + j] * z[j];
                }
                x[i] += self.sigma * lz;
            }
            population.push(x);
        }

        assert_eq!(population.len(), self.lambda);
        population
    }

    /// Update the distribution parameters given sorted (fitness, solution) pairs.
    ///
    /// # Preconditions
    /// - `ranked` is sorted by fitness (ascending, best first).
    /// - `ranked.len() >= mu`.
    pub fn update(&mut self, ranked: &[(f64, Vec<f64>)]) {
        assert!(
            ranked.len() >= self.mu,
            "Need at least {} ranked solutions, got {}",
            self.mu,
            ranked.len()
        );

        let n = self.dim;

        // Weighted mean of the mu best solutions
        let mut new_mean = vec![0.0; n];
        for i in 0..self.mu {
            for j in 0..n {
                new_mean[j] += self.weights[i] * ranked[i].1[j];
            }
        }

        // Mean shift (for path updates)
        let mean_shift: Vec<f64> = new_mean
            .iter()
            .zip(self.mean.iter())
            .map(|(new, old)| (new - old) / self.sigma)
            .collect();

        // Update evolution path for sigma (p_sigma)
        let sqrt_cov_inv = self.cholesky_inv();
        let invsqrt_times_shift: Vec<f64> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| sqrt_cov_inv[i * n + j] * mean_shift[j])
                    .sum::<f64>()
            })
            .collect();

        let cs_complement = (1.0 - self.c_sigma).sqrt();
        let cs_scale = (self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff).sqrt();
        for i in 0..n {
            self.p_sigma[i] = cs_complement * self.p_sigma[i] + cs_scale * invsqrt_times_shift[i];
        }

        // Update sigma
        let p_sigma_norm: f64 = self.p_sigma.iter().map(|x| x * x).sum::<f64>().sqrt();
        self.sigma *= ((self.c_sigma / self.d_sigma) * (p_sigma_norm / self.chi_n - 1.0)).exp();

        // Heaviside function for p_c update
        let h_sigma = if p_sigma_norm
            / (1.0 - (1.0 - self.c_sigma).powi(2 * (self.generation as i32 + 1))).sqrt()
            < (1.4 + 2.0 / (n as f64 + 1.0)) * self.chi_n
        {
            1.0
        } else {
            0.0
        };

        // Update evolution path for covariance (p_c)
        let cc_complement = (1.0 - self.c_c).sqrt();
        let cc_scale = h_sigma * (self.c_c * (2.0 - self.c_c) * self.mu_eff).sqrt();
        for i in 0..n {
            self.p_c[i] = cc_complement * self.p_c[i] + cc_scale * mean_shift[i];
        }

        // Rank-one update + rank-mu update of covariance
        let delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c);
        for i in 0..n {
            for j in 0..n {
                let idx = i * n + j;
                // Rank-one
                let rank1 = self.c_1 * self.p_c[i] * self.p_c[j];
                // Rank-mu
                let mut rank_mu = 0.0;
                for k in 0..self.mu {
                    let yi = (ranked[k].1[i] - self.mean[i]) / self.sigma;
                    let yj = (ranked[k].1[j] - self.mean[j]) / self.sigma;
                    rank_mu += self.weights[k] * yi * yj;
                }
                rank_mu *= self.c_mu;

                self.cov[idx] = (1.0 - self.c_1 - self.c_mu + delta_h * self.c_1) * self.cov[idx]
                    + rank1
                    + rank_mu;
            }
        }

        self.mean = new_mean;
        self.generation += 1;
    }

    /// Cholesky decomposition of the covariance matrix (lower triangular L).
    fn cholesky_decompose(&self) -> Vec<f64> {
        let n = self.dim;
        let mut l = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += l[i * n + k] * l[j * n + k];
                }
                if i == j {
                    let val = self.cov[i * n + i] - sum;
                    l[i * n + j] = if val > 0.0 { val.sqrt() } else { 1e-10 };
                } else {
                    l[i * n + j] = (self.cov[i * n + j] - sum) / l[j * n + j].max(1e-20);
                }
            }
        }
        l
    }

    /// Inverse of Cholesky factor (for p_sigma update).
    fn cholesky_inv(&self) -> Vec<f64> {
        let n = self.dim;
        let l = self.cholesky_decompose();
        let mut inv = vec![0.0; n * n];

        // Forward substitution on identity columns
        for col in 0..n {
            for i in 0..n {
                let mut val = if i == col { 1.0 } else { 0.0 };
                for j in 0..i {
                    val -= l[i * n + j] * inv[j * n + col];
                }
                inv[i * n + col] = val / l[i * n + i].max(1e-20);
            }
        }
        inv
    }

    /// Get the current best mean (distribution center).
    pub fn current_mean(&self) -> &[f64] {
        &self.mean
    }

    /// Get the current step size.
    pub fn current_sigma(&self) -> f64 {
        self.sigma
    }
}

/// Run CMA-ES optimization with a given objective function.
///
/// # Arguments
/// * `dim` - Number of decision variables.
/// * `initial_mean` - Starting point.
/// * `config` - CMA-ES configuration.
/// * `objective` - Closure that evaluates a candidate and returns fitness (minimize).
///
/// # Returns
/// `CmaEsResult` with the best solution found.
pub fn optimize<F>(
    dim: usize,
    initial_mean: &[f64],
    config: &CmaEsConfig,
    objective: F,
) -> CmaEsResult
where
    F: Fn(&[f64]) -> f64 + Sync,
{
    assert!(dim >= 1, "Dimension must be >= 1, got {}", dim);
    assert!(config.max_iterations >= 1);

    let mut state = CmaEsState::new(dim, initial_mean, config);
    let mut best_solution = initial_mean.to_vec();
    let mut best_fitness = f64::INFINITY;
    let mut fitness_history = Vec::with_capacity(config.max_iterations);
    let mut total_evals = 0usize;

    for _gen in 0..config.max_iterations {
        // 1. Sample population
        let population = state.sample_population();

        // 2. Evaluate in parallel
        #[cfg(feature = "parallel")]
        let fitnesses: Vec<f64> = population.par_iter().map(|x| objective(x)).collect();
        #[cfg(not(feature = "parallel"))]
        let fitnesses: Vec<f64> = population.iter().map(|x| objective(x)).collect();

        total_evals += population.len();

        // 3. Sort by fitness
        let mut ranked: Vec<(f64, Vec<f64>)> =
            fitnesses.into_iter().zip(population.into_iter()).collect();
        ranked.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Track best
        if ranked[0].0 < best_fitness {
            best_fitness = ranked[0].0;
            best_solution = ranked[0].1.clone();
        }
        fitness_history.push(best_fitness);

        // 4. Check termination
        if let Some(target) = config.target_fitness {
            if best_fitness <= target {
                break;
            }
        }
        if fitness_history.len() > 20 {
            let recent = &fitness_history[fitness_history.len() - 20..];
            let range = recent.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                - recent.iter().cloned().fold(f64::INFINITY, f64::min);
            if range < config.fitness_tolerance {
                break;
            }
        }

        // 5. Update distribution
        state.update(&ranked);
    }

    CmaEsResult {
        best_solution,
        best_fitness,
        fitness_history,
        generations: state.generation,
        evaluations: total_evals,
    }
}

/// Run CMA-ES optimization on a batch of torque coefficient candidates.
///
/// This is the high-level entry point that integrates with the pendulum
/// batch evaluator for parallel physics simulation.
///
/// # Arguments
/// * `n_joints` - Number of actuated joints (e.g., 2 for double, 7 for golfer).
/// * `n_coeffs_per_joint` - Polynomial coefficients per joint.
/// * `initial_coeffs` - Starting torque coefficients, shape (n_joints * n_coeffs_per_joint,).
/// * `evaluate_fn` - Function that takes flat coefficients and returns negative fitness.
/// * `config` - CMA-ES config.
pub fn optimize_torque_coefficients<F>(
    n_joints: usize,
    n_coeffs_per_joint: usize,
    initial_coeffs: &[f64],
    evaluate_fn: F,
    config: &CmaEsConfig,
) -> CmaEsResult
where
    F: Fn(&[f64]) -> f64 + Sync,
{
    let dim = n_joints * n_coeffs_per_joint;
    assert_eq!(
        initial_coeffs.len(),
        dim,
        "Expected {} coefficients, got {}",
        dim,
        initial_coeffs.len()
    );

    optimize(dim, initial_coeffs, config, evaluate_fn)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sphere_function() {
        // Minimize f(x) = sum(x_i^2), optimum at origin
        let config = CmaEsConfig {
            population_size: 10,
            max_iterations: 200,
            initial_sigma: 1.0,
            target_fitness: Some(1e-6),
            fitness_tolerance: 1e-12,
        };

        let dim = 3;
        let initial = vec![1.0, 2.0, 3.0];

        let result = optimize(dim, &initial, &config, |x| {
            x.iter().map(|xi| xi * xi).sum::<f64>()
        });

        assert!(
            result.best_fitness < 0.1,
            "Should find near-optimal: {}",
            result.best_fitness
        );
        assert_eq!(result.best_solution.len(), dim);
    }

    #[test]
    fn test_rosenbrock_2d() {
        let config = CmaEsConfig {
            population_size: 20,
            max_iterations: 500,
            initial_sigma: 0.5,
            target_fitness: Some(1e-4),
            fitness_tolerance: 1e-12,
        };

        let result = optimize(2, &[0.0, 0.0], &config, |x| {
            let a = 1.0 - x[0];
            let b = x[1] - x[0] * x[0];
            a * a + 100.0 * b * b
        });

        assert!(
            result.best_fitness < 1.0,
            "Should approach Rosenbrock optimum"
        );
    }

    #[test]
    fn test_torque_coefficient_api() {
        let config = CmaEsConfig {
            population_size: 8,
            max_iterations: 50,
            initial_sigma: 0.5,
            ..Default::default()
        };

        let result = optimize_torque_coefficients(
            2,
            3,
            &[0.0; 6],
            |coeffs| coeffs.iter().map(|c| c * c).sum::<f64>(),
            &config,
        );

        assert_eq!(result.best_solution.len(), 6);
        assert!(result.evaluations > 0);
    }

    #[test]
    #[should_panic(expected = "Dimension must be >= 1")]
    fn test_zero_dimension_panics() {
        let config = CmaEsConfig::default();
        optimize(0, &[], &config, |_| 0.0);
    }
}
