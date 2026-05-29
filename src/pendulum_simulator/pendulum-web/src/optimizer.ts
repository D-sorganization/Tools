/**
 * Torque profile optimizer — maximizes horizontal tip velocity at bottom of arc.
 *
 * Strategy: Nelder-Mead simplex on polynomial torque coefficients.
 * Respects joint limits (ROM) and torque saturation clamps.
 *
 * DbC: Pre/post-conditions on inputs and outputs.
 * DRY: Reuses physics engine's runSimulation, makePolynomialTorque, jointVelocities.
 * TDD: Pure functions, deterministic for given seed.
 *
 * @module optimizer
 */

import {
    PendulumParams, State, TorqueFunc, JointLimits, TorqueClamp,
    makePolynomialTorque, runSimulation, jointVelocities, clampTorque,
} from './physics';

// ── Types ────────────────────────────────────────────────────────────────────

export interface OptimizerConfig {
    params: PendulumParams;
    initialState: State;
    tEnd: number;
    dt: number;
    limits?: JointLimits;
    clamp?: TorqueClamp;
    shoulderDegree: number; // polynomial degree for shoulder torque
    wristDegree: number;    // polynomial degree for wrist torque
    maxIterations: number;
    tolerance: number;
}

export interface OptimizerProgress {
    iteration: number;
    maxIterations: number;
    bestCost: number;
    bestHorizontalSpeed: number;
    coeffsShoulder: number[];
    coeffsWrist: number[];
}

export type ProgressCallback = (progress: OptimizerProgress) => void;

export interface OptimizerResult {
    coeffsShoulder: number[];
    coeffsWrist: number[];
    bestHorizontalSpeed: number; // m/s
    iterations: number;
    converged: boolean;
}

// ── Objective function ──────────────────────────────────────────────────────

/**
 * Evaluate the negative of max horizontal tip speed at lowest point.
 * We negate because we minimize cost but want to maximize speed.
 *
 * "Bottom of arc" = the point where tip y-position is most negative.
 *
 * Pre: coeffs has shoulderDegree+1+wristDegree+1 elements.
 * Post: returns finite number.
 */
function evaluateCost(
    coeffs: number[],
    config: OptimizerConfig,
    coeffsShoulder: number[],
    coeffsWrist: number[],
): number {
    const nShoulder = config.shoulderDegree + 1;
    const nWrist = coeffs.length - nShoulder;
    for (let i = 0; i < nShoulder; i++) coeffsShoulder[i] = coeffs[i];
    for (let i = 0; i < nWrist; i++) coeffsWrist[i] = coeffs[nShoulder + i];

    let torqueFunc: TorqueFunc;
    try {
        torqueFunc = makePolynomialTorque(coeffsShoulder, coeffsWrist);
    } catch {
        return 1e10; // Invalid coefficients
    }

    // Wrap with clamping if needed
    const clampedTorque: TorqueFunc = config.clamp
        ? (t: number) => clampTorque(torqueFunc(t), config.clamp!)
        : torqueFunc;

    let result;
    try {
        result = runSimulation(
            config.params, config.initialState, config.tEnd,
            clampedTorque, config.dt, config.limits, config.clamp,
        );
    } catch {
        return 1e10; // Simulation failure (singular matrix, NaN, etc.)
    }

    // Find the frame where tip is lowest (most negative y)
    let bestSpeed = 0;
    let lowestY = Infinity;
    let lowestIdx = 0;

    for (let i = 0; i < result.states.length; i++) {
        const state = result.states[i];
        const theta1 = state[0], phi = state[1];
        const a2 = theta1 + phi;
        const tipY = -config.params.L1 * Math.cos(theta1)
                     - config.params.L2 * Math.cos(a2);
        if (tipY < lowestY) {
            lowestY = tipY;
            lowestIdx = i;
        }
    }

    // Get horizontal velocity at lowest point
    const jv = jointVelocities(result.states[lowestIdx], config.params);
    bestSpeed = Math.abs(jv.tipVel[0]); // horizontal component

    return -bestSpeed; // Negate: we minimize, so negative speed = maximize speed
}

// ── Nelder-Mead simplex optimizer ───────────────────────────────────────────

/**
 * Nelder-Mead simplex method for unconstrained optimization.
 * Robust, derivative-free, handles noisy objectives well.
 *
 * Pre: x0.length >= 1, maxIter > 0.
 * Post: returns best point found and cost.
 */
function nelderMead(
    f: (x: number[]) => number,
    x0: number[],
    maxIter: number,
    tol: number,
    onProgress?: (iter: number, bestCost: number, bestX: number[]) => void,
): { x: number[]; cost: number; iterations: number; converged: boolean } {
    const n = x0.length;
    const alpha = 1.0;  // reflection
    const gamma = 2.0;  // expansion
    const rho = 0.5;    // contraction
    const sigma = 0.5;  // shrink

    // Initialize simplex: x0 + unit perturbations
    const simplex: { x: number[]; cost: number }[] = [];
    simplex.push({ x: [...x0], cost: f(x0) });

    for (let i = 0; i < n; i++) {
        const xi = [...x0];
        xi[i] += Math.abs(xi[i]) > 1e-6 ? xi[i] * 0.1 : 1.0;
        simplex.push({ x: xi, cost: f(xi) });
    }

    const sortSimplex = () => simplex.sort((a, b) => a.cost - b.cost);
    sortSimplex();

    let iter = 0;
    let converged = false;

    // ⚡ Bolt Optimization: Pre-allocate working arrays outside the main loop
    // to eliminate continuous garbage collection overhead during algorithm iteration.
    const centroid = new Array<number>(n);
    const reflected = new Array<number>(n);
    const expanded = new Array<number>(n);
    const contracted = new Array<number>(n);

    for (; iter < maxIter; iter++) {
        sortSimplex();

        // Report progress every 5 iterations
        if (onProgress && iter % 5 === 0) {
            onProgress(iter, simplex[0].cost, simplex[0].x);
        }

        // Check convergence: spread of costs
        const costSpread = Math.abs(simplex[n].cost - simplex[0].cost);
        if (costSpread < tol) {
            converged = true;
            break;
        }

        // Centroid of all except worst
        centroid.fill(0);
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                centroid[j] += simplex[i].x[j];
            }
        }
        for (let j = 0; j < n; j++) centroid[j] /= n;

        const worst = simplex[n];
        const secondWorst = simplex[n - 1];
        const best = simplex[0];

        // ⚡ Bolt Optimization: Replace .map() calls with pre-allocated arrays and standard for-loops
        // to eliminate continuous callback allocation and garbage collection overhead in the tight loop.

        // Reflection
        for (let j = 0; j < n; j++) {
            reflected[j] = centroid[j] + alpha * (centroid[j] - worst.x[j]);
        }
        const fReflected = f(reflected);

        if (fReflected < secondWorst.cost && fReflected >= best.cost) {
            simplex[n] = { x: [...reflected], cost: fReflected };
            continue;
        }

        // Expansion
        if (fReflected < best.cost) {
            for (let j = 0; j < n; j++) {
                expanded[j] = centroid[j] + gamma * (reflected[j] - centroid[j]);
            }
            const fExpanded = f(expanded);
            simplex[n] = fExpanded < fReflected
                ? { x: [...expanded], cost: fExpanded }
                : { x: [...reflected], cost: fReflected };
            continue;
        }

        // Contraction
        for (let j = 0; j < n; j++) {
            contracted[j] = centroid[j] + rho * (worst.x[j] - centroid[j]);
        }
        const fContracted = f(contracted);
        if (fContracted < worst.cost) {
            simplex[n] = { x: [...contracted], cost: fContracted };
            continue;
        }

        // Shrink
        for (let i = 1; i <= n; i++) {
            const newX = new Array<number>(n);
            for (let j = 0; j < n; j++) {
                newX[j] = best.x[j] + sigma * (simplex[i].x[j] - best.x[j]);
            }
            simplex[i].x = newX;
            simplex[i].cost = f(simplex[i].x);
        }
    }

    sortSimplex();
    if (onProgress) {
        onProgress(iter, simplex[0].cost, simplex[0].x);
    }

    return {
        x: simplex[0].x,
        cost: simplex[0].cost,
        iterations: iter,
        converged,
    };
}

// ── Public optimizer API ────────────────────────────────────────────────────

/**
 * Optimize torque polynomial coefficients to maximize horizontal tip velocity
 * at the bottom of the swing arc.
 *
 * Pre: config has valid params and positive degrees.
 * Post: result contains optimized coefficients and speed.
 */
export function optimizeTorqueProfile(
    config: OptimizerConfig,
    onProgress?: ProgressCallback,
    initialCoeffs?: { shoulder: number[]; wrist: number[] },
): OptimizerResult {
    if (!(config.shoulderDegree >= 0)) throw new RangeError('[DbC] shoulderDegree must be >= 0');
    if (!(config.wristDegree >= 0)) throw new RangeError('[DbC] wristDegree must be >= 0');

    const nShoulder = config.shoulderDegree + 1;
    const nWrist = config.wristDegree + 1;

    // Initial guess: provided or zero
    const x0: number[] = [];
    if (initialCoeffs) {
        for (let i = 0; i < nShoulder; i++) {
            x0.push(i < initialCoeffs.shoulder.length ? initialCoeffs.shoulder[i] : 0);
        }
        for (let i = 0; i < nWrist; i++) {
            x0.push(i < initialCoeffs.wrist.length ? initialCoeffs.wrist[i] : 0);
        }
    } else {
        // Reasonable starting guess for golf swing
        x0.push(-20); // shoulder constant
        for (let i = 1; i < nShoulder; i++) x0.push(i === 1 ? 10 : 0);
        x0.push(0); // wrist constant
        for (let i = 1; i < nWrist; i++) x0.push(0);
    }

    const evalShoulder = new Array(nShoulder);
    const evalWrist = new Array(nWrist);
    const objective = (coeffs: number[]) => evaluateCost(coeffs, config, evalShoulder, evalWrist);

    const progressAdapter = onProgress
        ? (iter: number, bestCost: number, bestX: number[]) => {
            const coeffsShoulder = new Array(nShoulder);
            for (let i = 0; i < nShoulder; i++) coeffsShoulder[i] = bestX[i];
            const coeffsWrist = new Array(nWrist);
            for (let i = 0; i < nWrist; i++) coeffsWrist[i] = bestX[nShoulder + i];
            onProgress({
                iteration: iter,
                maxIterations: config.maxIterations,
                bestCost,
                bestHorizontalSpeed: -bestCost,
                coeffsShoulder,
                coeffsWrist,
            });
        }
        : undefined;

    const result = nelderMead(
        objective, x0, config.maxIterations, config.tolerance, progressAdapter,
    );

    const coeffsShoulder = new Array(nShoulder);
    for (let i = 0; i < nShoulder; i++) coeffsShoulder[i] = result.x[i];
    const coeffsWrist = new Array(nWrist);
    for (let i = 0; i < nWrist; i++) coeffsWrist[i] = result.x[nShoulder + i];

    return {
        coeffsShoulder,
        coeffsWrist,
        bestHorizontalSpeed: -result.cost,
        iterations: result.iterations,
        converged: result.converged,
    };
}

/** Default optimizer configuration factory. */
export function defaultOptimizerConfig(
    params: PendulumParams,
    initialState: State,
    tEnd: number,
    limits?: JointLimits,
    clamp?: TorqueClamp,
): OptimizerConfig {
    return {
        params,
        initialState,
        tEnd,
        dt: 0.01, // Coarser dt for speed during optimization
        limits,
        clamp,
        shoulderDegree: 2,
        wristDegree: 1,
        maxIterations: 300,
        tolerance: 1e-4,
    };
}
