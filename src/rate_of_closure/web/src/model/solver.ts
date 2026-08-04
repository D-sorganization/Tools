/**
 * Impact-parameter solver for the web clone (epic #4103, #4109/#4110).
 *
 * Practical TypeScript counterpart of shared/python/swing_sim/solver:
 * goals limited to what the ported TS physics computes (club path / face
 * / attack angle / dynamic loft at the delivery level; ball speed,
 * launch angles, and spin through solveImpact + deriveLaunch; carry
 * through the Waterloo/Penner flight port), the same
 * launch-monitor-resolution residual scales, and the same free-with-
 * bounds vs fixed variable partition over the delivery variables.
 *
 * The optimizer is a bounded Nelder-Mead over the free variables
 * (candidates clamped into the bounds before evaluation) with a small
 * deterministic multi-start — simple and honest for <= 7 smooth
 * variables. No swing-source mode on web, and no worker thread yet:
 * solves are quick at these sizes, and the WASM + web-worker upgrade
 * lands with the P7 kernels (see model/simulation.ts).
 *
 * Parity: solver.test.ts pins the same easy case as the pytest suite
 * (tests/rate_of_closure/test_solver_gui.py) — 150 mph ball speed from
 * a free clubhead speed solves to ~45.825 m/s in both implementations.
 */

import { deriveLaunch, simulateFlight } from "./flight";
import { MPH_PER_MPS, solveImpact, toFlightFrame } from "./simulation";

export const SOLVER_GOAL_KEYS = [
  "clubPathDeg",
  "faceAngleDeg",
  "attackAngleDeg",
  "dynamicLoftDeg",
  "ballSpeedMph",
  "launchAngleDeg",
  "launchAzimuthDeg",
  "spinRpm",
  "carryM",
] as const;
export type SolverGoalKey = (typeof SOLVER_GOAL_KEYS)[number];

export const SOLVER_VARIABLE_KEYS = [
  "clubheadSpeedMps",
  "clubPathDeg",
  "faceAngleDeg",
  "attackAngleDeg",
  "dynamicLoftDeg",
  "impactOffsetToeMm",
  "impactOffsetHighMm",
] as const;
export type SolverVariableKey = (typeof SOLVER_VARIABLE_KEYS)[number];

/** Defaults used for variables that are neither free nor fixed. */
export const VARIABLE_DEFAULTS: Record<SolverVariableKey, number> = {
  clubheadSpeedMps: 45.0,
  clubPathDeg: 0.0,
  faceAngleDeg: 0.0,
  attackAngleDeg: 0.0,
  dynamicLoftDeg: 10.5,
  impactOffsetToeMm: 0.0,
  impactOffsetHighMm: 0.0,
};

/** Same launch-monitor-resolution scales as swing_sim/solver/tuning.py. */
const RESIDUAL_SCALES: Record<SolverGoalKey, number> = {
  clubPathDeg: 0.5,
  faceAngleDeg: 0.5,
  attackAngleDeg: 0.5,
  dynamicLoftDeg: 0.5,
  ballSpeedMph: 1.0,
  launchAngleDeg: 0.5,
  launchAzimuthDeg: 0.5,
  spinRpm: 100.0,
  carryM: 1.0,
};

export interface GoalTermTs {
  target: number;
  weight: number;
}

export type SolverGoalTs = Partial<Record<SolverGoalKey, GoalTermTs>>;

export interface VariablePartitionTs {
  free: Partial<Record<SolverVariableKey, [number, number]>>;
  fixed: Partial<Record<SolverVariableKey, number>>;
}

export interface SolverResultTs {
  variables: Record<SolverVariableKey, number>;
  achieved: Partial<Record<SolverGoalKey, number>>;
  perGoalErrors: Partial<Record<SolverGoalKey, number>>;
  residualNorm: number;
  cost: number;
  converged: boolean;
  nEvals: number;
}

const MAX_ANGLE_DEG = 89.0;
const clampAngle = (v: number): number =>
  Math.max(-MAX_ANGLE_DEG, Math.min(MAX_ANGLE_DEG, v));

/** DbC-style validation mirroring VariablePartition/ImpactGoal. */
export function validateInputs(
  goal: SolverGoalTs,
  partition: VariablePartitionTs,
): void {
  const goalKeys = Object.keys(goal) as SolverGoalKey[];
  if (goalKeys.length === 0) {
    throw new Error("at least one goal quantity must be targeted");
  }
  for (const key of goalKeys) {
    const term = goal[key];
    if (!SOLVER_GOAL_KEYS.includes(key)) {
      throw new Error(`unknown goal quantity ${key}`);
    }
    if (!term || !Number.isFinite(term.target) || !(term.weight > 0)) {
      throw new Error(`goal ${key} needs a finite target and a weight > 0`);
    }
  }
  const freeKeys = Object.keys(partition.free) as SolverVariableKey[];
  const fixedKeys = Object.keys(partition.fixed) as SolverVariableKey[];
  for (const key of [...freeKeys, ...fixedKeys]) {
    if (!SOLVER_VARIABLE_KEYS.includes(key)) {
      throw new Error(`unknown variable ${key}`);
    }
  }
  for (const key of freeKeys) {
    if (fixedKeys.includes(key)) {
      throw new Error(`variable ${key} cannot be both free and fixed`);
    }
    const [lo, hi] = partition.free[key]!;
    if (!Number.isFinite(lo) || !Number.isFinite(hi) || !(lo < hi)) {
      throw new Error(`bounds for ${key} must be finite with lower < upper`);
    }
  }
  for (const key of fixedKeys) {
    if (!Number.isFinite(partition.fixed[key]!)) {
      throw new Error(`fixed value for ${key} must be finite`);
    }
  }
  if (freeKeys.length === 0) {
    throw new Error("at least one variable must be optimized");
  }
}

/** Full variable record from a free-variable vector (defaults filled). */
export function assembleVariables(
  x: number[],
  freeKeys: SolverVariableKey[],
  partition: VariablePartitionTs,
): Record<SolverVariableKey, number> {
  const variables = { ...VARIABLE_DEFAULTS, ...partition.fixed };
  freeKeys.forEach((key, i) => {
    variables[key] = x[i];
  });
  return variables;
}

/** Run delivery -> impact (-> flight) and report achieved quantities. */
export function achievedQuantities(
  variables: Record<SolverVariableKey, number>,
  goal: SolverGoalTs,
): Partial<Record<SolverGoalKey, number>> {
  const achieved: Partial<Record<SolverGoalKey, number>> = {
    clubPathDeg: clampAngle(variables.clubPathDeg),
    faceAngleDeg: clampAngle(variables.faceAngleDeg),
    attackAngleDeg: clampAngle(variables.attackAngleDeg),
    dynamicLoftDeg: clampAngle(variables.dynamicLoftDeg),
  };
  const needsLaunch =
    goal.ballSpeedMph !== undefined ||
    goal.launchAngleDeg !== undefined ||
    goal.launchAzimuthDeg !== undefined ||
    goal.spinRpm !== undefined ||
    goal.carryM !== undefined;
  if (!needsLaunch) return achieved;

  const impact = solveImpact({
    clubheadSpeedMps: Math.max(variables.clubheadSpeedMps, 1e-3),
    clubPathDeg: achieved.clubPathDeg!,
    faceAngleDeg: achieved.faceAngleDeg!,
    attackAngleDeg: achieved.attackAngleDeg!,
    dynamicLoftDeg: achieved.dynamicLoftDeg!,
    impactOffsetToeMm: variables.impactOffsetToeMm,
    impactOffsetHighMm: variables.impactOffsetHighMm,
  });
  const launch = deriveLaunch(
    toFlightFrame(impact.ballVelocity),
    toFlightFrame(impact.ballAngularVelocity),
  );
  achieved.ballSpeedMph = launch.ballSpeedMps * MPH_PER_MPS;
  achieved.launchAngleDeg = (launch.launchAngleRad * 180.0) / Math.PI;
  // Flight-frame azimuth is + toward +y (left); goals use + = right.
  achieved.launchAzimuthDeg = (-launch.azimuthRad * 180.0) / Math.PI;
  achieved.spinRpm = launch.spinRpm;
  if (goal.carryM !== undefined) {
    achieved.carryM = simulateFlight(launch).carryM;
  }
  return achieved;
}

/** Scalar cost: sum of squared weighted, scaled per-goal residuals. */
function costOf(
  x: number[],
  freeKeys: SolverVariableKey[],
  partition: VariablePartitionTs,
  goal: SolverGoalTs,
): number {
  const achieved = achievedQuantities(
    assembleVariables(x, freeKeys, partition),
    goal,
  );
  let cost = 0.0;
  for (const key of Object.keys(goal) as SolverGoalKey[]) {
    const term = goal[key]!;
    const r =
      (term.weight * (achieved[key]! - term.target)) / RESIDUAL_SCALES[key];
    cost += r * r;
  }
  return 0.5 * cost;
}

interface NmOptions {
  maxEvals: number;
  tol: number;
}

/** Bounded Nelder-Mead: candidates are clamped into the bounds. */
function nelderMead(
  f: (x: number[]) => number,
  x0: number[],
  lo: number[],
  hi: number[],
  options: NmOptions,
): { x: number[]; cost: number; nEvals: number; converged: boolean } {
  const n = x0.length;
  const clamp = (x: number[]): number[] =>
    x.map((v, i) => Math.max(lo[i], Math.min(hi[i], v)));
  let nEvals = 0;
  const evalAt = (x: number[]): number => {
    nEvals += 1;
    return f(clamp(x));
  };
  // Initial simplex: x0 plus 5%-of-range steps along each axis.
  let simplex = [clamp(x0), ...x0.map((_, i) => {
    const step = 0.05 * (hi[i] - lo[i]);
    const vertex = [...x0];
    vertex[i] = vertex[i] + step <= hi[i] ? vertex[i] + step : vertex[i] - step;
    return clamp(vertex);
  })].map((x) => ({ x, cost: evalAt(x) }));

  let converged = false;
  while (nEvals < options.maxEvals) {
    simplex.sort((a, b) => a.cost - b.cost);
    if (
      Math.abs(simplex[n].cost - simplex[0].cost) <
      options.tol * (1.0 + Math.abs(simplex[0].cost))
    ) {
      converged = true;
      break;
    }
    const centroid = new Array<number>(n).fill(0);
    for (let i = 0; i < n; i += 1) {
      for (let j = 0; j < n; j += 1) centroid[j] += simplex[i].x[j] / n;
    }
    const worst = simplex[n];
    const move = (factor: number): number[] =>
      centroid.map((c, j) => c + factor * (worst.x[j] - c));
    const reflected = move(-1.0);
    const reflectedCost = evalAt(reflected);
    if (reflectedCost < simplex[0].cost) {
      const expanded = move(-2.0);
      const expandedCost = evalAt(expanded);
      simplex[n] =
        expandedCost < reflectedCost
          ? { x: expanded, cost: expandedCost }
          : { x: reflected, cost: reflectedCost };
    } else if (reflectedCost < simplex[n - 1].cost) {
      simplex[n] = { x: reflected, cost: reflectedCost };
    } else {
      const contracted = move(0.5);
      const contractedCost = evalAt(contracted);
      if (contractedCost < worst.cost) {
        simplex[n] = { x: contracted, cost: contractedCost };
      } else {
        // Shrink toward the best vertex.
        simplex = simplex.map((v, i) =>
          i === 0
            ? v
            : (() => {
                const x = v.x.map(
                  (value, j) => simplex[0].x[j] + 0.5 * (value - simplex[0].x[j]),
                );
                return { x, cost: evalAt(x) };
              })(),
        );
      }
    }
  }
  simplex.sort((a, b) => a.cost - b.cost);
  return { x: clamp(simplex[0].x), cost: simplex[0].cost, nEvals, converged };
}

/** Deterministic multi-start fractions (midpoint first, no RNG). */
const START_FRACTIONS = [0.5, 0.25, 0.75];

/** Solve for the free-variable values that best achieve the goal. */
export function solveGoals(
  goal: SolverGoalTs,
  partition: VariablePartitionTs,
  maxEvalsPerStart = 400,
): SolverResultTs {
  validateInputs(goal, partition);
  const freeKeys = Object.keys(partition.free) as SolverVariableKey[];
  const lo = freeKeys.map((key) => partition.free[key]![0]);
  const hi = freeKeys.map((key) => partition.free[key]![1]);
  const f = (x: number[]): number => costOf(x, freeKeys, partition, goal);

  let best: ReturnType<typeof nelderMead> | null = null;
  let totalEvals = 0;
  for (const fraction of START_FRACTIONS) {
    const x0 = lo.map((l, i) => l + fraction * (hi[i] - l));
    const run = nelderMead(f, x0, lo, hi, {
      maxEvals: maxEvalsPerStart,
      tol: 1e-12,
    });
    totalEvals += run.nEvals;
    if (best === null || run.cost < best.cost) best = run;
  }
  const variables = assembleVariables(best!.x, freeKeys, partition);
  const achieved = achievedQuantities(variables, goal);
  const perGoalErrors: Partial<Record<SolverGoalKey, number>> = {};
  for (const key of Object.keys(goal) as SolverGoalKey[]) {
    perGoalErrors[key] = achieved[key]! - goal[key]!.target;
  }
  return {
    variables,
    achieved,
    perGoalErrors,
    residualNorm: Math.sqrt(2.0 * best!.cost),
    cost: best!.cost,
    converged: best!.converged,
    nEvals: totalEvals,
  };
}
