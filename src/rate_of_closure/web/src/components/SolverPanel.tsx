/**
 * Solver section for the web Simulation tab (epic #4103, #4109/#4110).
 *
 * Practical counterpart of the PyQt6 Solver panel: checkbox-enabled
 * weighted goal targets, a per-variable Optimize-with-bounds / Fix
 * partition over the delivery variables, a synchronous bounded
 * Nelder-Mead run through the parity-ported TS physics
 * (model/solver.ts), an achieved-vs-goal results table with per-goal
 * errors / residual norm / convergence, and an Apply button that loads
 * the solved clubhead speed and impact offsets into the scenario so the
 * simulation reruns with them. Validation problems surface as friendly
 * messages. The WASM + web-worker upgrade (progress, cancel, swing-
 * source mode) lands with the P7 kernels.
 */

import { useState } from "react";

import { type ImpactScenario } from "../model/impact";
import { MPH_PER_MPS } from "../model/simulation";
import {
  solveGoals,
  type SolverGoalKey,
  type SolverGoalTs,
  type SolverResultTs,
  type SolverVariableKey,
  type VariablePartitionTs,
  VARIABLE_DEFAULTS,
} from "../model/solver";
import { FIELD_GUIDANCE } from "../model/units";

interface GoalRowSpec {
  key: SolverGoalKey;
  label: string;
  unit: string;
  target: number;
}

const GOAL_ROWS: GoalRowSpec[] = [
  { key: "clubPathDeg", label: "Club Path", unit: "°", target: 0 },
  { key: "faceAngleDeg", label: "Face Angle", unit: "°", target: 0 },
  { key: "attackAngleDeg", label: "Attack Angle", unit: "°", target: -1 },
  { key: "dynamicLoftDeg", label: "Dynamic Loft", unit: "°", target: 12 },
  { key: "ballSpeedMph", label: "Ball Speed", unit: "mph", target: 150 },
  { key: "launchAngleDeg", label: "Launch Angle", unit: "°", target: 12 },
  { key: "launchAzimuthDeg", label: "Launch Azimuth", unit: "°", target: 0 },
  { key: "spinRpm", label: "Total Spin", unit: "rpm", target: 2600 },
  { key: "carryM", label: "Carry Distance", unit: "m", target: 230 },
];

interface VarRowSpec {
  key: SolverVariableKey;
  label: string;
  unit: string;
  bounds: [number, number];
  guidanceKey: string;
}

const VAR_ROWS: VarRowSpec[] = [
  {
    key: "clubheadSpeedMps",
    label: "Clubhead Speed",
    unit: "m/s",
    bounds: [30, 60],
    guidanceKey: "clubheadSpeedMph",
  },
  {
    key: "clubPathDeg",
    label: "Club Path",
    unit: "°",
    bounds: [-15, 15],
    guidanceKey: "solverClubPath",
  },
  {
    key: "faceAngleDeg",
    label: "Face Angle",
    unit: "°",
    bounds: [-15, 15],
    guidanceKey: "solverFaceAngle",
  },
  {
    key: "attackAngleDeg",
    label: "Attack Angle",
    unit: "°",
    bounds: [-10, 10],
    guidanceKey: "solverAttackAngle",
  },
  {
    key: "dynamicLoftDeg",
    label: "Dynamic Loft",
    unit: "°",
    bounds: [5, 25],
    guidanceKey: "clubLoftDeg",
  },
  {
    key: "impactOffsetToeMm",
    label: "Impact Toward Toe",
    unit: "mm",
    bounds: [-20, 20],
    guidanceKey: "impactOffsetToeMm",
  },
  {
    key: "impactOffsetHighMm",
    label: "Impact Above Center",
    unit: "mm",
    bounds: [-15, 15],
    guidanceKey: "impactOffsetHighMm",
  },
];

/** Solver-only guidance (FIELD_GUIDANCE covers the shared fields). */
const SOLVER_GUIDANCE: Record<string, string> = {
  solverClubPath:
    "Suggested range: -8 to +8 deg (positive = in-to-out). Source: " +
    "openly published tour launch-monitor averages.",
  solverFaceAngle:
    "Suggested range: -5 to +5 deg at impact (positive = open). Source: " +
    "standard launch-monitor sign conventions and published tour data.",
  solverAttackAngle:
    "Suggested range: -5 to +5 deg for drivers (tour mean near -1 deg). " +
    "Source: openly published tour launch-monitor averages.",
};

const guidance = (key: string): string =>
  FIELD_GUIDANCE[key] ?? SOLVER_GUIDANCE[key] ?? "";

interface GoalState {
  enabled: boolean;
  target: number;
  weight: number;
}

interface VarState {
  optimize: boolean;
  lo: number;
  hi: number;
  value: number;
}

interface Props {
  onApply: (updates: Partial<ImpactScenario>) => void;
}

const inputClass =
  "no-spinner w-20 rounded border border-slate-700 bg-slate-800 px-2 " +
  "py-1 text-slate-100 focus:border-blue-500 focus:outline-none " +
  "disabled:opacity-40";

export function SolverPanel({ onApply }: Props) {
  const [goals, setGoals] = useState<Record<string, GoalState>>(() =>
    Object.fromEntries(
      GOAL_ROWS.map((row) => [
        row.key,
        {
          enabled: row.key === "ballSpeedMph",
          target: row.target,
          weight: 1,
        },
      ]),
    ),
  );
  const [vars, setVars] = useState<Record<string, VarState>>(() =>
    Object.fromEntries(
      VAR_ROWS.map((row) => [
        row.key,
        {
          optimize: row.key === "clubheadSpeedMps",
          lo: row.bounds[0],
          hi: row.bounds[1],
          value: VARIABLE_DEFAULTS[row.key],
        },
      ]),
    ),
  );
  const [result, setResult] = useState<SolverResultTs | null>(null);
  const [message, setMessage] = useState<string>("");

  const setGoal = (key: string, patch: Partial<GoalState>) =>
    setGoals((g) => ({ ...g, [key]: { ...g[key], ...patch } }));
  const setVar = (key: string, patch: Partial<VarState>) =>
    setVars((v) => ({ ...v, [key]: { ...v[key], ...patch } }));

  const runSolver = () => {
    const goal: SolverGoalTs = {};
    for (const row of GOAL_ROWS) {
      const state = goals[row.key];
      if (state.enabled) {
        goal[row.key] = { target: state.target, weight: state.weight };
      }
    }
    const partition: VariablePartitionTs = { free: {}, fixed: {} };
    for (const row of VAR_ROWS) {
      const state = vars[row.key];
      if (state.optimize) partition.free[row.key] = [state.lo, state.hi];
      else partition.fixed[row.key] = state.value;
    }
    try {
      const solved = solveGoals(goal, partition);
      setResult(solved);
      setMessage(
        `${solved.converged ? "Converged" : "Did NOT converge"} — residual ` +
          `norm ${solved.residualNorm.toExponential(2)}, ${solved.nEvals} ` +
          "evaluations.",
      );
    } catch (error) {
      setResult(null);
      setMessage(`Cannot solve: ${(error as Error).message}`);
    }
  };

  const applySolution = () => {
    if (!result) return;
    onApply({
      clubheadSpeedMph: result.variables.clubheadSpeedMps * MPH_PER_MPS,
      impactOffsetToeMm: result.variables.impactOffsetToeMm,
      impactOffsetHighMm: result.variables.impactOffsetHighMm,
    });
    setMessage(
      "Applied: scenario clubhead speed and impact offsets updated — " +
        "run the simulation to see the optimized impact.",
    );
  };

  const numberField = (
    value: number,
    disabled: boolean,
    title: string,
    onChange: (value: number) => void,
  ) => (
    <input
      type="number"
      inputMode="decimal"
      value={value}
      disabled={disabled}
      title={title}
      aria-disabled={disabled}
      onChange={(e) => {
        const parsed = Number(e.target.value);
        if (Number.isFinite(parsed)) onChange(parsed);
      }}
      className={inputClass}
    />
  );

  return (
    <div className="rounded-xl border border-slate-800/80 bg-slate-900/60 p-5 shadow-lg shadow-black/20 backdrop-blur">
      <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">
        Solver — Goal-Driven Optimization
      </h2>

      <h3 className="mb-2 text-xs font-semibold uppercase text-slate-500">
        Goals (Check the Targets to Hit)
      </h3>
      <div className="mb-4 grid gap-1">
        {GOAL_ROWS.map((row) => {
          const state = goals[row.key];
          return (
            <label
              key={row.key}
              className="flex items-center gap-2 text-sm text-slate-300"
            >
              <input
                type="checkbox"
                checked={state.enabled}
                onChange={(e) => setGoal(row.key, { enabled: e.target.checked })}
              />
              <span className="flex-1">
                {row.label}{" "}
                <span className="text-slate-500">({row.unit})</span>
              </span>
              {numberField(state.target, !state.enabled, `Target ${row.unit}`, (v) =>
                setGoal(row.key, { target: v }),
              )}
              <span className="text-slate-500">w</span>
              {numberField(state.weight, !state.enabled, "Relative weight", (v) =>
                setGoal(row.key, { weight: v }),
              )}
            </label>
          );
        })}
      </div>

      <h3 className="mb-2 text-xs font-semibold uppercase text-slate-500">
        Variables (Optimize Within Bounds, or Fix)
      </h3>
      <div className="mb-4 grid gap-2">
        {VAR_ROWS.map((row) => {
          const state = vars[row.key];
          return (
            <div
              key={row.key}
              title={guidance(row.guidanceKey)}
              className="flex flex-wrap items-center gap-2 text-sm text-slate-300"
            >
              <span className="w-40">
                {row.label}{" "}
                <span className="text-slate-500">({row.unit})</span>
              </span>
              <label className="flex items-center gap-1">
                <input
                  type="radio"
                  name={`solver-var-${row.key}`}
                  checked={state.optimize}
                  onChange={() => setVar(row.key, { optimize: true })}
                />
                Optimize
              </label>
              {numberField(state.lo, !state.optimize, "Lower bound", (v) =>
                setVar(row.key, { lo: v }),
              )}
              {numberField(state.hi, !state.optimize, "Upper bound", (v) =>
                setVar(row.key, { hi: v }),
              )}
              <label className="flex items-center gap-1">
                <input
                  type="radio"
                  name={`solver-var-${row.key}`}
                  checked={!state.optimize}
                  onChange={() => setVar(row.key, { optimize: false })}
                />
                Fix
              </label>
              {numberField(state.value, state.optimize, "Fixed value", (v) =>
                setVar(row.key, { value: v }),
              )}
            </div>
          );
        })}
      </div>

      <div className="mb-3 flex gap-2">
        <button
          type="button"
          onClick={runSolver}
          className="flex-1 rounded-lg border border-sky-400/60 bg-sky-500/10 px-3 py-2 text-sm font-semibold text-sky-300 transition-all hover:bg-sky-500/20"
        >
          Run Solver
        </button>
        <button
          type="button"
          onClick={applySolution}
          disabled={!result}
          className="rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-sm text-slate-300 hover:border-slate-500 disabled:opacity-40"
        >
          Apply to Scenario
        </button>
      </div>

      {message && (
        <p aria-live="polite" className="mb-3 text-xs text-slate-400">
          {message}
        </p>
      )}

      {result && (
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="text-xs uppercase text-slate-500">
                <th className="py-1 pr-2">Quantity</th>
                <th className="py-1 pr-2">Target</th>
                <th className="py-1 pr-2">Achieved</th>
                <th className="py-1">Error</th>
              </tr>
            </thead>
            <tbody>
              {GOAL_ROWS.filter((row) => goals[row.key].enabled).map((row) => (
                <tr key={row.key} className="border-t border-slate-800/70">
                  <td className="py-1 pr-2 text-slate-400">{row.label}</td>
                  <td className="py-1 pr-2 tabular-nums">
                    {goals[row.key].target.toFixed(1)} {row.unit}
                  </td>
                  <td className="py-1 pr-2 tabular-nums">
                    {(result.achieved[row.key] ?? NaN).toFixed(1)} {row.unit}
                  </td>
                  <td className="py-1 tabular-nums">
                    {(result.perGoalErrors[row.key] ?? NaN).toFixed(2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-2 text-xs text-slate-500">
            Solved:{" "}
            {VAR_ROWS.filter((row) => vars[row.key].optimize)
              .map(
                (row) =>
                  `${row.label} = ${result.variables[row.key].toFixed(2)} ${row.unit}`,
              )
              .join(", ")}
            . Goals are limited to what the parity-ported TS physics
            computes; the swing-source mode, progress reporting, and
            cancellation arrive with the WASM worker in P7.
          </p>
        </div>
      )}
    </div>
  );
}
