import { useEffect, useMemo, useRef, useState } from "react";

import { BUTTON_CLASS, downloadText, INPUT_CLASS, PANEL_CLASS } from "./variationUi";
import { WindStrategyScatter } from "./WindStrategyScatter";
import type { Launch } from "../model/flight";
import { buildScalarEnsembleScatter, type ScalarEnsembleResult } from "../model/scalarEnsembleContract";
import { scalarEnsembleToCsv } from "../model/scalarEnsembleCsv";
import type { SpatialTargetTs } from "../model/spatialTarget";
import {
  WIND_UNCERTAINTY_SCHEMA_VERSION,
  type WindOutcomeStatus,
  type WindStrategyAnalysis,
  type WindStrategyRequest,
} from "../model/windUncertainty";
import { buildWindStrategyPlotData } from "../model/windStrategyPlotData";
import {
  runWindStrategyInWorker,
  type WindStrategyProgress,
  type WindStrategyRunController,
  type WindStrategyRunner,
} from "../model/windStrategyWorkerClient";

interface Props {
  readonly launch: Launch | null;
  readonly launchError?: string | null;
  readonly target: SpatialTargetTs;
  readonly runner?: WindStrategyRunner;
}

interface Controls {
  trials: number;
  seed: number;
  trueSpeedMps: number;
  trueBearingDeg: number;
  speedBiasMps: number;
  speedStdMps: number;
  bearingBiasDeg: number;
  bearingStdDeg: number;
  correlation: number;
  aimGainDegPerMps: number;
}

interface CompletedRun {
  request: WindStrategyRequest;
  analysis: WindStrategyAnalysis;
  ensemble: ScalarEnsembleResult<WindOutcomeStatus>;
}

const DEFAULT_CONTROLS: Controls = {
  trials: 200,
  seed: 4199,
  trueSpeedMps: 4.5,
  trueBearingDeg: 90,
  speedBiasMps: 0,
  speedStdMps: 0.8,
  bearingBiasDeg: 0,
  bearingStdDeg: 8,
  correlation: 0,
  aimGainDegPerMps: 0.2,
};

const CONTROL_FIELDS: ReadonlyArray<{
  key: keyof Controls;
  label: string;
  unit: string;
  min?: number;
  max?: number;
  step?: number;
}> = [
  { key: "trials", label: "Wind strategy trials", unit: "trials", min: 1, max: 100000, step: 1 },
  { key: "seed", label: "Wind strategy seed", unit: "uint32", min: 0, max: 4294967295, step: 1 },
  { key: "trueSpeedMps", label: "True wind speed", unit: "m/s", min: 0, step: 0.1 },
  { key: "trueBearingDeg", label: "True wind from bearing", unit: "deg", step: 1 },
  { key: "speedBiasMps", label: "Wind speed estimate bias", unit: "m/s", step: 0.1 },
  { key: "speedStdMps", label: "Wind speed estimate standard deviation", unit: "m/s", min: 0, step: 0.1 },
  { key: "bearingBiasDeg", label: "Wind bearing estimate bias", unit: "deg", step: 1 },
  { key: "bearingStdDeg", label: "Wind bearing estimate standard deviation", unit: "deg", min: 0, step: 1 },
  { key: "correlation", label: "Wind estimate correlation", unit: "r", min: -1, max: 1, step: 0.05 },
  { key: "aimGainDegPerMps", label: "Crosswind aim gain", unit: "deg/(m/s)", step: 0.05 },
];

const targetRadius = (target: SpatialTargetTs): number => {
  const tolerance = target.tolerance;
  if (tolerance.kind === "surface_corridor") {
    return Math.min(tolerance.halfLengthM, tolerance.halfWidthM);
  }
  if (tolerance.kind === "surface_circle") return tolerance.radiusM;
  throw new RangeError("Wind strategy landing targets require a surface tolerance");
};

function buildRequest(launch: Launch, target: SpatialTargetTs, controls: Controls): WindStrategyRequest {
  if (target.kind !== "landing_area") {
    throw new RangeError("Wind strategy analysis requires the current target to be a landing area");
  }
  return {
    uncertainty: {
      schema_version: WIND_UNCERTAINTY_SCHEMA_VERSION,
      trials: controls.trials,
      seed: controls.seed,
      true_speed_mps: { kind: "fixed", center: controls.trueSpeedMps, spread: 0, minimum: 0 },
      true_from_bearing_deg: { kind: "fixed", center: controls.trueBearingDeg, spread: 0 },
      estimate_error: {
        speed_bias_mps: controls.speedBiasMps,
        speed_std_mps: controls.speedStdMps,
        bearing_bias_deg: controls.bearingBiasDeg,
        bearing_std_deg: controls.bearingStdDeg,
        correlation: controls.correlation,
      },
      provenance: `flight-explorer/wind-strategy/seed-${controls.seed}`,
    },
    strategies: [{
      id: "current-launch", label: "Current launch", launch,
      crosswind_aim_gain_rad_per_mps: controls.aimGainDegPerMps * Math.PI / 180,
    }],
    target: {
      forward_m: target.point.appCoordinatesM[0],
      right_m: target.point.appCoordinatesM[2],
    },
    analysis: {
      model_name: "waterloo_penner", max_time_s: 10, time_step_s: 0.01,
      miss_scale_m: Math.max(targetRadius(target), 1), failure_cost: 100,
      target_radius_m: targetRadius(target), miss_distance_cvar_alpha: 0.9,
    },
  };
}

const inputSignature = (launch: Launch | null, target: SpatialTargetTs): string => JSON.stringify({
  launch: launch && [
    launch.ballSpeedMps, launch.launchAngleRad, launch.azimuthRad,
    launch.spinRpm, ...launch.spinAxis,
  ],
  target: {
    label: target.label, kind: target.kind, point: target.point.appCoordinatesM,
    tolerance: target.tolerance, elevationSource: target.elevationSource,
    groundSource: target.groundSource,
  },
});

export function WindStrategyPanel({
  launch, launchError = null, target, runner = runWindStrategyInWorker,
}: Props) {
  const [controls, setControls] = useState(DEFAULT_CONTROLS);
  const [progress, setProgress] = useState<WindStrategyProgress | null>(null);
  const [status, setStatus] = useState("Ready");
  const [error, setError] = useState<string | null>(null);
  const [completed, setCompleted] = useState<CompletedRun | null>(null);
  const [xKey, setXKey] = useState("actual_landing_forward_m");
  const [yKey, setYKey] = useState("actual_landing_right_m");
  const active = useRef<WindStrategyRunController | null>(null);
  const runId = useRef(0);
  const signature = inputSignature(launch, target);
  const previousSignature = useRef(signature);

  const invalidate = (message: string): void => {
    runId.current += 1;
    active.current?.cancel();
    active.current = null;
    setCompleted(null);
    setProgress(null);
    setError(null);
    setStatus(message);
  };

  useEffect(() => () => {
    runId.current += 1;
    active.current?.cancel();
  }, []);

  useEffect(() => {
    if (launch !== null) return;
    runId.current += 1;
    active.current?.cancel();
    active.current = null;
    setCompleted(null);
    setProgress(null);
    setStatus("Unavailable until launch inputs are valid");
  }, [launch]);

  useEffect(() => {
    if (signature === previousSignature.current) return;
    previousSignature.current = signature;
    invalidate("Launch or target changed — run again");
  }, [signature]);

  const run = (): void => {
    active.current?.cancel();
    const currentRunId = ++runId.current;
    setCompleted(null);
    setError(null);
    setProgress({ completed: 0, total: controls.trials });
    setStatus("Running wind strategy analysis");
    try {
      if (!launch) throw new RangeError(launchError ?? "Current launch inputs are invalid");
      const request = buildRequest(launch, target, controls);
      const controller = runner(request, (next) => {
        if (currentRunId === runId.current) setProgress(next);
      });
      active.current = controller;
      void controller.promise.then((analysis) => {
        if (currentRunId !== runId.current) return;
        // This adapter replays only the deterministic sampler to verify agreement;
        // it does not create another stochastic authority or rerun flight physics.
        const ensemble = buildWindStrategyPlotData(request, analysis);
        setCompleted({ request, analysis, ensemble });
        setStatus(`Completed ${analysis.wind_trials.length} trials`);
        active.current = null;
      }).catch((reason: unknown) => {
        if (currentRunId !== runId.current) return;
        setError(reason instanceof Error ? reason.message : String(reason));
        setStatus("Wind strategy analysis failed");
        active.current = null;
      });
    } catch (reason: unknown) {
      setError(reason instanceof Error ? reason.message : String(reason));
      setStatus("Wind strategy analysis failed");
    }
  };

  const cancel = (): void => {
    if (!active.current) return;
    runId.current += 1;
    active.current.cancel();
    active.current = null;
    setStatus("Cancelled");
    setProgress(null);
  };

  return <section className={PANEL_CLASS} aria-label="Wind strategy uncertainty analysis">
    <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-300">
      Wind Strategy Uncertainty
    </h2>
    <p className="mt-1 text-xs text-slate-400">
      Paired true-versus-estimated wind trials use the current launch and the current canonical landing target.
    </p>
    <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
      {CONTROL_FIELDS.map((field) => <label key={field.key} className="text-xs text-slate-300">
        <span className="mb-1 flex justify-between gap-2"><span>{field.label}</span>
          <span className="text-slate-500">{field.unit}</span></span>
        <input type="number" aria-label={field.label} className={INPUT_CLASS}
          value={controls[field.key]} min={field.min} max={field.max} step={field.step}
          onChange={(event) => {
            invalidate("Wind settings changed — run again");
            setControls((current) => ({
              ...current, [field.key]: Number(event.target.value),
            }));
          }} />
      </label>)}
    </div>
    <div className="mt-4 flex flex-wrap items-center gap-2">
      <button type="button" className={BUTTON_CLASS} onClick={run}
        title="Run paired wind trials in a background worker using the current launch and target"
        disabled={active.current !== null || launch === null}>Run wind strategy analysis</button>
      <button type="button" className={BUTTON_CLASS} onClick={cancel}
        title="Cancel the active analysis and terminate its background worker"
        disabled={active.current === null}>Cancel wind strategy analysis</button>
      <span role="status" aria-label="Wind strategy run status" className="text-xs text-slate-300">
        {status}{progress && active.current
          ? ` — ${progress.completed}/${progress.total} outcomes` : ""}
      </span>
    </div>
    {launch === null && <p role="alert" className="mt-3 text-xs text-amber-300">
      Wind strategy analysis is unavailable: {launchError ?? "current launch inputs are invalid"}.
    </p>}
    {error && <p role="alert" className="mt-3 text-xs text-rose-400">{error}</p>}
    {completed && launch && <WindStrategyResults run={completed} xKey={xKey} yKey={yKey}
      setXKey={setXKey} setYKey={setYKey} />}
  </section>;
}

function WindStrategyResults({ run, xKey, yKey, setXKey, setYKey }: {
  run: CompletedRun; xKey: string; yKey: string;
  setXKey: (key: string) => void; setYKey: (key: string) => void;
}): JSX.Element {
  const scatter = useMemo(
    () => buildScalarEnsembleScatter(run.ensemble, xKey, yKey),
    [run.ensemble, xKey, yKey],
  );
  return <div className="mt-5 space-y-4 border-t border-slate-800 pt-4">
    <CalculationBasis request={run.request} />
    <SummaryTable analysis={run.analysis} />
    <div className="grid gap-3 sm:grid-cols-2">
      <AxisSelect label="Wind scatter horizontal axis" value={xKey}
        variables={run.ensemble.variables} onChange={setXKey} />
      <AxisSelect label="Wind scatter vertical axis" value={yKey}
        variables={run.ensemble.variables} onChange={setYKey} />
    </div>
    <p className="text-xs text-slate-400" aria-live="polite">
      {run.ensemble.cohorts.map(({ key, label }) => {
        const availability = scatter.availability.by_cohort[key];
        return `${label}: ${availability.paired_finite}/${availability.total_rows} plotted`
          + (availability.unavailable ? `, ${availability.unavailable} unavailable` : "");
      }).join(" · ")}
    </p>
    <WindStrategyScatter scatter={scatter} />
    <button type="button" className={BUTTON_CLASS}
      aria-label="Export raw wind strategy CSV"
      title="Export every raw scalar row, cohort, variable, and availability-preserving null value"
      onClick={() => downloadText(
        `${run.ensemble.result_id}-raw.csv`, scalarEnsembleToCsv(run.ensemble),
        "text/csv;charset=utf-8",
      )}>Export raw wind strategy CSV</button>
  </div>;
}

function CalculationBasis({ request }: { request: WindStrategyRequest }): JSX.Element {
  const analysis = request.analysis;
  const policy = request.strategies.map((strategy) => {
    const gain = strategy.crosswind_aim_gain_rad_per_mps;
    return `${strategy.label}: ${gain.toPrecision(7)} rad/(m/s) `
      + `(${(gain * 180 / Math.PI).toFixed(4)} deg/(m/s))`;
  }).join(" · ");
  const entries = [
    ["Model", analysis.model_name],
    ["Trials / seed", `${request.uncertainty.trials} / ${request.uncertainty.seed}`],
    ["Target", `${request.target.forward_m.toFixed(3)} m forward, `
      + `${request.target.right_m.toFixed(3)} m right; radius ${analysis.target_radius_m.toFixed(3)} m`],
    ["Integration", `${analysis.time_step_s.toPrecision(5)} s step; ${analysis.max_time_s.toPrecision(5)} s maximum`],
    ["Failure cost", analysis.failure_cost.toPrecision(7)],
    ["Miss-distance CVaR alpha", analysis.miss_distance_cvar_alpha.toPrecision(5)],
    ["Aim-gain policy", policy],
  ];
  return <section aria-label="Wind strategy calculation basis"
    className="rounded-lg border border-slate-700/80 bg-slate-950/50 p-3">
    <h3 className="text-xs font-semibold uppercase tracking-wide text-sky-300">
      Captured Calculation Basis
    </h3>
    <dl className="mt-2 grid gap-x-4 gap-y-1 text-xs sm:grid-cols-[max-content_1fr]">
      {entries.map(([term, value]) => <div key={term} className="contents">
        <dt className="font-medium text-slate-400">{term}</dt>
        <dd className="break-words tabular-nums text-slate-200">{value}</dd>
      </div>)}
    </dl>
  </section>;
}

function SummaryTable({ analysis }: { analysis: WindStrategyAnalysis }): JSX.Element {
  return <div className="overflow-x-auto"><table aria-label="Wind strategy summary"
    className="w-full text-left text-xs text-slate-300"><thead><tr>
      <th className="px-2 py-1">Strategy</th><th className="px-2 py-1">Completed</th>
      <th className="px-2 py-1">Failed</th><th className="px-2 py-1">Mean miss</th>
      <th className="px-2 py-1">Target hold</th><th className="px-2 py-1">CVaR</th>
    </tr></thead><tbody>{analysis.summaries.map((summary) => <tr key={summary.strategy_id}
      className="border-t border-slate-800"><td className="px-2 py-1">{summary.label}</td>
      <td className="px-2 py-1">{summary.completed_trials}</td>
      <td className="px-2 py-1">{summary.failed_trials}</td>
      <td className="px-2 py-1">{meanMiss(analysis, summary.strategy_id)}</td>
      <td className="px-2 py-1">{(100 * summary.target_hold_probability).toFixed(1)}%</td>
      <td className="px-2 py-1">{summary.miss_distance_cvar_m.toFixed(2)} m</td>
    </tr>)}</tbody></table></div>;
}

const meanMiss = (analysis: WindStrategyAnalysis, strategyId: string): string => {
  const values = analysis.outcomes.filter((item) => item.strategy_id === strategyId)
    .map((item) => item.miss_distance_m).filter((value): value is number => value !== null);
  return values.length ? `${(values.reduce((sum, value) => sum + value, 0) / values.length).toFixed(2)} m` : "—";
};

function AxisSelect({ label, value, variables, onChange }: {
  label: string; value: string; variables: ScalarEnsembleResult<string>["variables"];
  onChange: (key: string) => void;
}): JSX.Element {
  return <label className="text-xs text-slate-300"><span className="mb-1 block">{label}</span>
    <select aria-label={label} value={value} className={INPUT_CLASS}
      title={`Choose the ${label.toLowerCase()} from all available scalar variables`}
      onChange={(event) => onChange(event.target.value)}>{variables.map((variable) =>
        <option key={variable.key} value={variable.key}>{variable.label} [{variable.unit}]</option>)}</select>
  </label>;
}
