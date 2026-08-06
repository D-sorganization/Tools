import { useMemo, useState } from "react";

import type { VariationDatasetTs } from "../model/variation";
import {
  buildScalarPlotVariables,
  buildScalarScatter,
  type ScalarPlotVariableTs,
} from "../model/variationPlotData";
import { INPUT_CLASS } from "./variationUi";
import type {
  SwingTrialStatusTs,
  SwingVariationResultTs,
} from "../model/variationSwingEnsemble";

interface VariationScatterProps {
  dataset: VariationDatasetTs;
  ensemble?: SwingVariationResultTs | null;
}

const WIDTH = 640;
const HEIGHT = 360;
const MARGIN = { left: 66, right: 24, top: 28, bottom: 62 };

const axisLabel = (variable: ScalarPlotVariableTs): string =>
  variable.unit ? `${variable.label} [${variable.unit}]` : variable.label;

export function VariationScatter({ dataset, ensemble = null }: VariationScatterProps): JSX.Element {
  const variables = useMemo(() => buildScalarPlotVariables(dataset), [dataset]);
  const defaultX = variables.find((item) => item.kind === "input")?.key ?? variables[0].key;
  const defaultY = variables.find((item) => item.key === "output:carry_m")?.key
    ?? variables.find((item) => item.kind === "shot")?.key
    ?? variables[variables.length - 1].key;
  const [xKey, setXKey] = useState(defaultX);
  const [yKey, setYKey] = useState(defaultY);
  const safeX = variables.some((item) => item.key === xKey) ? xKey : defaultX;
  const safeY = variables.some((item) => item.key === yKey) ? yKey : defaultY;
  const scatter = useMemo(
    () => buildScalarScatter(dataset, safeX, safeY),
    [dataset, safeX, safeY],
  );
  const bounds = plotBounds(scatter.points.map((point) => [point.x, point.y]));
  const cohort = (trialIndex: number): SwingTrialStatusTs | "evaluated" | "failure" =>
    ensemble?.runs[trialIndex]?.status ?? scatter.points.find(
      (point) => point.trialIndex === trialIndex,
    )?.cohort ?? "failure";
  const ensembleCounts = ensemble && ([
    "evaluated_hit", "evaluated_no_impact", "numerical_failure",
  ] as SwingTrialStatusTs[]).map((status) => {
    const total = ensemble.runs.filter((run) => run.status === status).length;
    const plotted = scatter.points.filter((point) => cohort(point.trialIndex) === status).length;
    return { status, total, plotted, unavailable: total - plotted };
  });

  return (
    <div className="space-y-3">
      <div className="grid gap-3 sm:grid-cols-2">
        <AxisSelect
          label="Scatter horizontal axis"
          value={safeX}
          variables={variables}
          onChange={setXKey}
        />
        <AxisSelect
          label="Scatter vertical axis"
          value={safeY}
          variables={variables}
          onChange={setYKey}
        />
      </div>
      <p className="text-xs text-slate-400" aria-live="polite">
        {ensembleCounts
          ? ensembleCounts.map((entry) => `${cohortLabel(entry.status)}: ${entry.plotted}/${entry.total} plotted${entry.unavailable ? `, ${entry.unavailable} unavailable` : ""}`).join(" · ")
          : `Evaluated: ${scatter.cohorts.evaluated.plotted}/${scatter.cohorts.evaluated.total} plotted · Failures: ${scatter.cohorts.failure.plotted}/${scatter.cohorts.failure.total} plotted${scatter.cohorts.failure.unavailable ? `, ${scatter.cohorts.failure.unavailable} unavailable` : ""}. Scalar studies do not expose a geometric no-impact cohort.`}
      </p>
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        className="h-auto w-full rounded-lg border border-slate-800 bg-slate-950/60"
        role="img"
        aria-label={`Variation scatter: ${axisLabel(scatter.xVariable)} versus ${axisLabel(scatter.yVariable)}`}
      >
        <line x1={MARGIN.left} y1={HEIGHT - MARGIN.bottom} x2={WIDTH - MARGIN.right} y2={HEIGHT - MARGIN.bottom} stroke="#64748b" />
        <line x1={MARGIN.left} y1={MARGIN.top} x2={MARGIN.left} y2={HEIGHT - MARGIN.bottom} stroke="#64748b" />
        {scatter.points.map((point) => (
          <circle
            key={point.trialIndex}
            cx={scaleX(point.x, bounds)}
            cy={scaleY(point.y, bounds)}
            r="4"
            fill={cohortColor(cohort(point.trialIndex))}
            opacity="0.78"
          >
            <title>{`Trial ${point.trialIndex + 1}: ${point.x.toPrecision(5)}, ${point.y.toPrecision(5)}`}</title>
          </circle>
        ))}
        <text x={(MARGIN.left + WIDTH - MARGIN.right) / 2} y={HEIGHT - 18} textAnchor="middle" fill="#cbd5e1" fontSize="13">
          {axisLabel(scatter.xVariable)}
        </text>
        <text transform={`translate(18 ${(MARGIN.top + HEIGHT - MARGIN.bottom) / 2}) rotate(-90)`} textAnchor="middle" fill="#cbd5e1" fontSize="13">
          {axisLabel(scatter.yVariable)}
        </text>
        {scatter.points.length === 0 && (
          <text x={WIDTH / 2} y={HEIGHT / 2} textAnchor="middle" fill="#94a3b8">
            No finite paired values for these axes
          </text>
        )}
      </svg>
    </div>
  );
}

const cohortLabel = (status: SwingTrialStatusTs): string => ({
  evaluated_hit: "Hits",
  evaluated_no_impact: "No impact",
  numerical_failure: "Numerical failures",
})[status];

const cohortColor = (
  status: SwingTrialStatusTs | "evaluated" | "failure",
): string => ({
  evaluated_hit: "#38bdf8",
  evaluated_no_impact: "#f59e0b",
  numerical_failure: "#ef6464",
  evaluated: "#38bdf8",
  failure: "#ef6464",
})[status];

function AxisSelect({
  label,
  value,
  variables,
  onChange,
}: {
  label: string;
  value: string;
  variables: ScalarPlotVariableTs[];
  onChange: (value: string) => void;
}): JSX.Element {
  return (
    <label className="text-xs text-slate-300">
      <span className="mb-1 block">{label}</span>
      <select
        aria-label={label}
        className={INPUT_CLASS}
        value={value}
        onChange={(event) => onChange(event.target.value)}
      >
        {variables.map((variable) => (
          <option key={variable.key} value={variable.key}>{axisLabel(variable)}</option>
        ))}
      </select>
    </label>
  );
}

interface PlotBounds {
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
}

function plotBounds(points: number[][]): PlotBounds {
  const x = points.map((point) => point[0]);
  const y = points.map((point) => point[1]);
  const extent = (values: number[]): [number, number] => {
    if (values.length === 0) return [-1, 1];
    const low = Math.min(...values);
    const high = Math.max(...values);
    const padding = Math.max((high - low) * 0.08, Math.max(Math.abs(low), 1) * 1e-6);
    return [low - padding, high + padding];
  };
  const [xMin, xMax] = extent(x);
  const [yMin, yMax] = extent(y);
  return { xMin, xMax, yMin, yMax };
}

const scaleX = (value: number, bounds: PlotBounds): number =>
  MARGIN.left + ((value - bounds.xMin) / (bounds.xMax - bounds.xMin))
    * (WIDTH - MARGIN.left - MARGIN.right);

const scaleY = (value: number, bounds: PlotBounds): number =>
  HEIGHT - MARGIN.bottom - ((value - bounds.yMin) / (bounds.yMax - bounds.yMin))
    * (HEIGHT - MARGIN.top - MARGIN.bottom);
