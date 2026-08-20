import { useMemo, useRef, useState } from "react";

import type { VariationDatasetTs } from "../model/variation";
import {
  buildScalarPlotVariables,
  buildScalarScatter,
  distributionMatrixToCsv,
  type ScalarPlotVariableTs,
  type ScalarScatterDataTs,
} from "../model/variationPlotData";
import {
  makeVariationPlotDefinition,
  variationPlotDefinitionToJson,
  variationResultFingerprint,
} from "../model/variationPlotDefinition";
import type {
  SwingTrialStatusTs,
  SwingVariationResultTs,
} from "../model/variationSwingEnsemble";
import { BUTTON_CLASS, downloadSvgElement, downloadText, INPUT_CLASS } from "./variationUi";

interface VariationScatterProps {
  dataset: VariationDatasetTs;
  ensemble?: SwingVariationResultTs | null;
  selectedTrialIndex: number | null;
  onSelectedTrialChange: (trialIndex: number | null) => void;
}

interface ScatterViewModel {
  variables: ScalarPlotVariableTs[];
  xKey: string;
  yKey: string;
  setXKey: (value: string) => void;
  setYKey: (value: string) => void;
  scatter: ScalarScatterDataTs;
  resultId: string;
  cohort: (trialIndex: number) => DisplayCohort;
  ensembleCounts: EnsembleCount[] | null;
}

type DisplayCohort = SwingTrialStatusTs | "evaluated" | "failure";
interface EnsembleCount {
  status: SwingTrialStatusTs;
  total: number;
  plotted: number;
  unavailable: number;
}

const WIDTH = 640;
const HEIGHT = 360;
const MARGIN = { left: 66, right: 24, top: 28, bottom: 62 };

const axisLabel = (variable: ScalarPlotVariableTs): string =>
  variable.unit ? `${variable.label} [${variable.unit}]` : variable.label;

export function VariationScatter(props: VariationScatterProps): JSX.Element {
  const svgRef = useRef<SVGSVGElement>(null);
  const model = useScatterViewModel(props.dataset, props.ensemble ?? null);
  return (
    <div className="space-y-3">
      <ScatterSelectors model={model} {...props} />
      <ScatterExports svgRef={svgRef} model={model} {...props} />
      <ScatterAvailability model={model} />
      <ScatterCanvas svgRef={svgRef} model={model} {...props} />
      <ScatterTable model={model} {...props} />
    </div>
  );
}

function useScatterViewModel(
  dataset: VariationDatasetTs,
  ensemble: SwingVariationResultTs | null,
): ScatterViewModel {
  const variables = useMemo(() => buildScalarPlotVariables(dataset), [dataset]);
  const defaultX = variables.find((item) => item.kind === "input")?.key ?? variables[0].key;
  const defaultY = variables.find((item) => item.key === "output:carry_m")?.key
    ?? variables.find((item) => item.kind === "shot")?.key
    ?? variables[variables.length - 1].key;
  const [requestedX, setXKey] = useState(defaultX);
  const [requestedY, setYKey] = useState(defaultY);
  const xKey = variables.some((item) => item.key === requestedX) ? requestedX : defaultX;
  const yKey = variables.some((item) => item.key === requestedY) ? requestedY : defaultY;
  const scatter = useMemo(
    () => buildScalarScatter(dataset, xKey, yKey), [dataset, xKey, yKey],
  );
  const cohort = (trialIndex: number): DisplayCohort =>
    ensemble?.runs[trialIndex]?.status
      ?? scatter.points.find((point) => point.trialIndex === trialIndex)?.cohort
      ?? "failure";
  return {
    variables, xKey, yKey, setXKey, setYKey, scatter, cohort,
    resultId: variationResultFingerprint(ensemble ?? dataset),
    ensembleCounts: ensemble ? buildEnsembleCounts(ensemble, scatter, cohort) : null,
  };
}

function buildEnsembleCounts(
  ensemble: SwingVariationResultTs,
  scatter: ScalarScatterDataTs,
  cohort: (trialIndex: number) => DisplayCohort,
): EnsembleCount[] {
  const statuses: SwingTrialStatusTs[] = [
    "evaluated_hit", "evaluated_no_impact", "numerical_failure",
  ];
  return statuses.map((status) => {
    const total = ensemble.runs.filter((run) => run.status === status).length;
    const plotted = scatter.points.filter((point) => cohort(point.trialIndex) === status).length;
    return { status, total, plotted, unavailable: total - plotted };
  });
}

function ScatterSelectors({ model, selectedTrialIndex, onSelectedTrialChange }:
  { model: ScatterViewModel } & Pick<VariationScatterProps,
  "selectedTrialIndex" | "onSelectedTrialChange">): JSX.Element {
  return <>
    <div className="grid gap-3 sm:grid-cols-2">
      <AxisSelect label="Scatter horizontal axis" value={model.xKey}
        variables={model.variables} onChange={model.setXKey} />
      <AxisSelect label="Scatter vertical axis" value={model.yKey}
        variables={model.variables} onChange={model.setYKey} />
    </div>
    <TrialSelect trialIndices={model.scatter.points.map((point) => point.trialIndex)}
      selectedTrialIndex={selectedTrialIndex} onSelectedTrialChange={onSelectedTrialChange} />
  </>;
}

function ScatterExports({ svgRef, model, dataset, ensemble, selectedTrialIndex }:
  { svgRef: React.RefObject<SVGSVGElement>; model: ScatterViewModel }
  & Omit<VariationScatterProps, "onSelectedTrialChange">): JSX.Element {
  const definition = makeVariationPlotDefinition(ensemble ?? dataset, {
    plotType: "scalar_scatter", coordinateFrame: null,
    xVariableKey: model.xKey, yVariableKey: model.yKey, pointId: null,
    positionUnit: null, alignmentBasis: null,
    dispersionMetric: null, dispersionUnit: null, quietThreshold: null,
    confidenceLevel: null, minQuietDurationS: null, minQuietSamples: null,
    selectedTrialIndex, cameraYawDeg: null, cameraPitchDeg: null, cameraZoom: null,
    outcomeFilter: null, phaseEndFraction: null, perturbationSourceKey: null,
    perturbationBand: null, variableKeys: null, showConfidenceEllipsoids: null,
  });
  const outcomes = ensemble?.runs.map((run) => run.status);
  return <div className="flex flex-wrap gap-2">
    <button type="button" className={BUTTON_CLASS}
      onClick={() => svgRef.current && downloadSvgElement(`${model.resultId}-scatter.svg`, svgRef.current)}>
      Scatter SVG
    </button>
    <button type="button" className={BUTTON_CLASS} onClick={() => downloadText(
      `${model.resultId}-scatter.csv`,
      distributionMatrixToCsv(dataset, [model.xKey, model.yKey], outcomes),
      "text/csv;charset=utf-8",
    )}>Scatter Selected CSV</button>
    <button type="button" className={BUTTON_CLASS} onClick={() => downloadText(
      `${model.resultId}-scatter.plot.json`, variationPlotDefinitionToJson(definition),
      "application/json",
    )}>Scatter Plot Definition JSON</button>
  </div>;
}

function ScatterAvailability({ model }: { model: ScatterViewModel }): JSX.Element {
  const scalar = model.scatter.cohorts;
  const message = model.ensembleCounts
    ? model.ensembleCounts.map((entry) => availabilityMessage(entry)).join(" · ")
    : `Evaluated: ${scalar.evaluated.plotted}/${scalar.evaluated.total} plotted · `
      + `Failures: ${scalar.failure.plotted}/${scalar.failure.total} plotted`
      + `${scalar.failure.unavailable ? `, ${scalar.failure.unavailable} unavailable` : ""}. `
      + "Scalar studies do not expose a geometric no-impact cohort.";
  return <p className="text-xs text-slate-400" aria-live="polite">{message}</p>;
}

function availabilityMessage(entry: EnsembleCount): string {
  const missing = entry.unavailable ? `, ${entry.unavailable} unavailable` : "";
  return `${cohortLabel(entry.status)}: ${entry.plotted}/${entry.total} plotted${missing}`;
}

function ScatterCanvas({ svgRef, model, selectedTrialIndex, onSelectedTrialChange }:
  { svgRef: React.RefObject<SVGSVGElement>; model: ScatterViewModel }
  & Pick<VariationScatterProps,
  "selectedTrialIndex" | "onSelectedTrialChange">): JSX.Element {
  const bounds = plotBounds(model.scatter.points.map((point) => [point.x, point.y]));
  return <svg ref={svgRef} viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
    className="h-auto w-full rounded-lg border border-slate-800 bg-slate-950/60"
    role="img" aria-label={`Variation scatter: ${axisLabel(model.scatter.xVariable)} versus ${axisLabel(model.scatter.yVariable)}`}>
    <line x1={MARGIN.left} y1={HEIGHT - MARGIN.bottom} x2={WIDTH - MARGIN.right}
      y2={HEIGHT - MARGIN.bottom} stroke="#64748b" />
    <line x1={MARGIN.left} y1={MARGIN.top} x2={MARGIN.left}
      y2={HEIGHT - MARGIN.bottom} stroke="#64748b" />
    {model.scatter.points.map((point) => <ScatterPoint key={point.trialIndex}
      point={point} bounds={bounds} cohort={model.cohort(point.trialIndex)}
      selected={point.trialIndex === selectedTrialIndex}
      dimmed={selectedTrialIndex !== null && point.trialIndex !== selectedTrialIndex}
      onSelect={() => onSelectedTrialChange(
        point.trialIndex === selectedTrialIndex ? null : point.trialIndex,
      )} />)}
    <ScatterAxisLabels scatter={model.scatter} />
  </svg>;
}

function ScatterPoint({ point, bounds, cohort, selected, dimmed, onSelect }:
  { point: ScalarScatterDataTs["points"][number]; bounds: PlotBounds;
    cohort: DisplayCohort; selected: boolean; dimmed: boolean; onSelect: () => void }): JSX.Element {
  return <circle cx={scaleX(point.x, bounds)} cy={scaleY(point.y, bounds)}
    r={selected ? 7 : 4} fill={cohortColor(cohort)} opacity={dimmed ? "0.32" : "0.9"}
    stroke={selected ? "#f8fafc" : "none"} strokeWidth={selected ? "2" : "0"}
    role="button" tabIndex={0} aria-label={`Select Trial ${point.trialIndex + 1}`}
    onClick={onSelect} onKeyDown={(event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault(); onSelect();
      }
    }}><title>{`Trial ${point.trialIndex + 1}: ${point.x.toPrecision(5)}, ${point.y.toPrecision(5)}`}</title></circle>;
}

function ScatterAxisLabels({ scatter }: { scatter: ScalarScatterDataTs }): JSX.Element {
  return <>
    <text x={(MARGIN.left + WIDTH - MARGIN.right) / 2} y={HEIGHT - 18}
      textAnchor="middle" fill="#cbd5e1" fontSize="13">{axisLabel(scatter.xVariable)}</text>
    <text transform={`translate(18 ${(MARGIN.top + HEIGHT - MARGIN.bottom) / 2}) rotate(-90)`}
      textAnchor="middle" fill="#cbd5e1" fontSize="13">{axisLabel(scatter.yVariable)}</text>
    {scatter.points.length === 0 && <text x={WIDTH / 2} y={HEIGHT / 2}
      textAnchor="middle" fill="#94a3b8">No finite paired values for these axes</text>}
  </>;
}

function ScatterTable({ model, selectedTrialIndex, onSelectedTrialChange }:
  { model: ScatterViewModel } & Pick<VariationScatterProps,
  "selectedTrialIndex" | "onSelectedTrialChange">): JSX.Element {
  return <details><summary className="cursor-pointer text-xs text-slate-300">
    Accessible Plotted-Trial Data ({model.scatter.points.length})
  </summary><div className="mt-2 max-h-64 overflow-auto rounded border border-slate-800">
    <table className="w-full text-left text-xs text-slate-300"><thead className="sticky top-0 bg-slate-950">
      <tr><th className="px-2 py-1">Trial</th><th className="px-2 py-1">Cohort</th>
        <th className="px-2 py-1">{axisLabel(model.scatter.xVariable)}</th>
        <th className="px-2 py-1">{axisLabel(model.scatter.yVariable)}</th></tr>
    </thead><tbody>{model.scatter.points.map((point) => <tr key={point.trialIndex}
      className={point.trialIndex === selectedTrialIndex ? "bg-sky-900/40" : "border-t border-slate-800/60"}>
      <td className="px-2 py-1"><button type="button" className="underline"
        onClick={() => onSelectedTrialChange(point.trialIndex)}>Trial {point.trialIndex + 1}</button></td>
      <td className="px-2 py-1">{model.cohort(point.trialIndex).replace(/_/g, " ")}</td>
      <td className="px-2 py-1 tabular-nums">{point.x.toPrecision(7)}</td>
      <td className="px-2 py-1 tabular-nums">{point.y.toPrecision(7)}</td>
    </tr>)}</tbody></table>
  </div></details>;
}

function TrialSelect({ trialIndices, selectedTrialIndex, onSelectedTrialChange }:
  { trialIndices: number[]; selectedTrialIndex: number | null;
    onSelectedTrialChange: (trialIndex: number | null) => void }): JSX.Element {
  return <label className="block text-xs text-slate-300">
    <span className="mb-1 block">Highlighted Trial (Linked Across Plots)</span>
    <select aria-label="Highlighted trial" className={INPUT_CLASS}
      value={selectedTrialIndex ?? ""} onChange={(event) => onSelectedTrialChange(
        event.target.value === "" ? null : Number(event.target.value),
      )}><option value="">All Trials</option>{trialIndices.map((trialIndex) =>
        <option key={trialIndex} value={trialIndex}>Trial {trialIndex + 1}</option>)}</select>
  </label>;
}

const cohortLabel = (status: SwingTrialStatusTs): string => ({
  evaluated_hit: "Hits", evaluated_no_impact: "No impact",
  numerical_failure: "Numerical failures",
})[status];

const cohortColor = (status: DisplayCohort): string => ({
  evaluated_hit: "#38bdf8", evaluated_no_impact: "#f59e0b",
  numerical_failure: "#ef6464", evaluated: "#38bdf8", failure: "#ef6464",
})[status];

function AxisSelect({ label, value, variables, onChange }:
  { label: string; value: string; variables: ScalarPlotVariableTs[];
    onChange: (value: string) => void }): JSX.Element {
  return <label className="text-xs text-slate-300"><span className="mb-1 block">{label}</span>
    <select aria-label={label} className={INPUT_CLASS} value={value}
      onChange={(event) => onChange(event.target.value)}>{variables.map((variable) =>
        <option key={variable.key} value={variable.key}>{axisLabel(variable)}</option>)}</select>
  </label>;
}

interface PlotBounds { xMin: number; xMax: number; yMin: number; yMax: number }

function plotBounds(points: number[][]): PlotBounds {
  const extent = (values: number[]): [number, number] => {
    if (values.length === 0) return [-1, 1];
    const low = Math.min(...values); const high = Math.max(...values);
    const padding = Math.max((high - low) * 0.08, Math.max(Math.abs(low), 1) * 1e-6);
    return [low - padding, high + padding];
  };
  const [xMin, xMax] = extent(points.map((point) => point[0]));
  const [yMin, yMax] = extent(points.map((point) => point[1]));
  return { xMin, xMax, yMin, yMax };
}

const scaleX = (value: number, bounds: PlotBounds): number =>
  MARGIN.left + ((value - bounds.xMin) / (bounds.xMax - bounds.xMin))
    * (WIDTH - MARGIN.left - MARGIN.right);
const scaleY = (value: number, bounds: PlotBounds): number =>
  HEIGHT - MARGIN.bottom - ((value - bounds.yMin) / (bounds.yMax - bounds.yMin))
    * (HEIGHT - MARGIN.top - MARGIN.bottom);
