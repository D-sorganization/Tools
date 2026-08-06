import { useMemo, useState } from "react";

import type { VariationDatasetTs } from "../model/variation";
import {
  buildScalarMarginal,
  buildScalarPlotVariables,
  buildScalarScatter,
  distributionMatrixToCsv,
  distributionMatrixToSvg,
  matrixCohortColor,
  scalarValues,
} from "../model/variationPlotData";
import {
  makeVariationPlotDefinition,
  variationPlotDefinitionToJson,
  variationResultFingerprint,
} from "../model/variationPlotDefinition";
import { BUTTON_CLASS, downloadText, INPUT_CLASS } from "./variationUi";
import type { SwingVariationResultTs } from "../model/variationSwingEnsemble";

interface Props {
  dataset: VariationDatasetTs;
  ensemble?: SwingVariationResultTs | null;
  selectedTrialIndex: number | null;
  onSelectedTrialChange: (trialIndex: number | null) => void;
}
const SIZE = 150;
const PAD = 14;

export function VariationDistributionMatrix({
  dataset,
  ensemble = null,
  selectedTrialIndex,
  onSelectedTrialChange,
}: Props): JSX.Element {
  const variables = useMemo(() => buildScalarPlotVariables(dataset), [dataset]);
  const defaults = useMemo(() => {
    const input = variables.find((item) => item.kind === "input");
    const impact = variables.find((item) => item.kind === "impact");
    const carry = variables.find((item) => item.key === "output:carry_m");
    const lateral = variables.find((item) => item.key === "output:lateral_m");
    return [input, impact, carry, lateral].filter(Boolean).map((item) => item!.key);
  }, [variables]);
  const [keys, setKeys] = useState(defaults);
  const selected = keys.map((key) => variables.find((item) => item.key === key) ?? variables[0]);
  const selectedKeys = selected.map((item) => item.key);
  const selectedColumns = selected.map((variable) => scalarValues(dataset, variable));
  const result = ensemble ?? dataset;
  const resultId = variationResultFingerprint(result);
  const outcomes = ensemble?.runs.map((run) => run.status);
  return (
    <div className="space-y-3">
      <div className="grid gap-2 md:grid-cols-4">
        {selected.map((variable, index) => (
          <label key={index} className="text-xs text-slate-300">
            <span className="mb-1 block">Matrix Variable {index + 1}</span>
            <select aria-label={`Matrix variable ${index + 1}`} className={INPUT_CLASS} value={variable.key} onChange={(event) => setKeys((current) => current.map((key, keyIndex) => keyIndex === index ? event.target.value : key))}>
              {variables.map((item) => <option key={item.key} value={item.key}>{item.label} [{item.unit || "unitless"}]</option>)}
            </select>
          </label>
        ))}
      </div>
      <div className="flex flex-wrap gap-2">
        <button type="button" className={BUTTON_CLASS} onClick={() => downloadText(
          `${resultId}-distribution-matrix.svg`,
          distributionMatrixToSvg(dataset, selectedKeys, outcomes),
          "image/svg+xml;charset=utf-8",
        )}>Matrix SVG</button>
        <button type="button" className={BUTTON_CLASS} onClick={() => downloadText(
          `${resultId}-distribution-matrix.csv`,
          distributionMatrixToCsv(dataset, selectedKeys, outcomes),
          "text/csv;charset=utf-8",
        )}>Matrix Selected CSV</button>
        <button type="button" className={BUTTON_CLASS} onClick={() => downloadText(
          `${resultId}-distribution-matrix.plot.json`,
          variationPlotDefinitionToJson(makeVariationPlotDefinition(result, {
            plotType: "distribution_matrix", coordinateFrame: ensemble?.coordinateFrame ?? null,
            xVariableKey: null, yVariableKey: null, pointId: null,
            positionUnit: null, alignmentBasis: null, quietThresholdM: null,
            selectedTrialIndex: null, cameraYawDeg: null, cameraPitchDeg: null,
            cameraZoom: null, outcomeFilter: null, phaseEndFraction: null,
            perturbationSourceKey: null, perturbationBand: null,
            variableKeys: selectedKeys,
          })),
          "application/json",
        )}>Matrix Plot Definition JSON</button>
      </div>
      <div className="overflow-auto">
        <div className="grid min-w-max" style={{ gridTemplateColumns: `repeat(${selected.length}, ${SIZE}px)` }} role="group" aria-label="Scatter matrix with marginal histograms">
          {selected.flatMap((row, rowIndex) => selected.map((column, columnIndex) => (
            <MatrixCell key={`${row.key}:${column.key}`} dataset={dataset} xKey={column.key} yKey={row.key} diagonal={rowIndex === columnIndex} outcomes={outcomes} selectedTrialIndex={selectedTrialIndex} onSelectedTrialChange={onSelectedTrialChange} />
          )))}
        </div>
      </div>
      <p className="text-xs text-slate-500">Diagonal cells are marginal histograms. Off-diagonal cells retain only finite paired values; misses and failures remain counted as unavailable in the canonical result and exports.</p>
      <details className="text-xs text-slate-400">
        <summary className="cursor-pointer">Accessible Selected Matrix Data</summary>
        <div className="mt-2 overflow-auto">
          <table className="w-full text-left">
            <thead><tr><th>Trial</th><th>Outcome</th>{selected.map((variable) => <th key={variable.key}>{variable.label} [{variable.unit || "unitless"}]</th>)}</tr></thead>
            <tbody>{dataset.success.map((success, trialIndex) => <tr key={trialIndex}>
              <td><button type="button" className="underline" aria-label={`Select matrix trial ${trialIndex + 1}`} onClick={() => onSelectedTrialChange(trialIndex)}>{trialIndex + 1}</button></td><td>{outcomes?.[trialIndex]?.split("_").join(" ") ?? (success ? "Evaluated" : "Failure")}</td>
              {selectedColumns.map((column, index) => <td key={selected[index].key}>{column[trialIndex] ?? "Unavailable"}</td>)}
            </tr>)}</tbody>
          </table>
        </div>
      </details>
    </div>
  );
}

function MatrixCell({ dataset, xKey, yKey, diagonal, outcomes, selectedTrialIndex, onSelectedTrialChange }: { dataset: VariationDatasetTs; xKey: string; yKey: string; diagonal: boolean; outcomes?: string[]; selectedTrialIndex: number | null; onSelectedTrialChange: (trialIndex: number | null) => void }): JSX.Element {
  if (diagonal) {
    const marginal = buildScalarMarginal(dataset, xKey);
    const maximum = Math.max(...marginal.counts, 1);
    return <svg viewBox={`0 0 ${SIZE} ${SIZE}`} role="img" aria-label={`${marginal.variable.label} marginal histogram; ${marginal.nAvailable} available, ${marginal.nMissing} missing`} className="border border-slate-800 bg-slate-950/60">
      {marginal.counts.map((count, index) => <rect key={index} x={PAD + index * (SIZE - 2 * PAD) / marginal.counts.length} y={SIZE - PAD - count / maximum * (SIZE - 2 * PAD)} width={Math.max((SIZE - 2 * PAD) / marginal.counts.length - 1, 1)} height={count / maximum * (SIZE - 2 * PAD)} fill="#38bdf8" opacity="0.75" />)}
    </svg>;
  }
  const scatter = buildScalarScatter(dataset, xKey, yKey);
  const xs = scatter.points.map((point) => point.x);
  const ys = scatter.points.map((point) => point.y);
  const scale = (value: number, values: number[]) => PAD + (value - Math.min(...values)) / Math.max(Math.max(...values) - Math.min(...values), 1e-12) * (SIZE - 2 * PAD);
  return <svg viewBox={`0 0 ${SIZE} ${SIZE}`} role="img" aria-label={`${scatter.xVariable.label} versus ${scatter.yVariable.label}; ${scatter.points.length} paired trials`} className="border border-slate-800 bg-slate-950/60">
    {scatter.points.map((point) => <circle key={point.trialIndex} cx={scale(point.x, xs)} cy={SIZE - scale(point.y, ys)} r={point.trialIndex === selectedTrialIndex ? "4" : "2.3"} fill={matrixCohortColor(outcomes?.[point.trialIndex] ?? point.cohort)} stroke={point.trialIndex === selectedTrialIndex ? "#f8fafc" : "none"} opacity="0.65" role="button" tabIndex={0} aria-label={`Select trial ${point.trialIndex + 1} in matrix`} onClick={() => onSelectedTrialChange(point.trialIndex)} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") onSelectedTrialChange(point.trialIndex); }}><title>{`Trial ${point.trialIndex + 1}`}</title></circle>)}
  </svg>;
}
