import { useMemo, useState } from "react";

import type { VariationDatasetTs } from "../model/variation";
import {
  buildScalarMarginal,
  buildScalarPlotVariables,
  buildScalarScatter,
} from "../model/variationPlotData";
import { INPUT_CLASS } from "./variationUi";

interface Props { dataset: VariationDatasetTs }
const SIZE = 150;
const PAD = 14;

export function VariationDistributionMatrix({ dataset }: Props): JSX.Element {
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
      <div className="overflow-auto">
        <div className="grid min-w-max" style={{ gridTemplateColumns: `repeat(${selected.length}, ${SIZE}px)` }} role="group" aria-label="Scatter matrix with marginal histograms">
          {selected.flatMap((row, rowIndex) => selected.map((column, columnIndex) => (
            <MatrixCell key={`${row.key}:${column.key}`} dataset={dataset} xKey={column.key} yKey={row.key} diagonal={rowIndex === columnIndex} />
          )))}
        </div>
      </div>
      <p className="text-xs text-slate-500">Diagonal cells are marginal histograms. Off-diagonal cells retain only finite paired values; misses and failures remain counted as unavailable in the canonical result and exports.</p>
    </div>
  );
}

function MatrixCell({ dataset, xKey, yKey, diagonal }: { dataset: VariationDatasetTs; xKey: string; yKey: string; diagonal: boolean }): JSX.Element {
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
    {scatter.points.map((point) => <circle key={point.trialIndex} cx={scale(point.x, xs)} cy={SIZE - scale(point.y, ys)} r="2.3" fill={point.cohort === "evaluated" ? "#38bdf8" : "#ef6464"} opacity="0.65"><title>{`Trial ${point.trialIndex + 1}`}</title></circle>)}
  </svg>;
}
