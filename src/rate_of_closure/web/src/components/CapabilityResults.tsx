import { useEffect, useMemo, useState } from "react";

import type { CapabilityRunOutput } from "../model/capabilityRun";
import { stableCapabilityObservationEnsembleJson } from "../model/capabilityObservationEnsemble";
import {
  capabilityAlternativesCsv, stableCapabilityResultExportJson,
} from "../model/capabilityResultExport";
import {
  buildScalarEnsembleScatter, type ScalarVariableDefinition,
} from "../model/scalarEnsembleContract";
import { nonCompleteReasonSummary, scalarEnsembleToCsv } from "../model/scalarEnsembleCsv";
import { ScalarEnsembleScatter } from "./ScalarEnsembleScatter";
import { BUTTON_CLASS, INPUT_CLASS, downloadText } from "./variationUi";

const PAGE_SIZE = 25;

const axisLabel = (
  variable: ScalarVariableDefinition, variables: readonly ScalarVariableDefinition[],
): string => {
  const duplicate = variables.some((item) => item.key !== variable.key
    && item.label === variable.label && item.unit === variable.unit);
  const stage = variable.stage_key.charAt(0).toUpperCase() + variable.stage_key.slice(1);
  return `${duplicate ? `${stage} · ` : ""}${variable.label} [${variable.unit}]`;
};

function Alternatives({ output }: { readonly output: CapabilityRunOutput }) {
  const units = Object.fromEntries(output.ensemble.variables
    .filter(({ key }) => key.startsWith("nominal."))
    .map(({ key, unit }) => [key.slice("nominal.".length), unit]));
  const recommendation = (item: CapabilityRunOutput["result"]["alternatives"][number]): string =>
    item.parameters.map((value) => {
      const unit = units[value.parameterId];
      if (unit === undefined) throw new RangeError(`missing unit for ${value.parameterId}`);
      return `${value.parameterId}=${value.value.toPrecision(5)} ${unit}`;
    }).join(" · ");
  return <div className="overflow-x-auto"><table aria-label="Ranked capability alternatives"
    className="w-full whitespace-nowrap text-left text-xs text-slate-300"><thead><tr>
      <th className="px-2 py-1">Rank / Club</th><th className="px-2 py-1">Recommendation</th>
      <th className="px-2 py-1">Score</th>
      <th className="px-2 py-1">Carry</th><th className="px-2 py-1">Mean miss</th>
      <th className="px-2 py-1">Dispersion</th><th className="px-2 py-1">Target hold</th>
      <th className="px-2 py-1">Miss CVaR</th><th className="px-2 py-1">Downside carry</th>
      <th className="px-2 py-1">Outcomes</th><th className="px-2 py-1">Failure rate</th>
      <th className="px-2 py-1">Evidence</th><th className="px-2 py-1">Pareto</th>
    </tr></thead><tbody>{output.result.alternatives.map((item) => <tr key={item.rank}
      className="border-t border-slate-800"><td className="px-2 py-1">{item.rank}. {item.clubId}</td>
      <td className="px-2 py-1">{recommendation(item)}</td>
      <td className="px-2 py-1">{item.score.toPrecision(5)}</td>
      <td className="px-2 py-1">{item.meanCarryM.toFixed(2)} m</td>
      <td className="px-2 py-1">{item.expectedMissM.toFixed(2)} m</td>
      <td className="px-2 py-1">{item.dispersionRmsM.toFixed(2)} m</td>
      <td className="px-2 py-1">{(100 * item.targetHoldProbability).toFixed(1)}%</td>
      <td className="px-2 py-1">{item.cvarMissM.toFixed(2)} m</td>
      <td className="px-2 py-1">{item.downsideCarryM.toFixed(2)} m</td>
      <td className="px-2 py-1">{item.successfulCount}/{item.sampleCount} complete · {
        item.noImpactCount} no impact · {item.failedCount} failed</td>
      <td className="px-2 py-1">{(100 * item.failureFraction).toFixed(1)}%</td>
      <td className="px-2 py-1">{(100 * item.confidence).toFixed(1)}% · {
        item.extrapolated ? "extrapolated" : "within envelope"} · {
        item.limitingConstraints.join(", ") || "no limits"}</td>
      <td className="px-2 py-1">{item.paretoEfficient ? "efficient" : "dominated"}</td>
    </tr>)}</tbody></table></div>;
}

function RawRows({ output }: { readonly output: CapabilityRunOutput }) {
  const [page, setPage] = useState(0);
  useEffect(() => setPage(0), [output.ensemble]);
  const rows = output.ensemble.rows.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);
  const pages = Math.max(1, Math.ceil(output.ensemble.rows.length / PAGE_SIZE));
  return <section aria-label="Capability raw observation rows" className="space-y-2">
    <div className="flex items-center gap-2 text-xs text-slate-300">
      <button className={BUTTON_CLASS} type="button" disabled={page === 0}
        onClick={() => setPage((value) => value - 1)}>Previous rows</button>
      <span>Page {page + 1} of {pages}</span>
      <button className={BUTTON_CLASS} type="button" disabled={page + 1 >= pages}
        onClick={() => setPage((value) => value + 1)}>Next rows</button>
    </div>
    <div className="max-h-72 overflow-auto"><table className="w-full text-left text-xs text-slate-300">
      <thead><tr><th className="px-2">Row</th><th className="px-2">Cohort</th>
        <th className="px-2">Series</th><th className="px-2">Available scalars</th></tr></thead>
      <tbody>{rows.map((row) => <tr key={row.row_id} className="border-t border-slate-800">
        <td className="px-2 py-1">{row.row_id}</td><td className="px-2">{row.cohort}</td>
        <td className="px-2">{row.series_id}</td><td className="px-2">{
          Object.values(row.values).filter((value) => value !== null).length}</td></tr>)}</tbody>
    </table></div>
  </section>;
}

export function CapabilityResults({ output }: { readonly output: CapabilityRunOutput }) {
  const variables = output.ensemble.variables;
  const defaultX = variables.find(({ key }) => key === "perturbed.ball_speed")?.key ?? variables[0].key;
  const defaultY = variables.find(({ key }) => key === "metric.carry_distance")?.key ?? variables[1].key;
  const [xKey, setXKey] = useState(defaultX); const [yKey, setYKey] = useState(defaultY);
  const scatter = useMemo(() => buildScalarEnsembleScatter(output.ensemble, xKey, yKey),
    [output.ensemble, xKey, yKey]);
  const availability = scatter.availability.overall;
  return <div className="mt-5 space-y-5 border-t border-slate-800 pt-4">
    <p className="text-xs text-slate-300">Attempted {output.result.evaluationsAttempted}; complete {
      output.result.evaluationsCompleted}; failed {output.result.failedCount}; no impact {
      output.result.noImpactCount}. Status: {output.result.status}.{
      nonCompleteReasonSummary(output.ensemble)}</p>
    <Alternatives output={output} />
    <div className="grid gap-3 sm:grid-cols-2">{[["Horizontal axis", xKey, setXKey],
      ["Vertical axis", yKey, setYKey]].map(([label, value, setter]) =>
      <label key={label as string} className="text-xs text-slate-300">{label as string}
        <select className={INPUT_CLASS} value={value as string}
          onChange={(event) => (setter as (value: string) => void)(event.target.value)}>{
          variables.map((item) => <option key={item.key} value={item.key}>{
            axisLabel(item, variables)}</option>)
        }</select></label>)}</div>
    <p className="text-xs text-slate-400">Paired finite {availability.paired_finite}/{
      availability.total_rows}; unavailable {availability.unavailable}.</p>
    <ScalarEnsembleScatter scatter={scatter} label="Capability observations" />
    <RawRows output={output} />
    <div className="flex flex-wrap gap-2"><button type="button" className={BUTTON_CLASS}
      onClick={() => downloadText("capability-observations.csv",
        scalarEnsembleToCsv(output.ensemble), "text/csv;charset=utf-8")}>Export raw CSV</button>
      <button type="button" className={BUTTON_CLASS}
        onClick={() => downloadText("capability-observations.json",
          stableCapabilityObservationEnsembleJson(output.ensemble), "application/json")}>Export stable JSON</button>
      <button type="button" className={BUTTON_CLASS}
        onClick={() => downloadText("capability-results.csv",
          capabilityAlternativesCsv(output.result, output.ensemble),
          "text/csv;charset=utf-8")}>Export results CSV</button>
      <button type="button" className={BUTTON_CLASS}
        onClick={() => downloadText("capability-results.json",
          stableCapabilityResultExportJson(output.result, output.ensemble),
          "application/json")}>Export results JSON</button></div>
  </div>;
}
