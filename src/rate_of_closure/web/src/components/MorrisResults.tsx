import { useMemo, useState } from "react";

import type { MorrisReport } from "../model/morrisGlobalSensitivityContract";
import { presentMorrisReport } from "../model/morrisPresentation";
import { INPUT_CLASS, PANEL_CLASS } from "./variationUi";

const metric = (value: number | null): string => value === null ? "—" : value.toFixed(4);

export function MorrisResults({ report }: { readonly report: MorrisReport }) {
  const targets = useMemo(() => [...new Set(report.estimates.map((estimate) => estimate.target.name))], [report]);
  const [targetName, setTargetName] = useState(targets[0]);
  const presentation = presentMorrisReport(report, targetName);
  return (
    <section aria-label="Morris screening results" className={`${PANEL_CLASS} min-w-0 space-y-4`}>
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div><h3 className="text-lg font-semibold">{presentation.target.label} Ranking</h3>
          <p className="text-xs text-slate-400">{presentation.target.unit} · {presentation.target.kind}
            {presentation.target.coordinateFrame ? ` · ${presentation.target.coordinateFrame}` : ""}</p></div>
        <label className="text-xs text-slate-300">Output target
          <select className={`${INPUT_CLASS} mt-1 min-w-56`} value={targetName}
            title="Choose the output whose factor effects are ranked"
            onChange={(event) => setTargetName(event.target.value)}>
            {targets.map((target) => <option key={target} value={target}>{presentMorrisReport(report, target).target.label}</option>)}
          </select>
        </label>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[920px] text-left text-xs">
          <thead className="text-slate-400"><tr>
            <th className="p-2">Rank</th><th>Factor</th><th>Bounds</th><th>μ*</th><th>μ</th>
            <th>SE(μ*)</th><th>σ</th><th>Adequacy</th><th>Valid / total</th><th>Unavailable detail</th>
          </tr></thead>
          <tbody>{presentation.rows.map((row) => <tr key={row.specId} className="border-t border-slate-800">
            <td className="p-2">{row.rank ?? "—"}</td><td>{row.label}</td>
            <td>{row.sourceLower}–{row.sourceUpper} {row.sourceUnit}</td>
            <td className="font-semibold text-sky-300">{metric(row.muStar)}</td><td>{metric(row.mu)}</td>
            <td>{metric(row.muStarStandardError)}</td><td>{metric(row.sigma)}</td>
            <td>{row.availability}; {row.sampleAdequacy}</td><td>{row.validPairs} / {row.totalPairs}</td>
            <td>no impact {row.noImpactUnavailablePairs} ({row.typedNoImpactPairs} typed); failed {row.failedPairs}; nonfinite {row.nonfinitePairs}</td>
          </tr>)}</tbody>
        </table>
      </div>
      <div className="grid gap-3 text-xs text-slate-300 sm:grid-cols-2">
        <p className="rounded border border-slate-800 p-3">Design: {report.design.trajectories} trajectories,
          {` ${report.design.levels} levels, ${report.design.totalSamples} evaluations, seed ${report.design.seed}. `}
          Normalized step {report.design.normalizedStep.toFixed(6)}.</p>
        <p className="rounded border border-amber-500/30 bg-amber-950/20 p-3 text-amber-100">{report.interactionCaveat}</p>
      </div>
      <details className="text-xs text-slate-400"><summary className="cursor-pointer text-slate-300">Assumptions</summary>
        <ul className="mt-2 list-disc space-y-1 pl-5">{report.assumptions.map((assumption) => <li key={assumption}>{assumption}</li>)}</ul>
      </details>
    </section>
  );
}
