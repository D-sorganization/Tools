import type { LaunchMonitorRow } from "../model/launchMonitorAnalysis";
import { importedAdvancedResults, type PcaScore, type RankedValue } from "../model/launchMonitorImportedResults";

const card = "rounded-xl border border-slate-800/80 bg-slate-900/60 p-4 shadow-lg shadow-black/20";

function RankedTable({ title, rows }: { title: string; rows: RankedValue[] }) {
  return <div className="overflow-x-auto">
    <h4 className="mb-2 font-medium text-slate-200">{title}</h4>
    <table className="w-full text-left text-sm">
      <thead className="text-slate-400"><tr><th>Rank</th><th>Variable</th><th>Value</th><th>Method</th></tr></thead>
      <tbody>{rows.slice(0, 12).map((item) => <tr key={`${item.label}-${item.method}`} className="border-t border-slate-800">
        <td className="py-2">{item.rank}</td><td>{item.label}</td><td>{item.value.toFixed(5)}</td><td>{item.method}</td>
      </tr>)}</tbody>
    </table>
  </div>;
}

function PcaScores({ scores }: { scores: PcaScore[] }) {
  const xBound = Math.max(1, ...scores.map((score) => Math.abs(score.pc1)));
  const yBound = Math.max(1, ...scores.map((score) => Math.abs(score.pc2)));
  return <svg viewBox="0 0 640 250" role="img" aria-label="Imported PCA component one versus component two scores"
    className="h-64 w-full rounded-lg border border-slate-800 bg-slate-950">
    <title>Imported PCA scores only; PCA fitting and scaling are performed by the source campaign</title>
    <line x1="320" y1="15" x2="320" y2="220" stroke="#64748b" /><line x1="45" y1="118" x2="615" y2="118" stroke="#64748b" />
    {scores.map((score) => <circle key={score.id} cx={320 + score.pc1 / xBound * 275}
      cy={118 - score.pc2 / yBound * 103} r="3" fill="#c084fc" opacity="0.7">
      <title>{`${score.id}: PC1 ${score.pc1.toFixed(4)}, PC2 ${score.pc2.toFixed(4)}`}</title>
    </circle>)}
    <text x="330" y="242" textAnchor="middle" fill="#94a3b8" fontSize="12">PC1 score (standardized unitless)</text>
    <text x="14" y="118" textAnchor="middle" fill="#94a3b8" fontSize="12" transform="rotate(-90 14 118)">PC2 score (standardized unitless)</text>
  </svg>;
}

export function LaunchMonitorImportedResults({ rows }: { rows: LaunchMonitorRow[] }) {
  const result = importedAdvancedResults(rows);
  const present = result.pcaLoadings.length || result.pcaScores.length ||
    result.featureImportance.length || result.performance.length || result.residualColumns.length;
  if (!present) return null;
  return <section className={`${card} space-y-5`} aria-label="Imported advanced model analysis">
    <div>
      <h3 className="font-semibold text-slate-200">Imported Advanced Model Analysis</h3>
      <p className="mt-1 text-xs text-slate-400">
        This view displays campaign-produced PCA, residual/model-spread importance, and held-out metrics.
        Tools does not refit or certify these imported results. Interpret rank within the named method only;
        loading magnitude is not causal importance, and held-out performance depends on the campaign split,
        leakage controls, preprocessing, and sample representativeness.
      </p>
    </div>
    {result.pcaScores.length > 1 && <PcaScores scores={result.pcaScores} />}
    {result.pcaLoadings.length > 0 && <RankedTable title="PCA Loading Magnitudes" rows={result.pcaLoadings} />}
    {result.featureImportance.length > 0 && <RankedTable title="Residual / Model-Spread Feature Importance" rows={result.featureImportance} />}
    {result.performance.length > 0 && <div>
      <h4 className="mb-2 font-medium text-slate-200">Held-Out Performance</h4>
      <div className="grid gap-2 sm:grid-cols-3">{result.performance.map((item) => <p
        key={`${item.method}-${item.metric}-${item.value}`} title="Imported held-out metric; consult the private campaign manifest for split and preprocessing provenance."
        className="rounded border border-slate-800 p-2 text-sm">{item.method} · {item.metric}: <strong>{item.value.toFixed(5)}</strong></p>)}</div>
    </div>}
    {result.residualColumns.length > 0 && <p className="text-xs text-slate-400">
      Recognized residual/model-spread fields: {result.residualColumns.join(", ")}. Retained rows remain exportable from Traceability.
    </p>}
  </section>;
}

