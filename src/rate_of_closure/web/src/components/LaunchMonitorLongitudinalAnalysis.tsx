import { useState } from "react";

import type { LaunchMonitorRow } from "../model/launchMonitorAnalysisTypes";
import { downloadJson, downloadSvg } from "../model/launchMonitorDownloads";
import { analyzeLongitudinalPerformance } from "../model/launchMonitorLongitudinal";
import { metricLabel } from "../model/launchMonitorMetricUnits";

const field = "rounded border border-slate-700 bg-slate-950 px-2 py-1 text-sm";
const button = "rounded border border-slate-700 px-3 py-2 text-sm disabled:opacity-40";
const shown = (value: number | null) => value === null ? "—" : value.toFixed(3);

export function LaunchMonitorLongitudinalAnalysis({ rows, columns, numeric }: {
  rows: LaunchMonitorRow[]; columns: string[]; numeric: string[];
}) {
  const [player, setPlayer] = useState(""); const [session, setSession] = useState("");
  const [order, setOrder] = useState(""); const [metric, setMetric] = useState("");
  const [playerAttested, setPlayerAttested] = useState(false);
  const [sessionAttested, setSessionAttested] = useState(false);
  const [higherIsBetter, setHigherIsBetter] = useState(true);
  const [minSessions, setMinSessions] = useState(3);
  const [result, setResult] = useState<ReturnType<typeof analyzeLongitudinalPerformance> | null>(null);
  const [error, setError] = useState("");
  const select = (label: string, value: string, update: (value: string) => void, choices: string[]) =>
    <label className="text-sm">{label}<select aria-label={label} title={`Select explicit ${label.toLowerCase()}.`}
      value={value} onChange={(event) => { update(event.target.value); setResult(null); }} className={`${field} ml-2`}>
      <option value="">Select</option>{choices.map((choice) => <option key={choice}>{choice}</option>)}
    </select></label>;
  const run = () => {
    try { setResult(analyzeLongitudinalPerformance(rows, {
      metricColumn: metric, sessionColumn: session, sessionOrderColumn: order, playerColumn: player,
      playerIdentityAttested: playerAttested, sessionIdentityAttested: sessionAttested,
      higherIsBetter, confidenceLevel: 0.95, minSessions,
    })); setError(""); }
    catch (caught) { setResult(null); setError(caught instanceof Error ? caught.message : String(caught)); }
  };
  const ready = Boolean(player && session && order && metric && playerAttested && sessionAttested);
  const orders = result?.sessionPoints.map((point) => point.sessionOrder) ?? [];
  const values = result?.sessionPoints.map((point) => point.mean) ?? [];
  const minX = Math.min(...orders, 0); const maxX = Math.max(...orders, 1);
  const minY = Math.min(...values, 0); const maxY = Math.max(...values, 1);
  const x = (value: number) => 45 + (value - minX) / Math.max(maxX - minX, 1) * 560;
  const y = (value: number) => 210 - (value - minY) / Math.max(maxY - minY, 1) * 170;

  return <div className="space-y-3 rounded border border-slate-700 p-3">
    <h4 className="font-semibold">Longitudinal Player and Population Analysis</h4>
    <p className="text-xs text-slate-400">Session means receive equal weight. Each eligible player gets an OLS slope with uncertainty; player slopes are synthesized with fixed and DerSimonian–Laird random effects. Identities and order are never inferred.</p>
    <div className="flex flex-wrap gap-3">
      {select("Longitudinal player column", player, (value) => { setPlayer(value); setPlayerAttested(false); }, columns)}
      {select("Longitudinal session column", session, (value) => { setSession(value); setSessionAttested(false); }, columns)}
      {select("Longitudinal order column", order, setOrder, numeric)}
      {select("Longitudinal metric", metric, setMetric, numeric)}
      <label><input type="checkbox" aria-label="Attest longitudinal player identity" title="Required explicit player identity attestation."
        checked={playerAttested} onChange={(event) => setPlayerAttested(event.target.checked)} /> Player trusted</label>
      <label><input type="checkbox" aria-label="Attest longitudinal session identity" title="Required explicit session identity and order attestation."
        checked={sessionAttested} onChange={(event) => setSessionAttested(event.target.checked)} /> Session/order trusted</label>
      <label><input type="checkbox" aria-label="Higher metric is better" title="Clear when lower values represent improvement."
        checked={higherIsBetter} onChange={(event) => setHigherIsBetter(event.target.checked)} /> Higher is better</label>
      <label>Minimum sessions/player<input type="number" min="3" step="1" value={minSessions}
        aria-label="Minimum longitudinal sessions per player" title="Players below this count remain visible but are excluded from synthesis."
        onChange={(event) => setMinSessions(Number(event.target.value))} className={`${field} ml-2 w-20`} /></label>
      <button type="button" disabled={!ready} onClick={run} className={button}
        title="Estimate session uncertainty, player trends, and population trend.">Run Longitudinal Inference</button>
    </div>
    {result && <div className="space-y-2"><p>Random-effects slope: {shown(result.population.randomEffectSlope)} {metricLabel(metric)}/session · improvement probability {shown(result.population.improvementProbability)} · I² {shown(result.population.iSquaredPct)}%</p>
      <svg id="launch-monitor-longitudinal-plot" role="img" aria-label={`Session trend plot for ${metricLabel(metric)}`} viewBox="0 0 640 240" className="h-60 w-full bg-slate-950">
        <text x="280" y="232" fill="white">Explicit session order</text><text x="8" y="18" fill="white">{metricLabel(metric)}</text>
        {result.sessionPoints.map((point) => <circle key={`${point.playerId}-${point.sessionId}`} cx={x(point.sessionOrder)} cy={y(point.mean)} r="4" fill="#38bdf8"><title>{point.playerId} · {point.sessionId} · mean {point.mean.toFixed(3)} · SE {shown(point.standardError)}</title></circle>)}
      </svg>
      <div className="overflow-x-auto"><table className="w-full text-left text-sm"><thead><tr><th>Player</th><th>Sessions</th><th>Slope/session</th><th>95% CI</th><th>First-to-last</th><th>Status</th></tr></thead><tbody>
        {result.players.map((item) => <tr key={item.playerId}><td>{item.playerId}</td><td>{item.sessionCount}</td><td>{shown(item.slopePerSession)}</td><td>[{shown(item.ciLower)}, {shown(item.ciUpper)}]</td><td>{shown(item.firstToLastChange)}</td><td>{item.status}</td></tr>)}</tbody></table></div>
      {result.warnings.map((warning) => <p key={warning} className="text-xs text-amber-200">{warning}</p>)}
      <button type="button" className={button} title="Export sessions, player estimates, population synthesis, formula, and warnings."
        onClick={() => downloadJson("launch-monitor-longitudinal.json", result)}>Export Longitudinal Analysis</button>
      <button type="button" className={button} title="Save the unit-labelled longitudinal plot as SVG."
        onClick={() => downloadSvg("launch-monitor-longitudinal.svg", "launch-monitor-longitudinal-plot")}>Save Longitudinal Plot</button></div>}
    {error && <p role="alert" className="text-red-300">{error}</p>}
  </div>;
}
