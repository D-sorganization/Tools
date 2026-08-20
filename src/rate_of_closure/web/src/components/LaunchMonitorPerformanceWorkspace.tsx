import { useMemo, useRef, useState } from "react";

import { numericLaunchMonitorColumns } from "../model/launchMonitorAnalysis";
import type { LaunchMonitorRow } from "../model/launchMonitorAnalysisTypes";
import {
  analyzeDispersion, analyzeSessionTrend, calculateStrokesGained, calculateTargetError,
  type DistanceUnit,
} from "../model/launchMonitorPerformance";
import { fingerprintLaunchMonitorRows } from "../model/launchMonitorWorkspace";

interface Props { rows: LaunchMonitorRow[]; sourceName: string }
const field = "rounded border border-slate-700 bg-slate-950 px-2 py-1 text-sm";
const button = "rounded border border-slate-700 px-3 py-2 text-sm disabled:opacity-40";

const download = (name: string, content: string, type: string) => {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const anchor = document.createElement("a"); anchor.href = url; anchor.download = name; anchor.click();
  URL.revokeObjectURL(url);
};
const csv = (rows: LaunchMonitorRow[]) => {
  const columns = [...new Set(rows.flatMap(Object.keys))];
  const cell = (value: unknown) => `"${String(value ?? "").replace(/"/g, '""')}"`;
  return [columns, ...rows.map((row) => columns.map((column) => row[column]))]
    .map((row) => row.map(cell).join(",")).join("\n");
};

export function LaunchMonitorPerformanceWorkspace({ rows, sourceName }: Props) {
  const numeric = useMemo(() => numericLaunchMonitorColumns(rows), [rows]);
  const columns = useMemo(() => [...new Set(rows.flatMap(Object.keys))].sort(), [rows]);
  const fingerprint = useMemo(() => fingerprintLaunchMonitorRows(rows), [rows]);
  const [carry, setCarry] = useState(""); const [lateral, setLateral] = useState("");
  const [carryUnit, setCarryUnit] = useState<DistanceUnit>("yd"); const [lateralUnit, setLateralUnit] = useState<DistanceUnit>("yd");
  const [target, setTarget] = useState(150); const [dispersion, setDispersion] = useState<ReturnType<typeof analyzeDispersion> | null>(null);
  const [proxy, setProxy] = useState<ReturnType<typeof calculateTargetError> | null>(null);
  const [before, setBefore] = useState(""); const [after, setAfter] = useState(""); const [baseline, setBaseline] = useState("");
  const [strokes, setStrokes] = useState<ReturnType<typeof calculateStrokesGained> | null>(null);
  const [player, setPlayer] = useState(""); const [session, setSession] = useState(""); const [order, setOrder] = useState(""); const [metric, setMetric] = useState("");
  const [playerAttested, setPlayerAttested] = useState(false); const [sessionAttested, setSessionAttested] = useState(false);
  const [trend, setTrend] = useState<ReturnType<typeof analyzeSessionTrend> | null>(null); const [error, setError] = useState("");
  const svg = useRef<SVGSVGElement>(null); const loadInput = useRef<HTMLInputElement>(null);

  const runDispersion = () => { try {
    const next = analyzeDispersion(rows, { lateralColumn: lateral, carryColumn: carry, lateralUnit, carryUnit });
    setDispersion(next); setProxy(calculateTargetError(rows, { carryColumn: carry, lateralColumn: lateral, carryUnit, lateralUnit, targetDistanceYards: target })); setError("");
  } catch (caught) { setError(caught instanceof Error ? caught.message : String(caught)); } };
  const runStrokes = () => { try { setStrokes(calculateStrokesGained(rows, { expectedBeforeColumn: before, expectedAfterColumn: after, baselineSourceUrl: baseline })); setError(""); }
    catch (caught) { setError(caught instanceof Error ? caught.message : String(caught)); } };
  const runTrend = () => { try { setTrend(analyzeSessionTrend(rows, { metricColumn: metric, sessionColumn: session, sessionOrderColumn: order, playerColumn: player, playerIdentityAttested: playerAttested, sessionIdentityAttested: sessionAttested })); setError(""); }
    catch (caught) { setError(caught instanceof Error ? caught.message : String(caught)); } };
  const save = () => download("performance.lmanalysis.json", JSON.stringify({ contractVersion: "launch-monitor-performance/1.0", datasetSha256: fingerprint, sourceName,
    settings: { carry, lateral, carryUnit, lateralUnit, target }, dispersion, proxy, strokes, trend }, null, 2), "application/json");
  const load = async (file: File) => { try { const payload = JSON.parse(await file.text()); if (payload.datasetSha256 !== fingerprint) throw new RangeError("Saved analysis references a different dataset");
    setCarry(payload.settings.carry); setLateral(payload.settings.lateral); setCarryUnit(payload.settings.carryUnit); setLateralUnit(payload.settings.lateralUnit); setTarget(payload.settings.target);
    setDispersion(payload.dispersion); setProxy(payload.proxy); setStrokes(payload.strokes); setTrend(payload.trend); setError("");
  } catch (caught) { setError(caught instanceof Error ? caught.message : String(caught)); } };

  const select = (label: string, value: string, update: (value: string) => void, choices = numeric) => <label className="text-sm">{label}<select aria-label={label} title={`Select ${label.toLowerCase()} with an explicit unit or identity role.`} value={value} onChange={(event) => update(event.target.value)} className={`${field} ml-2`}><option value="">Select</option>{choices.map((choice) => <option key={choice}>{choice}</option>)}</select></label>;
  const xRange = dispersion ? Math.max(...dispersion.points.map((point) => point.carryYards), 1) : 1;
  const yRange = dispersion ? Math.max(...dispersion.points.map((point) => Math.abs(point.lateralYards)), 1) : 1;

  return <section aria-label="Launch monitor performance analytics" className="space-y-4 rounded-xl border border-slate-800 bg-slate-900/60 p-4">
    <h3 className="font-semibold">Dispersion, Scoring & Session Trends</h3>
    <p className="text-xs text-slate-400">These descriptive bookkeeping calculations are explicitly local v1 compatibility/offline fallbacks. Inferential statistics use the validated UpstreamDrift v2 client seam; row-aligned residuals remain unavailable unless a canonical v2 response supplies aligned backing rows.</p>
    <details title="Show formulas, provenance, and availability rules"><summary>Calculations and backing-data rules</summary><p className="text-xs text-slate-300">Negative lateral is yards left; positive is yards right. RMS = √mean(lateral²). Radial target error = hypot(target − carry, lateral) in yards and is not strokes gained. User-supplied expected-strokes SG = E(before) − 1 − E(after); this mode does not validate or reproduce the cited baseline. Source-backed strokes gained remains unavailable until a versioned baseline table, SHA-256, state schema, and required course-state inputs are loaded. Session cumulative means equally weight attested sessions.</p></details>
    <div className="flex flex-wrap gap-3">{select("Dispersion carry column", carry, setCarry)}{select("Dispersion lateral column", lateral, setLateral)}
      <label>Carry unit<select aria-label="Carry source unit" title="Source distance unit; chart output is yards." value={carryUnit} onChange={(event) => setCarryUnit(event.target.value as DistanceUnit)} className={`${field} ml-2`}><option>yd</option><option>m</option></select></label>
      <label>Lateral unit<select aria-label="Lateral source unit" title="Source distance unit; chart output is yards left/right." value={lateralUnit} onChange={(event) => setLateralUnit(event.target.value as DistanceUnit)} className={`${field} ml-2`}><option>yd</option><option>m</option></select></label>
      <label>Target (yd)<input aria-label="Target distance yards" title="Target distance for the radial-error proxy." type="number" min="1" value={target} onChange={(event) => setTarget(Number(event.target.value))} className={`${field} ml-2 w-24`} /></label>
      <button type="button" className={button} title="Calculate unit-labeled directional dispersion and radial target error." onClick={runDispersion}>Analyze Dispersion</button></div>
    {dispersion && proxy && <div><p>{dispersion.leftCount} yards left · {dispersion.rightCount} yards right · {dispersion.centerCount} centered · RMS {dispersion.rmsYards.toFixed(2)} yd</p><p>Radial target error (not strokes gained): {proxy.mean.toFixed(2)} yd</p>
      <svg ref={svg} role="img" aria-label="Dispersion plot, carry yards versus lateral yards left and right" viewBox="0 0 640 240" className="h-60 w-full bg-slate-950"><line x1="40" x2="620" y1="120" y2="120" stroke="#64748b"/><text x="280" y="232" fill="white">Carry (yd)</text><text x="8" y="20" fill="white">Lateral (yd; left − / right +)</text>{dispersion.points.map((point) => <circle key={point.sourceIndex} cx={40 + point.carryYards / xRange * 560} cy={120 - point.lateralYards / yRange * 95} r="4" fill="#38bdf8"/>)}</svg></div>}
    <div className="flex flex-wrap gap-3">{select("Expected strokes before column", before, setBefore)}{select("Expected strokes after column", after, setAfter)}<label>User citation URL<input aria-label="User-supplied expected-strokes citation URL" title="User-declared HTTP(S) citation; the app does not validate its baseline table." value={baseline} onChange={(event) => setBaseline(event.target.value)} className={`${field} ml-2`} /></label><button type="button" disabled={!before || !after || !baseline} onClick={runStrokes} title="Calculate user-supplied expected-strokes SG; not source-backed baseline interpolation." className={button}>Calculate User-Supplied SG</button></div>
    {!strokes ? <p>Source-backed strokes gained unavailable: current data lacks required course-state inputs and no validated baseline manifest/table is loaded. User-supplied expected-strokes SG requires two explicit columns and a citation.</p> : <p>Mean user-supplied expected-strokes SG: {strokes.mean.toFixed(3)} strokes · <a href={strokes.sourceUrl}>user citation (not validated baseline)</a></p>}
    <div className="flex flex-wrap gap-3">{select("Trusted player identity column", player, (value) => { setPlayer(value); setPlayerAttested(false); }, columns)}{select("Trusted session identity column", session, (value) => { setSession(value); setSessionAttested(false); }, columns)}{select("Explicit session order column", order, setOrder)}{select("Session trend metric", metric, setMetric)}
      <label><input type="checkbox" aria-label="Attest trusted player identity" title="Identity must be supplied, never inferred." checked={playerAttested} onChange={(event) => setPlayerAttested(event.target.checked)} /> Player trusted</label><label><input type="checkbox" aria-label="Attest trusted session identity and order" title="Session identity and order must be supplied, never inferred." checked={sessionAttested} onChange={(event) => setSessionAttested(event.target.checked)} /> Session/order trusted</label><button type="button" disabled={!playerAttested || !sessionAttested || !player || !session || !order || !metric} onClick={runTrend} title="Calculate session and cumulative means using explicit identities and order." className={button}>Run Session Trend</button></div>
    {trend && <p>{trend.points.length} player-session points · {trend.formula}</p>}
    <div className="flex flex-wrap gap-2"><button type="button" title="Save fingerprint-bound settings, results, formulas, and provenance." onClick={save} className={button}>Save Performance Analysis</button><input ref={loadInput} type="file" className="hidden" aria-label="Load saved performance analysis" onChange={(event) => { const file = event.target.files?.[0]; if (file) void load(file); }}/><button type="button" title="Reload only when the current dataset fingerprint matches." onClick={() => loadInput.current?.click()} className={button}>Load Performance Analysis</button><button type="button" disabled={!svg.current} title="Export the current SVG with its visible units and direction convention." onClick={() => svg.current && download("dispersion.svg", svg.current.outerHTML, "image/svg+xml")} className={button}>Export Plot SVG</button><button type="button" title="Export every retained backing row as CSV." onClick={() => download("performance-backing.csv", csv(rows), "text/csv")} className={button}>Export Backing Data</button></div>
    {error && <p role="alert" className="text-red-300">{error}</p>}
  </section>;
}
