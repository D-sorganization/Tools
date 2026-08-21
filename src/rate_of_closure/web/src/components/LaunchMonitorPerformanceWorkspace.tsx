import { useMemo, useRef, useState } from "react";

import { numericLaunchMonitorColumns } from "../model/launchMonitorAnalysis";
import type { LaunchMonitorRow } from "../model/launchMonitorAnalysisTypes";
import {
  analyzeDispersion, analyzeSessionTrend, calculateStrokesGained, calculateTargetError,
  type DistanceUnit,
} from "../model/launchMonitorPerformance";
import { fingerprintLaunchMonitorRows } from "../model/launchMonitorWorkspace";
import {
  createPerformanceWorkspaceV3,
  loadPerformanceWorkspace,
} from "../model/launchMonitorPerformanceWorkspace";
import { serializeWorkspaceV3 } from "../model/launchMonitorWorkspaceV3";
import { LaunchMonitorSourceBackedStrokesGained } from "./LaunchMonitorSourceBackedStrokesGained";
import { LaunchMonitorLongitudinalAnalysis } from "./LaunchMonitorLongitudinalAnalysis";

interface Props { rows: LaunchMonitorRow[]; sourceName: string }
const field = "rounded border border-slate-700 bg-slate-950 px-2 py-1 text-sm";
const button = "rounded border border-slate-700 px-3 py-2 text-sm disabled:opacity-40";

const download = (name: string, content: BlobPart, type: string) => {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const anchor = document.createElement("a"); anchor.href = url; anchor.download = name; anchor.click();
  URL.revokeObjectURL(url);
};
const downloadPng = (element: SVGSVGElement) => {
  const image = new Image();
  image.onload = () => {
    const canvas = document.createElement("canvas"); canvas.width = 1280; canvas.height = 480;
    canvas.getContext("2d")?.drawImage(image, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => { if (blob) download("dispersion.png", blob, "image/png"); });
  };
  image.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(element.outerHTML)}`;
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
  const [loadedSummary, setLoadedSummary] = useState<Record<string, unknown> | null>(null);
  const svg = useRef<SVGSVGElement>(null); const loadInput = useRef<HTMLInputElement>(null);

  const runDispersion = () => { try {
    const next = analyzeDispersion(rows, { lateralColumn: lateral, carryColumn: carry, lateralUnit, carryUnit });
    setDispersion(next); setProxy(calculateTargetError(rows, { carryColumn: carry, lateralColumn: lateral, carryUnit, lateralUnit, targetDistanceYards: target })); setError("");
  } catch (caught) { setError(caught instanceof Error ? caught.message : String(caught)); } };
  const runStrokes = () => { try { setStrokes(calculateStrokesGained(rows, { expectedBeforeColumn: before, expectedAfterColumn: after, baselineSourceUrl: baseline })); setError(""); }
    catch (caught) { setError(caught instanceof Error ? caught.message : String(caught)); } };
  const runTrend = () => { try { setTrend(analyzeSessionTrend(rows, { metricColumn: metric, sessionColumn: session, sessionOrderColumn: order, playerColumn: player, playerIdentityAttested: playerAttested, sessionIdentityAttested: sessionAttested })); setError(""); }
    catch (caught) { setError(caught instanceof Error ? caught.message : String(caught)); } };
  const settings = () => ({ carry, lateral, carryUnit, lateralUnit, target, before, after, baseline,
    player, session, order, metric, playerAttested, sessionAttested });
  const save = () => download("performance.lmanalysis.json", `${serializeWorkspaceV3(createPerformanceWorkspaceV3({
    sourceName, datasetSha256: fingerprint, rowCount: rows.length, settings: settings(),
    results: { dispersion, proxy, strokes, trend },
  }))}\n`, "application/json");
  const load = async (file: File) => { try { const payload = loadPerformanceWorkspace(await file.text(), fingerprint); const saved = payload.settings;
    setCarry(String(saved.carry ?? "")); setLateral(String(saved.lateral ?? "")); setCarryUnit(String(saved.carryUnit ?? "yd") as DistanceUnit); setLateralUnit(String(saved.lateralUnit ?? "yd") as DistanceUnit); setTarget(Number(saved.target ?? 150));
    setBefore(String(saved.before ?? "")); setAfter(String(saved.after ?? "")); setBaseline(String(saved.baseline ?? ""));
    setPlayer(String(saved.player ?? "")); setSession(String(saved.session ?? "")); setOrder(String(saved.order ?? "")); setMetric(String(saved.metric ?? ""));
    setPlayerAttested(saved.playerAttested === true); setSessionAttested(saved.sessionAttested === true);
    setDispersion(null); setProxy(null); setStrokes(null); setTrend(null); setLoadedSummary(payload.results);
    setError(`${payload.importedFrom} settings and aggregate results loaded; rerun to regenerate row-aligned plots.`);
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
    <LaunchMonitorSourceBackedStrokesGained rows={rows} columns={columns} numeric={numeric} />
    <div className="flex flex-wrap gap-3">{select("Trusted player identity column", player, (value) => { setPlayer(value); setPlayerAttested(false); }, columns)}{select("Trusted session identity column", session, (value) => { setSession(value); setSessionAttested(false); }, columns)}{select("Explicit session order column", order, setOrder)}{select("Session trend metric", metric, setMetric)}
      <label><input type="checkbox" aria-label="Attest trusted player identity" title="Identity must be supplied, never inferred." checked={playerAttested} onChange={(event) => setPlayerAttested(event.target.checked)} /> Player trusted</label><label><input type="checkbox" aria-label="Attest trusted session identity and order" title="Session identity and order must be supplied, never inferred." checked={sessionAttested} onChange={(event) => setSessionAttested(event.target.checked)} /> Session/order trusted</label><button type="button" disabled={!playerAttested || !sessionAttested || !player || !session || !order || !metric} onClick={runTrend} title="Calculate session and cumulative means using explicit identities and order." className={button}>Run Session Trend</button></div>
    {trend && <p>{trend.points.length} player-session points · {trend.formula}</p>}
    {loadedSummary && <details title="Loaded aggregate results remain row-free"><summary>Loaded saved aggregate results</summary><pre aria-label="Loaded saved aggregate results" className="max-h-48 overflow-auto text-xs">{JSON.stringify(loadedSummary, null, 2)}</pre></details>}
    <LaunchMonitorLongitudinalAnalysis rows={rows} columns={columns} numeric={numeric} />
    <div className="flex flex-wrap gap-2"><button type="button" title="Save row-free v3 settings, aggregate results, formulas, exclusions, and provenance." onClick={save} className={button}>Save Performance Analysis</button><input ref={loadInput} type="file" className="hidden" aria-label="Load saved performance analysis" onChange={(event) => { const file = event.target.files?.[0]; if (file) void load(file); }}/><button type="button" title="Reload only when the current dataset fingerprint matches; v1 imports are labelled compatibility." onClick={() => loadInput.current?.click()} className={button}>Load Performance Analysis</button><button type="button" disabled={!dispersion} title="Export the current SVG with its visible units and direction convention." onClick={() => svg.current && download("dispersion.svg", svg.current.outerHTML, "image/svg+xml")} className={button}>Export Plot SVG</button><button type="button" disabled={!dispersion} title="Export the current plot as PNG with the same visible units." onClick={() => svg.current && downloadPng(svg.current)} className={button}>Export Plot PNG</button><button type="button" disabled title="Unavailable in browser: PDF plot export requires the desktop renderer." className={button}>Export Plot PDF</button><button type="button" disabled title="Unavailable in browser: restricted backing rows require explicit desktop approval." className={button}>Export Backing Data</button></div>
    {error && <p role="alert" className="text-red-300">{error}</p>}
  </section>;
}
