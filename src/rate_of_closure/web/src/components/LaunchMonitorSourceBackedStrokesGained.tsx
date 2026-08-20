import { useRef, useState } from "react";

import type { LaunchMonitorRow } from "../model/launchMonitorAnalysisTypes";
import { downloadJson } from "../model/launchMonitorDownloads";
import {
  calculateSourceBackedStrokesGained,
  parseStrokesGainedBaseline,
  type StrokesGainedBaseline,
} from "../model/launchMonitorSourceBackedStrokesGained";

const field = "rounded border border-slate-700 bg-slate-950 px-2 py-1 text-sm";
const button = "rounded border border-slate-700 px-3 py-2 text-sm disabled:opacity-40";

export function LaunchMonitorSourceBackedStrokesGained({ rows, columns, numeric }: {
  rows: LaunchMonitorRow[]; columns: string[]; numeric: string[];
}) {
  const [baseline, setBaseline] = useState<StrokesGainedBaseline | null>(null);
  const [beforeLie, setBeforeLie] = useState("");
  const [beforeDistance, setBeforeDistance] = useState("");
  const [afterLie, setAfterLie] = useState("");
  const [afterDistance, setAfterDistance] = useState("");
  const [beforeUnit, setBeforeUnit] = useState<"yd" | "m">("yd");
  const [afterUnit, setAfterUnit] = useState<"yd" | "m">("yd");
  const [result, setResult] = useState<ReturnType<typeof calculateSourceBackedStrokesGained> | null>(null);
  const [error, setError] = useState("");
  const input = useRef<HTMLInputElement>(null);
  const select = (label: string, value: string, update: (value: string) => void, choices: string[]) =>
    <label className="text-sm">{label}<select aria-label={label} title={`Select ${label.toLowerCase()}.`}
      value={value} onChange={(event) => { update(event.target.value); setResult(null); }} className={`${field} ml-2`}>
      <option value="">Select</option>{choices.map((choice) => <option key={choice}>{choice}</option>)}
    </select></label>;
  const load = async (file: File) => {
    try { setBaseline(await parseStrokesGainedBaseline(await file.text())); setResult(null); setError(""); }
    catch (caught) { setBaseline(null); setResult(null); setError(caught instanceof Error ? caught.message : String(caught)); }
  };
  const calculate = () => {
    if (!baseline) return;
    try {
      setResult(calculateSourceBackedStrokesGained(rows, baseline, {
        beforeLieColumn: beforeLie, beforeDistanceColumn: beforeDistance,
        afterLieColumn: afterLie, afterDistanceColumn: afterDistance,
        beforeDistanceUnit: beforeUnit, afterDistanceUnit: afterUnit,
      })); setError("");
    } catch (caught) { setResult(null); setError(caught instanceof Error ? caught.message : String(caught)); }
  };
  const ready = Boolean(baseline && beforeLie && beforeDistance && afterLie && afterDistance);

  return <div className="space-y-3 rounded border border-slate-700 p-3">
    <h4 className="font-semibold">Source-Backed Strokes Gained</h4>
    <p className="text-xs text-slate-400">Load a licensed, versioned expected-strokes artifact. The client verifies its canonical table SHA-256, source URL, license declaration, state schema, and interpolation bounds before enabling this calculation.</p>
    <input ref={input} type="file" accept=".json,application/json" className="hidden"
      aria-label="Load verified strokes-gained baseline" onChange={(event) => {
        const file = event.target.files?.[0]; if (file) void load(file);
      }} />
    <button type="button" className={button} title="Load and hash-verify a versioned expected-strokes baseline artifact."
      onClick={() => input.current?.click()}>Load Baseline Artifact</button>
    {baseline ? <p className="break-all text-xs text-emerald-200">Verified {baseline.baselineId} · version {baseline.version} · SHA-256 {baseline.tableSha256} · license {baseline.license}</p>
      : <p className="text-xs text-amber-200">Unavailable until a verified baseline artifact is loaded. No baseline table is bundled.</p>}
    <div className="flex flex-wrap gap-3">
      {select("Before lie column", beforeLie, setBeforeLie, columns)}
      {select("Before distance column", beforeDistance, setBeforeDistance, numeric)}
      <label className="text-sm">Before unit<select aria-label="Before course-state distance unit"
        title="Declare the source distance unit; baseline lookups use yards." value={beforeUnit}
        onChange={(event) => setBeforeUnit(event.target.value as "yd" | "m")} className={`${field} ml-2`}>
        <option value="yd">yd</option><option value="m">m</option></select></label>
      {select("After lie column", afterLie, setAfterLie, columns)}
      {select("After distance column", afterDistance, setAfterDistance, numeric)}
      <label className="text-sm">After unit<select aria-label="After course-state distance unit"
        title="Declare the source distance unit; baseline lookups use yards." value={afterUnit}
        onChange={(event) => setAfterUnit(event.target.value as "yd" | "m")} className={`${field} ml-2`}>
        <option value="yd">yd</option><option value="m">m</option></select></label>
      <button type="button" disabled={!ready} onClick={calculate} className={button}
        title="Calculate verified E(before) minus one stroke minus verified E(after).">Calculate Source-Backed SG</button>
    </div>
    {result && <div className="text-sm"><p>Mean source-backed SG: {result.mean.toFixed(3)} strokes across {result.values.length} complete shots.</p>
      <p className="text-xs text-slate-400">{result.formula} <a className="underline" href={result.sourceUrl} target="_blank" rel="noreferrer">Baseline source</a></p>
      <button type="button" className={button} title="Export baseline identity, formula, every lookup, and shot result."
        onClick={() => downloadJson("source-backed-strokes-gained.json", result)}>Export Source-Backed SG</button></div>}
    {error && <p role="alert" className="text-red-300">{error}</p>}
  </div>;
}
