import { useRef, useState } from "react";

import type { LaunchMonitorRow } from "../model/launchMonitorAnalysisTypes";
import { downloadJson } from "../model/launchMonitorDownloads";
import {
  buildSourceBackedStrokesGainedPayload,
  calculateSourceBackedStrokesGained,
  parseStrokesGainedBaseline,
  type SourceBackedStrokesGainedRequest,
  type StrokesGainedBaseline,
} from "../model/launchMonitorSourceBackedStrokesGained";
import { createLaunchMonitorStrokesGainedClient } from "../model/launchMonitorV2Client";

const field = "rounded border border-slate-700 bg-slate-950 px-2 py-1 text-sm";
const button = "rounded border border-slate-700 px-3 py-2 text-sm disabled:opacity-40";
type Unit = "yd" | "m";
interface DisplayResult { mean: number; count: number; formula: string; sourceUrl: string; payload: unknown; canonical: boolean }
interface Selection extends Omit<SourceBackedStrokesGainedRequest, "trustedSummary"> {
  authorityUrl: string; playerGroup: string; sessionGroup: string; clubGroup: string;
  orderColumn: string; summaryAttested: boolean;
}
const initial: Selection = {
  authorityUrl: "", beforeLieColumn: "", beforeContextColumn: "", beforeTargetColumn: "",
  beforeDistanceColumn: "", afterLieColumn: "", afterContextColumn: "", afterTargetColumn: "",
  afterDistanceColumn: "", beforeDistanceUnit: "yd", afterDistanceUnit: "yd",
  playerGroup: "", sessionGroup: "", clubGroup: "", orderColumn: "", summaryAttested: false,
};

function requestOf(selection: Selection): SourceBackedStrokesGainedRequest {
  const { authorityUrl, playerGroup, sessionGroup, clubGroup,
    orderColumn, summaryAttested, ...request } = selection;
  void authorityUrl;
  if (!summaryAttested) return request;
  return { ...request, trustedSummary: { playerColumn: playerGroup,
    sessionColumn: sessionGroup, clubColumn: clubGroup, orderColumn,
    orderUnit: "session", evidence: "Explicit user attestation in the Tools scoring UI." } };
}

async function score(rows: LaunchMonitorRow[], baseline: StrokesGainedBaseline, selection: Selection) {
  const request = requestOf(selection);
  if (!selection.authorityUrl.trim()) {
    const local = calculateSourceBackedStrokesGained(rows, baseline, request);
    return { mean: local.mean, count: local.values.length, formula: local.formula,
      sourceUrl: local.sourceUrl, payload: local, canonical: false } satisfies DisplayResult;
  }
  const canonical = await createLaunchMonitorStrokesGainedClient(selection.authorityUrl.trim())(
    buildSourceBackedStrokesGainedPayload(rows, baseline, request),
  );
  if (canonical.mean === null) throw new RangeError("Canonical strokes-gained estimate is unavailable");
  return { mean: canonical.mean, count: canonical.count,
    formula: String(canonical.payload.formula), sourceUrl: baseline.sourceUrl,
    payload: canonical.payload, canonical: true } satisfies DisplayResult;
}

function useScoring(rows: LaunchMonitorRow[]) {
  const [baseline, setBaseline] = useState<StrokesGainedBaseline | null>(null);
  const [selection, setSelection] = useState(initial);
  const [result, setResult] = useState<DisplayResult | null>(null);
  const [error, setError] = useState("");
  const update = <K extends keyof Selection>(key: K, value: Selection[K]) => {
    setSelection((current) => ({ ...current, [key]: value })); setResult(null);
  };
  const load = async (file: File) => {
    try { setBaseline(await parseStrokesGainedBaseline(await file.text())); setResult(null); setError(""); }
    catch (caught) { setBaseline(null); setResult(null); setError(caught instanceof Error ? caught.message : String(caught)); }
  };
  const calculate = async () => {
    if (!baseline) return;
    try { setResult(await score(rows, baseline, selection)); setError(""); }
    catch (caught) { setResult(null); setError(caught instanceof Error ? caught.message : String(caught)); }
  };
  const required = [selection.beforeLieColumn, selection.beforeContextColumn,
    selection.beforeTargetColumn, selection.beforeDistanceColumn, selection.afterLieColumn,
    selection.afterContextColumn, selection.afterTargetColumn, selection.afterDistanceColumn];
  return { baseline, selection, result, error, update, load, calculate,
    ready: Boolean(baseline && required.every(Boolean)) };
}

function BaselineLoader({ baseline, load }: { baseline: StrokesGainedBaseline | null; load: (file: File) => Promise<void> }) {
  const input = useRef<HTMLInputElement>(null);
  return <>
    <input ref={input} type="file" accept=".json,application/json" className="hidden"
      aria-label="Load verified strokes-gained baseline" onChange={(event) => {
        const file = event.target.files?.[0]; if (file) void load(file);
      }} />
    <button type="button" className={button} title="Load and hash-verify a versioned expected-strokes baseline artifact."
      onClick={() => input.current?.click()}>Load Baseline Artifact</button>
    {baseline ? <p className="break-all text-xs text-emerald-200">Verified {baseline.baselineId} · version {baseline.version} · SHA-256 {baseline.tableSha256} · license {baseline.license}</p>
      : <p className="text-xs text-amber-200">Unavailable until a verified baseline artifact is loaded. No baseline table is bundled.</p>}
  </>;
}

interface ControlsProps { selection: Selection; columns: string[]; numeric: string[];
  update: <K extends keyof Selection>(key: K, value: Selection[K]) => void;
  calculate: () => Promise<void>; ready: boolean }
function ScoringControls({ selection, columns, numeric, update, calculate, ready }: ControlsProps) {
  const select = (label: string, key: keyof Selection, choices: string[]) =>
    <label className="text-sm">{label}<select aria-label={label} title={`Select ${label.toLowerCase()}.`}
      value={String(selection[key])} onChange={(event) => update(key, event.target.value)} className={`${field} ml-2`}>
      <option value="">Select</option>{choices.map((choice) => <option key={choice}>{choice}</option>)}
    </select></label>;
  return <div className="flex flex-wrap gap-3">
    <label className="text-sm">Upstream authority URL<input aria-label="Upstream strokes-gained authority URL"
      title="HTTP(S) UpstreamDrift API authority. Blank runs the labeled local compatibility fallback."
      value={selection.authorityUrl} onChange={(event) => update("authorityUrl", event.target.value)}
      className={`${field} ml-2`} placeholder="https://authority.example" /></label>
    {select("Before lie column", "beforeLieColumn", columns)}
    {select("Before context column", "beforeContextColumn", columns)}
    {select("Before target or hole column", "beforeTargetColumn", columns)}
    {select("Before distance column", "beforeDistanceColumn", numeric)}
    <UnitSelect label="Before" value={selection.beforeDistanceUnit} update={(value) => update("beforeDistanceUnit", value)} />
    {select("After lie column", "afterLieColumn", columns)}
    {select("After context column", "afterContextColumn", columns)}
    {select("After target or hole column", "afterTargetColumn", columns)}
    {select("After distance column", "afterDistanceColumn", numeric)}
    <UnitSelect label="After" value={selection.afterDistanceUnit} update={(value) => update("afterDistanceUnit", value)} />
    {select("Trusted player identity column for SG", "playerGroup", columns)}
    {select("Trusted session identity column for SG", "sessionGroup", columns)}
    {select("Trusted club identity column for SG", "clubGroup", columns)}
    {select("Explicit longitudinal order column for SG", "orderColumn", numeric)}
    <label className="text-sm"><input type="checkbox" aria-label="Attest strokes-gained grouping identities and longitudinal order"
      title="Identity and chronology are never inferred. This attestation enables canonical grouped and longitudinal summaries."
      checked={selection.summaryAttested} onChange={(event) => update("summaryAttested", event.target.checked)} />
      Grouping/order are explicit and trustworthy</label>
    <button type="button" disabled={!ready} onClick={() => void calculate()} className={button}
      title="Calculate verified E(before) minus one stroke minus verified E(after).">Calculate Source-Backed SG</button>
  </div>;
}

function UnitSelect({ label, value, update }: { label: string; value: Unit; update: (value: Unit) => void }) {
  return <label className="text-sm">{label} unit<select aria-label={`${label} course-state distance unit`}
    title="Declare the source distance unit; baseline lookups use yards." value={value}
    onChange={(event) => update(event.target.value as Unit)} className={`${field} ml-2`}>
    <option value="yd">yd</option><option value="m">m</option></select></label>;
}

function ResultView({ result }: { result: DisplayResult }) {
  return <div className="text-sm"><p>{result.canonical ? "Canonical" : "Local compatibility"} mean source-backed SG: {result.mean.toFixed(3)} strokes across {result.count} complete shots.</p>
    <p className="text-xs text-slate-400">{result.formula} <a className="underline" href={result.sourceUrl} target="_blank" rel="noreferrer">Baseline source</a></p>
    <button type="button" className={button} title="Export baseline identity, formula, every lookup, and shot result."
      onClick={() => downloadJson("source-backed-strokes-gained.json", result.payload)}>Export Source-Backed SG</button></div>;
}

export function LaunchMonitorSourceBackedStrokesGained({ rows, columns, numeric }: {
  rows: LaunchMonitorRow[]; columns: string[]; numeric: string[];
}) {
  const scoring = useScoring(rows);
  return <div className="space-y-3 rounded border border-slate-700 p-3">
    <h4 className="font-semibold">Source-Backed Strokes Gained</h4>
    <p className="text-xs text-slate-400">Load a licensed, versioned expected-strokes artifact. The client verifies its canonical table SHA-256, source URL, license declaration, state schema, and interpolation bounds before enabling this calculation.</p>
    <BaselineLoader baseline={scoring.baseline} load={scoring.load} />
    <ScoringControls selection={scoring.selection} columns={columns} numeric={numeric}
      update={scoring.update} calculate={scoring.calculate} ready={scoring.ready} />
    {scoring.result && <ResultView result={scoring.result} />}
    {scoring.error && <p role="alert" className="text-red-300">{scoring.error}</p>}
  </div>;
}
