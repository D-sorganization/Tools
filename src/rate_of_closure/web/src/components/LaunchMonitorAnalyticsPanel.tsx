import { useMemo, useRef, useState } from "react";

import { MAX_LINKED_SCATTER_ROWS } from "../model/launchMonitorLinkedScatter";
import { LaunchMonitorLinkedScatter } from "./LaunchMonitorLinkedScatter";
import { LaunchMonitorPlayerWorkspace } from "./LaunchMonitorPlayerWorkspace";
import { LaunchMonitorPerformanceWorkspace } from "./LaunchMonitorPerformanceWorkspace";
import {
  analyzeLaunchMonitorData,
  numericLaunchMonitorColumns,
  readLaunchMonitorFile,
  type AnalysisMode,
  type CorrelationMethod,
  type LaunchMonitorAnalysisResult,
  type LaunchMonitorRow,
  type MissingPolicy,
} from "../model/launchMonitorAnalysis";
import {
  conventionRegistry,
  PARAMETER_IDS,
  type ConventionId,
  type ParameterId,
} from "../model/launchMonitorConventions";

const DEMO_ROWS: LaunchMonitorRow[] = Array.from({ length: 120 }, (_, index) => {
  const clubSpeed = 38 + index * 0.11;
  const attackAngle = -4 + (index % 17) * 0.4;
  const clubPath = -3 + (index % 13) * 0.5;
  const faceAngle = clubPath * 0.65 + Math.sin(index * 0.7) * 0.8;
  const ballSpeed = clubSpeed * 1.46 + attackAngle * 0.08 + Math.sin(index) * 0.25;
  return {
    shot_id: `demo-${index + 1}`,
    player_id: index < 60 ? "demo-player-a" : "demo-player-b",
    session_id: index < 60 ? "demo-a" : "demo-b",
    monitor_vendor: index % 2 ? "FlightScope" : "TrackMan",
    observation_kind: "shot",
    club_speed: clubSpeed,
    attack_angle: attackAngle,
    club_path: clubPath,
    face_angle: faceAngle,
    ball_speed: ballSpeed,
    carry_distance: ballSpeed * 3.25 + attackAngle * 0.9,
  };
});

const card = "rounded-xl border border-slate-800/80 bg-slate-900/60 p-4 shadow-lg shadow-black/20";
const field = "w-full rounded border border-slate-700 bg-slate-950 px-2 py-2 text-sm text-slate-100 focus:border-sky-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500";

const finiteText = (value: number | null | undefined, digits = 4) =>
  value === null || value === undefined || !Number.isFinite(value) ? "—" : value.toFixed(digits);

const conventionLabel = (id: ConventionId) => ({
  app_native: "App-Native",
  trackman_comparable: "TrackMan-Comparable",
  foresight_comparable: "Foresight-Comparable",
}[id]);

function download(name: string, payload: unknown) {
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  URL.revokeObjectURL(url);
}

export function LaunchMonitorAnalyticsPanel() {
  const [rows, setRows] = useState<LaunchMonitorRow[]>(DEMO_ROWS);
  const [sourceName, setSourceName] = useState("Built-In Demonstration Data");
  const [outcome, setOutcome] = useState("ball_speed");
  const [predictors, setPredictors] = useState<string[]>(["club_speed", "attack_angle"]);
  const [mode, setMode] = useState<AnalysisMode>("comprehensive");
  const [method, setMethod] = useState<CorrelationMethod>("pearson");
  const [missing, setMissing] = useState<MissingPolicy>("pairwise");
  const [groupBy, setGroupBy] = useState("monitor_vendor");
  const [confidence, setConfidence] = useState(0.95);
  const [minSamples, setMinSamples] = useState(10);
  const [convention, setConvention] = useState<ConventionId>("app_native");
  const [result, setResult] = useState<LaunchMonitorAnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedRawIndex, setSelectedRawIndex] = useState<number | null>(null);
  const input = useRef<HTMLInputElement>(null);
  const importEpoch = useRef(0);
  const numeric = useMemo(() => numericLaunchMonitorColumns(rows), [rows]);
  const grouping = useMemo(() => {
    const columns = new Set(rows.flatMap((row) => Object.keys(row)));
    return [...columns].filter((column) => new Set(rows.map((row) => row[column])).size <= 100).sort();
  }, [rows]);
  const parameter = PARAMETER_IDS.includes(outcome as ParameterId) ? outcome as ParameterId : "club_speed";
  const definition = conventionRegistry().definition(convention, parameter);
  const invalidate = () => { setResult(null); setError(null); };

  const run = () => {
    try {
      setResult(analyzeLaunchMonitorData(rows, {
        outcome, predictors, analysisMode: mode, correlationMethod: method,
        missingPolicy: missing, groupBy: groupBy || undefined,
        confidenceLevel: confidence, minSamples,
      }));
      setError(null);
    } catch (caught) {
      setResult(null);
      setError(caught instanceof Error ? caught.message : String(caught));
    }
  };

  const loadFile = async (file: File) => {
    const epoch = ++importEpoch.current;
    try {
      const next = await readLaunchMonitorFile(file);
      if (epoch !== importEpoch.current) return;
      if (next.length > MAX_LINKED_SCATTER_ROWS) {
        throw new RangeError(`The retained-data limit is ${MAX_LINKED_SCATTER_ROWS} rows.`);
      }
      const nextNumeric = numericLaunchMonitorColumns(next);
      if (nextNumeric.length < 2) throw new RangeError("The file needs at least two numeric columns with three values each.");
      setRows(next);
      setSourceName(file.name);
      const nextOutcome = nextNumeric.includes("ball_speed") ? "ball_speed" : nextNumeric[0];
      const nextPredictors = ["club_speed", "attack_angle"]
        .filter((column) => nextNumeric.includes(column) && column !== nextOutcome);
      setOutcome(nextOutcome);
      setPredictors(nextPredictors.length ? nextPredictors : [
        nextNumeric.find((column) => column !== nextOutcome) as string,
      ]);
      setGroupBy(next.some((row) => "monitor_vendor" in row) ? "monitor_vendor" : "");
      setResult(null);
      setSelectedRawIndex(null);
      setError(null);
    } catch (caught) {
      if (epoch !== importEpoch.current) return;
      setError(caught instanceof Error ? caught.message : String(caught));
    }
  };

  return (
    <section aria-label="Launch Monitor Analytics" className="space-y-5">
      <div className={card}>
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <h2 className="text-xl font-semibold text-sky-200">Launch Monitor Analytics</h2>
            <details className="mt-1 max-w-3xl text-sm text-slate-400">
              <summary className="cursor-pointer text-slate-300">Scientific and import boundary</summary>
              <p className="mt-2">Import CSV or JSON records without dropping source columns, then
                correlate, regress, stratify, diagnose, and export any compatible numeric variables.
                Associations and fitted models are not causal evidence.</p>
            </details>
          </div>
          <div className="flex flex-wrap gap-2">
            <input ref={input} type="file" accept=".csv,.json,text/csv,application/json"
              className="hidden" aria-label="Launch monitor CSV or JSON file"
              onChange={(event) => { const file = event.target.files?.[0]; if (file) void loadFile(file); }} />
            <button type="button" title="Import a local CSV or JSON launch-monitor export"
              onClick={() => input.current?.click()}
              className="rounded-lg bg-sky-700 px-4 py-2 text-sm font-semibold hover:bg-sky-600 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500">Import Data</button>
            <button type="button" title="Restore the built-in non-vendor demonstration dataset"
              onClick={() => { importEpoch.current += 1; setRows(DEMO_ROWS); setSourceName("Built-In Demonstration Data"); setOutcome("ball_speed"); setPredictors(["club_speed", "attack_angle"]); setGroupBy("monitor_vendor"); setResult(null); setSelectedRawIndex(null); setError(null); }}
              className="rounded-lg border border-slate-700 px-4 py-2 text-sm hover:bg-slate-800 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500">Load Demo</button>
          </div>
        </div>
        <p className="mt-3 text-xs text-slate-500">Source: {sourceName} · {rows.length} retained rows · {Object.keys(rows[0] ?? {}).length} source columns</p>
      </div>

      <div className="grid gap-5 xl:grid-cols-[340px_1fr]">
        <section aria-label="Analysis contract" className={`${card} space-y-4`}>
          <h3 className="font-semibold text-slate-200">Analysis Contract</h3>
          <label className="block text-sm text-slate-300">Interpretation Convention
            <select value={convention} title="Choose the documented parameter convention used to interpret canonical names"
              onChange={(event) => { setConvention(event.target.value as ConventionId); invalidate(); }} className={`${field} mt-1`}>
              {(["app_native", "trackman_comparable", "foresight_comparable"] as ConventionId[])
                .map((id) => <option key={id} value={id}>{conventionLabel(id)}</option>)}
            </select>
          </label>
          <div className="rounded-lg border border-amber-500/30 bg-amber-950/20 p-3 text-xs text-amber-100">
            <strong>{conventionLabel(convention)}</strong> is a documented comparability frame, not
            device emulation or certification. {definition.label}: {definition.referencePoint.replace(/_/g, " ")}, {definition.eventTime.replace(/_/g, " ")}.
            {" "}<a className="underline" href={definition.sourceUrl} target="_blank" rel="noreferrer">Source definition</a>
          </div>
          <label className="block text-sm text-slate-300">Outcome
            <select value={outcome} title="Select the numeric outcome variable"
              onChange={(event) => { setOutcome(event.target.value); invalidate(); }} className={`${field} mt-1`}>
              {numeric.map((column) => <option key={column}>{column}</option>)}
            </select>
          </label>
          <label className="block text-sm text-slate-300">Predictors
            <select multiple value={predictors} title="Select one or more numeric predictor variables"
              aria-label="Predictor Variables" onChange={(event) => {
                setPredictors([...event.currentTarget.selectedOptions].map((option) => option.value));
                invalidate();
              }} className={`${field} mt-1 min-h-36`}>
              {numeric.filter((column) => column !== outcome).map((column) => <option key={column}>{column}</option>)}
            </select>
          </label>
          <div className="grid grid-cols-2 gap-3">
            <label className="text-sm text-slate-300">Mode
              <select value={mode} title="Choose correlation, regression, or both"
                onChange={(event) => { setMode(event.target.value as AnalysisMode); invalidate(); }} className={`${field} mt-1`}>
                <option value="comprehensive">Comprehensive</option><option value="correlation">Correlation</option><option value="regression">Regression</option>
              </select>
            </label>
            <label className="text-sm text-slate-300">Correlation
              <select value={method} title="Choose the correlation estimator"
                onChange={(event) => { setMethod(event.target.value as CorrelationMethod); invalidate(); }} className={`${field} mt-1`}>
                <option value="pearson">Pearson</option><option value="spearman">Spearman</option><option value="kendall">Kendall</option>
              </select>
            </label>
            <label className="text-sm text-slate-300">Missing Data
              <select value={missing} title="Choose how missing numeric values are handled"
                onChange={(event) => { setMissing(event.target.value as MissingPolicy); invalidate(); }} className={`${field} mt-1`}>
                <option value="pairwise">Pairwise</option><option value="listwise">Listwise</option><option value="fail">Fail Closed</option>
              </select>
            </label>
            <label className="text-sm text-slate-300">Group By
              <select value={groupBy} title="Optionally compute separate results for each group"
                onChange={(event) => { setGroupBy(event.target.value); invalidate(); }} className={`${field} mt-1`}>
                <option value="">No Grouping</option>{grouping.map((column) => <option key={column}>{column}</option>)}
              </select>
            </label>
            <label className="text-sm text-slate-300">Confidence
              <input type="number" min="0.51" max="0.999" step="0.01" value={confidence}
                title="Set the confidence level for analytical intervals" aria-label="Confidence Level"
                onChange={(event) => { setConfidence(Number(event.target.value)); invalidate(); }} className={`${field} mt-1`} />
            </label>
            <label className="text-sm text-slate-300">Minimum N
              <input type="number" min="3" step="1" value={minSamples}
                title="Set the minimum observations per analysis" aria-label="Minimum Sample Count"
                onChange={(event) => { setMinSamples(Number(event.target.value)); invalidate(); }} className={`${field} mt-1`} />
            </label>
          </div>
          <button type="button" onClick={run} title="Run the selected traceable statistical analysis"
            className="w-full rounded-lg bg-emerald-700 px-4 py-3 font-semibold hover:bg-emerald-600 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500">Run Analysis</button>
          {error && <p role="alert" className="rounded border border-red-500/40 bg-red-950/30 p-3 text-sm text-red-200">{error}</p>}
        </section>

        <div className="order-first space-y-5 xl:order-none">
          <div className={card}>
            <h3 className="mb-3 font-semibold text-slate-200">Selected Relationship</h3>
            <LaunchMonitorLinkedScatter rows={rows} yField={outcome}
              xField={predictors[0] ?? ""} selectedRawIndex={selectedRawIndex}
              onSelectedRawIndex={setSelectedRawIndex} />
          </div>
          {!result ? (
            <div className={`${card} text-center text-slate-500`}>Run the analysis to populate uncertainty, diagnostics, grouping, and lineage.</div>
          ) : (
            <>
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                <div className={card}><p className="text-xs uppercase text-slate-500">Rows / Complete</p><p className="text-2xl font-semibold">{result.dataset.rowCount} / {result.dataset.completeRowCount}</p></div>
                <div className={card}><p className="text-xs uppercase text-slate-500">R² / Adjusted</p><p className="text-2xl font-semibold">{finiteText(result.regression?.rSquared)} / {finiteText(result.regression?.adjustedRSquared)}</p></div>
                <div className={card}><p className="text-xs uppercase text-slate-500">RMSE / MAE</p><p className="text-2xl font-semibold">{finiteText(result.regression?.residualDiagnostics.rmse)} / {finiteText(result.regression?.residualDiagnostics.mae)}</p></div>
                <div className={card}><p className="text-xs uppercase text-slate-500">Groups</p><p className="text-2xl font-semibold">{result.groups.length || "—"}</p></div>
              </div>
              <div className={`${card} overflow-x-auto`}>
                <h3 className="mb-3 font-semibold text-slate-200">Correlations and Multiplicity Control</h3>
                <table className="w-full text-left text-sm"><thead className="text-slate-400"><tr><th>Predictor</th><th>r / τ</th><th>p</th><th>BH q</th><th>Confidence Interval</th><th>N</th></tr></thead>
                  <tbody>{result.correlations.map((item) => <tr key={item.predictor} className="border-t border-slate-800"><td className="py-2">{item.predictor}</td><td>{finiteText(item.coefficient)}</td><td>{finiteText(item.pValue)}</td><td>{finiteText(item.adjustedPValue)}</td><td>[{finiteText(item.ciLower)}, {finiteText(item.ciUpper)}]</td><td>{item.sampleCount}</td></tr>)}</tbody></table>
              </div>
              {result.regression && <div className={`${card} overflow-x-auto`}>
                <h3 className="mb-3 font-semibold text-slate-200">OLS Coefficients</h3>
                <table className="w-full text-left text-sm"><thead className="text-slate-400"><tr><th>Term</th><th>Estimate</th><th>SE</th><th>t</th><th>p</th><th>Confidence Interval</th></tr></thead>
                  <tbody>{Object.entries(result.regression.coefficients).map(([name, item]) => <tr key={name} className="border-t border-slate-800"><td className="py-2">{name}</td><td>{finiteText(item.estimate)}</td><td>{finiteText(item.standardError)}</td><td>{finiteText(item.tStatistic)}</td><td>{finiteText(item.pValue)}</td><td>[{finiteText(item.ciLower)}, {finiteText(item.ciUpper)}]</td></tr>)}</tbody></table>
              </div>}
              <div className={card}>
                <h3 className="font-semibold text-slate-200">Traceability</h3>
                <p className="mt-2 break-all font-mono text-xs text-slate-400">SHA-256: {result.dataset.fingerprintSha256}</p>
                <p className="mt-2 text-sm text-slate-400">Monitors: {result.dataset.monitorVendors.join(", ") || "unrecorded"} · Sessions: {result.dataset.sessionIds.join(", ") || "unrecorded"} · Contract: {result.contractVersion}</p>
                {result.warnings.map((warning) => <p key={warning} className="mt-2 text-sm text-amber-200">{warning}</p>)}
                <div className="mt-4 flex flex-wrap gap-2">
                  <button type="button" title="Download every retained input record as JSON"
                    onClick={() => download("launch-monitor-records.json", rows)} className="rounded border border-slate-700 px-3 py-2 text-sm hover:bg-slate-800 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500">Export Retained Data</button>
                  <button type="button" title="Download the complete request, statistics, warnings, and lineage as JSON"
                    onClick={() => download("launch-monitor-analysis.json", result)} className="rounded border border-slate-700 px-3 py-2 text-sm hover:bg-slate-800 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500">Export Analysis</button>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
      <LaunchMonitorPlayerWorkspace rows={rows} sourceName={sourceName} />
      <LaunchMonitorPerformanceWorkspace rows={rows} sourceName={sourceName} />
    </section>
  );
}
