import { useEffect, useMemo, useRef, useState } from "react";

import { analyzeLaunchMonitorData, numericLaunchMonitorColumns } from "../model/launchMonitorAnalysis";
import type { LaunchMonitorAnalysisResult, LaunchMonitorRow } from "../model/launchMonitorAnalysisTypes";
import {
  createAnalysisExportBundle,
  fingerprintLaunchMonitorRows,
  parseLaunchMonitorProject,
  serializeLaunchMonitorProject,
  type LaunchMonitorProject,
} from "../model/launchMonitorWorkspace";
import { LaunchMonitorCovariation } from "./LaunchMonitorCovariation";
import {
  MAX_CANONICAL_INLINE_RECORDS,
  buildDatasetJobRequest,
  buildPlayerCovariationPayload,
  createCanonicalLaunchMonitorClient,
  parseCanonicalDatasetReference,
  type CanonicalDatasetReference,
} from "../model/launchMonitorV2Client";

interface Props { rows: LaunchMonitorRow[]; sourceName: string }

const field = "w-full rounded border border-slate-700 bg-slate-950 px-2 py-2 text-sm text-slate-100";

function download(name: string, content: string, type = "application/json") {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  URL.revokeObjectURL(url);
}

export function LaunchMonitorPlayerWorkspace({ rows, sourceName }: Props) {
  const columns = useMemo(() => [...new Set(rows.flatMap((row) => Object.keys(row)))].sort(), [rows]);
  const numeric = useMemo(() => numericLaunchMonitorColumns(rows), [rows]);
  const [identity, setIdentity] = useState("");
  const [attested, setAttested] = useState(false);
  const [x, setX] = useState(numeric.includes("face_angle") ? "face_angle" : numeric[0] ?? "");
  const [y, setY] = useState(numeric.includes("club_path") ? "club_path" : numeric[1] ?? "");
  const [datasetSha, setDatasetSha] = useState("");
  const [result, setResult] = useState<LaunchMonitorAnalysisResult | null>(null);
  const [message, setMessage] = useState("Select and attest an explicit player identity column.");
  const [authorityUrl, setAuthorityUrl] = useState("");
  const [canonicalReference, setCanonicalReference] = useState<CanonicalDatasetReference | null>(null);
  const [canonicalJobId, setCanonicalJobId] = useState("");
  const [canonicalResult, setCanonicalResult] = useState<Record<string, unknown> | null>(null);
  const loadInput = useRef<HTMLInputElement>(null);
  const corpusInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setDatasetSha(fingerprintLaunchMonitorRows(rows));
    setAttested(false);
    setIdentity("");
    setResult(null);
    setCanonicalResult(null);
  }, [rows]);

  const project = (): LaunchMonitorProject => ({
    contractVersion: "2.0.0",
    name: `${sourceName} player covariation`,
    dataset: {
      sourceName, repository: "local-user-data", revision: "unversioned",
      relativePath: sourceName, sha256: datasetSha, rowCount: rows.length,
    },
    playerIdentity: { column: identity, userAttested: attested },
    selection: { x, y, minSamples: 10, confidenceLevel: 0.95 },
    ...(canonicalReference ? { canonicalDataset: canonicalReference } : {}),
  });
  const ready = Boolean(identity && attested && x && y && x !== y && datasetSha);
  const canonicalReady = ready && Boolean(authorityUrl.trim()) && rows.length <= MAX_CANONICAL_INLINE_RECORDS;

  const loadCorpusReference = async (file: File) => {
    try {
      const reference = parseCanonicalDatasetReference(JSON.parse(await file.text()));
      setCanonicalReference(reference);
      setCanonicalJobId("");
      setMessage(`Authorized reference loaded: ${reference.expected_row_count.toLocaleString()} rows; no private rows were loaded.`);
    } catch (caught) {
      setCanonicalReference(null);
      setMessage(caught instanceof Error ? caught.message : String(caught));
    }
  };

  const inspectCorpus = async () => {
    if (!canonicalReference) return;
    try {
      const client = createCanonicalLaunchMonitorClient(authorityUrl);
      const status = await client.submitDatasetJob(buildDatasetJobRequest(canonicalReference, "source_summary"));
      setCanonicalJobId(String(status.job_id));
      setMessage(`Canonical corpus job ${String(status.job_id)} is ${String(status.status)}.`);
    } catch (caught) { setMessage(caught instanceof Error ? caught.message : String(caught)); }
  };

  const refreshCorpus = async () => {
    if (!canonicalJobId) return;
    try {
      const client = createCanonicalLaunchMonitorClient(authorityUrl);
      const status = await client.datasetJobStatus(canonicalJobId);
      if (status.status === "completed") {
        const page = await client.datasetJobResults(canonicalJobId);
        setMessage(`Canonical corpus job completed with ${String(page.total_items)} bounded aggregate items.`);
      } else setMessage(`Canonical corpus job is ${String(status.status)}.`);
    } catch (caught) { setMessage(caught instanceof Error ? caught.message : String(caught)); }
  };

  const runCanonical = async () => {
    try {
      const payload = buildPlayerCovariationPayload(rows, { playerColumn: identity, xColumn: x, yColumn: y, minSamples: 10, confidenceLevel: 0.95 });
      const response = await createCanonicalLaunchMonitorClient(authorityUrl).playerCovariation(payload);
      setResult(null);
      setCanonicalResult(response);
      setMessage(`Canonical Upstream player covariation completed (${String(response.status)}) with evidence-bearing lineage.`);
    } catch (caught) { setMessage(caught instanceof Error ? caught.message : String(caught)); }
  };

  const run = () => {
    try {
      const next = analyzeLaunchMonitorData(rows, {
        outcome: y, predictors: [x], analysisMode: "correlation",
        correlationMethod: "pearson", missingPolicy: "pairwise",
        groupBy: identity, confidenceLevel: 0.95, minSamples: 10,
      });
      setResult(next);
      setCanonicalResult(null);
      setMessage(`${next.groups.length} player groups analyzed. Associations are not causal; no player identity was inferred.`);
    } catch (caught) {
      setResult(null);
      setMessage(caught instanceof Error ? caught.message : String(caught));
    }
  };

  const load = async (file: File) => {
    try {
      const saved = parseLaunchMonitorProject(await file.text());
      if (saved.dataset.sha256 !== datasetSha) throw new RangeError("Saved project references a different dataset");
      setIdentity(saved.playerIdentity.column);
      setAttested(saved.playerIdentity.userAttested);
      setX(saved.selection.x);
      setY(saved.selection.y);
      setCanonicalReference(saved.canonicalDataset ?? null);
      setMessage("Saved project settings restored against the matching dataset fingerprint.");
    } catch (caught) {
      setMessage(caught instanceof Error ? caught.message : String(caught));
    }
  };

  const exportBundle = async () => {
    const analysis = result as unknown as Record<string, unknown> | null ?? canonicalResult;
    if (!analysis) return;
    const bundle = await createAnalysisExportBundle(project(), analysis, rows);
    download("launch-monitor-analysis-bundle.json", JSON.stringify(bundle, null, 2));
  };

  return <section aria-label="Player analytics workspace" className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
    <h3 className="font-semibold text-slate-200">Player Covariation Workspace</h3>
    <p className="mt-1 text-xs text-slate-400">Identity is never inferred from session, club, filename, or row order. The replaceable client validates canonical UpstreamDrift v2 responses. With no authority URL configured in this standalone build, Run is explicitly the local v1 compatibility/offline calculation; row-aligned residuals are unavailable.</p>
    <div className="mt-3 rounded border border-slate-700 p-3">
      <label className="text-sm text-slate-300">Canonical authority URL
        <input className={`${field} mt-1`} type="url" aria-label="Canonical Upstream authority URL"
          title="Authorized HTTP(S) UpstreamDrift analytics authority; private filesystem paths are not accepted"
          value={authorityUrl} onChange={(event) => { setAuthorityUrl(event.target.value); setCanonicalJobId(""); setCanonicalResult(null); }} />
      </label>
      <p className="mt-2 text-xs text-slate-400">Canonical inline limit is 20,000 rows. Larger authorized corpora use immutable reference-only aggregate jobs; private rows are never stored in browser projects.</p>
      <input ref={corpusInput} type="file" accept=".json,application/json" className="hidden"
        aria-label="Load authorized corpus reference" onChange={(event) => { const file = event.target.files?.[0]; if (file) void loadCorpusReference(file); }} />
      <div className="mt-2 flex flex-wrap gap-2">
        <button type="button" title="Load an opaque root alias and immutable hashes, never a private path or row"
          onClick={() => corpusInput.current?.click()} className="rounded border border-slate-700 px-3 py-2 text-sm">Select Authorized Corpus Reference</button>
        <button type="button" disabled={!canonicalReference || !authorityUrl.trim()} title="Submit a source-summary job using only the immutable authorized reference"
          onClick={() => void inspectCorpus()} className="rounded border border-slate-700 px-3 py-2 text-sm disabled:opacity-40">Inspect Authorized Corpus</button>
        <button type="button" disabled={!canonicalJobId} title="Refresh the data-free job state and fetch bounded aggregates after completion"
          onClick={() => void refreshCorpus()} className="rounded border border-slate-700 px-3 py-2 text-sm disabled:opacity-40">Refresh Corpus Job</button>
        <button type="button" disabled={!canonicalReady} title={rows.length > MAX_CANONICAL_INLINE_RECORDS ? "Unavailable: canonical inline covariation accepts at most 20,000 rows" : "Run the canonical evidence-bearing player covariation endpoint"}
          onClick={() => void runCanonical()} className="rounded bg-sky-700 px-3 py-2 text-sm disabled:opacity-40">Run Canonical Player Covariation</button>
      </div>
    </div>
    <div className="mt-3 grid gap-3 sm:grid-cols-3">
      <label className="text-sm text-slate-300">Player identity
        <select aria-label="Player identity column" title="Choose a real player identifier supplied by the dataset owner" value={identity}
          onChange={(event) => { setIdentity(event.target.value); setAttested(false); setResult(null); setCanonicalResult(null); }} className={`${field} mt-1`}>
          <option value="">Select column</option>{columns.map((column) => <option key={column}>{column}</option>)}
        </select>
      </label>
      <label className="text-sm text-slate-300">X variable
        <select aria-label="Player covariation X variable" title="Choose the first covariation variable" value={x}
          onChange={(event) => { setX(event.target.value); setResult(null); setCanonicalResult(null); }} className={`${field} mt-1`}>
          {numeric.map((column) => <option key={column}>{column}</option>)}
        </select>
      </label>
      <label className="text-sm text-slate-300">Y variable
        <select aria-label="Player covariation Y variable" title="Choose the second covariation variable" value={y}
          onChange={(event) => { setY(event.target.value); setResult(null); setCanonicalResult(null); }} className={`${field} mt-1`}>
          {numeric.map((column) => <option key={column}>{column}</option>)}
        </select>
      </label>
    </div>
    <label className="mt-3 flex items-start gap-2 text-sm text-amber-100">
      <input type="checkbox" aria-label="I attest this column identifies a player" title="Required identity-safety attestation"
        checked={attested} disabled={!identity} onChange={(event) => { setAttested(event.target.checked); setResult(null); setCanonicalResult(null); }} />
      I attest this column identifies a player; it was not inferred from session, club, filename, or row order.
    </label>
    <div className="mt-3 flex flex-wrap gap-2">
      <button type="button" disabled={!ready} title="Run the explicitly labelled local v1 compatibility calculation" onClick={run}
        className="rounded bg-emerald-700 px-3 py-2 text-sm disabled:opacity-40">Run Offline Compatibility Covariation</button>
      <button type="button" disabled={!ready} title="Save a persistent reference-only project that does not embed private rows"
        onClick={() => download("analysis.lmproject.json", serializeLaunchMonitorProject(project()))}
        className="rounded border border-slate-700 px-3 py-2 text-sm disabled:opacity-40">Save Project</button>
      <input ref={loadInput} type="file" accept=".json,application/json" className="hidden" aria-label="Load saved launch-monitor project"
        onChange={(event) => { const file = event.target.files?.[0]; if (file) void load(file); }} />
      <button type="button" title="Load settings from a saved project after fingerprint verification" onClick={() => loadInput.current?.click()}
        className="rounded border border-slate-700 px-3 py-2 text-sm">Load Project</button>
      <button type="button" disabled={!result && !canonicalResult} title="Export the project, result, manifest, hashes, and explicit backing rows"
        onClick={() => void exportBundle()} className="rounded border border-slate-700 px-3 py-2 text-sm disabled:opacity-40">Export Full Bundle</button>
    </div>
    <p role="status" className="mt-3 text-sm text-slate-400">{message}</p>
    {canonicalResult && <details className="mt-3 rounded border border-slate-700 p-3">
      <summary className="cursor-pointer text-sm text-slate-200">Canonical player covariation evidence</summary>
      <pre aria-label="Canonical player covariation evidence" className="mt-2 max-h-64 overflow-auto whitespace-pre-wrap text-xs text-slate-300">
        {JSON.stringify(canonicalResult, null, 2)}
      </pre>
    </details>}
    {identity && attested && <div className="mt-5 border-t border-slate-800 pt-5">
      <LaunchMonitorCovariation rows={rows} lockedPlayerColumn={identity}
        savedSettings={{
          playerColumn: identity, xColumn: x, yColumn: y, selectedPlayer: "",
          method: "pearson", minSamples: 10, confidenceLevel: 0.95,
        }}
        onSettingsChange={(settings) => { setX(settings.xColumn); setY(settings.yColumn); }} />
    </div>}
  </section>;
}
