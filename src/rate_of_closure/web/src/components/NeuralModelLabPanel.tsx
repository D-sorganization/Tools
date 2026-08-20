import { useMemo, useRef, useState } from "react";

import capabilityData from "../../../data/neural_vendor_capabilities.v2.json";
import { readLaunchMonitorFile, type LaunchMonitorRow } from "../model/launchMonitorAnalysis";
import { buildTrainingManifest, inferPortableModel, parseCapabilityManifest,
  parsePortableModel, type PortableModel } from "../model/neuralLabContract";

const card = "rounded-xl border border-slate-800/80 bg-slate-900/60 p-4";
const field = "rounded border border-slate-700 bg-slate-950 px-2 py-1 text-slate-100";
const defaultCapabilities = parseCapabilityManifest(capabilityData);

function download(name: string, content: string, type = "application/json") {
  const url = URL.createObjectURL(new Blob([content], { type })); const anchor = document.createElement("a");
  anchor.href = url; anchor.download = name; anchor.click(); URL.revokeObjectURL(url);
}

async function fileSha256(file: File): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", await file.arrayBuffer());
  return [...new Uint8Array(digest)].map((value) => value.toString(16).padStart(2, "0")).join("");
}

function CapabilityPlot({ vendors }: { readonly vendors: typeof defaultCapabilities.vendors }) {
  const maximum = Math.max(1, ...vendors.map((vendor) => vendor.strictRowCount));
  return <svg viewBox="0 0 640 190" role="img" aria-label="Vendor strict eligible input rows chart" className="mt-3 w-full rounded bg-slate-950">
    <title>Strict five-input rows by vendor; availability remains policy governed</title>
    <text x="12" y="18" fill="#94a3b8">Strict eligible input rows (count)</text>
    {vendors.map((vendor, index) => { const width = 480 * vendor.strictRowCount / maximum; const y = 38 + index * 46;
      return <g key={vendor.vendor}><text x="12" y={y + 18} fill="#cbd5e1">{vendor.vendor}</text>
        <rect x="120" y={y} width={Math.max(1, width)} height="24" fill="#38bdf8"/><text x={128 + width} y={y + 18} fill="#e2e8f0">{vendor.strictRowCount.toLocaleString()} rows</text></g>; })}
  </svg>;
}

function ResidualPlot({ model }: { readonly model: PortableModel }) {
  const rows = model.residuals.rows ?? [];
  if (model.residuals.state !== "available" || !rows.length) return <p role="status" className="text-amber-300">
    Residual plot unavailable: {model.residuals.reason ?? "row-aligned held-out residuals were not exported."}</p>;
  const points = rows.flatMap((row, index) => typeof row.residual === "number" ? [{ x: index, y: row.residual }] : []);
  if (!points.length) return <p role="status" className="text-amber-300">Residual plot unavailable: residual rows lack finite residual values.</p>;
  const extent = Math.max(1, ...points.map(({ y }) => Math.abs(y)));
  return <svg viewBox="0 0 640 220" role="img" aria-label="Held-out residual by aligned row plot" className="w-full rounded bg-slate-950">
    <title>Held-out residual by aligned row; zero is perfect prediction</title>
    <line x1="45" x2="625" y1="110" y2="110" stroke="#64748b"/><text x="5" y="18" fill="#94a3b8">Residual (target unit)</text>
    <text x="500" y="210" fill="#94a3b8">Aligned held-out row</text>
    {points.map(({ x, y }) => <circle key={x} cx={45 + x * 580 / Math.max(1, points.length - 1)} cy={110 - y * 90 / extent} r="3" fill="#38bdf8" />)}
  </svg>;
}

export function NeuralModelLabPanel() {
  const datasetInput = useRef<HTMLInputElement>(null); const modelInput = useRef<HTMLInputElement>(null);
  const capabilityInput = useRef<HTMLInputElement>(null);
  const [capabilities, setCapabilities] = useState(defaultCapabilities);
  const [rows, setRows] = useState<LaunchMonitorRow[]>([]); const [datasetName, setDatasetName] = useState("");
  const [datasetSha, setDatasetSha] = useState(""); const [repository, setRepository] = useState("");
  const [commit, setCommit] = useState(""); const [vendor, setVendor] = useState("Custom");
  const [features, setFeatures] = useState(""); const [targets, setTargets] = useState("");
  const [splitGroup, setSplitGroup] = useState(""); const [approved, setApproved] = useState(false);
  const [endpoint, setEndpoint] = useState(""); const [job, setJob] = useState<Record<string, unknown> | null>(null);
  const [model, setModel] = useState<PortableModel | null>(null); const [inputs, setInputs] = useState<Record<string, number>>({});
  const [prediction, setPrediction] = useState<ReturnType<typeof inferPortableModel> | null>(null);
  const [message, setMessage] = useState("No private training request submitted.");
  const columns = useMemo(() => [...new Set(rows.flatMap((row) => Object.keys(row)))].sort(), [rows]);

  const manifest = () => buildTrainingManifest({ datasetId: datasetName, repository, commit,
    datasetPath: datasetName, sha256: datasetSha, rowCount: rows.length }, rows,
  { vendor, features: features.split(",").map((item) => item.trim()).filter(Boolean),
    targets: targets.split(",").map((item) => item.trim()).filter(Boolean), splitGroup, splitGroupPolicyApproved: approved });

  const submit = async () => { try {
    const request = manifest();
    if (!endpoint.trim()) { download("neural-training-request.v2.json", `${JSON.stringify(request, null, 2)}\n`);
      setMessage("Request exported for submission by the private training CLI; no browser training occurred."); return; }
    const response = await fetch(`${endpoint.replace(/\/$/, "")}/jobs`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(request) });
    if (!response.ok) throw new Error(`Private training service rejected the request (${response.status}).`);
    const next = await response.json() as Record<string, unknown>; setJob(next); setMessage("Private training request submitted.");
  } catch (error) { setMessage(error instanceof Error ? error.message : String(error)); } };

  const monitor = async () => { try { const id = job?.job_id;
    if (!endpoint.trim() || typeof id !== "string") throw new Error("No private training job ID is available to monitor.");
    const response = await fetch(`${endpoint.replace(/\/$/, "")}/jobs/${encodeURIComponent(id)}`);
    if (!response.ok) throw new Error(`Private training status failed (${response.status}).`);
    const next = await response.json() as Record<string, unknown>; setJob(next); setMessage(`Job state: ${String(next.state ?? "unknown")}`);
  } catch (error) { setMessage(error instanceof Error ? error.message : String(error)); } };

  return <section aria-label="Neural Model Lab" className="space-y-5">
    <div className={card}><h2 className="text-xl font-semibold text-sky-200">Neural Model Lab</h2>
      <p className="mt-2 text-sm text-slate-300">Safe client for private, group-safe vendor-comparable surrogate training. This browser never trains on or persists private rows. Models are descriptive and are not device emulation or certification.</p>
      <input ref={capabilityInput} type="file" className="hidden" accept=".json" aria-label="Private capability manifest" onChange={(event) => { const file = event.target.files?.[0]; if (file) void file.text().then((value) => { try { setCapabilities(parseCapabilityManifest(JSON.parse(value))); setMessage("Loaded user-authorized private capability metadata; no private rows or path were persisted."); } catch (error) { setMessage(error instanceof Error ? error.message : String(error)); } }); }} />
      <button type="button" title="Load a user-authorized private capability manifest without persisting its path or rows" className="mt-2 rounded border border-slate-700 px-3 py-1" onClick={() => capabilityInput.current?.click()}>Load Capability Manifest</button>
      <CapabilityPlot vendors={capabilities.vendors}/>
      <div className="mt-3 grid gap-3 md:grid-cols-3">{capabilities.vendors.map((item) => <article key={item.vendor} className="rounded border border-slate-700 p-3" title={item.blockers.join(" ")}>
        <h3 className="font-semibold">{item.vendor}: unavailable</h3><p>{item.rowCount.toLocaleString()} rows / {item.strictRowCount.toLocaleString()} strict</p>
        <p className="text-xs text-amber-300">Artifact: {item.artifactState}</p><ul className="list-disc pl-4 text-xs text-slate-400">{item.blockers.map((reason) => <li key={reason}>{reason}</li>)}</ul>
      </article>)}</div>
    </div>
    <div className={card}><h3 className="font-semibold">Private training request</h3>
      <input ref={datasetInput} type="file" className="hidden" accept=".csv,.json" aria-label="Custom training dataset" onChange={(event) => { const file = event.target.files?.[0]; if (file) void Promise.all([readLaunchMonitorFile(file), fileSha256(file)]).then(([data, hash]) => { setRows(data); setDatasetName(file.name); setDatasetSha(hash); setMessage(`Loaded ${data.length} local rows by reference; rows are not submitted until a private authority is configured.`); }).catch((error) => setMessage(String(error))); }} />
      <button type="button" title="Select a custom local CSV or JSON dataset" className="mt-2 rounded bg-sky-700 px-3 py-1" onClick={() => datasetInput.current?.click()}>Select Custom Dataset</button>
      <p className="mt-2 text-xs text-slate-400">{datasetName || "No dataset"} — {rows.length} rows — SHA-256 {datasetSha || "unavailable"}</p>
      <div className="mt-3 grid gap-2 md:grid-cols-2">{[
        ["Private repository", repository, setRepository], ["Immutable 40-char commit", commit, setCommit], ["Vendor", vendor, setVendor],
        ["Features (comma separated)", features, setFeatures], ["Targets (comma separated)", targets, setTargets], ["Repeating split group", splitGroup, setSplitGroup],
        ["Private API base (optional)", endpoint, setEndpoint],
      ].map(([label, value, setter]) => <label key={label as string} title={label as string}>{label as string}<input aria-label={label as string} className={`${field} ml-2`} value={value as string} onChange={(event) => (setter as (value: string) => void)(event.target.value)} /></label>)}</div>
      <p className="mt-2 text-xs text-slate-500">Columns: {columns.join(", ") || "unavailable"}</p>
      <label title="Attest that the split column is policy-approved and represents repeatable independent groups"><input type="checkbox" aria-label="Policy-approved repeating split group" checked={approved} onChange={(event) => setApproved(event.target.checked)} /> Policy-approved repeating split group</label>
      <div className="mt-3 flex gap-2"><button type="button" title="Submit to a private API or export a CLI request" className="rounded bg-emerald-700 px-3 py-1" onClick={() => void submit()}>Submit / Export Request</button>
        <button type="button" title="Poll the configured private training API" className="rounded border border-slate-700 px-3 py-1" onClick={() => void monitor()}>Monitor Job</button></div>
      <p role="status" className="mt-2 text-sm text-amber-200">{message}</p>{job && <pre className="overflow-auto text-xs">{JSON.stringify(job, null, 2)}</pre>}
    </div>
    <div className={card}><h3 className="font-semibold">Validated portable inference</h3>
      <input ref={modelInput} type="file" className="hidden" accept=".json" aria-label="Portable neural model JSON" onChange={(event) => { const file = event.target.files?.[0]; if (file) void file.text().then((value) => { try { const next = parsePortableModel(JSON.parse(value)); setModel(next); setInputs(Object.fromEntries(next.features.map((feature) => [feature.name, feature.mean]))); setPrediction(null); setMessage(`Validated ${next.modelId}.`); } catch (error) { setModel(null); setMessage(error instanceof Error ? error.message : String(error)); } }); }} />
      <button type="button" title="Load a non-executable JSON model after schema and hash validation" className="mt-2 rounded bg-sky-700 px-3 py-1" onClick={() => modelInput.current?.click()}>Load Portable Model</button>
      {model && <><p className="mt-2">{model.modelId} — {model.vendor}</p><pre className="overflow-auto text-xs" title="Model card and held-out metrics">{JSON.stringify({ modelCard: model.modelCard, metrics: model.metrics }, null, 2)}</pre>
        <div className="flex flex-wrap gap-2">{model.features.map((feature) => <label key={feature.name} title={`${feature.name}; training range ${feature.min} to ${feature.max} ${feature.unit}`}>{feature.name} ({feature.unit})<input type="number" aria-label={`${feature.name} query in ${feature.unit}`} className={`${field} ml-1 w-28`} value={inputs[feature.name]} onChange={(event) => setInputs({ ...inputs, [feature.name]: Number(event.target.value) })}/></label>)}</div>
        <button type="button" title="Run validated local JSON inference" className="my-2 rounded bg-emerald-700 px-3 py-1" onClick={() => { try { setPrediction(inferPortableModel(model, inputs)); } catch (error) { setMessage(String(error)); } }}>Query Model</button>
        {prediction && <pre className="text-sm" title="Prediction and out-of-domain warnings">{JSON.stringify(prediction, null, 2)}</pre>}
        <ResidualPlot model={model}/><button type="button" title="Export model card, metrics, residual availability, and provenance" className="mt-2 rounded border border-slate-700 px-3 py-1" onClick={() => download(`${model.modelId}-inspection.json`, `${JSON.stringify(model, (_key, value) => value, 2)}\n`)}>Export Inspection</button></>}
    </div>
  </section>;
}
