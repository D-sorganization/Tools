import { useMemo, useState } from "react";

import { parseLaunchMonitorFile } from "../model/launchMonitorFileParsing";
import { applicabilityWarnings, inferPortableModel, parsePortableModelBundle, type PortableModelBundle } from "../model/neuralModelBundle";
import { createTrainingRequest, type TrainingDatasetReference } from "../model/neuralTrainingRequest";
import { LearningCurveChart, MetricCards, ResidualChart, type PredictionPoint } from "./NeuralModelCharts";

const card = "rounded-xl border border-slate-800/80 bg-slate-900/60 p-4 shadow-lg shadow-black/20";
const control = "rounded border border-slate-700 bg-slate-800 px-2 py-1.5 text-slate-100";
const vendors = [
  { name: "TrackMan-comparable", enabled: true, reason: "Approved row-level TrackMan targets are available." },
  { name: "Foresight-comparable", enabled: false, reason: "Aggregate evidence only; no approved row-level targets." },
  { name: "FlightScope-comparable", enabled: false, reason: "Aggregate evidence only; no approved row-level targets." },
  { name: "Custom", enabled: true, reason: "Requires traceable row-level targets supplied by the user." },
];

const download = (name: string, content: string, type: string) => {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  URL.revokeObjectURL(url);
};

const numberList = (value: string): number[] => value.split(",").map(Number);
const columnList = (value: string): string[] => value.split(",").map((item) => item.trim()).filter(Boolean);
const numeric = (value: unknown): number | undefined => {
  const converted = Number(value);
  return Number.isFinite(converted) ? converted : undefined;
};

const readFileText = (file: File): Promise<string> => new Promise((resolve, reject) => {
  const reader = new FileReader();
  reader.onerror = () => reject(new Error(`Unable to read ${file.name}`));
  reader.onload = () => resolve(String(reader.result ?? ""));
  reader.readAsText(file);
});

function fileMetadata(fileName: string, rows: Record<string, unknown>[]): TrainingDatasetReference {
  return { fileName, rowCount: rows.length, columns: rows.length ? Object.keys(rows[0]) : [] };
}

export function NeuralModelLabPanel() {
  const [vendor, setVendor] = useState("TrackMan-comparable");
  const [bundle, setBundle] = useState<PortableModelBundle | null>(null);
  const [dataset, setDataset] = useState<TrainingDatasetReference | null>(null);
  const [featureText, setFeatureText] = useState("");
  const [outputText, setOutputText] = useState("");
  const [layers, setLayers] = useState("64,32");
  const [activation, setActivation] = useState<"relu" | "tanh" | "linear">("relu");
  const [alpha, setAlpha] = useState(0.0001);
  const [epochs, setEpochs] = useState(500);
  const [learningRate, setLearningRate] = useState(0.001);
  const [error, setError] = useState("");
  const [inputs, setInputs] = useState<Record<string, string>>({});
  const [prediction, setPrediction] = useState<Record<string, number>>({});
  const [warnings, setWarnings] = useState<string[]>([]);
  const [batchRows, setBatchRows] = useState<Record<string, unknown>[]>([]);
  const [predictionRows, setPredictionRows] = useState<PredictionPoint[]>([]);
  const displayedVendor = bundle?.vendor ?? vendor;

  const modelInputs = useMemo(() => bundle?.features ?? [], [bundle]);

  const loadBundle = async (file?: File) => {
    if (!file) return;
    try {
      const parsed = parsePortableModelBundle(await readFileText(file));
      setBundle(parsed); setVendor(parsed.vendor); setError("");
      setInputs(Object.fromEntries(parsed.features.map((feature) => [feature.name, String(feature.mean)])));
    } catch (caught) { setError(String(caught)); }
  };

  const loadDataset = async (file?: File) => {
    if (!file) return;
    try {
      const rows = parseLaunchMonitorFile(file.name, await readFileText(file)) as Record<string, unknown>[];
      const metadata = fileMetadata(file.name, rows);
      setDataset(metadata); setFeatureText(metadata.columns.slice(0, Math.max(1, metadata.columns.length - 1)).join(", "));
      setOutputText(metadata.columns[metadata.columns.length - 1] ?? ""); setError("");
    } catch (caught) { setError(String(caught)); }
  };

  const exportRequest = () => {
    try {
      if (!dataset) throw new RangeError("Import the current or custom CSV before exporting a training request");
      const request = createTrainingRequest({ vendor, dataset, featureColumns: columnList(featureText),
        outputColumns: columnList(outputText), hiddenLayers: numberList(layers), activation, alpha, epochs, learningRate,
        validationFraction: 0.2, randomSeed: 42 });
      download("neural-training-request.json", JSON.stringify(request, null, 2), "application/json"); setError("");
    } catch (caught) { setError(String(caught)); }
  };

  const query = () => {
    if (!bundle) return;
    try {
      const values = Object.fromEntries(Object.entries(inputs).map(([name, value]) => [name, Number(value)]));
      setPrediction(inferPortableModel(bundle, values)); setWarnings(applicabilityWarnings(bundle, values)); setError("");
    } catch (caught) { setError(String(caught)); }
  };

  const loadBatch = async (file?: File) => {
    if (!file) return;
    try { setBatchRows(parseLaunchMonitorFile(file.name, await readFileText(file)) as Record<string, unknown>[]); setError(""); }
    catch (caught) { setError(String(caught)); }
  };

  const queryBatch = () => {
    if (!bundle) return;
    try {
      const results = batchRows.flatMap((row, rowIndex) => {
        const inferred = inferPortableModel(bundle, Object.fromEntries(bundle.features.map(
          ({ name }) => [name, numeric(row[name]) ?? Number.NaN])));
        return bundle.outputs.map(({ name, unit }) => ({ row: rowIndex + 1, output: name, unit,
          predicted: inferred[name], observed: numeric(row[name]) }));
      });
      setPredictionRows(results); setError("");
    } catch (caught) { setError(String(caught)); }
  };

  return <div className="space-y-5">
    <section className={card}>
      <h2 className="text-xl font-semibold text-sky-200">Neural Model Lab</h2>
      <p className="mt-2 text-sm text-slate-300">The browser prepares a training request for the private, reproducible CLI procedure and performs deterministic local inference from a validated portable bundle. It does not train models in-browser or transmit data.</p>
      <p className="mt-2 text-xs text-amber-300">Vendor-comparable models approximate observed outputs in the available evidence. They are not vendor emulation or certification, and sparse aggregate-only vendor evidence is not adequate for shot-level neural training.</p>
    </section>

    <section className={`${card} grid gap-4 lg:grid-cols-2`} aria-label="Training request configuration">
      <div className="space-y-3">
        <h3 className="font-semibold">Configure External Training</h3>
        <label className="block text-sm">Vendor target<select aria-label="Vendor target" title="Only vendors with approved row-level targets are enabled." value={vendor} onChange={(event) => setVendor(event.target.value)} className={`${control} mt-1 w-full`}>{vendors.map(({ name, enabled, reason }) => <option key={name} disabled={!enabled} title={reason}>{name}</option>)}</select></label>
        <p className="text-xs text-slate-400">{vendors.map(({ name, enabled, reason }) => `${name}: ${enabled ? "available" : "unavailable"} — ${reason}`).join(" ")}</p>
        <label className="block text-sm">Custom or current dataset CSV<input aria-label="Custom training CSV" title="Import the current app export or another local CSV; only metadata is placed in the request." type="file" accept=".csv" onChange={(event) => void loadDataset(event.target.files?.[0])} className="mt-1 block w-full text-sm" /></label>
        <p className="text-xs text-slate-400">{dataset ? `${dataset.fileName}: ${dataset.rowCount} rows, ${dataset.columns.length} columns` : "No training dataset selected."}</p>
        <label className="block text-sm">Feature columns<input aria-label="Feature columns" title="Comma-separated numeric input columns." value={featureText} onChange={(event) => setFeatureText(event.target.value)} className={`${control} mt-1 w-full`} /></label>
        <label className="block text-sm">Output columns<input aria-label="Output columns" title="Comma-separated numeric vendor outcome columns; do not repeat features." value={outputText} onChange={(event) => setOutputText(event.target.value)} className={`${control} mt-1 w-full`} /></label>
      </div>
      <div className="space-y-3">
        <h3 className="font-semibold">Network and Optimization</h3>
        <label className="block text-sm">Hidden layers<input aria-label="Hidden layer widths" title="Comma-separated neuron counts, one through eight layers." value={layers} onChange={(event) => setLayers(event.target.value)} className={`${control} mt-1 w-full`} /></label>
        <label className="block text-sm">Activation<select aria-label="Hidden activation" title="Nonlinear activation requested for hidden layers." value={activation} onChange={(event) => setActivation(event.target.value as typeof activation)} className={`${control} mt-1 w-full`}><option>relu</option><option>tanh</option><option>linear</option></select></label>
        <label className="block text-sm">L2 alpha<input aria-label="Regularization alpha" title="L2 weight regularization strength." type="number" min="0" max="1" step="0.0001" value={alpha} onChange={(event) => setAlpha(Number(event.target.value))} className={`${control} mt-1 w-full`} /></label>
        <label className="block text-sm">Epochs<input aria-label="Training epochs" title="Maximum optimization passes; the private CLI applies early stopping." type="number" min="1" value={epochs} onChange={(event) => setEpochs(Number(event.target.value))} className={`${control} mt-1 w-full`} /></label>
        <label className="block text-sm">Learning rate<input aria-label="Learning rate" title="Adam optimizer step size requested from the private CLI." type="number" min="0.000001" max="1" step="0.0001" value={learningRate} onChange={(event) => setLearningRate(Number(event.target.value))} className={`${control} mt-1 w-full`} /></label>
        <button type="button" aria-label="Export Training Request" title="Save a metadata-only JSON request for the private training procedure." onClick={exportRequest} className={`${control} w-full`}>Export Training Request</button>
      </div>
    </section>

    <section className={card} aria-label="Portable model query">
      <h3 className="font-semibold">Portable Model Query</h3>
      <label className="mt-3 block text-sm">Model artifact<input aria-label="Portable model bundle JSON" title="Import a validated portable dense-network JSON artifact; inference remains local." type="file" accept=".json" onChange={(event) => void loadBundle(event.target.files?.[0])} className="mt-1 block w-full text-sm" /></label>
      {bundle ? <div className="mt-4 space-y-4">
        <p><strong>{bundle.modelId}</strong> · {displayedVendor} · {bundle.provenance.sampleCount} training rows</p>
        <div className="grid gap-2 sm:grid-cols-3">{modelInputs.map((feature) => <label key={feature.name} className="text-sm">{feature.name}<input aria-label={`${feature.name} (${feature.unit})`} title={`Input ${feature.name} in ${feature.unit}; training mean ${feature.mean}.`} type="number" value={inputs[feature.name] ?? ""} onChange={(event) => setInputs((current) => ({ ...current, [feature.name]: event.target.value }))} className={`${control} mt-1 w-full`} /></label>)}</div>
        <button type="button" title="Run this artifact locally for one input row." onClick={query} className={control}>Query Model</button>
        <div aria-live="polite" className="flex gap-3">{bundle.outputs.map((output) => prediction[output.name] === undefined ? null : <strong key={output.name}>{prediction[output.name].toFixed(3)} {output.unit}</strong>)}</div>
        {warnings.map((warning) => <p key={warning} className="text-xs text-amber-300">{warning}</p>)}
        <MetricCards metrics={bundle.metrics} /><LearningCurveChart points={bundle.learningCurve} />
        <p className="text-xs text-slate-400">Dataset SHA-256: {bundle.provenance.datasetSha256}<br />Training rows: {bundle.provenance.sampleCount}<br />Provenance: {JSON.stringify(bundle.provenance.details)}</p>
      </div> : <p className="mt-3 text-sm text-slate-400">No portable model loaded for {vendor}. Availability depends on traceable row-level vendor targets; aggregate comparisons remain descriptive only.</p>}
    </section>

    <section className={card} aria-label="Batch model query">
      <h3 className="font-semibold">Batch Query and Residuals</h3>
      <input aria-label="Batch query CSV" title="CSV must contain every model feature; matching output columns enable residuals." type="file" accept=".csv" onChange={(event) => void loadBatch(event.target.files?.[0])} className="mt-3 block w-full text-sm" />
      <div className="mt-3 flex flex-wrap gap-2"><button type="button" title="Predict every loaded batch row locally." onClick={queryBatch} disabled={!bundle || !batchRows.length} className={control}>Query Batch</button>
        <button type="button" aria-label="Export Prediction Data" title="Export predicted, observed, and residual backing rows as CSV." disabled={!predictionRows.length} onClick={() => download("neural-predictions.csv", ["row,output,unit,predicted,observed,residual", ...predictionRows.map((row) => `${row.row},${row.output},${row.unit},${row.predicted},${row.observed ?? ""},${row.observed === undefined ? "" : row.predicted - row.observed}`)].join("\n"), "text/csv") } className={control}>Export Prediction Data</button></div>
      <div className="mt-4"><ResidualChart points={predictionRows} /></div>
    </section>
    <details className={card} title="Calculation definitions and responsible-use guidance"><summary className="cursor-pointer font-semibold">Calculations, Evidence, and Procedure</summary><p className="mt-2 text-sm text-slate-300">Each feature is standardized as z = (x − training mean) / training scale, passed through the stored dense layers, then each output is restored as y = standardized output × training scale + training mean. ReLU uses max(0, x); tanh and linear activations have their standard definitions. A residual is predicted minus observed in the named output unit.</p><p className="mt-2 text-sm text-slate-300">Compare models only on held-out rows that were excluded from fitting. Preserve the dataset digest, split policy, random seed, feature list, units, training curve, and test metrics. A model for one vendor requires row-level targets from that vendor; relabeling TrackMan rows as Foresight or FlightScope would be invalid.</p></details>
    {error && <p role="alert" className="rounded border border-red-700 bg-red-950/50 p-3 text-sm text-red-200">{error}</p>}
  </div>;
}
