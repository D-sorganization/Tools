/**
 * Variation tab for the web clone (epic #4120, V3) — mirror of the
 * desktop Variation tab over the shared plan schema: noise rows from
 * the registry, seeded bounded runs (≤ 500, worker-less; the WASM +
 * web-worker upgrade removes the cap), summary + one-at-a-time
 * sensitivity tables, the landing scatter with its 2σ ellipse on a
 * canvas, and CSV/JSON downloads plus plan save/load.
 */

import { useMemo, useState } from "react";

import {
  MAX_RUNS,
  keysForMode,
  planFromJson,
  planToJson,
  runVariation,
  variableDef,
  variableLabel,
  type Distribution,
  type NoiseSpecTs,
  type VariationDatasetTs,
  type VariationMode,
  type VariationPlanTs,
} from "../model/variation";
import { LandingCanvas } from "./VariationLanding";
import {
  datasetToCsv,
  datasetToJson,
  oneAtATimeSensitivity,
  spearmanMatrix,
  summaryStats,
  type SensitivityResultTs,
} from "../model/variationAnalysis";

const DISTRIBUTIONS: Distribution[] = ["normal", "uniform", "triangular"];

const MODE_LABELS: Record<VariationMode, string> = {
  delivery: "Delivery → Impact → Flight",
  launch: "Launch Conditions → Flight",
};

const defaultSpec = (mode: VariationMode): NoiseSpecTs => {
  const key = keysForMode(mode)[0];
  return {
    variableKey: key,
    distribution: "normal",
    scale: variableDef(key)?.typicalScale ?? 1.0,
    lower: null,
    upper: null,
  };
};

const download = (name: string, text: string, type: string) => {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  URL.revokeObjectURL(url);
};

const heat = (fraction: number): string => {
  const f = Math.min(Math.max(fraction, 0), 1);
  const mix = (a: number, b: number) => Math.round(a + f * (b - a));
  return `rgb(${mix(37, 235)}, ${mix(66, 106)}, ${mix(96, 60)})`;
};

const panelClass =
  "rounded-xl border border-slate-800/80 bg-slate-900/60 p-5 shadow-lg shadow-black/20 backdrop-blur";
const inputClass =
  "no-spinner w-full rounded border border-slate-700 bg-slate-800 px-2 py-1 text-slate-100 focus:border-blue-500 focus:outline-none";
const buttonClass =
  "rounded border border-slate-700 bg-slate-800 px-3 py-1.5 text-sm text-slate-200 transition-colors hover:border-slate-500 disabled:opacity-40";

export function VariationPanel(): JSX.Element {
  const [mode, setMode] = useState<VariationMode>("delivery");
  const [noise, setNoise] = useState<NoiseSpecTs[]>([defaultSpec("delivery")]);
  const [nRuns, setNRuns] = useState(200);
  const [seed, setSeed] = useState(0);
  const [status, setStatus] = useState("Ready.");
  const [dataset, setDataset] = useState<VariationDatasetTs | null>(null);
  const [sensitivity, setSensitivity] = useState<SensitivityResultTs | null>(null);

  const plan = useMemo(
    (): VariationPlanTs => ({
      mode,
      baseVariables: {},
      noise,
      nRuns,
      seed,
      flightModel: "waterloo_penner",
    }),
    [mode, noise, nRuns, seed],
  );

  const stats = useMemo(() => (dataset ? summaryStats(dataset) : []), [dataset]);
  const spearman = useMemo(
    () => (dataset ? spearmanMatrix(dataset) : null),
    [dataset],
  );

  const setSpec = (index: number, updates: Partial<NoiseSpecTs>) => {
    setNoise((rows) =>
      rows.map((row, i) => (i === index ? { ...row, ...updates } : row)),
    );
  };

  const changeMode = (next: VariationMode) => {
    setMode(next);
    setNoise([defaultSpec(next)]);
    setDataset(null);
    setSensitivity(null);
  };

  const run = () => {
    try {
      const result = runVariation(plan);
      setDataset(result);
      setSensitivity(oneAtATimeSensitivity(plan));
      const ok = result.success.filter(Boolean).length;
      const failed = plan.nRuns - ok;
      setStatus(
        `Done: ${ok}/${plan.nRuns} runs${failed ? ` (${failed} failed)` : ""}.`,
      );
    } catch (error) {
      setStatus(`Cannot run: ${(error as Error).message}`);
    }
  };

  const loadPlan = (file: File) => {
    void file.text().then((text) => {
      try {
        const loaded = planFromJson(text);
        setMode(loaded.mode);
        setNoise(loaded.noise);
        setNRuns(loaded.nRuns);
        setSeed(loaded.seed);
        setDataset(null);
        setSensitivity(null);
        setStatus(`Plan loaded (${loaded.noise.length} noise rows).`);
      } catch (error) {
        setStatus(`Cannot load plan: ${(error as Error).message}`);
      }
    });
  };

  return (
    <div className="grid gap-6 xl:grid-cols-[420px_1fr]">
      <section aria-label="Variation setup" className="space-y-4">
        <div className={panelClass}>
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">
            Study Setup
          </h2>
          <label
            className="mb-3 block text-sm"
            title="Which pipeline slice each run exercises. The pendulum swing mode is desktop-only until the WASM kernels land."
          >
            <span className="mb-1 block text-slate-300">Pipeline</span>
            <select
              value={mode}
              onChange={(e) => changeMode(e.target.value as VariationMode)}
              className={inputClass}
            >
              {(Object.keys(MODE_LABELS) as VariationMode[]).map((m) => (
                <option key={m} value={m}>
                  {MODE_LABELS[m]}
                </option>
              ))}
            </select>
          </label>
          <div className="grid grid-cols-2 gap-3">
            <label
              className="block text-sm"
              title={`Monte-Carlo runs per study (browser-capped at ${MAX_RUNS}; the sensitivity pass repeats this once per noise row). The WASM + web-worker upgrade removes the cap.`}
            >
              <span className="mb-1 block text-slate-300">Runs (≤ {MAX_RUNS})</span>
              <input
                type="number"
                min={2}
                max={MAX_RUNS}
                value={nRuns}
                onChange={(e) =>
                  setNRuns(
                    Math.max(2, Math.min(MAX_RUNS, Number(e.target.value) || 2)),
                  )
                }
                className={inputClass}
              />
            </label>
            <label
              className="block text-sm"
              title="Master RNG seed — the same plan and seed always reproduce the same dataset (per-variable seeded streams)."
            >
              <span className="mb-1 block text-slate-300">Seed</span>
              <input
                type="number"
                min={0}
                value={seed}
                onChange={(e) => setSeed(Math.max(0, Number(e.target.value) || 0))}
                className={inputClass}
              />
            </label>
          </div>
        </div>

        <div className={panelClass}>
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">
            Varied Variables (Noise)
          </h2>
          {noise.map((spec, index) => {
            const def = variableDef(spec.variableKey);
            return (
              <div
                key={`${spec.variableKey}-${index}`}
                className="mb-3 rounded-lg border border-slate-800 bg-slate-950/40 p-3"
              >
                <div className="mb-2 flex gap-2">
                  <select
                    value={spec.variableKey}
                    onChange={(e) => {
                      const key = e.target.value;
                      setSpec(index, {
                        variableKey: key,
                        scale: variableDef(key)?.typicalScale ?? spec.scale,
                      });
                    }}
                    title={`${spec.variableKey} — ${def?.guidance ?? ""}`}
                    className={`${inputClass} flex-1`}
                  >
                    {keysForMode(mode).map((key) => (
                      <option key={key} value={key}>
                        {variableLabel(key)}
                      </option>
                    ))}
                  </select>
                  <button
                    type="button"
                    onClick={() =>
                      setNoise((rows) =>
                        rows.length > 1 ? rows.filter((_r, i) => i !== index) : rows,
                      )
                    }
                    title="Remove this noise row."
                    className={buttonClass}
                  >
                    ✕
                  </button>
                </div>
                <div className="grid grid-cols-2 gap-2 text-sm sm:grid-cols-4">
                  <select
                    value={spec.distribution}
                    onChange={(e) =>
                      setSpec(index, { distribution: e.target.value as Distribution })
                    }
                    title="Sampling distribution about the base value: normal (scale = std), uniform or triangular (scale = half-width)."
                    className={inputClass}
                  >
                    {DISTRIBUTIONS.map((d) => (
                      <option key={d} value={d}>
                        {d}
                      </option>
                    ))}
                  </select>
                  <input
                    type="number"
                    step="any"
                    value={spec.scale}
                    onChange={(e) => setSpec(index, { scale: Number(e.target.value) })}
                    title={`Noise scale [${def?.unit ?? ""}]. ${def?.guidance ?? ""}`}
                    className={inputClass}
                  />
                  <input
                    type="number"
                    step="any"
                    placeholder="min (opt.)"
                    value={spec.lower ?? ""}
                    onChange={(e) =>
                      setSpec(index, {
                        lower: e.target.value === "" ? null : Number(e.target.value),
                      })
                    }
                    title="Optional truncation: samples are clipped to stay at or above this bound."
                    className={inputClass}
                  />
                  <input
                    type="number"
                    step="any"
                    placeholder="max (opt.)"
                    value={spec.upper ?? ""}
                    onChange={(e) =>
                      setSpec(index, {
                        upper: e.target.value === "" ? null : Number(e.target.value),
                      })
                    }
                    title="Optional truncation: samples are clipped to stay at or below this bound."
                    className={inputClass}
                  />
                </div>
              </div>
            );
          })}
          <button
            type="button"
            onClick={() => setNoise((rows) => [...rows, defaultSpec(mode)])}
            title="Add another noise row (one per variable)."
            className={buttonClass}
          >
            Add Variable
          </button>
        </div>

        <div className={panelClass}>
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={run}
              title="Sample every noise row, run the pipeline once per run, and populate the results (runs synchronously in the page)."
              className={`${buttonClass} border-sky-500/60 text-sky-300`}
            >
              Run Variation Study
            </button>
            <button
              type="button"
              disabled={!dataset}
              onClick={() =>
                dataset &&
                download("variation_dataset.csv", datasetToCsv(dataset), "text/csv")
              }
              title="Download the runs table (inputs, outputs, success flags) as CSV."
              className={buttonClass}
            >
              Dataset CSV
            </button>
            <button
              type="button"
              disabled={!dataset}
              onClick={() =>
                dataset &&
                download(
                  "variation_dataset.json",
                  datasetToJson(dataset),
                  "application/json",
                )
              }
              title="Download the full dataset including the plan as JSON (same schema as the desktop tool)."
              className={buttonClass}
            >
              Dataset JSON
            </button>
            <button
              type="button"
              onClick={() =>
                download("variation_plan.json", planToJson(plan), "application/json")
              }
              title="Save just the plan — the same schema the desktop Variation tab reads."
              className={buttonClass}
            >
              Save Plan
            </button>
            <label className={`${buttonClass} cursor-pointer`} title="Load a plan JSON.">
              Load Plan
              <input
                type="file"
                accept="application/json"
                className="hidden"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) loadPlan(file);
                  e.target.value = "";
                }}
              />
            </label>
          </div>
          <p aria-live="polite" className="mt-3 text-xs text-slate-400">
            {status}
          </p>
        </div>
      </section>

      <section aria-label="Variation results" className="space-y-6">
        {dataset && (
          <div className={panelClass}>
            <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">
              Summary — Dispersion per Output
            </h2>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs text-slate-300">
                <thead>
                  <tr className="text-slate-500">
                    {["Output", "Mean", "Std", "P5", "Median", "P95", "N"].map((h) => (
                      <th key={h} className="px-2 py-1 font-medium">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {stats.map((s) => (
                    <tr key={s.name} className="border-t border-slate-800/60">
                      <td className="px-2 py-1 text-slate-200">{s.name}</td>
                      <td className="px-2 py-1 tabular-nums">{s.mean.toFixed(2)}</td>
                      <td className="px-2 py-1 tabular-nums">{s.std.toFixed(3)}</td>
                      <td className="px-2 py-1 tabular-nums">{s.p5.toFixed(2)}</td>
                      <td className="px-2 py-1 tabular-nums">{s.p50.toFixed(2)}</td>
                      <td className="px-2 py-1 tabular-nums">{s.p95.toFixed(2)}</td>
                      <td className="px-2 py-1 tabular-nums">{s.n}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {sensitivity && (
          <div className={panelClass}>
            <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">
              One-at-a-Time Sensitivity — Which Input Drives Which Output
            </h2>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="text-slate-500">
                    <th className="px-2 py-1 font-medium">Input \ Output</th>
                    {sensitivity.outputNames.map((name) => (
                      <th key={name} className="px-2 py-1 font-medium">
                        {name}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {sensitivity.inputKeys.map((key, i) => (
                    <tr key={key} className="border-t border-slate-800/60">
                      <td className="px-2 py-1 text-slate-200">{variableLabel(key)}</td>
                      {sensitivity.outputNames.map((name, j) => (
                        <td
                          key={name}
                          className="px-2 py-1 tabular-nums text-white"
                          style={{ backgroundColor: heat(sensitivity.normalized[i][j]) }}
                          title={`${variableLabel(key)} → ${name}: std ${sensitivity.matrix[i][j].toPrecision(3)} (column-normalized ${sensitivity.normalized[i][j].toFixed(2)}); Spearman ρ ${spearman?.[i]?.[j]?.toFixed(2) ?? "—"}`}
                        >
                          {sensitivity.matrix[i][j].toPrecision(3)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="mt-2 text-xs text-slate-500">
              Cell = std induced in the output when only that input varies (paired
              draws, same seed). Hot cells dominate their column; hover for the
              Spearman rank-correlation cross-check from the full study.
            </p>
          </div>
        )}

        {dataset && (
          <div className={panelClass}>
            <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">
              Landing Dispersion (2σ Ellipse)
            </h2>
            <LandingCanvas dataset={dataset} />
          </div>
        )}

        {!dataset && (
          <div className={`${panelClass} text-sm text-slate-400`}>
            Configure noise rows on the left and run a study to see dispersion
            statistics, the sensitivity matrix, and the landing scatter. Same
            plan schema and seeded behaviour as the desktop Variation tab
            (statistically compatible dispersion; exact RNG parity is not
            required — see model/variation.ts).
          </div>
        )}
      </section>
    </div>
  );
}
