import { useMemo, useState } from "react";

import type { TargetRegionTs } from "../model/targets";
import { DISTANCE_UNITS } from "../model/units";
import { variableLabel, type VariationDatasetTs } from "../model/variation";
import {
  spearmanMatrix,
  summaryStats,
  type SensitivityResultTs,
} from "../model/variationAnalysis";
import { LandingCanvas } from "./VariationLanding";
import { VariationScatter } from "./VariationScatter";
import { VariationArcOverlay } from "./VariationArcOverlay";
import { VariationDistributionMatrix } from "./VariationDistributionMatrix";
import type { SwingVariationResultTs } from "../model/variationSwingEnsemble";
import {
  chipForgivenessStudyToCsv,
  chipForgivenessStudyToJson,
  chipForgivenessVariationDataset,
  type ChipForgivenessStudyTs,
} from "../model/chipForgivenessEnsemble";
import { ChipForgivenessPanel } from "./ChipForgivenessPanel";
import { downloadText, PANEL_CLASS, sensitivityHeat } from "./variationUi";

interface VariationResultsProps {
  dataset: VariationDatasetTs | null;
  sensitivity: SensitivityResultTs | null;
  target?: TargetRegionTs;
  distanceUnit: string;
  ensemble?: SwingVariationResultTs | null;
  forgiveness?: ChipForgivenessStudyTs | null;
}

export function VariationResults({
  dataset,
  sensitivity,
  target,
  distanceUnit,
  ensemble = null,
  forgiveness = null,
}: VariationResultsProps): JSX.Element {
  const [selectedTrialIndex, setSelectedTrialIndex] = useState<number | null>(null);
  const stats = useMemo(() => dataset ? summaryStats(dataset) : [], [dataset]);
  const spearman = useMemo(() => dataset ? spearmanMatrix(dataset) : null, [dataset]);
  const forgivenessDataset = useMemo(
    () => forgiveness ? chipForgivenessVariationDataset(forgiveness) : null,
    [forgiveness],
  );

  return (
    <section aria-label="Variation results" className="min-w-0 space-y-6">
      {forgiveness && (
        <ChipForgivenessPanel
          summary={forgiveness.summary}
          limitations={forgiveness.metadata.limitations}
          onExportJson={() => downloadText(
            "chip-forgiveness-study.json",
            chipForgivenessStudyToJson(forgiveness),
            "application/json;charset=utf-8",
          )}
          onExportCsv={() => downloadText(
            "chip-forgiveness-trials.csv",
            chipForgivenessStudyToCsv(forgiveness),
            "text/csv;charset=utf-8",
          )}
        />
      )}
      {forgivenessDataset && (
        <div className={PANEL_CLASS}>
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">
            Forgiveness Metric Scatter and Marginal Distributions
          </h2>
          <VariationDistributionMatrix
            dataset={forgivenessDataset}
            ensemble={ensemble}
            selectedTrialIndex={selectedTrialIndex}
            onSelectedTrialChange={setSelectedTrialIndex}
          />
          <div className="mt-6 border-t border-slate-800 pt-5">
            <VariationScatter
              dataset={forgivenessDataset}
              ensemble={ensemble}
              selectedTrialIndex={selectedTrialIndex}
              onSelectedTrialChange={setSelectedTrialIndex}
            />
          </div>
        </div>
      )}
      {dataset && (
        <div className={PANEL_CLASS}>
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">Scatter Matrix and Marginal Distributions</h2>
          <VariationDistributionMatrix
            dataset={dataset}
            ensemble={ensemble}
            selectedTrialIndex={selectedTrialIndex}
            onSelectedTrialChange={setSelectedTrialIndex}
          />
        </div>
      )}

      {dataset && (
        <div className={PANEL_CLASS}>
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">
            Impact and Shot-Outcome Scatter
          </h2>
          <VariationScatter
            dataset={dataset}
            ensemble={ensemble}
            selectedTrialIndex={selectedTrialIndex}
            onSelectedTrialChange={setSelectedTrialIndex}
          />
        </div>
      )}

      {ensemble && (
        <div className={PANEL_CLASS}>
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">
            All Swing Arcs — Interactive 3D Overlay
          </h2>
          <VariationArcOverlay
            ensemble={ensemble}
            selectedTrialIndex={selectedTrialIndex}
            onSelectedTrialChange={setSelectedTrialIndex}
          />
        </div>
      )}

      {dataset && (
        <div className={PANEL_CLASS}>
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">
            Summary — Dispersion per Output
          </h2>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs text-slate-300">
              <thead>
                <tr className="text-slate-500">
                  {["Output", "Mean", "Std", "P5", "Median", "P95", "N"].map((heading) => (
                    <th key={heading} className="px-2 py-1 font-medium">{heading}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {stats.map((statistic) => {
                  const isDistance = statistic.name === "carry_m" || statistic.name === "lateral_m";
                  const factor = isDistance ? DISTANCE_UNITS[distanceUnit] : 1;
                  const name = isDistance ? `${statistic.name} [${distanceUnit}]` : statistic.name;
                  return (
                    <tr key={statistic.name} className="border-t border-slate-800/60">
                      <td className="px-2 py-1 text-slate-200">{name}</td>
                      <td className="px-2 py-1 tabular-nums">{(statistic.mean / factor).toFixed(2)}</td>
                      <td className="px-2 py-1 tabular-nums">{(statistic.std / factor).toFixed(3)}</td>
                      <td className="px-2 py-1 tabular-nums">{(statistic.p5 / factor).toFixed(2)}</td>
                      <td className="px-2 py-1 tabular-nums">{(statistic.p50 / factor).toFixed(2)}</td>
                      <td className="px-2 py-1 tabular-nums">{(statistic.p95 / factor).toFixed(2)}</td>
                      <td className="px-2 py-1 tabular-nums">{statistic.n}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {sensitivity && (
        <div className={PANEL_CLASS}>
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">
            One-at-a-Time Sensitivity — Which Input Drives Which Output
          </h2>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="text-slate-500">
                  <th className="px-2 py-1 font-medium">Input \ Output</th>
                  {sensitivity.outputNames.map((name) => (
                    <th key={name} className="px-2 py-1 font-medium">{name}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {sensitivity.inputKeys.map((key, inputIndex) => (
                  <tr key={key} className="border-t border-slate-800/60">
                    <td className="px-2 py-1 text-slate-200">{variableLabel(key)}</td>
                    {sensitivity.outputNames.map((name, outputIndex) => (
                      <td
                        key={name}
                        className="px-2 py-1 tabular-nums text-white"
                        style={{
                          backgroundColor: sensitivityHeat(
                            sensitivity.normalized[inputIndex][outputIndex],
                          ),
                        }}
                        title={`${variableLabel(key)} → ${name}: std ${sensitivity.matrix[inputIndex][outputIndex].toPrecision(3)}; Spearman ρ ${spearman?.[inputIndex]?.[outputIndex]?.toFixed(2) ?? "not requested"}`}
                      >
                        {sensitivity.matrix[inputIndex][outputIndex].toPrecision(3)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-2 text-xs text-slate-500">
            Each row runs independently with paired seeded draws. Spearman correlation is
            displayed only when the jointly enabled analysis was also requested.
          </p>
        </div>
      )}

      {dataset && (
        <div className={PANEL_CLASS}>
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">
            Landing Dispersion (2σ Ellipse)
          </h2>
          <LandingCanvas dataset={dataset} target={target} ensemble={ensemble} />
        </div>
      )}

      {!dataset && !sensitivity && (
        <div className={`${PANEL_CLASS} flex h-full min-h-[720px] flex-col`}>
          <div className="max-w-3xl">
            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-sky-400">
              Analysis Workspace
            </p>
            <h2 className="mt-3 text-2xl font-semibold text-slate-100">
              Ready to Analyze Variation
            </h2>
            <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-400">
              Configure the noise model, choose the analyses, and run the study. The resulting
              views share trial selection so an observation can be followed from its input draw
              through swing geometry, impact, and landing.
            </p>
          </div>

          <div className="mt-8 grid gap-4 md:grid-cols-2">
            {[
              ["Distribution Matrix", "Inputs, outputs, marginal distributions, and linked trial selection."],
              ["Swing Geometry", "Rotatable 3D traces, principal spread, RMS dispersion, and quiet zones."],
              ["Impact and Flight", "Typed hits, no-impact runs, numerical failures, and landing dispersion."],
              ["Sensitivity", "Paired one-at-a-time effects and rank correlations with explicit units."],
            ].map(([title, description], index) => (
              <article
                key={title}
                className="rounded-xl border border-slate-800 bg-slate-950/45 p-4"
              >
                <div className="flex items-start gap-3">
                  <span
                    aria-hidden="true"
                    className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-sky-500/40 bg-sky-500/10 text-xs font-semibold text-sky-300"
                  >
                    {index + 1}
                  </span>
                  <div>
                    <h3 className="text-sm font-semibold text-slate-200">{title}</h3>
                    <p className="mt-1 text-xs leading-5 text-slate-500">{description}</p>
                  </div>
                </div>
              </article>
            ))}
          </div>

          <div className="mt-auto border-t border-slate-800/80 pt-5 text-xs leading-5 text-slate-500">
            Plan files and the named library retain the complete v2 physical plan. Every trial
            keeps its typed outcome; misses and numerical failures are never converted into
            fabricated landing coordinates.
          </div>
        </div>
      )}
    </section>
  );
}
