import { useMemo, useState, type RefObject } from "react";

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
import { PANEL_CLASS, sensitivityHeat } from "./variationUi";
import { VariationLocalizedSources } from "./VariationLocalizedSources";
import { VisualStateFrame } from "./VisualStateFrame";
import type { VariationVisualState } from "../model/variationVisualState";
import { LocalizedAttributionPanel } from "./LocalizedAttributionPanel";
import type { AttributionAuthorityTs } from "../model/localizedAttribution";

interface VariationResultsProps {
  dataset: VariationDatasetTs | null;
  sensitivity: SensitivityResultTs | null;
  target?: TargetRegionTs;
  distanceUnit: string;
  ensemble?: SwingVariationResultTs | null;
  visualState?: VariationVisualState;
  visualAnnouncement?: string;
  prominenceRef?: RefObject<HTMLElement>;
  onReturnToControls?: (focusRun: boolean) => void;
  localizedAttributionAuthority?: AttributionAuthorityTs | null;
}

interface TrialSelection {
  dataset: VariationDatasetTs | null;
  ensemble: SwingVariationResultTs | null;
  trialIndex: number;
}

export function VariationResults({
  dataset,
  sensitivity,
  target,
  distanceUnit,
  ensemble = null,
  visualState = { phase: "empty", visualOrigin: "empty-preview", announcementRole: "status" },
  visualAnnouncement = "Ready.",
  prominenceRef,
  onReturnToControls,
  localizedAttributionAuthority = null,
}: VariationResultsProps): JSX.Element {
  const [selection, setSelection] = useState<TrialSelection | null>(null);
  const trialCount = dataset?.plan.nRuns ?? ensemble?.dataset.plan.nRuns ?? 0;
  const validSelectedTrialIndex = selection !== null
    && selection.dataset === dataset
    && selection.ensemble === ensemble
    && Number.isInteger(selection.trialIndex)
    && selection.trialIndex >= 0
    && selection.trialIndex < trialCount
    ? selection.trialIndex
    : null;
  const selectTrial = (trialIndex: number | null): void => {
    if (
      trialIndex === null
      || !Number.isInteger(trialIndex)
      || trialIndex < 0
      || trialIndex >= trialCount
    ) {
      setSelection(null);
      return;
    }
    setSelection({ dataset, ensemble, trialIndex });
  };
  const stats = useMemo(() => dataset ? summaryStats(dataset) : [], [dataset]);
  const spearman = useMemo(() => dataset ? spearmanMatrix(dataset) : null, [dataset]);
  const returnControl = visualState.visualOrigin !== "empty-preview" && onReturnToControls
    ? <button type="button" onClick={(event) => onReturnToControls(event.detail === 0)}
      className="mb-3 rounded-lg border border-sky-500/50 bg-slate-950 px-3 py-2 text-xs text-sky-200">
      Return to variation controls
    </button>
    : null;
  const localizedRunAvailable = ensemble?.runs.some(
    (trial) => trial.localizedTorqueCommands.length > 0,
  ) ?? false;

  return <VisualStateFrame state={visualState} announcement={visualAnnouncement}>
    <section aria-label="Variation results" className="min-w-0 space-y-6">
      {ensemble && <VariationLocalizedSources ensemble={ensemble} />}
      <LocalizedAttributionPanel
        authority={localizedAttributionAuthority}
        localizedRunAvailable={localizedRunAvailable}
      />
      {dataset && (
        <div className={PANEL_CLASS}>
          {returnControl}
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">Scatter Matrix and Marginal Distributions</h2>
          <VariationDistributionMatrix
            dataset={dataset}
            ensemble={ensemble}
            selectedTrialIndex={validSelectedTrialIndex}
            onSelectedTrialChange={selectTrial}
            primaryVisualRef={prominenceRef as RefObject<HTMLDivElement>}
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
            selectedTrialIndex={validSelectedTrialIndex}
            onSelectedTrialChange={selectTrial}
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
            selectedTrialIndex={validSelectedTrialIndex}
            onSelectedTrialChange={selectTrial}
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
          {!dataset && returnControl}
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">
            One-at-a-Time Sensitivity — Which Input Drives Which Output
          </h2>
          <div ref={!dataset ? prominenceRef as RefObject<HTMLDivElement> : undefined}
            role="group" aria-label="One-at-a-time sensitivity matrix"
            className="min-h-[180px] overflow-x-auto xl:min-h-[240px]">
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
          <svg viewBox="0 0 720 190" role="img"
            aria-label="Variation analysis workflow preview"
            className="h-60 w-full rounded-xl border border-slate-800 bg-slate-950/45 p-3">
            <defs><marker id="variation-preview-arrow" markerWidth="8" markerHeight="8"
              refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#38bdf8" /></marker></defs>
            {["Noise model", "Typed trials", "Linked geometry", "Impact & landing"].map((label, index) => (
              <g key={label} transform={`translate(${20 + index * 178} 55)`}>
                <rect width="150" height="74" rx="12" fill="#0f172a" stroke="#334155" />
                <text x="75" y="43" textAnchor="middle" fill="#cbd5e1" fontSize="14">{label}</text>
                {index < 3 && <line x1="150" y1="37" x2="174" y2="37" stroke="#38bdf8"
                  strokeWidth="2" markerEnd="url(#variation-preview-arrow)" />}
              </g>
            ))}
          </svg>
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
  </VisualStateFrame>;
}
