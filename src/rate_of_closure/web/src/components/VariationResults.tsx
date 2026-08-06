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
import { PANEL_CLASS, sensitivityHeat } from "./variationUi";

interface VariationResultsProps {
  dataset: VariationDatasetTs | null;
  sensitivity: SensitivityResultTs | null;
  target?: TargetRegionTs;
  distanceUnit: string;
  ensemble?: SwingVariationResultTs | null;
}

export function VariationResults({
  dataset,
  sensitivity,
  target,
  distanceUnit,
  ensemble = null,
}: VariationResultsProps): JSX.Element {
  const [selectedTrialIndex, setSelectedTrialIndex] = useState<number | null>(null);
  const stats = useMemo(() => dataset ? summaryStats(dataset) : [], [dataset]);
  const spearman = useMemo(() => dataset ? spearmanMatrix(dataset) : null, [dataset]);

  return (
    <section aria-label="Variation results" className="space-y-6">
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
        <div className={`${PANEL_CLASS} text-sm text-slate-400`}>
          Configure noise rows, select the analyses to execute, and run the study.
          Plan files and the named library retain the complete v2 physical plan.
        </div>
      )}
    </section>
  );
}
