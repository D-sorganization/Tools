import {
  CHIP_COHORTS,
  type ChipStudySummaryTs,
  type ChipTrialCohortTs,
} from "../model/chipForgiveness";
import { PANEL_CLASS } from "./variationUi";
import { OUTPUT_LABELS, OUTPUT_UNITS } from "../model/variationPlotData";

const COHORT_LABELS: Record<ChipTrialCohortTs, string> = {
  ball_first: "Ball First",
  ball_only: "Ball Only",
  ground_first: "Ground First",
  simultaneous_or_grazing: "Simultaneous / Grazing",
  ground_only_miss: "Ground Only — Ball Missed",
  no_contact_miss: "No Contact — Ball Missed",
  numerical_failure: "Numerical Failure",
};

const percent = (value: number): string => `${(100 * value).toFixed(2)}%`;
const metricValue = (value: number | null): string =>
  value === null ? "Unavailable" : value.toPrecision(5);
const metricLabel = (value: string): string => OUTPUT_LABELS[value] ?? value;

interface Props {
  summary: ChipStudySummaryTs;
  limitations: string;
  onExportJson?: () => void;
  onExportCsv?: () => void;
}

/** Present qualified all-trial risk and availability-aware physical metrics. */
export function ChipForgivenessPanel({
  summary,
  limitations,
  onExportJson,
  onExportCsv,
}: Props): JSX.Element {
  const decisions = [
    [
      "Expected Loss (95% Bootstrap CI)",
      `${summary.expectedLoss.toFixed(3)} [${summary.expectedLossCi[0].toFixed(3)}, ${summary.expectedLossCi[1].toFixed(3)}]`,
    ],
    [
      "Worst-Tail CVaR",
      `${summary.cvarLoss.toFixed(3)} (worst ${(100 * summary.cvarTailFraction).toFixed(0)}%)`,
    ],
    ["Clean-Contact Probability", percent(summary.cleanContactProbability)],
    ["Constraint-Violation Rate", percent(summary.constraintViolationRate)],
  ];
  return (
    <section aria-label="Chip forgiveness decision analysis" className={PANEL_CLASS}>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-emerald-400">
            All-Trial Decision Analysis
          </p>
          <h2 className="mt-1 text-lg font-semibold text-slate-100">
            Conditional Chip-Shot Forgiveness
          </h2>
        </div>
        <span className={`rounded-full border px-3 py-1 text-xs font-semibold ${
          summary.supportsTurfRankings
            ? "border-emerald-400/50 bg-emerald-500/10 text-emerald-200"
            : "border-amber-400/50 bg-amber-500/10 text-amber-200"
        }`}>
          {summary.supportsTurfRankings ? "Calibrated Turf" : "Turf Ranking Restricted"}
        </span>
      </div>
      <p className="mt-3 text-sm leading-6 text-slate-300">{summary.rankingScope}</p>
      <div className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        {decisions.map(([label, value]) => (
          <article key={label} className="rounded-lg border border-slate-700/80 bg-slate-950/50 p-3">
            <h3 className="text-xs text-slate-400">{label}</h3>
            <p className="mt-1 font-mono text-sm font-semibold text-slate-100">{value}</p>
          </article>
        ))}
      </div>
      <div className="mt-5 overflow-x-auto">
        <table className="w-full text-left text-xs text-slate-300">
          <caption className="mb-2 text-left font-semibold uppercase tracking-wide text-slate-400">
            Contact Cohorts — All {summary.sampleCount} Trials
          </caption>
          <thead><tr className="text-slate-500">
            {['Cohort', 'Count', 'Probability', '95% Wilson CI'].map((heading) =>
              <th key={heading} className="px-2 py-1 font-medium">{heading}</th>)}
          </tr></thead>
          <tbody>{CHIP_COHORTS.map((cohort) => {
            const estimate = summary.cohorts[cohort];
            return <tr key={cohort} className="border-t border-slate-800/70">
              <td className="px-2 py-1 text-slate-200">{COHORT_LABELS[cohort]}</td>
              <td className="px-2 py-1 tabular-nums">{estimate.count}</td>
              <td className="px-2 py-1 tabular-nums">{percent(estimate.probability)}</td>
              <td className="px-2 py-1 tabular-nums">
                {percent(estimate.ciLow)} — {percent(estimate.ciHigh)}
              </td>
            </tr>;
          })}</tbody>
        </table>
      </div>
      <details className="group mt-5 rounded-lg border border-slate-700/80 bg-slate-950/35 p-3">
        <summary className="cursor-pointer font-semibold text-sky-300 hover:text-sky-200">
          Advanced Metric Quantiles — Click for Support and Censoring
        </summary>
        <div className="mt-3 overflow-x-auto">
          <table className="w-full text-left text-xs text-slate-300">
            <thead><tr className="text-slate-500">
              {['Metric', 'Unit', 'P5', 'Median', 'P95', 'Support / Unavailable'].map((heading) =>
                <th key={heading} className="px-2 py-1 font-medium">{heading}</th>)}
            </tr></thead>
            <tbody>{summary.metricDistributions.map((distribution) =>
              <tr key={distribution.name} className="border-t border-slate-800/70">
                <td className="px-2 py-1 text-slate-200">{metricLabel(distribution.name)}</td>
                <td className="px-2 py-1 text-slate-400">{OUTPUT_UNITS[distribution.name] ?? "—"}</td>
                <td className="px-2 py-1 font-mono">{metricValue(distribution.p05)}</td>
                <td className="px-2 py-1 font-mono">{metricValue(distribution.p50)}</td>
                <td className="px-2 py-1 font-mono">{metricValue(distribution.p95)}</td>
                <td className="px-2 py-1 tabular-nums">
                  {distribution.supportCount} / {distribution.unavailableCount}
                </td>
              </tr>)}</tbody>
          </table>
        </div>
      </details>
      <details className="group mt-3 rounded-lg border border-slate-700/80 bg-slate-950/35 p-3">
        <summary className="cursor-pointer font-semibold text-sky-300 hover:text-sky-200">
          Expected-Loss Convergence — Click for Prefix Evidence
        </summary>
        <table className="mt-3 w-full text-left text-xs text-slate-300">
          <thead><tr className="text-slate-500">
            {['Trials', 'Running Mean Loss', 'Standard Error'].map((heading) =>
              <th key={heading} className="px-2 py-1 font-medium">{heading}</th>)}
          </tr></thead>
          <tbody>{summary.convergence.map((point) =>
            <tr key={point.sampleCount} className="border-t border-slate-800/70">
              <td className="px-2 py-1">{point.sampleCount}</td>
              <td className="px-2 py-1 font-mono">{point.meanLoss.toPrecision(5)}</td>
              <td className="px-2 py-1 font-mono">
                {point.standardError === null ? "Unavailable" : point.standardError.toPrecision(5)}
              </td>
            </tr>)}</tbody>
        </table>
      </details>
      <p className="mt-3 text-xs leading-5 text-slate-500">
        <b className="text-slate-400">Model Boundary:</b> {limitations}
      </p>
      {(onExportJson || onExportCsv) && (
        <div className="mt-4 flex flex-wrap gap-2">
          {onExportJson && <button
            type="button"
            onClick={onExportJson}
            className="rounded border border-sky-500/40 bg-sky-500/10 px-3 py-1.5 text-xs font-semibold text-sky-200 hover:bg-sky-500/20"
          >
            Forgiveness Study JSON
          </button>}
          {onExportCsv && <button
            type="button"
            onClick={onExportCsv}
            className="rounded border border-slate-600 bg-slate-800 px-3 py-1.5 text-xs font-semibold text-slate-200 hover:border-slate-500"
          >
            Forgiveness Trials CSV
          </button>}
        </div>
      )}
    </section>
  );
}
