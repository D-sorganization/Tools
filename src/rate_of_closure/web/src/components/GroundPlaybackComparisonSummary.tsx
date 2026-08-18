/** Accessible direct-delta and provenance tables for ground playback comparison. */

import type { GroundPlaybackComparison } from "../model/groundPlaybackComparison";

const numeric = (value: number): string => Number.isInteger(value)
  ? value.toString() : value.toFixed(6);

export function GroundPlaybackComparisonSummary({ comparison }: {
  readonly comparison: GroundPlaybackComparison;
}) {
  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(28rem,2fr)_minmax(20rem,1fr)]">
      <section className="overflow-x-auto rounded-lg border border-cyan-500/30 p-3">
        <h3 className="mb-1 font-semibold">Direct result comparison</h3>
        <p className="mb-2 text-xs text-slate-400">
          Delta is comparison minus primary. Values are reported result fields; no causal claim is made.
        </p>
        <table className="min-w-full text-left text-xs" aria-label="Ground result comparison table">
          <thead><tr>{["Metric", "Primary", "Comparison", "Comparison − primary"].map(
            (label) => <th scope="col" key={label} className="pr-3">{label}</th>,
          )}</tr></thead>
          <tbody>{comparison.metricRows.map((row) => (
            <tr key={row.metricId}>
              <th scope="row" className="pr-3">{row.label} [{row.unit}]</th>
              <td>{numeric(row.primary)}</td><td>{numeric(row.comparison)}</td>
              <td>{row.delta >= 0 ? "+" : ""}{numeric(row.delta)}</td>
            </tr>
          ))}</tbody>
        </table>
      </section>
      <section className="overflow-x-auto rounded-lg border border-cyan-500/30 p-3"
        aria-label="Ground comparison provenance">
        <h3 className="mb-2 font-semibold">Identity, status & provenance</h3>
        <table className="min-w-full text-left text-xs">
          <thead><tr><th scope="col">Field</th><th scope="col">Primary</th>
            <th scope="col">Comparison</th></tr></thead>
          <tbody>{comparison.provenanceRows.map((row) => (
            <tr key={row.field}><th scope="row" className="pr-3">{row.field}</th>
              <td className="break-all pr-3">{row.primary}</td>
              <td className="break-all">{row.comparison}</td></tr>
          ))}</tbody>
        </table>
      </section>
    </div>
  );
}
