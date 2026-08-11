/** Loaded ground result presentation, including raw primary/comparison evidence. */

import { useMemo } from "react";

import type { FlightToGroundResult } from "../model/flightGroundTypes";
import { GroundPlaybackTimeline } from "../model/groundPlayback";
import type { GroundPlaybackComparison } from "../model/groundPlaybackComparison";
import {
  GroundPlayback3D,
  type GroundPlaybackPortableState,
} from "./GroundPlayback3D";
import { GroundPlaybackComparisonSummary } from "./GroundPlaybackComparisonSummary";
import { GroundPlaybackResultEvidence } from "./GroundPlaybackResultEvidence";

function metricRows(result: FlightToGroundResult): Array<[string, string]> {
  const summary = result.summary;
  if (summary === null) return [];
  const endpoint = result.status === "complete" ? "Total" : "Observed total";
  return [
    ["Carry", `${summary.carry_distance_m.toFixed(3)} m`],
    [endpoint, `${summary.total_distance_m.toFixed(3)} m`],
    ["Bounce air", `${summary.bounce_air_distance_m.toFixed(3)} m`],
    ["Skid", `${summary.skid_distance_m.toFixed(3)} m`],
    ["Roll", `${summary.roll_distance_m.toFixed(3)} m`],
    ["Surface path", `${summary.surface_path_distance_m.toFixed(3)} m`],
  ];
}

function Summary({ result }: { readonly result: FlightToGroundResult }) {
  return (
    <section
      className="rounded-lg border border-slate-800 bg-slate-950/40 p-3"
      aria-label="Ground result summary"
    >
      <h3 className="mb-2 font-semibold text-slate-100">Result summary</h3>
      <table className="w-full text-left text-sm">
        <thead>
          <tr>
            <th scope="col">Metric</th>
            <th scope="col">Value</th>
          </tr>
        </thead>
        <tbody>
          {metricRows(result).map(([label, value]) => (
            <tr key={label}>
              <th scope="row" className="py-1 font-medium">
                {label}
              </th>
              <td>{value}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <dl className="mt-3 grid grid-cols-1 gap-1 text-xs text-slate-300 sm:grid-cols-2">
        <div>
          <dt className="font-semibold">Schema</dt>
          <dd>{result.schema_version}</dd>
        </div>
        <div>
          <dt className="font-semibold">Status</dt>
          <dd>{result.status}</dd>
        </div>
        <div>
          <dt className="font-semibold">Unit system</dt>
          <dd>{result.unit_system}</dd>
        </div>
        <div>
          <dt className="font-semibold">Frame</dt>
          <dd>{result.frame}</dd>
        </div>
        <div>
          <dt className="font-semibold">Surface ID</dt>
          <dd>{result.surface_id}</dd>
        </div>
        <div>
          <dt className="font-semibold">Termination</dt>
          <dd>
            {result.termination.reason} · completed=
            {String(result.termination.completed)} ·{" "}
            {result.termination.time_s.toFixed(6)} s
          </dd>
        </div>
        <div>
          <dt className="font-semibold">Model</dt>
          <dd>
            {result.model_id} {result.model_version}
          </dd>
        </div>
        <div>
          <dt className="font-semibold">Request</dt>
          <dd>{result.request_id}</dd>
        </div>
      </dl>
    </section>
  );
}

function Evidence({ result }: { readonly result: FlightToGroundResult }) {
  return (
    <section
      className="rounded-lg border border-slate-800 bg-slate-950/40 p-3"
      aria-label="Ground warnings and provenance"
    >
      <h3 className="mb-2 font-semibold text-slate-100">
        Warnings, calibration & provenance
      </h3>
      {result.warnings.length === 0 ? (
        <p className="text-sm text-slate-400">No warnings reported.</p>
      ) : (
        <ul className="space-y-2 text-sm">
          {result.warnings.map((warning, index) => (
            <li
              key={`${warning.code}-${index}`}
              className="rounded border border-amber-400/20 p-2"
            >
              <strong>{warning.code}</strong> · {warning.severity}
              <br />
              {warning.message}
            </li>
          ))}
        </ul>
      )}
      <dl className="mt-3 grid gap-2 text-xs sm:grid-cols-2">
        <div>
          <dt className="font-semibold">Producer</dt>
          <dd>
            {result.provenance.producer} {result.provenance.producer_version}
          </dd>
        </div>
        <div>
          <dt className="font-semibold">Source revision</dt>
          <dd>{result.provenance.source_revision}</dd>
        </div>
        <div>
          <dt className="font-semibold">Input SHA-256</dt>
          <dd className="break-all">{result.provenance.input_sha256}</dd>
        </div>
        <div>
          <dt className="font-semibold">Calibration ID</dt>
          <dd>{result.calibration.calibration_id}</dd>
        </div>
        <div>
          <dt className="font-semibold">Calibration</dt>
          <dd>
            {result.calibration.kind} · {result.calibration.source}
          </dd>
        </div>
        <div>
          <dt className="font-semibold">Confidence</dt>
          <dd>{result.calibration.confidence.toFixed(2)}</dd>
        </div>
      </dl>
    </section>
  );
}

export function GroundPlaybackLoadedResult({
  result,
  comparison,
  showComparison,
  initialState,
  onStateChange,
}: {
  readonly result: FlightToGroundResult;
  readonly comparison: GroundPlaybackComparison | null;
  readonly showComparison: boolean;
  readonly initialState: GroundPlaybackPortableState;
  readonly onStateChange: (state: GroundPlaybackPortableState) => void;
}) {
  const timeline = useMemo(() => new GroundPlaybackTimeline(result), [result]);
  return (
    <>
      <div className="grid gap-4 lg:grid-cols-[minmax(15rem,22rem)_1fr]">
        <div className="space-y-4">
          <Summary result={result} />
          <Evidence result={result} />
        </div>
        <GroundPlayback3D
          timeline={timeline}
          comparisonTimeline={comparison?.comparison}
          showComparison={showComparison}
          initialState={initialState}
          onStateChange={onStateChange}
        />
      </div>
      {comparison && (
        <GroundPlaybackComparisonSummary comparison={comparison} />
      )}
      <GroundPlaybackResultEvidence result={result} subject="primary" />
      {comparison && (
        <GroundPlaybackResultEvidence
          result={comparison.comparison.result}
          subject="comparison"
        />
      )}
    </>
  );
}
