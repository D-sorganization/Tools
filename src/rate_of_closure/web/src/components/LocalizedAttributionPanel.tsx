import { useMemo, useState } from "react";

import {
  ATTRIBUTION_CAVEAT,
  ATTRIBUTION_VIEW_SCHEMA_ID,
  attributionObservationsToCsv,
  attributionViewToJson,
  buildAttributionView,
  type AttributionAuthorityTs,
  type AttributionPairTs,
  type AttributionViewDefinitionTs,
} from "../model/localizedAttribution";
import { BUTTON_CLASS, PANEL_CLASS, downloadText } from "./variationUi";

interface Props {
  authority: AttributionAuthorityTs | null;
  localizedRunAvailable: boolean;
}

interface Selection {
  sourceSpecId: string;
  targetId: string;
  baselineTrialIndex: number;
  perturbedTrialIndex: number;
}

const pairLabel = (row: AttributionPairTs): string =>
  `Trial ${row.baselineTrialIndex} → ${row.perturbedTrialIndex}`;
const valueLabel = (value: number | null, unit: string): string =>
  value === null ? "Unavailable" : `${value.toPrecision(7)} ${unit}`;

const firstPair = (
  authority: AttributionAuthorityTs,
  sourceSpecId: string,
): AttributionPairTs | null => authority.pairs.find((row) =>
  row.sourceSpecId === sourceSpecId) ?? null;

const initialSelection = (authority: AttributionAuthorityTs): Selection => {
  const first = authority.observations[0];
  if (!first) throw new Error("attribution authority requires an observation");
  return {
    sourceSpecId: first.sourceSpecId,
    targetId: first.targetId,
    baselineTrialIndex: first.baselineTrialIndex,
    perturbedTrialIndex: first.perturbedTrialIndex,
  };
};

const definition = (
  authority: AttributionAuthorityTs,
  selection: Selection,
): AttributionViewDefinitionTs => ({
  schemaId: ATTRIBUTION_VIEW_SCHEMA_ID,
  schemaVersion: 1,
  authorityId: authority.authorityId,
  ...selection,
});

export function LocalizedAttributionPanel({
  authority,
  localizedRunAvailable,
}: Props): JSX.Element | null {
  const [stored, setStored] = useState<{
    authority: AttributionAuthorityTs;
    selection: Selection;
  } | null>(null);
  const selection = authority === null
    ? null
    : stored?.authority === authority ? stored.selection : initialSelection(authority);
  const view = useMemo(() => authority && selection
    ? buildAttributionView(authority, definition(authority, selection))
    : null, [authority, selection]);
  if (!localizedRunAvailable && authority === null) return null;

  const chooseSource = (sourceSpecId: string): void => {
    if (!authority || !selection) return;
    const pair = firstPair(authority, sourceSpecId);
    const target = authority.targets[0];
    if (!pair || !target) return;
    setStored({ authority, selection: {
      sourceSpecId, targetId: target.targetId,
      baselineTrialIndex: pair.baselineTrialIndex,
      perturbedTrialIndex: pair.perturbedTrialIndex,
    } });
  };
  const chooseTarget = (targetId: string): void => {
    if (!authority || !selection) return;
    const pair = firstPair(authority, selection.sourceSpecId);
    if (!pair) return;
    setStored({ authority, selection: {
      ...selection, targetId,
      baselineTrialIndex: pair.baselineTrialIndex,
      perturbedTrialIndex: pair.perturbedTrialIndex,
    } });
  };
  const choosePair = (pairKey: string): void => {
    if (!authority || !selection) return;
    const [baselineTrialIndex, perturbedTrialIndex] = pairKey.split(":").map(Number);
    setStored({ authority, selection: {
      ...selection, baselineTrialIndex, perturbedTrialIndex,
    } });
  };

  return (
    <section aria-label="Localized torque attribution" className={PANEL_CLASS}>
      <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-400">
        Localized Source → Target Response
      </h2>
      <p className="mt-2 text-xs leading-5 text-amber-300">{ATTRIBUTION_CAVEAT}</p>
      {!authority || !view || !selection ? (
        <div role="status" className="mt-3 rounded border border-amber-700/50 bg-amber-950/30 p-3 text-xs text-amber-200">
          Attribution unavailable: this Monte Carlo result retains perturbed traces and
          scalar outcomes, but not isolated baseline/perturbed pairs. Scatter and rank
          correlation are not substituted for planted-intervention authority.
        </div>
      ) : (
        <>
          <div className="mt-4 grid gap-3 md:grid-cols-3">
            <label className="text-xs text-slate-400">
              Source specification
              <select
                aria-label="Localized attribution source specification"
                title="Stable spec ID, topological joint, and required half-open torque window."
                value={selection.sourceSpecId}
                onChange={(event) => chooseSource(event.target.value)}
                className="mt-1 w-full rounded border border-slate-700 bg-slate-950 p-2 text-slate-200"
              >
                {authority.sources.map((source) => (
                  <option key={source.specId} value={source.specId}>{source.specId}</option>
                ))}
              </select>
            </label>
            <label className="text-xs text-slate-400">
              Target state / impact / shot
              <select
                aria-label="Localized attribution target"
                title="State targets retain spatial swing.* point and time; impact/shot targets use typed outcome availability."
                value={selection.targetId}
                onChange={(event) => chooseTarget(event.target.value)}
                className="mt-1 w-full rounded border border-slate-700 bg-slate-950 p-2 text-slate-200"
              >
                {authority.targets.map((target) => (
                  <option key={target.targetId} value={target.targetId}>
                    {target.kind}: {target.name}
                  </option>
                ))}
              </select>
            </label>
            <label className="text-xs text-slate-400">
              Retained pair
              <select
                aria-label="Localized attribution retained pair"
                title="Baseline and perturbed trial IDs from an explicitly retained pair."
                value={`${selection.baselineTrialIndex}:${selection.perturbedTrialIndex}`}
                onChange={(event) => choosePair(event.target.value)}
                className="mt-1 w-full rounded border border-slate-700 bg-slate-950 p-2 text-slate-200"
              >
                {authority.pairs.filter((row) =>
                  row.sourceSpecId === selection.sourceSpecId).map((row) => (
                  <option
                    key={`${row.baselineTrialIndex}:${row.perturbedTrialIndex}`}
                    value={`${row.baselineTrialIndex}:${row.perturbedTrialIndex}`}
                  >
                    {pairLabel(row)}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <dl className="mt-4 grid gap-2 text-xs md:grid-cols-2">
            <div><dt className="text-slate-500">Source locus</dt><dd className="text-slate-200">
              {view.source.jointId} · [{view.source.timeWindowS[0]}, {view.source.timeWindowS[1]}) s · {view.source.unit}
            </dd></div>
            <div><dt className="text-slate-500">Target locus</dt><dd className="text-slate-200">
              {view.target.kind === "state"
                ? `${view.target.pointId} at ${view.target.timeS} s · ${view.target.coordinateFrame}`
                : `${view.target.kind} outcome · ${view.target.name}`} · {view.target.convention}
              {` · opaque stable ID ${view.target.targetId}`}
            </dd></div>
            <div><dt className="text-slate-500">Baseline</dt><dd className="text-slate-200">
              {valueLabel(view.selected.baselineTargetValue, view.target.unit)} · {view.selected.baselineStatus}
            </dd></div>
            <div><dt className="text-slate-500">Perturbed</dt><dd className="text-slate-200">
              {valueLabel(view.selected.perturbedTargetValue, view.target.unit)} · {view.selected.perturbedStatus}
            </dd></div>
            <div><dt className="text-slate-500">Response (perturbed − baseline)</dt><dd className="font-semibold text-sky-300">
              {valueLabel(view.selected.response, view.target.unit)}
            </dd></div>
            <div><dt className="text-slate-500">Availability</dt><dd className="text-slate-200">
              {view.selected.availability}
            </dd></div>
          </dl>
          <p className="mt-3 text-xs text-slate-500">
            Denominator: {view.denominator.availablePairs}/{view.denominator.totalPairs} available ·
            {` ${view.denominator.typedNoImpactPairs} typed no-impact · ${view.denominator.unavailableNoImpactPairs} no-impact unavailable · ${view.denominator.failedPairs} failed · ${view.denominator.nonfinitePairs} nonfinite unavailable.`}
          </p>
          <div className="mt-3 flex flex-wrap gap-2">
            <button type="button" className={BUTTON_CLASS} onClick={() => downloadText(
              "localized_attribution_observations.csv",
              attributionObservationsToCsv(authority), "text/csv",
            )}>Export Raw Observations CSV</button>
            <button type="button" className={BUTTON_CLASS} onClick={() => downloadText(
              "localized_attribution_view.json",
              attributionViewToJson(definition(authority, selection)), "application/json",
            )}>Export View Definition JSON</button>
          </div>
        </>
      )}
    </section>
  );
}
