/** Formula-safe raw-row export for localized paired attribution. */

import { spreadsheetSafeCsvCell } from "./csvSecurity";
import {
  ATTRIBUTION_AUTHORITY_SCHEMA_ID,
  ATTRIBUTION_SCHEMA_VERSION,
  attributionAuthorityToValue,
  type AttributionAuthorityTs,
} from "./localizedAttribution";

const HEADER = [
  "schema_id", "schema_version", "authority_id", "interpretation",
  "source_spec_id", "source_variable", "source_unit", "joint_id",
  "window_start_s", "window_end_s", "target_id", "target_kind", "target_name",
  "target_unit", "target_frame", "target_convention", "target_time_s",
  "target_point_id", "baseline_trial", "perturbed_trial", "baseline_status",
  "perturbed_status", "baseline_source_value", "perturbed_source_value",
  "baseline_target_value", "perturbed_target_value", "response", "availability",
] as const;
const FLOAT_COLUMNS = new Set([8, 9, 16, 22, 23, 24, 25, 26]);
const NUMERIC_COLUMNS = new Set([1, 8, 9, 16, 18, 19, 22, 23, 24, 25, 26]);

const cell = (value: unknown): string => value === null ? "" : String(value);
const rowCell = (value: unknown, column: number): string =>
  typeof value === "number" && Number.isInteger(value) && FLOAT_COLUMNS.has(column)
    ? value.toFixed(1) : cell(value);

export function attributionObservationsToRows(
  authority: AttributionAuthorityTs,
): string[][] {
  attributionAuthorityToValue(authority);
  const sources = new Map(authority.sources.map((source) => [source.specId, source]));
  const targets = new Map(authority.targets.map((target) => [target.targetId, target]));
  const rows = authority.observations.map((observation) => {
    const source = sources.get(observation.sourceSpecId)!;
    const target = targets.get(observation.targetId)!;
    return [
      ATTRIBUTION_AUTHORITY_SCHEMA_ID, ATTRIBUTION_SCHEMA_VERSION,
      authority.authorityId, authority.interpretation, source.specId,
      source.variableKey, source.unit, source.jointId, ...source.timeWindowS,
      target.targetId, target.kind, target.name, target.unit, target.coordinateFrame,
      target.convention, target.timeS, target.pointId, observation.baselineTrialIndex,
      observation.perturbedTrialIndex, observation.baselineStatus,
      observation.perturbedStatus, observation.baselineSourceValue,
      observation.perturbedSourceValue, observation.baselineTargetValue,
      observation.perturbedTargetValue, observation.response, observation.availability,
    ].map(rowCell);
  });
  return [[...HEADER], ...rows];
}

export function attributionObservationsToCsv(
  authority: AttributionAuthorityTs,
): string {
  return attributionObservationsToRows(authority).map((row, rowIndex) => row.map(
    (value, column) => spreadsheetSafeCsvCell(
      rowIndex > 0 && value !== "" && NUMERIC_COLUMNS.has(column) ? Number(value) : value,
    ),
  ).join(",")).join("\n") + "\n";
}
