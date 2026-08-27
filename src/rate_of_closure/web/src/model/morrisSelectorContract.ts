/** Provenance-complete selection over immutable Morris reports. */

import type {
  MorrisReport,
  MorrisTargetProvenance,
} from "./morrisGlobalSensitivityContract";
import {
  morrisTargetLabel,
  presentMorrisTarget,
  type MorrisReportPresentation,
} from "./morrisPresentation";
import { variableLabel } from "./variationRegistry";

export const MORRIS_TARGET_SELECTION_SCHEMA_ID = "rate-of-closure/morris-target-selection" as const;
export const MORRIS_TARGET_SELECTION_SCHEMA_VERSION = 1 as const;

export interface MorrisTargetIdentity extends MorrisTargetProvenance {
  readonly schemaId: typeof MORRIS_TARGET_SELECTION_SCHEMA_ID;
  readonly schemaVersion: typeof MORRIS_TARGET_SELECTION_SCHEMA_VERSION;
}

export interface MorrisTargetOption {
  readonly identity: MorrisTargetIdentity;
  readonly label: string;
}

export interface MorrisSourceOption {
  readonly specId: string;
  readonly variableKey: string;
  readonly label: string;
}

export interface MorrisReportSelection {
  readonly target: MorrisTargetIdentity;
  readonly sourceSpecId: string | null;
}

const targetKey = (target: MorrisTargetProvenance): string => JSON.stringify([
  target.kind,
  target.name,
  target.unit,
  target.pointId,
  target.timeS,
  target.coordinateFrame,
]);

const identity = (target: MorrisTargetProvenance): MorrisTargetIdentity => Object.freeze({
  schemaId: MORRIS_TARGET_SELECTION_SCHEMA_ID,
  schemaVersion: MORRIS_TARGET_SELECTION_SCHEMA_VERSION,
  ...target,
});

const optionLabel = (target: MorrisTargetProvenance): string => {
  const context: string[] = [target.kind];
  if (target.pointId !== null) context.push(target.pointId);
  if (target.timeS !== null) context.push(`t=${target.timeS} s`);
  return `${morrisTargetLabel(target.name)} — ${context.join(" · ")}`;
};

export function listMorrisTargetOptions(report: MorrisReport): readonly MorrisTargetOption[] {
  const targets = new Map<string, MorrisTargetProvenance>();
  report.estimates.forEach((estimate) => targets.set(targetKey(estimate.target), estimate.target));
  return Object.freeze([...targets.values()]
    .sort((left, right) => targetKey(left).localeCompare(targetKey(right)))
    .map((target) => Object.freeze({ identity: identity(target), label: optionLabel(target) })));
}

const targetFromIdentity = (value: MorrisTargetIdentity): MorrisTargetProvenance => {
  if (value.schemaId !== MORRIS_TARGET_SELECTION_SCHEMA_ID
      || value.schemaVersion !== MORRIS_TARGET_SELECTION_SCHEMA_VERSION) {
    throw new RangeError("unsupported Morris target selection identity");
  }
  return {
    name: value.name,
    unit: value.unit,
    kind: value.kind,
    timeS: value.timeS,
    pointId: value.pointId,
    coordinateFrame: value.coordinateFrame,
  };
};

export function listMorrisSourceOptions(
  report: MorrisReport,
  selected: MorrisTargetIdentity,
): readonly MorrisSourceOption[] {
  const key = targetKey(targetFromIdentity(selected));
  const sources = new Map<string, MorrisSourceOption>();
  report.estimates.filter((estimate) => targetKey(estimate.target) === key)
    .forEach((estimate) => sources.set(estimate.source.specId, Object.freeze({
      specId: estimate.source.specId,
      variableKey: estimate.source.variableKey,
      label: variableLabel(estimate.source.variableKey),
    })));
  if (sources.size === 0) throw new RangeError("Morris target is not present in the report");
  return Object.freeze([...sources.values()].sort((left, right) => (
    left.variableKey.localeCompare(right.variableKey) || left.specId.localeCompare(right.specId)
  )));
}

export function selectMorrisReport(
  report: MorrisReport,
  selection: MorrisReportSelection,
): MorrisReportPresentation {
  const presentation = presentMorrisTarget(report, targetFromIdentity(selection.target));
  if (selection.sourceSpecId === null) return presentation;
  const rows = presentation.rows.filter((row) => row.specId === selection.sourceSpecId);
  if (rows.length === 0) throw new RangeError("Morris source is not present for the selected target");
  return Object.freeze({ target: presentation.target, rows: Object.freeze(rows) });
}
