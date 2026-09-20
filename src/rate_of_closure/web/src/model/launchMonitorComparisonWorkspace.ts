/** Side-by-side launch-monitor comparison workspace model. */

import {
  PARAMETER_IDS,
  compareDefinitions,
  conventionRegistry,
  parameterGroup,
  parameterGroupLabel,
  type ParameterGroup,
  type ParameterId,
} from "./launchMonitorConventions";

export interface ComparisonRowTs {
  readonly parameterId: ParameterId;
  readonly label: string;
  readonly group: ParameterGroup;
  readonly groupLabel: string;
  readonly unit: string;
  readonly trackmanValue: number | null;
  readonly foresightValue: number | null;
  readonly difference: number | null;
  readonly differenceText: string;
  readonly isComparable: boolean;
  readonly reasons: readonly string[];
  readonly reasonsText: string;
  readonly trackmanRef: string;
  readonly foresightRef: string;
  readonly trackmanTime: string;
  readonly foresightTime: string;
  readonly trackmanStatus: string;
  readonly foresightStatus: string;
  readonly definition: string;
}

export function buildComparisonRows(
  trackmanValues?: Record<string, number | null | undefined>,
  foresightValues?: Record<string, number | null | undefined>,
): ComparisonRowTs[] {
  const registry = conventionRegistry();
  const tmVals = trackmanValues ?? {};
  const fsVals = foresightValues ?? {};

  return PARAMETER_IDS.map((paramId) => {
    const tmDef = registry.definition("trackman_comparable", paramId);
    const fsDef = registry.definition("foresight_comparable", paramId);
    const compat = compareDefinitions(tmDef, fsDef);
    const grp = parameterGroup(paramId);
    const grpLabel = parameterGroupLabel(grp);

    const tmRaw = tmVals[paramId];
    const fsRaw = fsVals[paramId];
    const tmVal = tmRaw !== undefined && tmRaw !== null && Number.isFinite(tmRaw) ? tmRaw : null;
    const fsVal = fsRaw !== undefined && fsRaw !== null && Number.isFinite(fsRaw) ? fsRaw : null;

    let diff: number | null = null;
    let diffText: string;
    let reasons: readonly string[] = [];
    let reasonsText = "—";

    if (compat.comparable) {
      if (tmVal !== null && fsVal !== null) {
        diff = tmVal - fsVal;
        diffText = diff > 0 ? `+${diff.toFixed(2)}` : diff.toFixed(2);
      } else {
        diffText = "—";
      }
    } else {
      reasons = compat.reasons;
      reasonsText = reasons.join(", ");
      diffText = `Not comparable (${reasonsText})`;
    }

    return Object.freeze({
      parameterId: paramId,
      label: tmDef.label,
      group: grp,
      groupLabel: grpLabel,
      unit: tmDef.unit,
      trackmanValue: tmVal,
      foresightValue: fsVal,
      difference: diff,
      differenceText: diffText,
      isComparable: compat.comparable,
      reasons: Object.freeze(reasons),
      reasonsText,
      trackmanRef: tmDef.referencePoint.replace(/_/g, " "),
      foresightRef: fsDef.referencePoint.replace(/_/g, " "),
      trackmanTime: tmDef.eventTime.replace(/_/g, " "),
      foresightTime: fsDef.eventTime.replace(/_/g, " "),
      trackmanStatus: tmDef.quantityStatus.replace(/_/g, " "),
      foresightStatus: fsDef.quantityStatus.replace(/_/g, " "),
      definition: tmDef.definition,
    });
  });
}

export function filterComparisonRows(
  rows: readonly ComparisonRowTs[],
  selectedGroup: string,
  searchQuery: string,
): ComparisonRowTs[] {
  const query = searchQuery.trim().toLowerCase();
  return rows.filter((row) => {
    if (selectedGroup !== "all" && row.group !== selectedGroup) {
      return false;
    }
    if (query) {
      const match =
        row.label.toLowerCase().includes(query) ||
        row.parameterId.toLowerCase().includes(query) ||
        row.groupLabel.toLowerCase().includes(query) ||
        row.definition.toLowerCase().includes(query);
      if (!match) return false;
    }
    return true;
  });
}

export function exportComparisonJson(
  rows: readonly ComparisonRowTs[],
  sourceName: string,
): string {
  const payload = {
    schema_version: "launch-monitor-comparison/v1",
    source_name: sourceName,
    rows,
    total_count: rows.length,
  };
  return JSON.stringify(payload, null, 2);
}

export function exportComparisonCsv(rows: readonly ComparisonRowTs[]): string {
  const headers = [
    "Group",
    "Parameter",
    "TrackMan",
    "Foresight",
    "Signed Difference (TM - FS)",
    "Unit",
    "TM Ref Point",
    "FS Ref Point",
    "TM Event Time",
    "FS Event Time",
    "Comparability",
    "Reason",
    "Definition",
  ];

  const escape = (val: string) => `"${val.replace(/"/g, '""')}"`;

  const lines = [headers.join(",")];
  for (const row of rows) {
    const tmStr = row.trackmanValue !== null ? row.trackmanValue.toFixed(4) : "";
    const fsStr = row.foresightValue !== null ? row.foresightValue.toFixed(4) : "";
    const diffStr = row.difference !== null ? row.difference.toFixed(4) : row.differenceText;
    const compStr = row.isComparable ? "Comparable" : "Not Comparable";
    lines.push([
      escape(row.groupLabel),
      escape(row.label),
      tmStr,
      fsStr,
      escape(diffStr),
      escape(row.unit),
      escape(row.trackmanRef),
      escape(row.foresightRef),
      escape(row.trackmanTime),
      escape(row.foresightTime),
      escape(compStr),
      escape(row.reasonsText),
      escape(row.definition),
    ].join(","));
  }
  return lines.join("\n");
}
