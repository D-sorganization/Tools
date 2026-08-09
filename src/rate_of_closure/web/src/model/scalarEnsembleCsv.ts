/** Lossless CSV serialization and honest summaries for the scalar ensemble. */

import type { ScalarEnsembleResult } from "./scalarEnsembleContract";

const csvCell = (value: string | number | null | undefined): string => {
  if (value === null || value === undefined) return "";
  const text = String(value);
  const safe = typeof value === "string" && /^[=+\-@]/.test(text) ? `'${text}` : text;
  return /[",\r\n]/.test(safe) ? `"${safe.replace(/"/g, '""')}"` : safe;
};

/** Serialize all raw rows and values; nulls remain explicit empty CSV cells. */
export function scalarEnsembleToCsv<Cohort extends string>(
  ensemble: ScalarEnsembleResult<Cohort>,
): string {
  const attributeKeys = [...new Set(ensemble.rows.flatMap(
    (row) => Object.keys(row.attributes ?? {}),
  ))].sort();
  const variableKeys = ensemble.variables.map(({ key }) => key);
  const header = [
    "row_id", "trial_index", "series_id", "cohort", ...variableKeys,
    ...attributeKeys.map((key) => `attribute:${key}`),
  ];
  const rows = ensemble.rows.map((row) => [
    row.row_id, row.trial_index, row.series_id, row.cohort,
    ...variableKeys.map((key) => row.values[key]),
    ...attributeKeys.map((key) => row.attributes?.[key]),
  ]);
  return [header, ...rows].map((row) => row.map(csvCell).join(",")).join("\n");
}

/**
 * Report why non-complete rows ended so counts are not read as defects.
 *
 * A horizon nonconvergence is normalized into the `failed` cohort by the
 * observation contract, which has no separate member for it. Naming the
 * retained reason keeps the count truthful.
 */
export function nonCompleteReasonSummary<Cohort extends string>(
  ensemble: ScalarEnsembleResult<Cohort>,
): string {
  const counts = new Map<string, number>();
  for (const row of ensemble.rows) {
    if (row.cohort === "complete") continue;
    const reason = row.attributes?.reason_code ?? "unspecified";
    const key = typeof reason === "string" && reason ? reason : "unspecified";
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  if (counts.size === 0) return "";
  const detail = [...counts.entries()]
    .sort(([leftKey, leftCount], [rightKey, rightCount]) =>
      rightCount - leftCount || leftKey.localeCompare(rightKey))
    .map(([reason, count]) => `${reason} x${count}`)
    .join("; ");
  return ` Non-complete reasons: ${detail}.`;
}
