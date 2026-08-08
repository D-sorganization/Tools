/** Lossless CSV serialization for the UI-neutral scalar ensemble contract. */

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
