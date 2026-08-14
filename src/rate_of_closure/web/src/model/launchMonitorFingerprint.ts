/** Deterministic provenance fingerprints for browser analysis exports. */

import type { LaunchMonitorRow } from "./launchMonitorAnalysisTypes";
import { sha256Text } from "./sha256";

export { sha256Text } from "./sha256";

export const uniqueStrings = (rows: LaunchMonitorRow[], column: string): string[] =>
  [...new Set(rows.map((row) => row[column])
    .filter((value) => value !== null && value !== undefined)
    .map(String).filter((value) => value.trim()))].sort();

export const canonicalFingerprint = (rows: LaunchMonitorRow[], selected: string[]): string => {
  const identity = ["shot_id", "session_id", "source_row", "monitor_vendor"]
    .filter((column) => rows.some((row) => column in row) && !selected.includes(column));
  const columns = [...identity, ...selected];
  return sha256Text(JSON.stringify(rows.map((row) => Object.fromEntries(
    columns.map((column) => [column, row[column] ?? null]),
  ))));
};
