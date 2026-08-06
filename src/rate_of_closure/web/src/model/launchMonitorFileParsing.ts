/** CSV and JSON ingestion for launch-monitor analysis. */

import type { LaunchMonitorRow, LaunchMonitorScalar } from "./launchMonitorAnalysisTypes";

const coerceCell = (value: string): LaunchMonitorScalar => {
  const trimmed = value.trim();
  if (!trimmed) return null;
  const numeric = Number(trimmed);
  return Number.isFinite(numeric) ? numeric : trimmed;
};

const parseCsvRows = (text: string): string[][] => {
  const rows: string[][] = [];
  let row: string[] = [];
  let cell = "";
  let quoted = false;
  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];
    if (character === '"') {
      if (quoted && text[index + 1] === '"') {
        cell += '"';
        index += 1;
      } else quoted = !quoted;
    } else if (character === "," && !quoted) {
      row.push(cell);
      cell = "";
    } else if ((character === "\n" || character === "\r") && !quoted) {
      if (character === "\r" && text[index + 1] === "\n") index += 1;
      row.push(cell);
      cell = "";
      if (row.some((value) => value.length)) rows.push(row);
      row = [];
    } else cell += character;
  }
  if (cell.length || row.length) {
    row.push(cell);
    rows.push(row);
  }
  if (quoted) throw new RangeError("CSV contains an unterminated quoted field");
  return rows;
};

export function parseLaunchMonitorFile(fileName: string, text: string): LaunchMonitorRow[] {
  if (fileName.toLowerCase().endsWith(".json")) {
    const parsed: unknown = JSON.parse(text);
    if (!Array.isArray(parsed) || parsed.some((row) =>
      !row || typeof row !== "object" || Array.isArray(row))) {
      throw new RangeError("JSON launch-monitor data must be an array of record objects");
    }
    return parsed as LaunchMonitorRow[];
  }
  const parsed = parseCsvRows(text);
  if (parsed.length < 2) throw new RangeError("CSV must contain a header and at least one row");
  const headers = parsed[0].map((header) => header.trim());
  if (headers.some((header) => !header) || new Set(headers).size !== headers.length) {
    throw new RangeError("CSV headers must be non-empty and unique");
  }
  return parsed.slice(1).map((values) => Object.fromEntries(headers.map(
    (header, index) => [header, coerceCell(values[index] ?? "")],
  ))) as LaunchMonitorRow[];
}
