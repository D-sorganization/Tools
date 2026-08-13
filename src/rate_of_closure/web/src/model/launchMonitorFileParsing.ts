/** CSV and JSON ingestion for launch-monitor analysis. */

import {
  finiteLaunchMonitorScalar,
  type LaunchMonitorRow,
  type LaunchMonitorScalar,
} from "./launchMonitorAnalysisTypes";

const coerceCell = (value: string): LaunchMonitorScalar => {
  const trimmed = value.trim();
  if (!trimmed) return null;
  const numeric = finiteLaunchMonitorScalar(trimmed);
  return numeric === null || (Number.isInteger(numeric) && !Number.isSafeInteger(numeric))
    ? trimmed : numeric;
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
  const suffix = fileName.toLowerCase();
  if (suffix.endsWith(".json")) {
    const parsed: unknown = JSON.parse(text);
    if (!Array.isArray(parsed) || parsed.some((row) =>
      !row || typeof row !== "object" || Array.isArray(row))) {
      throw new RangeError("JSON launch-monitor data must be an array of record objects");
    }
    const rows = parsed as Record<string, unknown>[];
    if (rows.some((row) => Object.keys(row).some((key) => !key.trim()))) {
      throw new RangeError("JSON launch-monitor field names must be non-empty");
    }
    if (rows.some((row) => Object.values(row).some((value) => {
      if (value === null || typeof value === "string" || typeof value === "boolean") return false;
      return typeof value !== "number" || !Number.isFinite(value) ||
        (Number.isInteger(value) && !Number.isSafeInteger(value));
    }))) {
      throw new RangeError("JSON launch-monitor record values must be portable finite scalars");
    }
    return rows as LaunchMonitorRow[];
  }
  if (!suffix.endsWith(".csv")) {
    throw new RangeError("Launch-monitor import supports CSV and JSON");
  }
  const parsed = parseCsvRows(text);
  if (parsed.length < 2) throw new RangeError("CSV must contain a header and at least one row");
  const headers = parsed[0].map((header) => header.trim());
  if (headers.some((header) => !header) || new Set(headers).size !== headers.length) {
    throw new RangeError("CSV headers must be non-empty and unique");
  }
  return parsed.slice(1).map((values) => {
    if (values.length !== headers.length) {
      throw new RangeError("Every CSV data row must match the header width");
    }
    return Object.fromEntries(headers.map(
      (header, index) => [header, coerceCell(values[index])],
    )) as LaunchMonitorRow;
  });
}
