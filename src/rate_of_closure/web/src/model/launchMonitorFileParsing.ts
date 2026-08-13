/** CSV and JSON ingestion for launch-monitor analysis. */

import {
  finiteLaunchMonitorScalar,
  type LaunchMonitorRow,
  type LaunchMonitorScalar,
} from "./launchMonitorAnalysisTypes";
import { parseUniqueJson } from "./strictJson";

export const MAX_LAUNCH_MONITOR_IMPORT_BYTES = 8 * 1024 * 1024;
export const MAX_LAUNCH_MONITOR_IMPORT_FIELD_UTF8_BYTES = 64 * 1024;
export const MAX_LAUNCH_MONITOR_IMPORT_ROWS = 250_000;
export const MAX_LAUNCH_MONITOR_IMPORT_COLUMNS = 256;
export const MAX_LAUNCH_MONITOR_IMPORT_CELLS = 2_000_000;

const assertTextBudget = (text: string): void => {
  if (text.length > MAX_LAUNCH_MONITOR_IMPORT_BYTES ||
      new TextEncoder().encode(text).byteLength > MAX_LAUNCH_MONITOR_IMPORT_BYTES) {
    throw new RangeError(`Launch-monitor import exceeds ${MAX_LAUNCH_MONITOR_IMPORT_BYTES} bytes`);
  }
};

const assertFieldBudget = (value: string): void => {
  for (let index = 0; index < value.length; index += 1) {
    const codeUnit = value.charCodeAt(index);
    if (codeUnit >= 0xd800 && codeUnit <= 0xdbff) {
      const low = value.charCodeAt(index + 1);
      if (!(low >= 0xdc00 && low <= 0xdfff)) {
        throw new RangeError("Launch-monitor text must be well-formed Unicode");
      }
      index += 1;
    } else if (codeUnit >= 0xdc00 && codeUnit <= 0xdfff) {
      throw new RangeError("Launch-monitor text must be well-formed Unicode");
    }
  }
  if (new TextEncoder().encode(value).byteLength > MAX_LAUNCH_MONITOR_IMPORT_FIELD_UTF8_BYTES) {
    throw new RangeError(
      `Launch-monitor field exceeds ${MAX_LAUNCH_MONITOR_IMPORT_FIELD_UTF8_BYTES} UTF-8 bytes`,
    );
  }
};

/** @internal Exact resource-shape authority used by both decoded formats and tests. */
export function assertLaunchMonitorImportShape(rowCount: number, columnCount: number): void {
  if (!Number.isSafeInteger(rowCount) || !Number.isSafeInteger(columnCount) ||
      rowCount < 0 || columnCount < 0) {
    throw new TypeError("launch-monitor import shape must use nonnegative safe integers");
  }
  if (rowCount > MAX_LAUNCH_MONITOR_IMPORT_ROWS) {
    throw new RangeError(`Launch-monitor import exceeds ${MAX_LAUNCH_MONITOR_IMPORT_ROWS} rows`);
  }
  if (columnCount > MAX_LAUNCH_MONITOR_IMPORT_COLUMNS) {
    throw new RangeError(`Launch-monitor import exceeds ${MAX_LAUNCH_MONITOR_IMPORT_COLUMNS} columns`);
  }
  if (rowCount * columnCount > MAX_LAUNCH_MONITOR_IMPORT_CELLS) {
    throw new RangeError(`Launch-monitor import exceeds ${MAX_LAUNCH_MONITOR_IMPORT_CELLS} dense cells`);
  }
}

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
      if (rows.length > MAX_LAUNCH_MONITOR_IMPORT_ROWS + 1) {
        throw new RangeError(`Launch-monitor import exceeds ${MAX_LAUNCH_MONITOR_IMPORT_ROWS} rows`);
      }
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
  if (!suffix.endsWith(".csv") && !suffix.endsWith(".json")) {
    throw new RangeError("Launch-monitor import supports CSV and JSON");
  }
  assertTextBudget(text);
  if (suffix.endsWith(".json")) {
    const parsed = parseUniqueJson(text, "launch-monitor JSON");
    if (!Array.isArray(parsed) || parsed.some((row) =>
      !row || typeof row !== "object" || Array.isArray(row))) {
      throw new RangeError("JSON launch-monitor data must be an array of record objects");
    }
    const rows = parsed as Record<string, unknown>[];
    assertLaunchMonitorImportShape(rows.length, 0);
    const union = new Set<string>();
    for (const row of rows) {
      for (const [key, value] of Object.entries(row)) {
        union.add(key);
        assertLaunchMonitorImportShape(0, union.size);
        assertFieldBudget(key);
        if (typeof value === "string") assertFieldBudget(value);
      }
    }
    assertLaunchMonitorImportShape(rows.length, union.size);
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
  const parsed = parseCsvRows(text);
  if (parsed.length > MAX_LAUNCH_MONITOR_IMPORT_ROWS + 1) {
    throw new RangeError(`Launch-monitor import exceeds ${MAX_LAUNCH_MONITOR_IMPORT_ROWS} rows`);
  }
  if (parsed.length < 2) throw new RangeError("CSV must contain a header and at least one row");
  parsed[0].forEach(assertFieldBudget);
  const headers = parsed[0].map((header) => header.trim());
  assertLaunchMonitorImportShape(parsed.length - 1, headers.length);
  if (headers.some((header) => !header) || new Set(headers).size !== headers.length) {
    throw new RangeError("CSV headers must be non-empty and unique");
  }
  return parsed.slice(1).map((values) => {
    if (values.length !== headers.length) {
      throw new RangeError("Every CSV data row must match the header width");
    }
    values.forEach(assertFieldBudget);
    return Object.fromEntries(headers.map(
      (header, index) => [header, coerceCell(values[index])],
    )) as LaunchMonitorRow;
  });
}

export async function readLaunchMonitorFile(file: File): Promise<LaunchMonitorRow[]> {
  if (!/\.(csv|json)$/i.test(file.name)) {
    throw new RangeError("Launch-monitor import supports CSV and JSON");
  }
  if (file.size > MAX_LAUNCH_MONITOR_IMPORT_BYTES) {
    throw new RangeError(`Launch-monitor import exceeds ${MAX_LAUNCH_MONITOR_IMPORT_BYTES} bytes`);
  }
  const bytes = await file.arrayBuffer();
  let text: string;
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch {
    throw new RangeError("Launch-monitor import must be valid UTF-8");
  }
  return parseLaunchMonitorFile(file.name, text);
}
