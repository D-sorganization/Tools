/**
 * CSV parsing for the Data Explorer's local-file import.
 *
 * {@link parseCsv} turns raw CSV text into a column-oriented table the explorer
 * can plot directly: a numeric epoch-ms `index` (when a time-like column is
 * detected) plus per-column numeric values where non-numeric cells become
 * `null` gaps. The parser is a small hand-rolled state machine that understands
 * the comma delimiter, RFC-4180 double-quoted fields (with `""` escaping), and
 * both `\n` and `\r\n` line endings — no external dependency.
 *
 * Pure, DOM-free, and unit-testable. Every public function validates its input
 * (DbC): empty or header-only text throws an `Error`.
 */

/** One parsed column: a name plus numeric values with `null` for gaps. */
export interface CsvColumn {
  name: string;
  values: (number | null)[];
}

/** Result of {@link parseCsv}: an optional epoch-ms index and value columns. */
export interface CsvTable {
  /** Epoch-ms index when a time-like column was found, else `null`. */
  index: number[] | null;
  /** Value columns in source order (the index column is excluded). */
  columns: CsvColumn[];
}

/** Header names (case-insensitive) treated as the time index. */
const TIME_NAMES = new Set(["time", "timestamp", "datetime", "date"]);

/**
 * Split CSV text into rows of string fields.
 *
 * Handles double-quoted fields (commas/newlines inside quotes are literal,
 * `""` is an escaped quote) and `\r\n` / `\n` line endings. A trailing newline
 * does not produce a spurious empty row.
 */
function tokenize(text: string): string[][] {
  const rows: string[][] = [];
  let field = "";
  let row: string[] = [];
  let inQuotes = false;
  let sawAnyChar = false;

  const pushField = (): void => {
    row.push(field);
    field = "";
  };
  const pushRow = (): void => {
    pushField();
    rows.push(row);
    row = [];
  };

  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (inQuotes) {
      if (ch === '"') {
        if (text[i + 1] === '"') {
          field += '"';
          i += 1;
        } else {
          inQuotes = false;
        }
      } else {
        field += ch;
      }
      sawAnyChar = true;
      continue;
    }
    if (ch === '"') {
      inQuotes = true;
      sawAnyChar = true;
    } else if (ch === ",") {
      pushField();
      sawAnyChar = true;
    } else if (ch === "\n") {
      pushRow();
      sawAnyChar = false;
    } else if (ch === "\r") {
      // Swallow; the following \n (if any) ends the row.
      if (text[i + 1] !== "\n") {
        pushRow();
        sawAnyChar = false;
      }
    } else {
      field += ch;
      sawAnyChar = true;
    }
  }
  // Flush a final unterminated row (no trailing newline).
  if (sawAnyChar || field.length > 0 || row.length > 0) {
    pushRow();
  }
  return rows;
}

/** Parse a single cell to a number, or `null` when it is not numeric. */
function parseNumber(cell: string): number | null {
  const trimmed = cell.trim();
  if (trimmed === "") return null;
  const n = Number(trimmed);
  return Number.isFinite(n) ? n : null;
}

/**
 * Parse a cell as an epoch-ms timestamp.
 *
 * Accepts a numeric epoch (interpreted as milliseconds) or an ISO-8601 / other
 * `Date`-parseable string. Returns `null` when the cell is not a valid time.
 */
function parseTime(cell: string): number | null {
  const trimmed = cell.trim();
  if (trimmed === "") return null;
  const asNumber = Number(trimmed);
  if (trimmed !== "" && Number.isFinite(asNumber)) {
    return asNumber;
  }
  const ms = Date.parse(trimmed);
  return Number.isNaN(ms) ? null : ms;
}

/** Does this column of raw cells look like parseable timestamps? */
function looksLikeTime(cells: string[]): boolean {
  let parsed = 0;
  let nonEmpty = 0;
  for (const cell of cells) {
    if (cell.trim() === "") continue;
    nonEmpty += 1;
    // A bare number is "time-like" only if the name said so; here we require
    // an actual date parse so we don't hijack ordinary numeric columns.
    if (Number.isFinite(Number(cell.trim()))) continue;
    if (!Number.isNaN(Date.parse(cell.trim()))) parsed += 1;
    else return false;
  }
  return nonEmpty > 0 && parsed > 0;
}

/**
 * Parse CSV text into an indexed, column-oriented numeric table.
 *
 * Preconditions: `text` must be a string with at least a header row and one
 * data row.
 *
 * Index detection: the first column whose header (case-insensitive) is one of
 * `time`/`timestamp`/`datetime`/`date`, OR whose data cells parse as ISO dates,
 * becomes the epoch-ms `index` and is excluded from `columns`. A name-matched
 * time column is parsed as epoch-ms (numeric epoch or ISO string). When no time
 * column is found, `index` is `null` and the caller synthesizes `0..n-1`.
 *
 * Non-numeric value cells become `null`.
 *
 * @throws TypeError if `text` is not a string.
 * @throws Error if `text` is empty or contains no data rows (header only).
 */
export function parseCsv(text: string): CsvTable {
  if (typeof text !== "string") {
    throw new TypeError("parseCsv: text must be a string");
  }
  if (text.trim() === "") {
    throw new Error("parseCsv: input is empty");
  }

  const rows = tokenize(text).filter(
    (r) => !(r.length === 1 && r[0].trim() === ""),
  );
  if (rows.length === 0) {
    throw new Error("parseCsv: input is empty");
  }
  if (rows.length < 2) {
    throw new Error("parseCsv: no data rows (header only)");
  }

  const header = rows[0].map((h) => h.trim());
  const dataRows = rows.slice(1);
  const colCount = header.length;

  // Materialize each source column's raw cells (padding short rows).
  // ⚡ Bolt Optimization: Avoid Array.from({ length }) overhead by using new Array + for-loop
  const rawColumns: string[][] = new Array(colCount);
  for (let c = 0; c < colCount; c += 1) {
    rawColumns[c] = new Array(dataRows.length);
  }
  for (let r = 0; r < dataRows.length; r += 1) {
    const row = dataRows[r];
    for (let c = 0; c < colCount; c += 1) {
      rawColumns[c][r] = c < row.length ? row[c] : "";
    }
  }

  // Pick the index column: prefer a name match, else a date-parseable column.
  let timeCol = -1;
  for (let c = 0; c < colCount; c += 1) {
    if (TIME_NAMES.has(header[c].toLowerCase())) {
      timeCol = c;
      break;
    }
  }
  if (timeCol === -1) {
    for (let c = 0; c < colCount; c += 1) {
      if (looksLikeTime(rawColumns[c])) {
        timeCol = c;
        break;
      }
    }
  }

  let index: number[] | null = null;
  if (timeCol !== -1) {
    // ⚡ Bolt Optimization: Replace multiple passes (.map, .every) with single pass
    const rawTimes = rawColumns[timeCol];
    const times = new Array(rawTimes.length);
    let allValid = true;
    for (let i = 0; i < rawTimes.length; i++) {
      const t = parseTime(rawTimes[i]);
      if (t === null) {
        allValid = false;
        break;
      }
      times[i] = t as number;
    }
    if (allValid) {
      index = times;
    } else {
      timeCol = -1;
    }
  }

  const columns: CsvColumn[] = [];
  for (let c = 0; c < colCount; c += 1) {
    if (c === timeCol) continue;

    // ⚡ Bolt Optimization: Replace rawColumns[c].map with single-pass loop pre-allocating the array
    const rawCells = rawColumns[c];
    const values = new Array(rawCells.length);
    for (let i = 0; i < rawCells.length; i++) {
      values[i] = parseNumber(rawCells[i]);
    }

    columns.push({
      name: header[c] === "" ? `column_${c + 1}` : header[c],
      values,
    });
  }

  return { index, columns };
}
