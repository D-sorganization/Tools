/**
 * Centralized display-formatting helpers (#3546).
 *
 * `.toFixed(...)` with `?? 0` fallbacks plus ad-hoc byte/duration formatters
 * were scattered across components. Consolidating them keeps numeric display
 * consistent and null-safe.
 */

/**
 * Format a possibly-missing number to a fixed number of decimals.
 *
 * @param value - the value (may be undefined/null/NaN).
 * @param digits - decimal places (default 2).
 * @param fallback - returned when `value` is not a finite number.
 */
export function fmtNumber(
  value: number | null | undefined,
  digits = 2,
  fallback = "—",
): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return fallback;
  }
  return value.toFixed(digits);
}

/** Human-readable byte size (e.g. `1.5 MB`). */
export function fmtBytes(bytes: number | null | undefined, digits = 1): string {
  if (bytes === null || bytes === undefined || !Number.isFinite(bytes)) {
    return "—";
  }
  if (bytes < 1) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB", "PB"];
  const exponent = Math.min(
    Math.floor(Math.log(bytes) / Math.log(1024)),
    units.length - 1,
  );
  const value = bytes / Math.pow(1024, exponent);
  return `${value.toFixed(exponent === 0 ? 0 : digits)} ${units[exponent]}`;
}

/** Human-readable duration from seconds (e.g. `1h 02m 03s`). */
export function fmtDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) {
    return "—";
  }
  const total = Math.max(0, Math.floor(seconds));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const pad = (n: number) => String(n).padStart(2, "0");
  if (h > 0) return `${h}h ${pad(m)}m ${pad(s)}s`;
  if (m > 0) return `${m}m ${pad(s)}s`;
  return `${s}s`;
}
