/**
 * Pure commit logic for inline-editable value fields.
 *
 * Extracted from the React component so the parse/validate/clamp/"did it
 * actually change" decision can be unit-tested without rendering (DRY/DbC). The
 * component stays a thin shell around {@link decideCommit}.
 */

export interface CommitOptions {
  /** Inclusive lower bound; the parsed value is clamped up to it. */
  min?: number;
  /** Inclusive upper bound; the parsed value is clamped down to it. */
  max?: number;
  /** Custom parser (defaults to a strict Number parse). */
  parse?: (raw: string) => number;
}

export interface CommitDecision {
  /** True when the raw text parsed to a finite number. */
  ok: boolean;
  /** The clamped value to commit (only meaningful when `ok`). */
  value: number;
  /** True when `value` differs from `current` (so a write is warranted). */
  changed: boolean;
  /** Why the parse was rejected (only when `!ok`). */
  reason?: string;
}

/** Strict numeric parse: rejects "", whitespace, and trailing garbage. */
export function parseNumber(raw: string): number {
  if (typeof raw !== "string") return Number.NaN;
  const trimmed = raw.trim();
  if (trimmed === "") return Number.NaN;
  // Number() (unlike parseFloat) rejects "12abc", which is what we want for a
  // committed edit — but it accepts "" (0) and whitespace, guarded above.
  return Number(trimmed);
}

/**
 * Decide what to do with an edited value.
 *
 * Parses `raw`, clamps to [min, max] when provided, and reports whether the
 * result differs from `current`. Never throws on bad input — returns
 * `{ ok: false }` so the caller can reject the edit and keep the old value.
 *
 * @throws RangeError if `min` and `max` are both given and `min > max` (a
 *   caller bug, not user input).
 */
export function decideCommit(
  raw: string,
  current: number,
  opts: CommitOptions = {},
): CommitDecision {
  const { min, max, parse = parseNumber } = opts;
  if (
    typeof min === "number" &&
    typeof max === "number" &&
    Number.isFinite(min) &&
    Number.isFinite(max) &&
    min > max
  ) {
    throw new RangeError(`decideCommit: min (${min}) must not exceed max (${max})`);
  }

  const parsed = parse(raw);
  if (typeof parsed !== "number" || !Number.isFinite(parsed)) {
    return { ok: false, value: current, changed: false, reason: "not a finite number" };
  }

  let value = parsed;
  if (typeof min === "number" && Number.isFinite(min)) value = Math.max(min, value);
  if (typeof max === "number" && Number.isFinite(max)) value = Math.min(max, value);

  return { ok: true, value, changed: value !== current };
}
