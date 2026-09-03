import type { InterlockConfig } from "../types";

/**
 * Parse an interlock-limit input box. Blank means "disabled" (`null`); a
 * non-finite number is refused by the backend (422, Tools #3974), so it is
 * treated as blank here rather than sent.
 */
export function parseLimitInput(raw: string): number | null {
  const trimmed = raw.trim();
  if (trimmed === "") {
    return null;
  }
  const value = Number(trimmed);
  return Number.isFinite(value) ? value : null;
}

/** Render a limit for an input box: `null` shows as an empty field. */
export function limitInputValue(limit: number | null | undefined): string | number {
  return limit === null || limit === undefined ? "" : limit;
}

/**
 * Whether `value` is outside the tag's trip band (low/high, the tiers the
 * firmware trips on). A disabled side (`null`) can never be violated.
 */
export function isOutsideTripBand(value: number, interlock: InterlockConfig | undefined): boolean {
  if (!interlock) {
    return false;
  }
  const aboveHigh = interlock.high_limit !== null && value > interlock.high_limit;
  const belowLow = interlock.low_limit !== null && value < interlock.low_limit;
  return aboveHigh || belowLow;
}

/** All-disabled interlock: the backend default for an unrouted tag. */
export const DISABLED_INTERLOCK: InterlockConfig = {
  lolo_limit: null,
  low_limit: null,
  high_limit: null,
  hihi_limit: null,
};
