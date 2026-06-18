/**
 * Centralized tag helpers (#3546).
 *
 * The magic number `32` and the `TAG_<n>` naming convention were duplicated
 * across App.tsx, RoutingMatrix, DataCapturePanel and others. This module is
 * the single source of truth for both.
 */

/** Number of broker tag registers exposed by the PLC. */
export const TAG_COUNT = 32;

/** `[0, 1, ..., TAG_COUNT - 1]` — handy for rendering tag grids/rows. */
export const TAG_INDICES: number[] = Array.from(
  { length: TAG_COUNT },
  (_, i) => i,
);

/** Canonical `TAG_<index>` name for a numeric tag index. */
export function tagName(index: number): string {
  return `TAG_${index}`;
}

/**
 * Parse the numeric index out of a `TAG_<n>` name.
 *
 * @returns the index when `name` is a well-formed in-range `TAG_<n>`, else null.
 */
export function parseTagId(name: string): number | null {
  if (!name.startsWith("TAG_")) {
    return null;
  }
  const id = Number.parseInt(name.slice(4), 10);
  if (Number.isNaN(id) || id < 0 || id >= TAG_COUNT) {
    return null;
  }
  return id;
}

/** True when `name` is a valid in-range `TAG_<n>` register name. */
export function isTagName(name: string): boolean {
  return parseTagId(name) !== null;
}
