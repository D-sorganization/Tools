/**
 * Wire `swing_sim.green_surface/1` — TypeScript mirror of the wire
 * half of `shared/python/swing_sim/putting/surface.py` (#4800 P2).
 *
 * Versioned, fail-closed JSON with the `delivery_interchange`
 * posture: sorted keys, compact separators, finite values only,
 * unknown fields refused, missing fields refused, byte-identical
 * round-trips within a runtime. Float formatting is runtime-local
 * (JS shortest-round-trip prints `90` where Python repr prints
 * `90.0`); cross-runtime interchange is by JSON value. The payload is
 * geometry only — stimp and friction stay in the simulation call —
 * and is the seam UpstreamDrift's `putting_green` `_surface_io`
 * adapter targets (epic #4800 Amendment 1, P9).
 */

import {
  gridSurface,
  planarSurface,
  type GreenSurface,
} from "./puttingGreen";

export const GREEN_SURFACE_FORMAT = "swing_sim.green_surface/1";

function canonicalJson(value: unknown): string {
  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      throw new Error("canonical JSON requires finite numbers");
    }
    return JSON.stringify(value);
  }
  if (typeof value === "string") return JSON.stringify(value);
  if (Array.isArray(value)) {
    return `[${value.map(canonicalJson).join(",")}]`;
  }
  if (typeof value === "object" && value !== null) {
    const record = value as Record<string, unknown>;
    const parts = Object.keys(record)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${canonicalJson(record[key])}`);
    return `{${parts.join(",")}}`;
  }
  throw new Error("unsupported canonical JSON value");
}

/** Serialize with deterministic key ordering (runtime-local bytes). */
export function greenSurfaceToJson(surface: GreenSurface): string {
  if (surface.kind === "planar") {
    return canonicalJson({
      format: GREEN_SURFACE_FORMAT,
      kind: "planar",
      grade_percent: surface.gradePercent,
      aspect_deg: surface.aspectDeg,
    });
  }
  return canonicalJson({
    format: GREEN_SURFACE_FORMAT,
    kind: "grid",
    origin_m: surface.originM,
    spacing_m: surface.spacingM,
    heights_m: surface.heightsM,
  });
}

/**
 * A strict JSON number: finite, and never a bare boolean (`typeof
 * true !== "number"` already excludes it, unlike Python where `bool`
 * subclasses `int`). Shared with `puttingGreenUdAdapter.ts`, mirroring
 * `ud_adapter.py`'s import of `_finite_number` from `.surface`.
 */
export function finiteNumber(value: unknown, name: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${name} must be a finite number`);
  }
  return value;
}

function requireExactFields(
  data: Record<string, unknown>,
  expected: string[],
): void {
  const keys = Object.keys(data).sort();
  const wanted = [...expected].sort();
  if (
    keys.length !== wanted.length ||
    keys.some((key, index) => key !== wanted[index])
  ) {
    throw new Error(`surface fields must be exactly ${wanted.join(", ")}`);
  }
}

/** Parse and validate; unknown fields and wrong formats are refused. */
export function greenSurfaceFromJson(text: string): GreenSurface {
  const data: unknown = JSON.parse(text);
  if (typeof data !== "object" || data === null || Array.isArray(data)) {
    throw new Error("green surface must be an object");
  }
  const record = data as Record<string, unknown>;
  if (record.format !== GREEN_SURFACE_FORMAT) {
    throw new Error(`format must be ${GREEN_SURFACE_FORMAT}`);
  }
  if (record.kind === "planar") {
    requireExactFields(record, ["format", "kind", "grade_percent", "aspect_deg"]);
    return planarSurface(
      finiteNumber(record.grade_percent, "grade_percent"),
      finiteNumber(record.aspect_deg, "aspect_deg"),
    );
  }
  if (record.kind === "grid") {
    requireExactFields(record, [
      "format",
      "kind",
      "origin_m",
      "spacing_m",
      "heights_m",
    ]);
    const origin = record.origin_m;
    if (!Array.isArray(origin) || origin.length !== 2) {
      throw new Error("origin_m must be an [x, y] pair");
    }
    const rows = record.heights_m;
    if (!Array.isArray(rows)) throw new Error("heights_m must be a list");
    const heights = rows.map((row, j) => {
      if (!Array.isArray(row)) throw new Error("each heights_m row must be a list");
      return row.map((item, i) => finiteNumber(item, `heights_m[${j}][${i}]`));
    });
    return gridSurface(
      [finiteNumber(origin[0], "origin_m[0]"), finiteNumber(origin[1], "origin_m[1]")],
      finiteNumber(record.spacing_m, "spacing_m"),
      heights,
    );
  }
  throw new Error("kind must be 'planar' or 'grid'");
}
