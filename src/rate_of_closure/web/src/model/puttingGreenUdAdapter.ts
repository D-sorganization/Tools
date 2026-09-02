/**
 * UpstreamDrift `putting_green` topography adapter — TypeScript mirror
 * of the *import* direction of
 * `shared/python/swing_sim/putting/ud_adapter.py` (#4800 P9,
 * ADR-0045 F2).
 *
 * Runtime-free format adapter between UpstreamDrift's serialized green
 * topography and the `swing_sim.green_surface/1` heightfield forms in
 * `puttingGreen.ts` — the same posture as the Python adapter:
 * UpstreamDrift is never imported, only the *documented file format*
 * is parsed. Only `green_surface_from_ud_json` is ported: the React
 * Putting tab only ever *reads* a green authored in `putting_green`
 * (ADR-0045 F2's "Import green…" action); the export direction
 * (`green_surface_to_ud_json`) has no caller on this side and stays
 * Python-only.
 *
 * Format mapping (full derivation in the Python module docstring):
 * UD's JSON topography (`_surface_io.py`'s `_load_json_topography`)
 * carries `contours` (`{x, y, elevation}` triples forming a complete
 * regular square grid — one shared spacing, the wire's `spacing_m`),
 * an optional `hole_position` `[x, y]` pair (import metadata, not
 * surface geometry), and refuses a `slopes` field: UD applies slope
 * regions as a weighted, generally non-conservative *slope* field with
 * no elevation function behind it, so no heightfield reproduces it —
 * refused by name rather than approximated. Scattered (non-grid)
 * contours are refused for the same reason UD's thin-plate-spline RBF
 * interpolation of them cannot be reproduced runtime-free.
 */

import { finiteNumber } from "./puttingGreenWire";
import { gridSurface, MAX_GRID_NODES, type GridGreenSurface } from "./puttingGreen";

export interface UdGreenTopography {
  surface: GridGreenSurface;
  /**
   * UD's `hole_position` [m, m] in the same (UD green-local)
   * coordinates as the surface, or null when the document does not
   * carry one. Metadata, not geometry.
   */
  holePositionM: [number, number] | null;
}

const TOP_LEVEL_FIELDS = ["contours", "hole_position"] as const;
const CONTOUR_FIELDS = ["elevation", "x", "y"] as const;

//: Relative tolerance for grid-regularity checks (node coordinates are
//: compared against the ideal `first + i * spacing` lattice).
const GRID_REL_TOL = 1e-9;

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function regularAxisSpacing(values: readonly number[], name: string): number {
  const count = values.length;
  if (count < 2) {
    throw new Error(`contours must span at least 2 distinct ${name} values`);
  }
  if (count > MAX_GRID_NODES) {
    throw new Error(`too many distinct ${name} values`);
  }
  const span = values[values.length - 1] - values[0];
  const spacing = span / (count - 1);
  if (!(spacing > 0)) {
    throw new Error(`${name} spacing must be positive`);
  }
  const tolerance = GRID_REL_TOL * Math.max(1, Math.abs(span));
  values.forEach((value, index) => {
    const ideal = values[0] + index * spacing;
    if (Math.abs(value - ideal) > tolerance) {
      throw new Error(`contour ${name} values must be evenly spaced`);
    }
  });
  return spacing;
}

interface ParsedContours {
  points: Map<string, number>;
  xs: number[];
  ys: number[];
}

/** Contour objects -> `{(x, y): elevation}`; duplicates refused. */
function parseContourPoints(rows: unknown): ParsedContours {
  if (!Array.isArray(rows)) throw new Error("contours must be a list");
  const points = new Map<string, number>();
  const xs = new Set<number>();
  const ys = new Set<number>();
  rows.forEach((row, index) => {
    if (!isPlainObject(row)) {
      throw new Error(`each contour must be an object (index ${index})`);
    }
    const keys = Object.keys(row).sort();
    const matches =
      keys.length === CONTOUR_FIELDS.length &&
      keys.every((key, i) => key === CONTOUR_FIELDS[i]);
    if (!matches) {
      throw new Error(
        `contour fields must be exactly ${CONTOUR_FIELDS.join(", ")} (index ${index})`,
      );
    }
    const x = finiteNumber(row.x, `contours[${index}].x`);
    const y = finiteNumber(row.y, `contours[${index}].y`);
    const elevation = finiteNumber(row.elevation, `contours[${index}].elevation`);
    const key = `${x}|${y}`;
    if (points.has(key)) {
      throw new Error(`duplicate contour node at (${x}, ${y})`);
    }
    points.set(key, elevation);
    xs.add(x);
    ys.add(y);
  });
  return {
    points,
    xs: [...xs].sort((a, b) => a - b),
    ys: [...ys].sort((a, b) => a - b),
  };
}

function parseHolePosition(value: unknown): [number, number] {
  if (!Array.isArray(value) || value.length !== 2) {
    throw new Error("hole_position must be an [x, y] pair");
  }
  return [
    finiteNumber(value[0], "hole_position[0]"),
    finiteNumber(value[1], "hole_position[1]"),
  ];
}

/**
 * Parse an UpstreamDrift `_surface_io` topography document.
 *
 * The contour points must form a complete regular square grid (see
 * the module docstring for why scattered contours and `slopes` are
 * refused). Unknown fields are refused fail-closed.
 *
 * @param text - The JSON topography document.
 * @returns The parsed topography: surface plus optional hole position
 *   metadata.
 * @throws Error if the document is not a representable topography, is
 *   not a string, or a field has the wrong type.
 */
export function greenSurfaceFromUdJson(text: string): UdGreenTopography {
  if (typeof text !== "string") {
    throw new TypeError("text must be a string");
  }
  let data: unknown;
  try {
    data = JSON.parse(text);
  } catch {
    throw new Error("topography must be valid JSON");
  }
  if (!isPlainObject(data)) {
    throw new Error("topography must be an object");
  }
  if ("slopes" in data) {
    throw new Error(
      "slopes are refused: UD's slope-region field is a non-conservative " +
        "slope (no heightfield reproduces it) — bake regions into contour " +
        "elevations for interchange",
    );
  }
  const unknown = Object.keys(data).filter(
    (key) => !(TOP_LEVEL_FIELDS as readonly string[]).includes(key),
  );
  if (unknown.length > 0) {
    throw new Error(`unknown topography fields: ${unknown.sort().join(", ")}`);
  }
  if (!("contours" in data)) {
    throw new Error("topography must carry contours");
  }

  const { points, xs, ys } = parseContourPoints(data.contours);
  if (points.size !== xs.length * ys.length) {
    throw new Error("contours must cover every node of a complete regular grid");
  }
  const spacingX = regularAxisSpacing(xs, "x");
  const spacingY = regularAxisSpacing(ys, "y");
  if (Math.abs(spacingX - spacingY) > GRID_REL_TOL * Math.max(spacingX, spacingY)) {
    throw new Error("the green_surface/1 wire carries one spacing: x and y must match");
  }
  const heights = ys.map((y) =>
    xs.map((x) => {
      const elevation = points.get(`${x}|${y}`);
      if (elevation === undefined) {
        throw new Error("internal: missing grid node");
      }
      return elevation;
    }),
  );
  const surface = gridSurface([xs[0], ys[0]], spacingX, heights);
  const holePositionM = "hole_position" in data ? parseHolePosition(data.hole_position) : null;
  return { surface, holePositionM };
}
