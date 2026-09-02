/**
 * Dispatch an imported green document between the two fail-closed
 * readers — TypeScript mirror of `rate_of_closure/putting.py`'s
 * `green_surface_from_document` (#4800 P6, ADR-0045 F2 "Import
 * green…" action).
 *
 * Dispatch is on the declared `format` field alone, never on shape:
 * present means the Tools `swing_sim.green_surface/1` wire
 * (`puttingGreenWire.ts`); absent means an UpstreamDrift
 * `_surface_io` topography (`puttingGreenUdAdapter.ts`, #4800 P9's
 * adapter). `swing_sim.green_surface/1` documents always carry
 * `format`; UpstreamDrift topographies refuse unknown top-level
 * fields and therefore never do, so the dispatch is exact and
 * fail-closed in both directions — the same posture as the Qt
 * Putting tab's `PuttingGreenControls.adopt_green_document`.
 */

import { GREEN_SURFACE_FORMAT, greenSurfaceFromJson } from "./puttingGreenWire";
import { greenSurfaceFromUdJson } from "./puttingGreenUdAdapter";
import type { GreenSurface } from "./puttingGreen";

//: Provenance label for a green read through the P9 UD adapter — the
//: web analogue of the Python dispatcher's second return value.
export const UPSTREAMDRIFT_TOPOGRAPHY_WIRE = "upstreamdrift.putting_green topography";

export interface ImportedGreen {
  surface: GreenSurface;
  /** The wire the surface actually came through (displayed authority). */
  wire: string;
}

/**
 * Read one imported green through the declared-format reader.
 *
 * @param text - The JSON document, as read from a File.
 * @returns The parsed surface and the provenance label naming the
 *   wire it came through.
 * @throws Error if the text is not JSON, is not an object, or the
 *   selected reader refuses it (its named reason is the thrown
 *   message — never silence).
 */
export function greenSurfaceFromDocument(text: string): ImportedGreen {
  if (typeof text !== "string") {
    throw new TypeError("green document text must be a string");
  }
  let data: unknown;
  try {
    data = JSON.parse(text);
  } catch {
    throw new Error("green document must be valid JSON");
  }
  if (typeof data !== "object" || data === null || Array.isArray(data)) {
    throw new Error("green document must be a JSON object");
  }
  if ("format" in data) {
    return { surface: greenSurfaceFromJson(text), wire: GREEN_SURFACE_FORMAT };
  }
  return {
    surface: greenSurfaceFromUdJson(text).surface,
    wire: UPSTREAMDRIFT_TOPOGRAPHY_WIRE,
  };
}
