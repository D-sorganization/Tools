/**
 * Green document dispatch (ADR-0045 F2's "Import green…" action).
 *
 * Mirrors `tests/rate_of_closure/test_putting.py::TestGreenDocumentBridge`
 * test-for-test: dispatch is on the declared `format` field alone,
 * never on shape, and an unrecognized or malformed document is
 * refused rather than guessed.
 */

import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import { greenSurfaceFromDocument, UPSTREAMDRIFT_TOPOGRAPHY_WIRE } from "./puttingGreenImport";
import { GREEN_SURFACE_FORMAT, greenSurfaceToJson } from "./puttingGreenWire";
import { planarSurface, type GridGreenSurface } from "./puttingGreen";

describe("greenSurfaceFromDocument: dispatch on declared format", () => {
  it("reads a swing_sim.green_surface/1 document with the Tools reader", () => {
    const surface = planarSurface(2.0, 90.0);
    const { surface: parsed, wire } = greenSurfaceFromDocument(
      greenSurfaceToJson(surface),
    );
    expect(parsed).toEqual(surface);
    expect(wire).toBe(GREEN_SURFACE_FORMAT);
  });

  it("reads an UpstreamDrift topography with the P9 adapter", () => {
    const text = JSON.stringify({
      contours: Array.from({ length: 4 }, (_, y) =>
        Array.from({ length: 4 }, (_, x) => ({
          x: x * 0.5,
          y: y * 0.5,
          elevation: -0.01 * x,
        })),
      ).flat(),
    });
    const { surface, wire } = greenSurfaceFromDocument(text);
    expect(surface.kind).toBe("grid");
    expect((surface as GridGreenSurface).spacingM).toBeCloseTo(0.5, 10);
    expect(wire).toBe(UPSTREAMDRIFT_TOPOGRAPHY_WIRE);
  });

  it("refuses a declared but unknown format rather than falling through to the UD reader", () => {
    expect(() =>
      greenSurfaceFromDocument('{"format": "swing_sim.green_surface/9"}'),
    ).toThrow();
  });

  it("refuses non-object and non-string documents", () => {
    expect(() => greenSurfaceFromDocument("[]")).toThrow();
    expect(() => greenSurfaceFromDocument("42")).toThrow();
    // @ts-expect-error - exercising the runtime guard for a non-string input
    expect(() => greenSurfaceFromDocument({})).toThrow(TypeError);
  });

  it("refuses malformed JSON", () => {
    expect(() => greenSurfaceFromDocument("{not json")).toThrow();
  });
});

describe("ud.putt_scene/1 mirror pattern consumer", () => {
  const sceneFixturePath = path.resolve(
    path.dirname(fileURLToPath(import.meta.url)),
    "..",
    "vendored",
    "fixtures",
    "ud_putt_scene.v1.json",
  );

  it("pins the mirrored artifact digest to the UD export", () => {
    const raw = fs.readFileSync(sceneFixturePath);
    const digest = crypto.createHash("sha256").update(raw).digest("hex");
    expect(digest).toBe(
      "43c483f2aaf841e281a5968deebbe9ec0e3673914f927bb761d63ef658eb6070", // pragma: allowlist secret
    );
  });

  it("loads a mirrored ud.putt_scene/1 artifact through greenSurfaceFromDocument", () => {
    const text = fs.readFileSync(sceneFixturePath, "utf8");
    const { surface, wire } = greenSurfaceFromDocument(text);
    expect(wire).toBe("ud.putt_scene/1");
    expect(surface.kind).toBe("grid");
    const grid = surface as GridGreenSurface;
    expect(grid.heightsM.length).toBe(36);
    expect(grid.heightsM[0].length).toBe(48);
  });

  it("rejects an unknown schema value", () => {
    const bad = JSON.stringify({
      schema: "ud.putt_scene/2",
      units: {},
      frame: {},
      roll_model: "test",
      terrain: { x: [0, 1], y: [0, 1], z: [[0, 0], [0, 0]] },
      trajectory: {},
      lines: {},
      green_size_m: [1, 1],
      metrics: {},
    });
    expect(() => greenSurfaceFromDocument(bad)).toThrow(/unsupported schema/);
  });

  it("rejects an export with missing required fields", () => {
    const bad = JSON.stringify({
      schema: "ud.putt_scene/1",
      units: {},
    });
    expect(() => greenSurfaceFromDocument(bad)).toThrow(/missing required field/);
  });
});

