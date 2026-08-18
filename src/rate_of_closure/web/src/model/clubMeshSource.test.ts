import { describe, expect, it } from "vitest";

import { generatedHeadFor } from "./clubHeadGeneration";
import { getClub } from "./club";
import {
  generatedMeshSource,
  importedMeshSource,
  proceduralMeshSource,
} from "./clubMeshSource";
import { writeBinaryStlFixture } from "./meshTestSupport";

describe("immutable club mesh source bundles", () => {
  it("distinguishes procedural and generated authored geometry", () => {
    const procedural = proceduralMeshSource(2);
    expect(procedural.status).toContain("0.110 m");
    const generated = generatedMeshSource(
      generatedHeadFor(getClub("Mallet Putter")), "Mallet Putter", 3,
    );
    expect(generated.mesh?.triangles).toHaveLength(2_176);
    expect(generated.status).toContain("authored SI geometry");
  });

  it("binds imported identity to SHA-256 and sanitizes display-only names", async () => {
    const buffer = writeBinaryStlFixture();
    const source = await importedMeshSource("../\u202E<b>head</b>.stl", buffer, 4);
    expect(source.sha256).toMatch(/^[0-9a-f]{64}$/);
    expect(source.rawBytes).toBe(buffer.byteLength);
    expect(source.rawTriangles).toBe(4);
    expect(source.retainedTriangles).toBe(4);
    expect(source.normalizationRevision).toBe("roc-stl-display-v1");
    expect(source.status).not.toContain("\u202E");
    expect(source.status).toContain("no physical registration or mass centroid inferred");
    expect(source.geometricCentroid).toBeNull();
    expect(Object.isFrozen(source.mesh?.triangles)).toBe(true);
  });

  it.each([-1, Number.MAX_SAFE_INTEGER + 1, 0.5])(
    "rejects unsafe generation %s", (generation) => {
      expect(() => proceduralMeshSource(generation)).toThrow(/generation/);
    },
  );
});
