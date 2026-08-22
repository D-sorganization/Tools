// Anti-drift gate for the vendored canonical JSON copies.
//
// web/ must be self-contained so the public mirror (rate-of-closure-explorer)
// builds standalone, but the canonical files stay owned by the monorepo.
// This test asserts every vendored copy is identical to its canonical source.
// In the standalone mirror the canonical paths do not exist, so the suite
// detects that and skips — the monorepo run is the one that blocks drift
// (together with tests/rate_of_closure/test_web_vendored_sync.py on the
// Python side).
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import vendoredMap from "./vendored_map.json";

const webRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
// In the Tools monorepo, web/ lives at <repo>/src/rate_of_closure/web.
const repoRoot = path.resolve(webRoot, "..", "..", "..");

const mappings = Object.entries(vendoredMap.mappings);
const inMonorepo = mappings.every(([canonicalRel]) =>
  fs.existsSync(path.join(repoRoot, canonicalRel)),
);

describe("vendored canonical JSON copies", () => {
  it("declares at least the ten known vendored files", () => {
    expect(mappings.length).toBeGreaterThanOrEqual(10);
  });

  it("has every vendored file present inside web/", () => {
    for (const [, vendoredRel] of mappings) {
      expect(fs.existsSync(path.join(webRoot, vendoredRel)), vendoredRel).toBe(true);
    }
  });

  describe.skipIf(!inMonorepo)("monorepo drift gate", () => {
    it.each(mappings)("%s matches its vendored copy", (canonicalRel, vendoredRel) => {
      const canonical = JSON.parse(
        fs.readFileSync(path.join(repoRoot, canonicalRel), "utf8"),
      );
      const vendored = JSON.parse(
        fs.readFileSync(path.join(webRoot, vendoredRel), "utf8"),
      );
      expect(
        vendored,
        `${vendoredRel} drifted from ${canonicalRel}; run \`node scripts/refresh-vendored.mjs\``,
      ).toStrictEqual(canonical);
    });
  });
});
