// Ratchet: no source file may import anything that resolves above the web/
// root. The public mirror (rate-of-closure-explorer) is a verbatim copy of
// web/, so an escaping relative import builds in the monorepo yet breaks the
// standalone mirror. Data the app needs from the monorepo is vendored into
// src/vendored/ instead (see vendored_map.json / scripts/refresh-vendored.mjs).
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const webRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");

const SCANNED_EXTENSIONS = /\.(ts|tsx|js|jsx|mjs|cjs)$/;
const SKIPPED_DIRECTORIES = new Set(["node_modules", "dist", ".git"]);
const IMPORT_PATTERN =
  /(?:from\s+|import\s*\(\s*|require\s*\(\s*)["'](\.\.?\/[^"']+)["']/g;

function collectSourceFiles(dir: string, out: string[] = []): string[] {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    if (SKIPPED_DIRECTORIES.has(entry.name)) continue;
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) collectSourceFiles(full, out);
    else if (SCANNED_EXTENSIONS.test(entry.name)) out.push(full);
  }
  return out;
}

describe("web/ import boundary", () => {
  it("no source file imports a path that resolves above the web root", () => {
    const escapes: string[] = [];
    for (const file of collectSourceFiles(webRoot)) {
      const source = fs.readFileSync(file, "utf8");
      for (const match of source.matchAll(IMPORT_PATTERN)) {
        const resolved = path.resolve(path.dirname(file), match[1]);
        const relative = path.relative(webRoot, resolved);
        if (relative.startsWith("..")) {
          escapes.push(
            `${path.relative(webRoot, file)} -> ${match[1]}`,
          );
        }
      }
    }
    expect(
      escapes,
      "Imports escaping web/ break the standalone public mirror. " +
        "Vendor the data into src/vendored/ (see vendored_map.json and " +
        "scripts/refresh-vendored.mjs) instead of importing across the boundary.",
    ).toStrictEqual([]);
  });
});
