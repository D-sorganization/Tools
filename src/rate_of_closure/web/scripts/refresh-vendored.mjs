#!/usr/bin/env node
// Refresh web/src/vendored/ from the canonical monorepo files.
//
// The public mirror (rate-of-closure-explorer) is a verbatim copy of web/,
// so every JSON the app or its tests import must live inside web/. The
// canonical copies stay where the Python side owns them; this script copies
// them byte-for-byte into src/vendored/ per src/vendored/vendored_map.json.
//
// Run from anywhere inside the Tools monorepo:
//   node src/rate_of_closure/web/scripts/refresh-vendored.mjs
//
// In the standalone mirror the canonical paths do not exist; the script
// reports this and exits non-zero (refreshing is a monorepo-only operation).

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const webRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
// web/ lives at <repo>/src/rate_of_closure/web
const repoRoot = path.resolve(webRoot, "..", "..", "..");

const mapPath = path.join(webRoot, "src", "vendored", "vendored_map.json");
const { mappings } = JSON.parse(fs.readFileSync(mapPath, "utf8"));

let missing = 0;
let copied = 0;
for (const [canonicalRel, vendoredRel] of Object.entries(mappings)) {
  const canonical = path.join(repoRoot, canonicalRel);
  const vendored = path.join(webRoot, vendoredRel);
  if (!fs.existsSync(canonical)) {
    console.error(`MISSING canonical file: ${canonicalRel} (not in a Tools monorepo checkout?)`);
    missing += 1;
    continue;
  }
  fs.mkdirSync(path.dirname(vendored), { recursive: true });
  fs.copyFileSync(canonical, vendored);
  copied += 1;
  console.log(`${canonicalRel} -> ${path.relative(webRoot, vendored).replaceAll("\\", "/")}`);
}

console.log(`Refreshed ${copied}/${Object.keys(mappings).length} vendored files.`);
if (missing > 0) process.exit(1);
