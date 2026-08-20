import assert from "node:assert/strict";
import { mkdtemp, mkdir, symlink, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import {
  ASSET_MANIFEST_SCHEMA,
  ASSET_MANIFEST_NAME,
  buildAssetManifest,
  embedRuntimeDescriptor,
  runtimeDescriptor,
  stableJson,
  verifyReleaseArtifacts,
  WEB_RUNTIME_NAME,
  WEB_RUNTIME_SCHEMA,
} from "./releaseArtifactContract.mjs";

const fixture = async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "roc-assets-"));
  await mkdir(path.join(root, "assets"));
  const descriptor = JSON.stringify(runtimeDescriptor("development"));
  await writeFile(
    path.join(root, "index.html"),
    `<main>Rate of Closure</main><script id="rate-of-closure-web-runtime" type="application/json">${descriptor}</script>\n`,
  );
  await writeFile(path.join(root, "assets", "app-123.js"), "export const app = 1;\n");
  return root;
};

test("builds one sorted deterministic manifest with exact hashes", async () => {
  const root = await fixture();
  const first = await buildAssetManifest(root, "development");
  const second = await buildAssetManifest(root, "development");
  assert.equal(first.schema_version, ASSET_MANIFEST_SCHEMA);
  assert.deepEqual(first, second);
  assert.deepEqual(first.assets.map((asset) => asset.path), [
    "assets/app-123.js",
    "index.html",
  ]);
  assert.match(first.assets[0].sha256, /^[0-9a-f]{64}$/);
  assert.equal(stableJson(first), stableJson(second));
});

test("rejects source maps and unsupported release files", async () => {
  const root = await fixture();
  await writeFile(path.join(root, "assets", "app.js.map"), "{}\n");
  await assert.rejects(buildAssetManifest(root, "development"), /unsupported/i);
});

test("rejects nonportable or reserved asset paths", async () => {
  for (const name of [".hidden.json", "CON.json", "caf\u00e9.json"]) {
    const root = await fixture();
    await writeFile(path.join(root, name), "{}\n");
    await assert.rejects(buildAssetManifest(root, "development"), /portable/i);
  }
});

test("rejects symbolic-link substitution", async (context) => {
  const root = await fixture();
  try {
    await symlink(path.join(root, "index.html"), path.join(root, "linked.html"));
  } catch (error) {
    context.skip(`symbolic links unavailable: ${error.code}`);
    return;
  }
  await assert.rejects(buildAssetManifest(root, "development"), /symbolic/i);
});

test("rejects ambiguous or non-release revision identities", async () => {
  const root = await fixture();
  await assert.rejects(buildAssetManifest(root, "main"), /revision/i);
  await assert.rejects(buildAssetManifest(root, "A".repeat(40)), /revision/i);
});

test("verification rejects any post-manifest asset substitution", async () => {
  const root = await fixture();
  await writeFile(path.join(root, WEB_RUNTIME_NAME), stableJson({
    schema_version: WEB_RUNTIME_SCHEMA,
    mode: "static_inspection",
    release_revision: "development",
  }));
  const manifest = await buildAssetManifest(root, "development");
  await writeFile(path.join(root, ASSET_MANIFEST_NAME), stableJson(manifest));
  await verifyReleaseArtifacts(root, "development");
  await writeFile(path.join(root, "index.html"), "substituted\n");
  await assert.rejects(
    verifyReleaseArtifacts(root, "development"),
    /runtime descriptor|manifest does not match/i,
  );
});

test("embeds exactly the canonical descriptor in the production document", () => {
  const descriptor = runtimeDescriptor("development");
  const source = '<script id="rate-of-closure-web-runtime" type="application/json">{}</script>';
  const embedded = embedRuntimeDescriptor(source, descriptor);
  assert.match(embedded, new RegExp(JSON.stringify(descriptor).replace(/[{}]/g, "\\$&")));
  assert.throws(() => embedRuntimeDescriptor(`${source}${source}`, descriptor), /one/i);
});
