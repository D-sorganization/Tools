import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  ASSET_MANIFEST_NAME,
  buildAssetManifest,
  embedRuntimeDescriptor,
  runtimeDescriptor,
  stableJson,
  validateReleaseRevision,
  verifyReleaseArtifacts,
  WEB_RUNTIME_NAME,
} from "./releaseArtifactContract.mjs";

const releaseDirectory = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../dist",
);
const releaseRevision = validateReleaseRevision(
  process.env.ROC_RELEASE_REVISION ?? "development",
);
const descriptor = runtimeDescriptor(releaseRevision);
await mkdir(releaseDirectory, { recursive: false }).catch((error) => {
  if (error?.code !== "EEXIST") throw error;
});
const indexPath = path.join(releaseDirectory, "index.html");
const indexSource = await readFile(indexPath, "utf8");
await writeFile(indexPath, embedRuntimeDescriptor(indexSource, descriptor), "utf8");
await writeFile(path.join(releaseDirectory, WEB_RUNTIME_NAME), stableJson(descriptor), {
  encoding: "utf8",
  flag: "wx",
});
const manifest = await buildAssetManifest(releaseDirectory, releaseRevision);
await writeFile(
  path.join(releaseDirectory, ASSET_MANIFEST_NAME),
  stableJson(manifest),
  { encoding: "utf8", flag: "wx" },
);
await verifyReleaseArtifacts(releaseDirectory, releaseRevision);
