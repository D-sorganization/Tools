import { createHash } from "node:crypto";
import { lstat, readdir, readFile } from "node:fs/promises";
import path from "node:path";

export const ASSET_MANIFEST_SCHEMA =
  "rate-of-closure/web-asset-manifest/v1";
export const WEB_RUNTIME_SCHEMA = "rate-of-closure/web-runtime/v1";
export const ASSET_MANIFEST_NAME = "rate-of-closure-assets.v1.json";
export const WEB_RUNTIME_NAME = "rate-of-closure-runtime.v1.json";
export const WEB_RUNTIME_ELEMENT_ID = "rate-of-closure-web-runtime";
const MAX_ASSET_BYTES = 16 * 1024 * 1024;
const MAX_TOTAL_BYTES = 64 * 1024 * 1024;
const MAX_ASSETS = 128;
const COMMIT_PATTERN = /^[0-9a-f]{40}$/;
const PORTABLE_COMPONENT = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;
const WINDOWS_RESERVED = /^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)/i;
const MEDIA_TYPES = new Map([
  [".css", "text/css; charset=utf-8"],
  [".html", "text/html; charset=utf-8"],
  [".ico", "image/x-icon"],
  [".js", "text/javascript; charset=utf-8"],
  [".json", "application/json; charset=utf-8"],
  [".png", "image/png"],
  [".svg", "image/svg+xml"],
  [".ttf", "font/ttf"],
  [".woff", "font/woff"],
  [".woff2", "font/woff2"],
]);

const compareText = (left, right) => left < right ? -1 : left > right ? 1 : 0;

export const validateReleaseRevision = (value) => {
  if (value !== "development" && !COMMIT_PATTERN.test(value)) {
    throw new Error("release revision must be development or a lowercase commit");
  }
  return value;
};

const mediaType = (assetPath) => {
  const selected = MEDIA_TYPES.get(path.posix.extname(assetPath));
  if (selected === undefined || assetPath.endsWith(".map")) {
    throw new Error(`unsupported release asset: ${assetPath}`);
  }
  return selected;
};

const walk = async (root, directory = root) => {
  const entries = await readdir(directory, { withFileTypes: true });
  const files = [];
  for (const entry of entries.sort((left, right) => compareText(left.name, right.name))) {
    if (entry.isSymbolicLink()) throw new Error("release assets must not be symbolic links");
    const candidate = path.join(directory, entry.name);
    if (entry.isDirectory()) files.push(...await walk(root, candidate));
    else if (entry.isFile()) files.push(candidate);
    else throw new Error("release assets must be regular files");
  }
  return files;
};

const assetPath = (root, candidate) => {
  const relative = path.relative(root, candidate).split(path.sep).join("/");
  if (!relative || relative.startsWith("../") || path.posix.isAbsolute(relative)) {
    throw new Error("release asset escaped its root");
  }
  const parts = relative.split("/");
  if (relative.length > 512 || parts.some((part) =>
    part.length > 128 || !PORTABLE_COMPONENT.test(part) ||
    part.endsWith(".") || part.endsWith(" ") || WINDOWS_RESERVED.test(part))) {
    throw new Error(`release asset path is not portable: ${relative}`);
  }
  return relative;
};

const describeAsset = async (root, candidate) => {
  const relative = assetPath(root, candidate);
  const before = await lstat(candidate, { bigint: true });
  if (!before.isFile() || before.isSymbolicLink()) {
    throw new Error(`release asset is not a regular non-link file: ${relative}`);
  }
  const source = await readFile(candidate);
  const after = await lstat(candidate, { bigint: true });
  if (before.dev !== after.dev || before.ino !== after.ino ||
      before.size !== after.size || before.mtimeNs !== after.mtimeNs) {
    throw new Error(`release asset changed during generation: ${relative}`);
  }
  const metadata = after;
  if (metadata.size <= 0n || metadata.size > BigInt(MAX_ASSET_BYTES)) {
    throw new Error(`release asset violates its byte bound: ${relative}`);
  }
  return Object.freeze({
    path: relative,
    bytes: Number(metadata.size),
    sha256: createHash("sha256").update(source).digest("hex"),
    media_type: mediaType(relative),
    executable: false,
  });
};

export const buildAssetManifest = async (root, releaseRevision) => {
  const revision = validateReleaseRevision(releaseRevision);
  const candidates = (await walk(root))
    .filter((candidate) => assetPath(root, candidate) !== ASSET_MANIFEST_NAME)
    .sort((left, right) => compareText(assetPath(root, left), assetPath(root, right)));
  if (candidates.length === 0 || candidates.length > MAX_ASSETS) {
    throw new Error("release asset count violates its bound");
  }
  const assets = [];
  const foldedPaths = new Set();
  let totalBytes = 0;
  for (const candidate of candidates) {
    const asset = await describeAsset(root, candidate);
    const folded = asset.path.toLowerCase();
    if (foldedPaths.has(folded)) throw new Error("release asset paths collide by case");
    foldedPaths.add(folded);
    totalBytes += asset.bytes;
    assets.push(asset);
  }
  if (totalBytes > MAX_TOTAL_BYTES) throw new Error("release assets exceed total byte bound");
  return Object.freeze({
    schema_version: ASSET_MANIFEST_SCHEMA,
    release_revision: revision,
    total_bytes: totalBytes,
    assets: Object.freeze(assets),
  });
};

export const stableJson = (value) => `${JSON.stringify(value, null, 2)}\n`;

export const runtimeDescriptor = (releaseRevision) => Object.freeze({
  schema_version: WEB_RUNTIME_SCHEMA,
  mode: "static_inspection",
  release_revision: validateReleaseRevision(releaseRevision),
});

export const embedRuntimeDescriptor = (indexSource, descriptor) => {
  const start = `<script id="${WEB_RUNTIME_ELEMENT_ID}" type="application/json">`;
  const end = "</script>";
  const first = indexSource.indexOf(start);
  const second = indexSource.indexOf(start, first + start.length);
  if (first < 0 || second >= 0) throw new Error("index must contain one runtime descriptor");
  const close = indexSource.indexOf(end, first + start.length);
  if (close < 0) throw new Error("runtime descriptor script is not closed");
  const compact = JSON.stringify(descriptor);
  return `${indexSource.slice(0, first + start.length)}${compact}${indexSource.slice(close)}`;
};

export const verifyReleaseArtifacts = async (root, releaseRevision) => {
  const revision = validateReleaseRevision(releaseRevision);
  const descriptor = runtimeDescriptor(revision);
  const expectedRuntime = stableJson(descriptor);
  const runtime = await readFile(path.join(root, WEB_RUNTIME_NAME), "utf8");
  if (runtime !== expectedRuntime) throw new Error("web runtime descriptor is not canonical");
  const index = await readFile(path.join(root, "index.html"), "utf8");
  if (!index.includes(`<script id="${WEB_RUNTIME_ELEMENT_ID}" type="application/json">${JSON.stringify(descriptor)}</script>`)) {
    throw new Error("index runtime descriptor does not match release metadata");
  }
  const expectedManifest = stableJson(await buildAssetManifest(root, revision));
  const manifest = await readFile(path.join(root, ASSET_MANIFEST_NAME), "utf8");
  if (manifest !== expectedManifest) throw new Error("web asset manifest does not match assets");
};
