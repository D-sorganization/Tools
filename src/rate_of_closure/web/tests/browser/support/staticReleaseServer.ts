import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { createServer, type Server, type ServerResponse } from "node:http";
import path from "node:path";

const MOUNT_PATH = "/release/candidate/";
const MANIFEST_NAME = "rate-of-closure-assets.v1.json";
const COMMIT_PATTERN = /^[0-9a-f]{40}$/;
const SAFE_ASSET = /^(?:assets\/[A-Za-z0-9._-]+|index\.html|rate-of-closure-runtime\.v1\.json)$/;

interface ManifestAsset {
  readonly path: string;
  readonly bytes: number;
  readonly sha256: string;
  readonly media_type: string;
}

interface ReleaseManifest {
  readonly release_revision: string;
  readonly assets: readonly ManifestAsset[];
}

export interface StaticReleaseServer {
  readonly origin: string;
  readonly mountUrl: string;
  close(): Promise<void>;
}

const releaseRoot = path.resolve(import.meta.dirname, "../../../dist");

async function verifiedAssets(): Promise<Map<string, ManifestAsset>> {
  const source = await readFile(path.join(releaseRoot, MANIFEST_NAME), "utf8");
  const manifest = JSON.parse(source) as ReleaseManifest;
  if (!COMMIT_PATTERN.test(manifest.release_revision) || !Array.isArray(manifest.assets)) {
    throw new Error("static fixture requires an exact release manifest");
  }
  const assets = new Map<string, ManifestAsset>();
  for (const asset of manifest.assets) {
    if (!SAFE_ASSET.test(asset.path) || assets.has(asset.path)) {
      throw new Error("static fixture rejected an unsafe or duplicate asset path");
    }
    const body = await readFile(path.join(releaseRoot, ...asset.path.split("/")));
    const digest = createHash("sha256").update(body).digest("hex");
    if (body.byteLength !== asset.bytes || digest !== asset.sha256) {
      throw new Error("static fixture rejected substituted release bytes");
    }
    assets.set(asset.path, asset);
  }
  return assets;
}

function reject(response: ServerResponse): void {
  response.writeHead(404, { "Cache-Control": "no-store" });
  response.end();
}

export async function startStaticReleaseServer(): Promise<StaticReleaseServer> {
  const assets = await verifiedAssets();
  const server = createServer(async (request, response) => {
    try {
      const target = new URL(request.url ?? "", "http://fixture.invalid");
      if (request.method !== "GET" || target.search || !target.pathname.startsWith(MOUNT_PATH)) {
        reject(response);
        return;
      }
      const relative = target.pathname.slice(MOUNT_PATH.length) || "index.html";
      const asset = assets.get(relative);
      if (asset === undefined) {
        reject(response);
        return;
      }
      const body = await readFile(path.join(releaseRoot, ...relative.split("/")));
      response.writeHead(200, {
        "Cache-Control": relative.startsWith("assets/")
          ? "public, max-age=31536000, immutable" : "no-store",
        "Content-Type": asset.media_type,
        "X-Content-Type-Options": "nosniff",
      });
      response.end(body);
    } catch {
      response.writeHead(500, { "Cache-Control": "no-store" });
      response.end();
    }
  });
  await new Promise<void>((resolve, rejectStart) => {
    server.once("error", rejectStart);
    server.listen(0, "127.0.0.1", resolve);
  });
  return describeServer(server);
}

function describeServer(server: Server): StaticReleaseServer {
  const address = server.address();
  if (address === null || typeof address === "string") {
    throw new Error("static fixture did not own a TCP listener");
  }
  const origin = `http://127.0.0.1:${address.port}`;
  return {
    origin,
    mountUrl: `${origin}${MOUNT_PATH}`,
    close: () => new Promise<void>((resolve, rejectClose) => {
      server.close((error) => error === undefined ? resolve() : rejectClose(error));
    }),
  };
}
