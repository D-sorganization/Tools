/** Transport-only Worker for the Python-owned durable ensemble authority. */

import {
  parseDurableEnsembleCapability,
  parseDurableEnsembleJob,
  type DurableEnsembleJob,
} from "../model/durableEnsembleAuthorityContract";
import type {
  DurableWorkerRequest,
  DurableWorkerResponse,
} from "../model/durableEnsembleWorkerClient";

interface WorkerScope {
  addEventListener(type: "message", listener: (event: MessageEvent<DurableWorkerRequest>) => void): void;
  postMessage(message: DurableWorkerResponse): void;
}
const scope = globalThis as unknown as WorkerScope;
const ROOT = "/api/rate-of-closure/v1/durable-ensembles";
const MAX_BYTES = 16 * 1024 * 1024;
let activeJob: DurableEnsembleJob | null = null;

async function document(response: Response): Promise<unknown> {
  if (response.headers.get("content-type")?.split(";", 1)[0]?.trim() !== "application/json") {
    throw new Error("Durable ensemble authority returned non-JSON content.");
  }
  const declared = response.headers.get("content-length");
  if (declared !== null && (!/^\d+$/.test(declared) || Number(declared) > MAX_BYTES)) {
    throw new Error("Durable ensemble authority response exceeds its byte limit.");
  }
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (bytes.byteLength > MAX_BYTES) throw new Error("Durable ensemble authority response exceeds its byte limit.");
  let value: unknown;
  try { value = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes)) as unknown; }
  catch { throw new Error("Durable ensemble authority returned invalid JSON."); }
  if (!response.ok) throw new Error("Durable ensemble authority rejected the request.");
  return value;
}

async function call(path: string, init: RequestInit): Promise<DurableEnsembleJob> {
  return parseDurableEnsembleJob(await document(await fetch(`${ROOT}${path}`, { ...init, redirect: "error" })));
}

async function run(request: DurableWorkerRequest & { readonly type: "run" }): Promise<void> {
  activeJob = await call("/jobs", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(request.request),
  });
  scope.postMessage({ type: "job", job: activeJob });
  while (activeJob.status === "queued" || activeJob.status === "running") {
    await new Promise((resolve) => setTimeout(resolve, 500));
    activeJob = await call(`/jobs/${encodeURIComponent(activeJob.jobId)}`, { method: "GET" });
    scope.postMessage({ type: "job", job: activeJob });
  }
}

scope.addEventListener("message", (event) => {
  const request = event.data;
  if (request.type === "capability") {
    void fetch(`${ROOT}/capabilities`, { redirect: "error" })
      .then(document).then(parseDurableEnsembleCapability)
      .then((capability) => scope.postMessage({ type: "capability", capability }))
      .catch((error: unknown) => scope.postMessage({ type: "error", message: error instanceof Error ? error.message : "Authority unavailable." }));
  } else if (request.type === "run") {
    void run(request).catch((error: unknown) => scope.postMessage({ type: "error", message: error instanceof Error ? error.message : "Durable ensemble run failed." }));
  } else if (activeJob !== null && (activeJob.status === "queued" || activeJob.status === "running")) {
    void call(`/jobs/${encodeURIComponent(activeJob.jobId)}`, { method: "DELETE" })
      .then((job) => { activeJob = job; scope.postMessage({ type: "job", job }); })
      .catch((error: unknown) => scope.postMessage({ type: "error", message: error instanceof Error ? error.message : "Cancellation failed." }));
  }
});
