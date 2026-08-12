/** Injected transport client for the Rate Morris authority; no physics fallback. */

import { parseMorrisJobEnvelope, type MorrisJobEnvelope } from "./morrisAuthorityContract";

export interface MorrisAuthorityClient {
  create(request: unknown, signal?: AbortSignal): Promise<MorrisJobEnvelope>;
  status(jobId: string, signal?: AbortSignal): Promise<MorrisJobEnvelope>;
  cancel(jobId: string, signal?: AbortSignal): Promise<MorrisJobEnvelope>;
}

export interface MorrisAuthorityClientOptions {
  readonly baseUrl?: string;
  readonly fetchImpl?: typeof fetch;
}

const record = (value: unknown): Record<string, unknown> => {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("Morris authority error must be a JSON object");
  }
  const item = value as Record<string, unknown>;
  if (Object.keys(item).length !== 1 || !("error" in item) || typeof item.error !== "string") {
    throw new Error("Morris authority error fields do not match the contract");
  }
  return item;
};

const stableJobId = (value: string): string => {
  if (typeof value !== "string" || value === "" || value !== value.trim()) {
    throw new TypeError("jobId must be a nonempty trimmed string");
  }
  return value;
};

const normalizedBaseUrl = (value: string): string => {
  if (typeof value !== "string" || value !== value.trim()) throw new TypeError("baseUrl must be a trimmed string");
  return value.endsWith("/") ? value.slice(0, -1) : value;
};

const responseDocument = async (response: Response): Promise<unknown> => {
  const mediaType = response.headers.get("content-type")?.split(";", 1)[0]?.trim().toLowerCase();
  if (mediaType !== "application/json") throw new Error("Morris authority returned non-JSON content");
  let document: unknown;
  try {
    document = await response.json() as unknown;
  } catch {
    throw new Error("Morris authority returned invalid JSON");
  }
  if (!response.ok) throw new Error(String(record(document).error));
  return document;
};

export function createMorrisAuthorityClient(options: MorrisAuthorityClientOptions = {}): MorrisAuthorityClient {
  const baseUrl = normalizedBaseUrl(options.baseUrl ?? "");
  const fetchImpl = options.fetchImpl ?? globalThis.fetch;
  if (typeof fetchImpl !== "function") throw new TypeError("fetchImpl must be callable");

  const call = async (path: string, init: RequestInit): Promise<MorrisJobEnvelope> => {
    const response = await fetchImpl(`${baseUrl}${path}`, init);
    return parseMorrisJobEnvelope(await responseDocument(response));
  };
  const jobPath = (jobId: string): string => `/morris/jobs/${encodeURIComponent(stableJobId(jobId))}`;
  return Object.freeze({
    create: (request: unknown, signal?: AbortSignal) => call("/morris/jobs", {
      method: "POST", signal, headers: { "Content-Type": "application/json" }, body: JSON.stringify(request),
    }),
    status: (jobId: string, signal?: AbortSignal) => call(jobPath(jobId), { method: "GET", signal }),
    cancel: (jobId: string, signal?: AbortSignal) => call(jobPath(jobId), { method: "DELETE", signal }),
  });
}
