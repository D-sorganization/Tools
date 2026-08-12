/** Injected transport client for the Rate Morris authority; no physics fallback. */

import { parseMorrisJobEnvelope, type MorrisJobEnvelope } from "./morrisAuthorityContract";
import {
  parseMorrisAuthorityCapability,
  type MorrisAuthorityCapability,
} from "./morrisAuthorityCapability";
import { morrisStableId } from "./morrisStableId";

export type MorrisAuthorityClientErrorCode =
  | "http_error"
  | "invalid_content_type"
  | "invalid_content_length"
  | "response_too_large"
  | "invalid_utf8"
  | "invalid_json"
  | "invalid_error_envelope"
  | "invalid_response"
  | "transport_error";

const MAX_ERROR_MESSAGE_LENGTH = 256;
const MAX_ERROR_RESPONSE_BYTES = 8_192;
const MAX_SUCCESS_RESPONSE_BYTES = 16 * 1_024 * 1_024;

export class MorrisAuthorityClientError extends Error {
  readonly code: MorrisAuthorityClientErrorCode;
  readonly status: number | null;

  constructor(code: MorrisAuthorityClientErrorCode, message: string, status: number | null = null) {
    super(message.slice(0, MAX_ERROR_MESSAGE_LENGTH));
    this.name = "MorrisAuthorityClientError";
    this.code = code;
    this.status = status;
  }
}

export interface MorrisAuthorityClient {
  capability(signal?: AbortSignal): Promise<MorrisAuthorityCapability>;
  create(request: unknown, signal?: AbortSignal): Promise<MorrisJobEnvelope>;
  status(jobId: string, signal?: AbortSignal): Promise<MorrisJobEnvelope>;
  cancel(jobId: string, signal?: AbortSignal): Promise<MorrisJobEnvelope>;
}

export interface MorrisAuthorityClientOptions {
  readonly fetchImpl?: typeof fetch;
}

const AUTHORITY_BASE_URL = "/api/rate-of-closure/v1";

const errorMessage = (value: unknown, status: number): string => {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new MorrisAuthorityClientError("invalid_error_envelope", "Morris authority returned an invalid error envelope.", status);
  }
  const item = value as Record<string, unknown>;
  if (Object.keys(item).length !== 1 || typeof item.error !== "string"
      || item.error === "" || item.error !== item.error.trim()
      || Array.from(item.error).some((character) => {
        const code = character.codePointAt(0) ?? 0;
        return code <= 0x1f || (code >= 0x7f && code <= 0x9f);
      })) {
    throw new MorrisAuthorityClientError("invalid_error_envelope", "Morris authority returned an invalid error envelope.", status);
  }
  return item.error;
};

const stableJobId = (value: string): string => {
  return morrisStableId(value, "jobId");
};

const cancelBody = (response: Response): void => {
  try {
    void response.body?.cancel().catch(() => undefined);
  } catch {
    // Cancellation is best effort after the response has already failed its contract.
  }
};

const declaredLength = (response: Response, limit: number): number | null => {
  const header = response.headers.get("content-length");
  if (header === null) return null;
  if (!/^(0|[1-9][0-9]*)$/.test(header) || !Number.isSafeInteger(Number(header))) {
    throw new MorrisAuthorityClientError("invalid_content_length", "Morris authority returned an invalid Content-Length.", response.status);
  }
  const length = Number(header);
  if (length > limit) {
    throw new MorrisAuthorityClientError("response_too_large", "Morris authority response exceeds its byte limit.", response.status);
  }
  return length;
};

const boundedBytes = async (response: Response, limit: number): Promise<Uint8Array> => {
  try {
    declaredLength(response, limit);
  } catch (error: unknown) {
    cancelBody(response);
    throw error;
  }
  if (response.body === null) return new Uint8Array();
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let length = 0;
  try {
    while (true) {
      const result = await reader.read();
      if (result.done) break;
      length += result.value.byteLength;
      if (length > limit) {
        void reader.cancel().catch(() => undefined);
        throw new MorrisAuthorityClientError("response_too_large", "Morris authority response exceeds its byte limit.", response.status);
      }
      chunks.push(result.value);
    }
  } finally {
    reader.releaseLock();
  }
  const body = new Uint8Array(length);
  let offset = 0;
  chunks.forEach((chunk) => { body.set(chunk, offset); offset += chunk.byteLength; });
  return body;
};

const responseDocument = async (response: Response): Promise<unknown> => {
  const mediaType = response.headers.get("content-type")?.split(";", 1)[0]?.trim().toLowerCase();
  if (mediaType !== "application/json") {
    cancelBody(response);
    throw new MorrisAuthorityClientError("invalid_content_type", "Morris authority returned non-JSON content.", response.status);
  }
  const limit = response.ok ? MAX_SUCCESS_RESPONSE_BYTES : MAX_ERROR_RESPONSE_BYTES;
  const raw = await boundedBytes(response, limit);
  let source: string;
  try {
    source = new TextDecoder("utf-8", { fatal: true }).decode(raw);
  } catch {
    throw new MorrisAuthorityClientError("invalid_utf8", "Morris authority returned invalid UTF-8.", response.status);
  }
  let document: unknown;
  try {
    // This same-origin client relies on the private authority to emit unique-key JSON;
    // the browser JSON API cannot preserve duplicate keys for independent rejection.
    document = JSON.parse(source) as unknown;
  } catch {
    throw new MorrisAuthorityClientError("invalid_json", "Morris authority returned invalid JSON.", response.status);
  }
  if (!response.ok) {
    throw new MorrisAuthorityClientError("http_error", errorMessage(document, response.status), response.status);
  }
  return document;
};

export function createMorrisAuthorityClient(options: MorrisAuthorityClientOptions = {}): MorrisAuthorityClient {
  if (Object.keys(options).some((key) => key !== "fetchImpl")) {
    throw new TypeError("Morris authority client options contain an unsupported key");
  }
  const fetchImpl = options.fetchImpl ?? globalThis.fetch;
  if (typeof fetchImpl !== "function") throw new TypeError("fetchImpl must be callable");

  const sameOriginFetch = async (path: string, init: RequestInit): Promise<Response> => {
    try {
      return await fetchImpl(`${AUTHORITY_BASE_URL}${path}`, { ...init, redirect: "error" });
    } catch (error: unknown) {
      if (error instanceof DOMException && error.name === "AbortError") throw error;
      throw new MorrisAuthorityClientError(
        "transport_error", "Morris authority transport failed.", null,
      );
    }
  };

  const call = async (
    path: string, init: RequestInit, statuses: readonly number[],
  ): Promise<MorrisJobEnvelope> => {
    const response = await sameOriginFetch(path, init);
    const document = await responseDocument(response);
    if (!statuses.includes(response.status)) {
      throw new MorrisAuthorityClientError(
        "invalid_response", "Morris authority returned an unexpected success status.", response.status,
      );
    }
    try {
      return parseMorrisJobEnvelope(document);
    } catch {
      throw new MorrisAuthorityClientError("invalid_response", "Morris authority job response failed validation.", response.status);
    }
  };
  const jobPath = (jobId: string): string => `/morris/jobs/${encodeURIComponent(stableJobId(jobId))}`;
  return Object.freeze({
    capability: async (signal?: AbortSignal) => {
      const response = await sameOriginFetch("/morris/capabilities", { method: "GET", signal });
      const document = await responseDocument(response);
      if (response.status !== 200) {
        throw new MorrisAuthorityClientError(
          "invalid_response", "Morris authority returned an unexpected success status.", response.status,
        );
      }
      try {
        return parseMorrisAuthorityCapability(document);
      } catch {
        throw new MorrisAuthorityClientError("invalid_response", "Morris authority capability failed validation.", response.status);
      }
    },
    create: (request: unknown, signal?: AbortSignal) => call("/morris/jobs", {
      method: "POST", signal, headers: { "Content-Type": "application/json" }, body: JSON.stringify(request),
    }, [202]),
    status: (jobId: string, signal?: AbortSignal) => call(jobPath(jobId), { method: "GET", signal }, [200]),
    cancel: (jobId: string, signal?: AbortSignal) => call(jobPath(jobId), { method: "DELETE", signal }, [200, 202]),
  });
}
