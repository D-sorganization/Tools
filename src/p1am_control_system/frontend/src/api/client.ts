import type { ZodType } from "zod";

/**
 * Shared typed HTTP client for the P1AM SCADA backend.
 *
 * Previously every component re-implemented `fetch` + `res.ok` + try/catch with
 * inconsistent error handling and the literal `/api` base scattered across the
 * codebase. `apiFetch` centralizes:
 *   - the `/api` base path,
 *   - JSON serialization of request bodies,
 *   - `res.ok` checking with a typed {@link ApiError},
 *   - optional runtime validation of the response via a zod schema (#3545).
 */

/** Base path for all backend calls. Vite proxies this to the FastAPI server. */
export const API_BASE = "/api";

/**
 * Error thrown by {@link apiFetch} when the response is not ok, the body cannot
 * be parsed, or runtime validation fails. Carries the HTTP status (0 for
 * network/parse failures) and any structured `detail` returned by FastAPI.
 */
export class ApiError extends Error {
  readonly status: number;
  readonly detail?: unknown;

  constructor(message: string, status: number, detail?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

export interface ApiFetchOptions<T> extends Omit<RequestInit, "body"> {
  /** Request body — plain objects are JSON-encoded automatically. */
  json?: unknown;
  /** Optional zod schema; when present the parsed response is validated. */
  schema?: ZodType<T>;
}

function joinPath(path: string): string {
  if (path.startsWith("http://") || path.startsWith("https://")) {
    return path;
  }
  if (path.startsWith(API_BASE)) {
    return path;
  }
  return `${API_BASE}${path.startsWith("/") ? "" : "/"}${path}`;
}

/** Execute one checked request while leaving successful response decoding to callers. */
export async function apiResponse(
  path: string,
  init: RequestInit = {},
): Promise<Response> {
  let res: Response;
  try {
    res = await fetch(joinPath(path), init);
  } catch (cause) {
    throw new ApiError(
      `Network error calling ${path}`,
      0,
      cause instanceof Error ? cause.message : cause,
    );
  }
  if (!res.ok) {
    let detail: unknown;
    try {
      detail = await res.json();
    } catch {
      detail = undefined;
    }
    const message =
      detail && typeof detail === "object" && "detail" in detail
        ? String((detail as { detail: unknown }).detail)
        : `Request to ${path} failed with status ${res.status}`;
    throw new ApiError(message, res.status, detail);
  }
  return res;
}

/**
 * Perform a JSON request against the backend.
 *
 * @typeParam T - the expected (and, when `schema` is supplied, validated) shape.
 * @param path - endpoint path; may be absolute (`/api/...`) or relative (`routing`).
 * @throws {ApiError} on non-2xx responses, network errors, or schema mismatch.
 */
export async function apiFetch<T = unknown>(
  path: string,
  options: ApiFetchOptions<T> = {},
): Promise<T> {
  const { json, schema, headers, ...rest } = options;

  const init: RequestInit = { ...rest };
  const finalHeaders = new Headers(headers);
  if (json !== undefined) {
    init.body = JSON.stringify(json);
    if (!finalHeaders.has("Content-Type")) {
      finalHeaders.set("Content-Type", "application/json");
    }
    if (!init.method) {
      init.method = "POST";
    }
  }
  init.headers = finalHeaders;

  const res = await apiResponse(path, init);

  // No-content responses (e.g. 204) resolve to undefined.
  if (res.status === 204) {
    return undefined as T;
  }

  let data: unknown;
  try {
    data = await res.json();
  } catch (cause) {
    throw new ApiError(
      `Invalid JSON in response from ${path}`,
      res.status,
      cause instanceof Error ? cause.message : cause,
    );
  }

  if (schema) {
    const parsed = schema.safeParse(data);
    if (!parsed.success) {
      throw new ApiError(
        `Response from ${path} did not match the expected contract`,
        res.status,
        parsed.error.issues,
      );
    }
    return parsed.data;
  }

  return data as T;
}
