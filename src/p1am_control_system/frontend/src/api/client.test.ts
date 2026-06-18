import { describe, it, expect, vi, afterEach } from "vitest";
import { z } from "zod";
import { apiFetch, ApiError, API_BASE } from "./client";

function mockFetchOnce(response: Partial<Response> & { jsonValue?: unknown }) {
  const { jsonValue, ...rest } = response;
  const res = {
    ok: true,
    status: 200,
    json: async () => jsonValue,
    ...rest,
  } as Response;
  vi.stubGlobal(
    "fetch",
    vi.fn(() => Promise.resolve(res)),
  );
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("apiFetch", () => {
  it("prefixes the API base path and returns parsed JSON", async () => {
    const fetchMock = vi.fn(() =>
      Promise.resolve({ ok: true, status: 200, json: async () => ({ a: 1 }) } as Response),
    );
    vi.stubGlobal("fetch", fetchMock);

    const data = await apiFetch<{ a: number }>("/routing");
    expect(data).toEqual({ a: 1 });
    expect(fetchMock).toHaveBeenCalledWith(`${API_BASE}/routing`, expect.any(Object));
  });

  it("does not double-prefix an already-absolute /api path", async () => {
    const fetchMock = vi.fn(() =>
      Promise.resolve({ ok: true, status: 200, json: async () => ({}) } as Response),
    );
    vi.stubGlobal("fetch", fetchMock);

    await apiFetch("/api/routing");
    expect(fetchMock).toHaveBeenCalledWith("/api/routing", expect.any(Object));
  });

  it("serializes a json body and defaults to POST", async () => {
    const fetchMock = vi.fn(
      (_input: RequestInfo | URL, _init?: RequestInit) =>
        Promise.resolve({ ok: true, status: 200, json: async () => ({}) } as Response),
    );
    vi.stubGlobal("fetch", fetchMock);

    await apiFetch("/tags/1", { json: { value: 5 } });
    const init = fetchMock.mock.calls[0][1] as RequestInit;
    expect(init.method).toBe("POST");
    expect(init.body).toBe(JSON.stringify({ value: 5 }));
    expect((init.headers as Headers).get("Content-Type")).toBe("application/json");
  });

  it("validates the response against a schema when provided", async () => {
    mockFetchOnce({ jsonValue: { id: 7 } });
    const schema = z.object({ id: z.number() });
    await expect(apiFetch("/x", { schema })).resolves.toEqual({ id: 7 });
  });

  it("throws ApiError when the response fails schema validation", async () => {
    mockFetchOnce({ jsonValue: { id: "not-a-number" } });
    const schema = z.object({ id: z.number() });
    await expect(apiFetch("/x", { schema })).rejects.toBeInstanceOf(ApiError);
  });

  it("throws ApiError carrying status and detail on a non-ok response", async () => {
    mockFetchOnce({ ok: false, status: 422, jsonValue: { detail: "bad input" } });
    const err = await apiFetch("/x").catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).status).toBe(422);
    expect((err as ApiError).message).toBe("bad input");
  });

  it("wraps network failures as an ApiError with status 0", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() => Promise.reject(new Error("offline"))),
    );
    const err = await apiFetch("/x").catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).status).toBe(0);
  });

  it("returns undefined for 204 No Content", async () => {
    mockFetchOnce({ status: 204 });
    await expect(apiFetch("/x")).resolves.toBeUndefined();
  });
});
