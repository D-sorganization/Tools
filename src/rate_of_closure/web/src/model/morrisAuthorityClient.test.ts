/** Morris authority client capability and bounded-error tests. */

import { describe, expect, it, vi } from "vitest";

import {
  createMorrisAuthorityClient,
  MorrisAuthorityClientError,
} from "./morrisAuthorityClient";

const capability = {
  schema_id: "rate-of-closure/morris-authority-capability",
  schema_version: 1,
  available: true,
  api_prefix: "/api/rate-of-closure/v1",
  request_schema_id: "rate-of-closure/morris-request",
  job_schema_id: "rate-of-closure/morris-job",
};

describe("Morris authority client", () => {
  it("discovers capability through the canonical endpoint", async () => {
    const fetcher = vi.fn<typeof fetch>(async () => new Response(JSON.stringify(capability), {
      status: 200, headers: { "Content-Type": "application/json" },
    }));
    const client = createMorrisAuthorityClient({ fetchImpl: fetcher });

    await expect(client.capability()).resolves.toMatchObject({ available: true });
    expect(fetcher.mock.calls[0]?.[0]).toBe("/api/rate-of-closure/v1/morris/capabilities");
    expect(fetcher.mock.calls[0]?.[1]).toMatchObject({ method: "GET", redirect: "error" });
  });

  it("has no cross-origin override and rejects non-portable job IDs", async () => {
    const fetcher = vi.fn<typeof fetch>();
    const client = createMorrisAuthorityClient({ fetchImpl: fetcher });

    for (const jobId of ["slash/id", "míssing", "space id", ""] as const) {
      expect(() => client.status(jobId)).toThrow(/portable stable identifier/);
    }
    expect(fetcher).not.toHaveBeenCalled();
    expect(() => createMorrisAuthorityClient({
      fetchImpl: fetcher,
      baseUrl: "https://attacker.invalid",
    } as never)).toThrow(/unsupported key/);
  });

  it("rejects unexpected 2xx statuses for each endpoint contract", async () => {
    const json = (status: number) => new Response("{}", {
      status, headers: { "Content-Type": "application/json" },
    });

    const capabilityClient = createMorrisAuthorityClient({ fetchImpl: async () => json(201) });
    await expect(capabilityClient.capability()).rejects.toMatchObject({
      code: "invalid_response", status: 201,
    });

    const createClient = createMorrisAuthorityClient({ fetchImpl: async () => json(200) });
    await expect(createClient.create({})).rejects.toMatchObject({
      code: "invalid_response", status: 200,
    });
  });

  it("types transport failures without swallowing AbortError", async () => {
    const failed = createMorrisAuthorityClient({ fetchImpl: async () => {
      throw new Error("private network detail");
    } });
    await expect(failed.capability()).rejects.toMatchObject({ code: "transport_error" });

    const abort = new DOMException("cancelled", "AbortError");
    const cancelled = createMorrisAuthorityClient({ fetchImpl: async () => {
      throw abort;
    } });
    await expect(cancelled.capability()).rejects.toBe(abort);
  });

  it("returns a typed bounded HTTP error", async () => {
    const longMessage = "x".repeat(1_000);
    const client = createMorrisAuthorityClient({ fetchImpl: async () => new Response(
      JSON.stringify({ error: longMessage }),
      { status: 503, headers: { "Content-Type": "application/json" } },
    ) });

    const error = await client.capability().catch((caught: unknown) => caught);

    expect(error).toBeInstanceOf(MorrisAuthorityClientError);
    expect(error).toMatchObject({ code: "http_error", status: 503 });
    expect((error as Error).message.length).toBeLessThanOrEqual(256);
  });

  it("types malformed content without reflecting response bodies", async () => {
    const secret = "do-not-reflect-this-body";
    const client = createMorrisAuthorityClient({ fetchImpl: async () => new Response(secret, {
      status: 502, headers: { "Content-Type": "text/plain" },
    }) });

    const error = await client.capability().catch((caught: unknown) => caught);

    expect(error).toMatchObject({ code: "invalid_content_type", status: 502 });
    expect(String(error)).not.toContain(secret);
  });

  it("rejects an oversized declared error body before reading it", async () => {
    const cancel = vi.fn(async () => undefined);
    const stream = new ReadableStream<Uint8Array>({ cancel });
    const client = createMorrisAuthorityClient({ fetchImpl: async () => new Response(stream, {
      status: 503,
      headers: { "Content-Type": "application/json", "Content-Length": "8193" },
    }) });

    const error = await client.capability().catch((caught: unknown) => caught);

    expect(error).toMatchObject({ code: "response_too_large", status: 503 });
    expect(cancel).toHaveBeenCalledOnce();
  });

  it("applies the 16 MiB success-response contract before reading", async () => {
    const cancel = vi.fn(async () => undefined);
    const stream = new ReadableStream<Uint8Array>({ cancel });
    const client = createMorrisAuthorityClient({ fetchImpl: async () => new Response(stream, {
      status: 200,
      headers: { "Content-Type": "application/json", "Content-Length": "16777217" },
    }) });

    const error = await client.capability().catch((caught: unknown) => caught);

    expect(error).toMatchObject({ code: "response_too_large", status: 200 });
    expect(cancel).toHaveBeenCalledOnce();
  });

  it("cancels a chunked response as soon as its bounded body overflows", async () => {
    const cancel = vi.fn(async () => undefined);
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new Uint8Array(5_000));
        controller.enqueue(new Uint8Array(4_000));
      },
      cancel,
    });
    const client = createMorrisAuthorityClient({ fetchImpl: async () => new Response(stream, {
      status: 503, headers: { "Content-Type": "application/json" },
    }) });

    const error = await client.capability().catch((caught: unknown) => caught);

    expect(error).toMatchObject({ code: "response_too_large", status: 503 });
    expect(cancel).toHaveBeenCalledOnce();
  });

  it("rejects malformed UTF-8 without replacement decoding", async () => {
    const client = createMorrisAuthorityClient({ fetchImpl: async () => new Response(
      new Uint8Array([0xc3, 0x28]),
      { status: 200, headers: { "Content-Type": "application/json" } },
    ) });

    const error = await client.capability().catch((caught: unknown) => caught);

    expect(error).toMatchObject({ code: "invalid_utf8", status: 200 });
  });
});
