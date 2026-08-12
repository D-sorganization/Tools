/** Strict Morris authority envelope and injected client tests. */

import { describe, expect, it, vi } from "vitest";

import fixture from "./__fixtures__/morris_global_sensitivity_golden_v1.json";
import { createMorrisAuthorityClient } from "./morrisAuthorityClient";
import { parseMorrisJobEnvelope } from "./morrisAuthorityContract";

const completed = (): Record<string, unknown> => ({
  schema_id: "rate-of-closure/morris-job",
  schema_version: 1,
  job_id: "job-1",
  request_id: "request-17",
  status: "completed",
  completed_samples: 36,
  total_samples: 36,
  cancel_requested: false,
  report: structuredClone(fixture),
  error: null,
});

describe("Morris authority contract", () => {
  it("strictly parses a completed job through the existing report parser", () => {
    const parsed = parseMorrisJobEnvelope(completed());
    expect(parsed.status).toBe("completed");
    expect(parsed.report?.schemaVersion).toBe(1);
    expect(Object.isFrozen(parsed)).toBe(true);
  });

  it.each([
    ["unknown field", (item: Record<string, unknown>) => { item.extra = true; }],
    ["version", (item: Record<string, unknown>) => { item.schema_version = 2; }],
    ["partial report", (item: Record<string, unknown>) => { item.status = "running"; }],
    ["failed without error", (item: Record<string, unknown>) => { item.status = "failed"; item.report = null; }],
    ["progress overflow", (item: Record<string, unknown>) => { item.completed_samples = 37; }],
  ])("rejects %s", (_name, mutate) => {
    const payload = completed();
    mutate(payload);
    expect(() => parseMorrisJobEnvelope(payload)).toThrow();
  });

  it("uses only the injected base URL and forwards AbortSignal for status", async () => {
    const fetcher = vi.fn<typeof fetch>(async (input, init) => {
      void input;
      void init;
      return new Response(JSON.stringify(completed()), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    });
    const client = createMorrisAuthorityClient({
      baseUrl: "http://127.0.0.1:8765/mount",
      fetchImpl: fetcher,
    });
    const signal = new AbortController().signal;

    await client.status("job-1", signal);
    await client.cancel("job-1");
    await client.create({ schema_id: "request" });

    expect(fetcher.mock.calls[0]?.[0]).toBe("http://127.0.0.1:8765/mount/morris/jobs/job-1");
    expect(fetcher.mock.calls[0]?.[1]).toMatchObject({ method: "GET", signal });
    expect(fetcher.mock.calls[1]?.[1]).toMatchObject({ method: "DELETE" });
    expect(fetcher.mock.calls[2]?.[1]).toMatchObject({ method: "POST" });
  });
});
