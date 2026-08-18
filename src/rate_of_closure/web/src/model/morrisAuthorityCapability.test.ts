/** Strict Morris authority capability tests. */

import { describe, expect, it } from "vitest";

import { parseMorrisAuthorityCapability } from "./morrisAuthorityCapability";

const capability = (): Record<string, unknown> => ({
  schema_id: "rate-of-closure/morris-authority-capability",
  schema_version: 1,
  available: true,
  api_prefix: "/api/rate-of-closure/v1",
  request_schema_id: "rate-of-closure/morris-request",
  job_schema_id: "rate-of-closure/morris-job",
});

describe("Morris authority capability", () => {
  it("parses and freezes the exact v1 document", () => {
    const parsed = parseMorrisAuthorityCapability(capability());

    expect(parsed).toEqual({
      schemaId: "rate-of-closure/morris-authority-capability",
      schemaVersion: 1,
      available: true,
      apiPrefix: "/api/rate-of-closure/v1",
      requestSchemaId: "rate-of-closure/morris-request",
      jobSchemaId: "rate-of-closure/morris-job",
    });
    expect(Object.isFrozen(parsed)).toBe(true);
  });

  it.each([
    ["unknown field", (item: Record<string, unknown>) => { item.extra = true; }],
    ["schema ID", (item: Record<string, unknown>) => { item.schema_id = "other"; }],
    ["version", (item: Record<string, unknown>) => { item.schema_version = 2; }],
    ["availability", (item: Record<string, unknown>) => { item.available = "yes"; }],
    ["API prefix", (item: Record<string, unknown>) => { item.api_prefix = "https://example.test"; }],
    ["alternate API prefix", (item: Record<string, unknown>) => { item.api_prefix = "/api/other/v1"; }],
    ["request schema", (item: Record<string, unknown>) => { item.request_schema_id = "other"; }],
  ])("rejects invalid %s", (_name, mutate) => {
    const item = capability();
    mutate(item);
    expect(() => parseMorrisAuthorityCapability(item)).toThrow();
  });
});
