import { describe, expect, it } from "vitest";

import fixture from "./__fixtures__/ground_repeated_bounce_wire_golden_v1.json";
import {
  MAX_REPEATED_BOUNCE_WIRE_BYTES,
  repeatedBounceResultFromJson,
  stableRepeatedBounceResultJson,
} from "./repeatedBounceWire";
import { sha256Text } from "./sha256";

type MutableRecord = Record<string, unknown>;
type Mutator = (value: MutableRecord) => void;

const clone = (): MutableRecord =>
  JSON.parse(JSON.stringify(fixture.result)) as MutableRecord;
const rows = (value: MutableRecord, key: string): MutableRecord[] =>
  value[key] as MutableRecord[];

describe("repeated-bounce evidence wire v1", () => {
  it("matches Python canonical JSON and SHA-256 without running browser physics", () => {
    const result = repeatedBounceResultFromJson(JSON.stringify(fixture.result));
    const text = stableRepeatedBounceResultJson(result);

    expect(text).toBe(stableRepeatedBounceResultJson(fixture.result));
    expect(sha256Text(text)).toBe(fixture.sha256);
    expect(repeatedBounceResultFromJson(text)).toEqual(result);
    expect(result.schema_version).toBe("ground-repeated-bounce-result/v1");
    expect(result.unit_system).toBe("SI");
  });

  it("round-trips a valid pre-contact cancellation with empty prefix evidence", () => {
    const value = clone();
    value.trajectory = [];
    value.events = [];
    value.impacts = [];
    value.airborne_segments = [];
    value.handoff_state = null;
    value.termination = { reason: "cancelled", time_s: 1, elapsed_time_s: 0 };

    const text = stableRepeatedBounceResultJson(value);
    const parsed = repeatedBounceResultFromJson(text);

    expect(parsed.trajectory).toEqual([]);
    expect(parsed.events).toEqual([]);
    expect(parsed.impacts).toEqual([]);
    expect(parsed.airborne_segments).toEqual([]);
    expect(parsed.handoff_state).toBeNull();
    expect(parsed.termination).toEqual({
      reason: "cancelled", time_s: 1, elapsed_time_s: 0,
    });
    expect(stableRepeatedBounceResultJson(parsed)).toBe(text);
  });

  it.each<readonly [string, Mutator]>([
    ["extra top-level key", (value) => { value.extra = true; }],
    ["missing events", (value) => { delete value.events; }],
    ["unsupported version", (value) => { value.schema_version = "future"; }],
    ["wrong unit system", (value) => { value.unit_system = "imperial"; }],
    ["wrong frame", (value) => { value.frame = "world"; }],
    ["bad fingerprint", (value) => { value.request_fingerprint_sha256 = "0"; }],
    ["nested extra key", (value) => {
      (rows(value, "impacts")[0].energy as MutableRecord).extra = true;
    }],
    ["non-finite evidence", (value) => {
      rows(value, "airborne_segments")[0].end_time_s = Infinity;
    }],
    ["missing settled handoff", (value) => { value.handoff_state = null; }],
    ["termination time drift", (value) => {
      (value.termination as MutableRecord).time_s = 999;
    }],
    ["termination elapsed drift", (value) => {
      (value.termination as MutableRecord).elapsed_time_s = 999;
    }],
    ["energy arithmetic drift", (value) => {
      const energy = rows(value, "impacts")[0].energy as MutableRecord;
      energy.dissipation_j = 999;
    }],
    ["cross-runtime time tolerance boundary", (value) => {
      const impact = rows(value, "impacts")[0];
      (impact.state_before as MutableRecord).time_s = 1.0050000005;
      (impact.state_after as MutableRecord).time_s = 1.0050000005;
    }],
    ["trajectory and handoff drift from event ledger", (value) => {
      const trajectory = rows(value, "trajectory");
      const finalPoint = trajectory[trajectory.length - 1];
      const handoff = value.handoff_state as MutableRecord;
      (finalPoint.position_m as number[])[0] = 1.14;
      (handoff.position_m as number[])[0] = 1.14;
    }],
    ["eventful record without trajectory", (value) => { value.trajectory = []; }],
  ])("rejects %s", (_name, mutate) => {
    const value = clone();
    mutate(value);
    expect(() => repeatedBounceResultFromJson(JSON.stringify(value))).toThrow();
  });

  it("rejects duplicate keys and enforces the UTF-8 byte bound", () => {
    const canonical = stableRepeatedBounceResultJson(fixture.result);
    const duplicate = canonical.replace(
      '"request_id":"surface-run-analytic"',
      '"request_id":"duplicate","request_id":"surface-run-analytic"',
    );
    expect(() => repeatedBounceResultFromJson(duplicate)).toThrow(/duplicate/i);
    expect(() => repeatedBounceResultFromJson(
      "é".repeat(Math.floor(MAX_REPEATED_BOUNCE_WIRE_BYTES / 2) + 1),
    )).toThrow(/maximum wire size/i);
  });

  it("rejects cross-record handoff drift", () => {
    const value = clone();
    const handoff = value.handoff_state as MutableRecord;
    const position = handoff.position_m as number[];
    position[0] += 1;
    expect(() => repeatedBounceResultFromJson(JSON.stringify(value))).toThrow(
      /handoff state must match/i,
    );
  });
});
