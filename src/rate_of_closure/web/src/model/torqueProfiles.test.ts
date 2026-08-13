import { describe, expect, it } from "vitest";

import parityPayload from "./__fixtures__/torque_profile_parity.json";
import {
  COEFFICIENT_ORDER,
  FitMetadata,
  JointTorqueAssignment,
  PrescribedTorqueProfile,
  TORQUE_PROFILE_SCHEMA_VERSION,
  TORQUE_UNIT,
  TorquePolynomial,
  TorqueProfileSource,
  evaluateAscendingPolynomial,
} from "./torqueProfiles";

const literalPayload = (): Record<string, unknown> =>
  structuredClone(parityPayload) as Record<string, unknown>;

describe("torque-profile schema parity", () => {
  it("pins schema constants and the complete Python source vocabulary", () => {
    expect(TORQUE_PROFILE_SCHEMA_VERSION).toBe(1);
    expect(TORQUE_UNIT).toBe("N*m");
    expect(COEFFICIENT_ORDER).toBe("ascending_c0_first");
    expect(new Set(Object.values(TorqueProfileSource))).toEqual(
      new Set(["direct", "drawn", "imported", "optimized", "fitted_run"]),
    );
  });

  it("loads the literal Python payload and evaluates ascending coefficients", () => {
    const profile = PrescribedTorqueProfile.fromJsonObject(literalPayload());
    expect(profile.evaluate(0.5)).toEqual({
      "joint.shoulder": 9,
      "joint.wrist": 1.375,
    });
    expect(evaluateAscendingPolynomial([2, 3, 4], 2)).toBe(24);
    expect(profile.toJsonObject()).toEqual(parityPayload);
  });

  it("round-trips deterministically with sorted metadata and immutable data", () => {
    const profile = PrescribedTorqueProfile.loads(JSON.stringify(parityPayload));
    const first = profile.dumps();
    expect(PrescribedTorqueProfile.loads(first).dumps()).toBe(first);
    expect(first.indexOf('"author"')).toBeLessThan(first.indexOf('"run_id"'));
    expect(Object.isFrozen(profile)).toBe(true);
    expect(Object.isFrozen(profile.sourceMetadata)).toBe(true);
    expect(Object.isFrozen(profile.assignments)).toBe(true);
    expect(() => {
      (profile.sourceMetadata as Record<string, string>).author = "changed";
    }).toThrow(TypeError);
  });

  it("preserves optional fit metadata and enforces its polynomial degree", () => {
    const metadata = new FitMetadata({
      degree: 2,
      rmseNm: 0.05,
      maxAbsErrorNm: 0.1,
      rSquared: 0.99,
      conditionNumber: 2.5,
      originalSampleSha256: "a".repeat(64),
    });
    expect(new TorquePolynomial([0, 3, -0.5], metadata).fitMetadata).toBe(metadata);
    expect(() => new TorquePolynomial([1, 2], metadata)).toThrow(/degree/i);
  });

  it("accepts evaluation only at finite times inside the inclusive domain", () => {
    const profile = PrescribedTorqueProfile.fromJsonObject(literalPayload());
    expect(profile.evaluate(0)["joint.shoulder"]).toBe(10);
    expect(profile.evaluate(1.25)["joint.shoulder"]).toBe(7.5);
    for (const time of [-0.001, 1.251, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(() => profile.evaluate(time)).toThrow();
    }
  });
});

describe("torque-profile validation", () => {
  it.each([
    ["schema version", (data: Record<string, unknown>) => { data.schema_version = 2; }],
    ["boolean schema version", (data: Record<string, unknown>) => { data.schema_version = true; }],
    ["torque unit", (data: Record<string, unknown>) => { data.torque_unit = "lbf*ft"; }],
    ["coefficient order", (data: Record<string, unknown>) => { data.coefficient_order = "descending"; }],
    ["unknown field", (data: Record<string, unknown>) => { data.unexpected = true; }],
    ["missing field", (data: Record<string, unknown>) => { delete data.model_id; }],
    ["invalid source", (data: Record<string, unknown>) => { data.source = "measured"; }],
    ["unstable ID", (data: Record<string, unknown>) => { data.profile_id = "has spaces"; }],
    ["empty metadata", (data: Record<string, unknown>) => { data.source_metadata = {}; }],
    ["invalid metadata key", (data: Record<string, unknown>) => { data.source_metadata = { "bad key": "x" }; }],
    ["empty metadata value", (data: Record<string, unknown>) => { data.source_metadata = { author: " " }; }],
    ["noncanonical timestamp", (data: Record<string, unknown>) => { data.created_at_utc = "2026-08-05 12:00:00"; }],
    ["timestamp order", (data: Record<string, unknown>) => { data.modified_at_utc = "2026-08-05T11:59:59Z"; }],
    ["microsecond timestamp order", (data: Record<string, unknown>) => {
      data.created_at_utc = "2026-08-05T12:00:00.000002Z";
      data.modified_at_utc = "2026-08-05T12:00:00.000001Z";
    }],
    ["unordered domain", (data: Record<string, unknown>) => { data.time_domain_s = [1, 1]; }],
    ["nonfinite domain", (data: Record<string, unknown>) => { data.time_domain_s = [0, Number.NaN]; }],
    ["empty assignments", (data: Record<string, unknown>) => { data.assignments = []; }],
  ])("rejects %s", (_label, mutate) => {
    const data = literalPayload();
    mutate(data);
    expect(() => PrescribedTorqueProfile.fromJsonObject(data)).toThrow();
  });

  it("rejects duplicate joint assignments and malformed assignment fields", () => {
    const duplicate = literalPayload();
    const assignments = duplicate.assignments as Array<Record<string, unknown>>;
    assignments.push(structuredClone(assignments[0]));
    expect(() => PrescribedTorqueProfile.fromJsonObject(duplicate)).toThrow(/unique/i);

    const malformed = literalPayload();
    (malformed.assignments as Array<Record<string, unknown>>)[0].extra = true;
    expect(() => PrescribedTorqueProfile.fromJsonObject(malformed)).toThrow(/fields/i);

    const unstable = literalPayload();
    (unstable.assignments as Array<Record<string, unknown>>)[0].joint_id = "bad joint";
    expect(() => PrescribedTorqueProfile.fromJsonObject(unstable)).toThrow(/joint_id/i);
  });

  it("rejects empty and nonfinite polynomial data", () => {
    expect(() => new TorquePolynomial([])).toThrow(/empty/i);
    expect(() => new TorquePolynomial([1, Number.NaN])).toThrow(/finite/i);
    expect(() => evaluateAscendingPolynomial([Number.MAX_VALUE, Number.MAX_VALUE], 2)).toThrow(/finite/i);
  });

  it("rejects nonfinite or out-of-range fit quality metadata", () => {
    const valid = {
      degree: 1,
      rmseNm: 0,
      maxAbsErrorNm: 0,
      rSquared: 1,
      conditionNumber: 1,
    };
    expect(() => new FitMetadata({ ...valid, rmseNm: Number.NaN })).toThrow(/finite/i);
    expect(() => new FitMetadata({ ...valid, rSquared: 1.01 })).toThrow(/r_squared/i);
    expect(() => new FitMetadata({ ...valid, conditionNumber: 0 })).toThrow(/condition/i);
  });

  it("rejects malformed JSON and duplicate JSON object fields", () => {
    expect(() => PrescribedTorqueProfile.loads("not-json")).toThrow(/JSON/i);
    const duplicated = JSON.stringify(parityPayload).replace(
      '"profile_id":"profile.web_parity.v1"',
      '"profile_id":"profile.web_parity.v1","profile_id":"duplicate"',
    );
    expect(() => PrescribedTorqueProfile.loads(duplicated)).toThrow(/duplicate/i);
  });

  it("constructs unique assignments from immutable polynomial values", () => {
    const polynomial = new TorquePolynomial([1, 2]);
    const assignment = new JointTorqueAssignment("joint.one", polynomial);
    expect(assignment.polynomial.evaluate(0.5)).toBe(2);
    expect(Object.isFrozen(polynomial.coefficients)).toBe(true);
    expect(Object.isFrozen(assignment)).toBe(true);
  });
});
