import { describe, expect, it } from "vitest";

import {
  parseCapabilitySampleObservation,
  type CapabilitySampleObservation,
} from "./capabilityObservationContract";
import {
  CapabilityObservationEnsembleBuilder,
  buildCapabilityObservationEnsemble,
  stableCapabilityObservationEnsembleJson,
} from "./capabilityObservationEnsemble";
import type { TargetDefinition } from "./capabilityContract";
import { flightMetricCatalog } from "./ballFlightMetricContract";

const target: TargetDefinition = {
  kind: "green", distanceM: 100, lateralM: 2, radiusM: 10,
  bandHalfLengthM: 15, halfWidthM: 16,
};

const observation = (
  attemptOrdinal: number,
  effectiveStatus: "complete" | "no_impact" | "failed",
): CapabilitySampleObservation => ({
  schemaVersion: "capability-sample-observation/v1",
  problemId: "observer-fixture", attemptOrdinal,
  attemptedCount: attemptOrdinal + 1, totalCount: 2,
  candidateOrdinal: attemptOrdinal, clubCandidateOrdinal: attemptOrdinal,
  sampleOrdinal: 0, clubId: attemptOrdinal === 0 ? "iron-7" : "driver",
  parameters: [
    { parameterId: "ball_speed", unit: "m/s", nominalValue: 50, perturbedValue: 51 + attemptOrdinal },
    { parameterId: "launch_angle", unit: "deg", nominalValue: 12, perturbedValue: 13 + attemptOrdinal },
  ],
  sourceStatus: effectiveStatus, effectiveStatus,
  reasonCode: effectiveStatus === "complete" ? null : "missed_ball",
  sourceReason: effectiveStatus === "complete" ? null : "missed_ball",
  metrics: effectiveStatus === "complete" ? [
    { metricId: "carry_distance", value: 105, provenance: "fixture.carry" },
    { metricId: "carry_offline", value: 5, provenance: "fixture.offline" },
    { metricId: "apex_height", value: 31, provenance: "fixture.apex" },
  ] : [],
});

const build = (observations: readonly CapabilitySampleObservation[], maxRows = 4) =>
  buildCapabilityObservationEnsemble({
    observations, target, maxRows, sourceProvenance: "fixture/evaluator-v1",
  });

const records = (value: unknown): Record<string, unknown>[] =>
  value as Record<string, unknown>[];

const mutate = (
  apply: (draft: Record<string, unknown>) => void,
): CapabilitySampleObservation => {
  const draft = structuredClone(observation(0, "complete")) as unknown as Record<string, unknown>;
  apply(draft);
  return draft as unknown as CapabilitySampleObservation;
};

describe("capability observation scalar ensemble", () => {
  it("declares every parameter/metric and emits deterministic nullable rows", () => {
    const result = build([observation(1, "no_impact"), observation(0, "complete")]);
    expect(result.rows.map(({ row_id }) => row_id)).toEqual([
      "series:candidate%3A0%2Fclub%3Airon-7/trial:0",
      "series:candidate%3A1%2Fclub%3Adriver/trial:0",
    ]);
    const variableKeys = result.variables.map(({ key }) => key);
    expect(variableKeys.slice(0, 4)).toEqual([
      "nominal.ball_speed", "perturbed.ball_speed",
      "nominal.launch_angle", "perturbed.launch_angle",
    ]);
    expect(variableKeys.filter((key) => key.startsWith("metric.")))
      .toEqual(flightMetricCatalog().definitions
        .filter(({ signRule }) => signRule !== "vector_components")
        .map(({ metricId }) => `metric.${metricId}`));
    expect(variableKeys.slice(-6)).toEqual([
      "target_downrange_residual", "target_lateral_residual", "target_residual",
      "target_signed_distance", "target_solver_residual", "target_contains",
    ]);
    const [complete, noImpact] = result.rows;
    expect(complete.values).toMatchObject({
      "nominal.ball_speed": 50, "perturbed.ball_speed": 51,
      "metric.carry_distance": 105, target_downrange_residual: 5,
      target_lateral_residual: 3, target_contains: 1,
    });
    expect(complete.values.target_residual).toBeCloseTo(Math.sqrt(34));
    expect(complete.values.target_signed_distance).toBeCloseTo(Math.sqrt(34) - 10);
    expect(complete.values.target_solver_residual).toBeCloseTo(0.05 * Math.sqrt(34));
    expect(complete.values["metric.total_distance"]).toBeNull();
    expect(complete.attributes).toMatchObject({
      effective_status: "complete", source_status: "complete",
      "metric.carry_distance.provenance": "fixture.carry",
    });
    expect(noImpact.cohort).toBe("no_impact");
    expect(Object.entries(noImpact.values)
      .filter(([key]) => key.startsWith("metric.") || key.startsWith("target_"))
      .every(([, value]) => value === null)).toBe(true);
    expect(noImpact.attributes).toMatchObject({
      effective_status: "no_impact", source_status: "no_impact",
      reason_code: "missed_ball", source_reason: "missed_ball",
      "metric.carry_distance.provenance": null,
    });
  });
});

describe("capability observation ensemble bounds", () => {
  it("rejects row overflow instead of truncating and bounds maxRows", () => {
    expect(() => build([observation(0, "complete"), observation(1, "failed")], 1))
      .toThrow(/exceeds maxRows/);
    for (const maxRows of [0, -1, 100_001, 1.5, true] as unknown as number[]) {
      expect(() => build([observation(0, "complete")], maxRows)).toThrow(/maxRows/);
    }
    const builder = new CapabilityObservationEnsembleBuilder({
      target, maxRows: 1, sourceProvenance: "fixture/evaluator-v1",
    });
    builder.accept(observation(0, "complete"));
    expect(() => builder.accept(observation(1, "failed"))).toThrow(/maxRows/);
    expect(builder.retainedCount).toBe(1);
    expect(builder.build().rows).toHaveLength(1);
  });

  it("rejects a non-contract observation without retaining it", () => {
    const builder = new CapabilityObservationEnsembleBuilder({
      target, maxRows: 2, sourceProvenance: "fixture/evaluator-v1",
    });
    const invalid = {
      ...observation(0, "complete"), schemaVersion: "capability-sample-observation/v0",
    } as unknown as CapabilitySampleObservation;
    expect(() => builder.accept(invalid)).toThrow(/schemaVersion/);
    expect(builder.retainedCount).toBe(0);
  });
});

describe("capability observation validation", () => {
  it("deep-validates every observation field before retention", () => {
    const invalid = [
      mutate((row) => { row.problemId = " "; }),
      mutate((row) => { row.attemptOrdinal = 0.5; }),
      mutate((row) => { row.totalCount = 0; }),
      mutate((row) => { records(row.parameters)[1].parameterId = "ball_speed"; }),
      mutate((row) => { records(row.parameters)[0].nominalValue = Number.NaN; }),
      mutate((row) => { records(row.metrics)[1].metricId = "carry_distance"; }),
      mutate((row) => { records(row.metrics)[0].metricId = "unknown_metric"; }),
      mutate((row) => { records(row.metrics)[0].value = Number.POSITIVE_INFINITY; }),
      mutate((row) => { records(row.metrics)[0].provenance = " "; }),
      mutate((row) => { row.sourceReason = "unexpected"; }),
      mutate((row) => { row.sourceStatus = "failed"; row.effectiveStatus = "complete"; }),
    ];
    invalid.forEach((row) => {
      const builder = new CapabilityObservationEnsembleBuilder({
        target, maxRows: 2, sourceProvenance: "fixture/evaluator-v1",
      });
      expect(() => builder.accept(row)).toThrow();
      expect(builder.retainedCount).toBe(0);
    });
  });

  it("defensively copies and freezes accepted observations", () => {
    const raw = structuredClone(observation(0, "complete"));
    const mutableTarget = structuredClone(target);
    const parsed = parseCapabilitySampleObservation(raw);
    expect(Object.isFrozen(parsed)).toBe(true);
    expect(Object.isFrozen(parsed.parameters)).toBe(true);
    expect(Object.isFrozen(parsed.parameters[0])).toBe(true);
    const builder = new CapabilityObservationEnsembleBuilder({
      target: mutableTarget, maxRows: 2, sourceProvenance: "fixture/evaluator-v1",
    });
    builder.accept(raw);
    (raw as { problemId: string }).problemId = "mutated";
    (raw.parameters as unknown as { parameterId: string }[])[0].parameterId = "mutated";
    (mutableTarget as { distanceM: number }).distanceM = 999;
    expect(builder.build().result_id).toBe("observer-fixture");
    expect(builder.build().variables[0].key).toBe("nominal.ball_speed");
    expect(builder.build().rows[0].values.target_downrange_residual).toBe(5);
  });

  it("enforces the exact status-to-metrics contract", () => {
    const incompleteLanding = mutate((row) => {
      row.sourceStatus = "complete"; row.effectiveStatus = "failed";
      row.reasonCode = "missing_required_landing_metrics"; row.sourceReason = null;
      row.metrics = [records(row.metrics)[0]];
    });
    expect(() => build([incompleteLanding])).not.toThrow();
    const invalid = [
      mutate((row) => { row.metrics = [records(row.metrics)[0]]; }),
      mutate((row) => {
        row.sourceStatus = "no_impact"; row.effectiveStatus = "no_impact";
        row.reasonCode = "missed_ball"; row.sourceReason = "missed_ball";
      }),
      mutate((row) => {
        row.sourceStatus = "failed"; row.effectiveStatus = "failed";
        row.reasonCode = "solver_failed"; row.sourceReason = "solver_failed";
      }),
      mutate((row) => {
        row.sourceStatus = "nonconverged"; row.effectiveStatus = "failed";
        row.reasonCode = "iteration_limit"; row.sourceReason = "iteration_limit";
      }),
      mutate((row) => {
        row.sourceStatus = null; row.effectiveStatus = "failed";
        row.reasonCode = "evaluator_exception"; row.sourceReason = null;
      }),
      mutate((row) => {
        row.sourceStatus = null; row.effectiveStatus = "failed";
        row.reasonCode = "invalid_evaluator_result"; row.sourceReason = null;
      }),
    ];
    records(invalid[0].metrics).splice(1);
    invalid.forEach((row) => expect(() => build([row])).toThrow(/status|metrics|carry/));
  });
});

describe("capability observation identity", () => {
  it("rejects ambiguous identities and declaration drift", () => {
    const duplicate = { ...observation(1, "complete"), attemptOrdinal: 0, attemptedCount: 1 };
    expect(() => build([observation(0, "complete"), duplicate])).toThrow(/attemptOrdinal/);
    const prefix = { ...observation(0, "complete"), totalCount: 3 };
    const gap = { ...observation(2, "complete"), totalCount: 3 };
    expect(() => build([prefix, gap]))
      .toThrow(/contiguous prefix/);
    const drift = {
      ...observation(1, "complete"),
      parameters: [
        { parameterId: "launch_angle", unit: "deg", nominalValue: 12, perturbedValue: 13 },
        { parameterId: "ball_speed", unit: "mph", nominalValue: 50, perturbedValue: 51 },
      ],
    } satisfies CapabilitySampleObservation;
    expect(() => build([observation(0, "complete"), drift])).toThrow(/parameter declarations/);
  });

  it("cannot confuse structurally different declarations containing delimiters", () => {
    const first = mutate((row) => {
      row.clubId = "same-club";
      row.parameters = [
        { parameterId: "a\u0000b\u0001c", unit: "d", nominalValue: 1, perturbedValue: 2 },
      ];
    });
    const second = structuredClone(first);
    (second as { attemptOrdinal: number }).attemptOrdinal = 1;
    (second as { attemptedCount: number }).attemptedCount = 2;
    (second as { candidateOrdinal: number }).candidateOrdinal = 1;
    (second as { clubCandidateOrdinal: number }).clubCandidateOrdinal = 1;
    (second as { parameters: CapabilitySampleObservation["parameters"] }).parameters = [
      { parameterId: "a", unit: "b", nominalValue: 1, perturbedValue: 2 },
      { parameterId: "c", unit: "d", nominalValue: 3, perturbedValue: 4 },
    ];
    expect(() => build([first, second])).toThrow(/parameter declarations/);
  });

  it("derives labels using ASCII-only initial-letter casing", () => {
    const unicode = mutate((row) => {
      row.parameters = [
        { parameterId: "alpha_\u00e9LAN_\u03c9mega", unit: "1", nominalValue: 1, perturbedValue: 2 },
      ];
    });
    expect(build([unicode]).variables.slice(0, 2).map(({ label }) => label)).toEqual([
      "Nominal Alpha \u00e9LAN \u03c9mega", "Perturbed Alpha \u00e9LAN \u03c9mega",
    ]);
  });
});

describe("capability observation wire parity", () => {
  it("has a stable runtime-neutral wire representation", async () => {
    const json = stableCapabilityObservationEnsembleJson(
      build([observation(1, "no_impact"), observation(0, "complete")]),
    );
    const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(json));
    expect(Array.from(new Uint8Array(digest), (byte) =>
      byte.toString(16).padStart(2, "0")).join(""))
      .toBe("df36f765afdf508d00a3d264911ce5b6f07e25da3744b187596d67487ea3be5f");
  });

  it("has stable Unicode code-point ordering across runtimes", async () => {
    const unicode = mutate((row) => {
      row.problemId = "観測-ß"; row.clubId = "ドライバー"; row.totalCount = 1;
      row.parameters = [
        { parameterId: "zeta", unit: "m/s", nominalValue: 1, perturbedValue: 2 },
        { parameterId: "Ω", unit: "deg", nominalValue: 3, perturbedValue: 4 },
      ];
      records(row.metrics)[0].provenance = "測定.é";
    });
    const json = stableCapabilityObservationEnsembleJson(build([unicode]));
    expect(json).toContain("観測-ß");
    const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(json));
    expect(Array.from(new Uint8Array(digest), (byte) =>
      byte.toString(16).padStart(2, "0")).join(""))
      .toBe("18086b5e97d576598bbfa63407b6eda786a3a7ce20509654de282400bd32efd0");
  });

  it("emits canonical raw number tokens at rounding and exponent edges", () => {
    const edge = mutate((row) => {
      row.totalCount = 1;
      row.parameters = [
        { parameterId: "half", unit: "1", nominalValue: 1.000000000005, perturbedValue: -1.000000000005 },
        { parameterId: "tiny", unit: "1", nominalValue: -0, perturbedValue: 0.0000001 },
        { parameterId: "large", unit: "1", nominalValue: 1e21, perturbedValue: 1e-12 },
      ];
    });
    const json = stableCapabilityObservationEnsembleJson(build([edge]));
    expect(json).toContain('"nominal.half":1.00000000001');
    expect(json).toContain('"perturbed.half":-1.00000000001');
    expect(json).toContain('"nominal.tiny":0');
    expect(json).toContain('"perturbed.tiny":0.0000001');
    expect(json).toContain('"nominal.large":1000000000000000000000');
    expect(json).toContain('"perturbed.large":0');
    expect(json).not.toMatch(/"(?:nominal|perturbed)\.[^"]+":"/);
    expect(json).not.toContain("e+21");
  });
});
