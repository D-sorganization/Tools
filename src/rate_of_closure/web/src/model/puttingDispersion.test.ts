/**
 * Putt dispersion summary + `swing_sim.putt_dispersion/1` wire gates —
 * the TypeScript mirror of the summary/wire half of
 * `swing_sim/putting/tests/test_variation.py` (#4800 P5).
 *
 * The statistics block pins the Python reference values for the shared
 * Welford sample spread and NumPy's linear-interpolation percentiles,
 * so the two runtimes summarize a cohort identically. Monte-Carlo
 * *execution* stays Python-authoritative (one canonical seeded
 * sampler), so there is nothing to twin there — and deliberately so.
 */

import { describe, expect, it } from "vitest";

import {
  PUTT_DISPERSION_FORMAT,
  finiteSampleStandardDeviation,
  percentile,
  puttDispersionFromJson,
  puttDispersionToJson,
  summarizePuttOutcomes,
  type PuttDispersionReport,
  type PuttOutcome,
} from "./puttingDispersion";

function outcome(patch: Partial<PuttOutcome> = {}): PuttOutcome {
  return {
    holed: false,
    startAzimuthDeg: 0.0,
    leaveDistanceM: 0.5,
    totalDistanceM: 3.2,
    breakM: 0.1,
    captureMarginM: -0.02,
    ...patch,
  };
}

const COHORT: PuttOutcome[] = [
  outcome({ holed: true, leaveDistanceM: 0.0, startAzimuthDeg: 0.12 }),
  outcome({ leaveDistanceM: 0.55, startAzimuthDeg: -0.4 }),
  outcome({ leaveDistanceM: 0.03, startAzimuthDeg: 0.9 }),
  outcome({ leaveDistanceM: 1.4, startAzimuthDeg: -1.1 }),
];

describe("shared statistics parity", () => {
  const values = [0.12, 0.55, 0.03, 1.4, 0.77, 0.21];

  it("matches the Python Welford sample spread", () => {
    expect(finiteSampleStandardDeviation(values)).toBeCloseTo(
      0.5164752333526426,
      15,
    );
  });

  it("matches NumPy's linear-interpolation percentiles", () => {
    expect(percentile(values, 50)).toBeCloseTo(0.38, 15);
    expect(percentile(values, 95)).toBeCloseTo(1.2425, 15);
    expect(percentile(values, 5)).toBeCloseTo(0.0525, 15);
  });

  it("has no spread below two samples", () => {
    expect(finiteSampleStandardDeviation([1.0])).toBeNaN();
  });

  it("refuses a non-finite cohort", () => {
    expect(() => finiteSampleStandardDeviation([1.0, Number.NaN])).toThrow();
  });
});

describe("putt dispersion summary", () => {
  it("needs two runs", () => {
    expect(() => summarizePuttOutcomes([outcome()])).toThrow();
  });

  it("refuses a holed putt that left something", () => {
    expect(() =>
      summarizePuttOutcomes([
        outcome({ holed: true, leaveDistanceM: 0.4 }),
        outcome(),
      ]),
    ).toThrow();
  });

  it("reports the holed fraction as the make percentage", () => {
    const summary = summarizePuttOutcomes(COHORT);
    expect(summary.nRuns).toBe(4);
    expect(summary.holedCount).toBe(1);
    expect(summary.makePercent).toBeCloseTo(25.0, 12);
  });

  it("reports the leave distribution", () => {
    const summary = summarizePuttOutcomes(COHORT);
    expect(summary.leaveMaxM).toBeCloseTo(1.4, 12);
    expect(summary.leaveMeanM).toBeCloseTo((0.0 + 0.55 + 0.03 + 1.4) / 4, 12);
    expect(summary.leaveP50M).toBeCloseTo(
      percentile([0, 0.55, 0.03, 1.4], 50),
      15,
    );
    expect(summary.leaveP95M).toBeGreaterThanOrEqual(summary.leaveP50M);
  });

  it("reports the start-line dispersion", () => {
    const summary = summarizePuttOutcomes(COHORT);
    expect(summary.startLineSigmaDeg).toBeCloseTo(
      finiteSampleStandardDeviation(COHORT.map((item) => item.startAzimuthDeg)),
      15,
    );
    expect(summary.startLineP05Deg).toBeLessThan(summary.startLineP95Deg);
  });

  it("has zero dispersion over an identical cohort", () => {
    const summary = summarizePuttOutcomes([outcome(), outcome(), outcome()]);
    expect(summary.startLineSigmaDeg).toBe(0);
    expect(summary.totalDistanceSigmaM).toBe(0);
  });
});

function report(): PuttDispersionReport {
  return {
    scenarioId: "p5-gate",
    seed: 8,
    variables: [
      {
        variableKey: "swing_sim.putting.aim_deg",
        distribution: "normal",
        scale: 0.6,
      },
      {
        variableKey: "swing_sim.putting.face_angle_deg",
        distribution: "normal",
        scale: 0.4,
      },
    ],
    summary: summarizePuttOutcomes(COHORT),
  };
}

describe("putt_dispersion/1 wire posture", () => {
  it("round-trips byte identically", () => {
    const text = puttDispersionToJson(report());
    expect(puttDispersionToJson(puttDispersionFromJson(text))).toBe(text);
  });

  it("declares its format and the varied inputs", () => {
    const payload = JSON.parse(puttDispersionToJson(report()));
    expect(payload.format).toBe(PUTT_DISPERSION_FORMAT);
    expect(
      payload.variables.map(
        (item: { variable_key: string }) => item.variable_key,
      ),
    ).toEqual([
      "swing_sim.putting.aim_deg",
      "swing_sim.putting.face_angle_deg",
    ]);
  });

  it("sorts keys", () => {
    const keys = Object.keys(JSON.parse(puttDispersionToJson(report())));
    expect(keys).toEqual([...keys].sort());
  });

  it("refuses an unknown field", () => {
    const payload = JSON.parse(puttDispersionToJson(report()));
    payload.extra = 1;
    expect(() => puttDispersionFromJson(JSON.stringify(payload))).toThrow();
  });

  it("refuses a wrong format", () => {
    const payload = JSON.parse(puttDispersionToJson(report()));
    payload.format = "swing_sim.putt_dispersion/2";
    expect(() => puttDispersionFromJson(JSON.stringify(payload))).toThrow();
  });

  it("refuses a non-finite summary value", () => {
    const payload = JSON.parse(puttDispersionToJson(report()));
    payload.summary.leave_mean_m = "NaN";
    expect(() => puttDispersionFromJson(JSON.stringify(payload))).toThrow();
  });

  it("refuses more holed runs than runs", () => {
    const payload = JSON.parse(puttDispersionToJson(report()));
    payload.summary.holed_count = payload.summary.n_runs + 1;
    expect(() => puttDispersionFromJson(JSON.stringify(payload))).toThrow();
  });

  it("refuses a non-positive declared scale", () => {
    const payload = JSON.parse(puttDispersionToJson(report()));
    payload.variables[0].scale = 0;
    expect(() => puttDispersionFromJson(JSON.stringify(payload))).toThrow();
  });

  it("refuses a negative seed", () => {
    expect(() => puttDispersionToJson({ ...report(), seed: -1 })).toThrow();
  });
});
