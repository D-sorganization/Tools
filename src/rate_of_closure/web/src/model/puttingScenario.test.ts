/**
 * Deterministic putt-evaluation gates (#4800 P5, twinned in P7).
 *
 * Mirrors the deterministic half of
 * `swing_sim/putting/tests/test_variation.py`. The Monte-Carlo sampler
 * itself is Python-authoritative and deliberately not twinned, so the
 * gates that need a seeded draw are replaced by an explicit offset
 * sweep — the assertions they carry are unchanged:
 *
 * - **The start line follows P1's effective-mass law exactly.** Every
 *   evaluated putt is checked against the closed form
 *   `aim + face + atan2((2/7) sin(fp), T cos(fp))` with
 *   `T = (1+e) M_eff / (M_eff + m)`.
 * - **Higher MOI tightens the start line by the MOI ratio.**
 * - **A square stroke makes the start line MOI-free.**
 * - Fail-closed refusals, and the outcome vocabulary
 *   (`puttOutcome` -> `summarizePuttOutcomes`).
 */

import { describe, expect, it } from "vitest";

import {
  finiteSampleStandardDeviation,
  summarizePuttOutcomes,
} from "./puttingDispersion";
import { planarSurface } from "./puttingGreen";
import {
  GOLF_BALL_MASS_KG,
  MINIMAL_PUTTERS,
  DEFAULT_PUTTER_MOI_KG_M2,
} from "./putting";
import {
  PUTTING_RESULT_KERNEL,
  puttingResultToJson,
  type PuttingResultProvenance,
} from "./puttingResultWire";
import {
  PUTT_AIM_KEY,
  PUTT_FACE_KEY,
  PUTT_PATH_KEY,
  PUTT_SPEED_KEY,
  PUTT_STRIKE_TOE_KEY,
  PUTT_VARIABLE_KEYS,
  evaluatePutt,
  evaluatePuttWithTrajectory,
  puttOutcome,
  puttStroke,
  strokeBaseValues,
  type PuttScenario,
  type PuttStroke,
} from "./puttingScenario";

const BLADE = MINIMAL_PUTTERS[0];
const FLAT = planarSurface(0, 0);
const ROLLING_CAP = 2.0 / 7.0;

const PROVENANCE: PuttingResultProvenance = {
  putterSource: "minimal",
  putterName: "Blade Putter",
  strokeSource: "declared",
  captureModel: "effective_radius",
  putterMeshSha256: null,
  putterLibraryName: null,
  strokeSourceId: null,
  kernel: PUTTING_RESULT_KERNEL,
};

function scenario(
  overrides: Partial<PuttScenario> & { stroke?: PuttStroke } = {},
): PuttScenario {
  return {
    scenarioId: "p7-gate",
    putter: BLADE,
    stroke: puttStroke(1.6),
    surface: FLAT,
    stimpFt: 10.0,
    holeDistanceM: 3.0,
    provenance: PROVENANCE,
    ...overrides,
  };
}

/** P1's effective-mass start line (the Python gate's closed form). */
function closedFormStartDeg(
  aimDeg: number,
  faceDeg: number,
  pathDeg: number,
  toeMm: number,
  moi: number,
): number {
  const radiusM = Math.abs(toeMm) * 1e-3;
  const headMassEff =
    radiusM === 0.0
      ? BLADE.headMassKg
      : 1.0 / (1.0 / BLADE.headMassKg + (radiusM * radiusM) / moi);
  const transfer =
    ((1.0 + BLADE.cor) * headMassEff) / (headMassEff + GOLF_BALL_MASS_KG);
  const faceToPath = ((pathDeg - faceDeg) * Math.PI) / 180.0;
  const deflection = Math.atan2(
    ROLLING_CAP * Math.sin(faceToPath),
    transfer * Math.cos(faceToPath),
  );
  return aimDeg + faceDeg + (deflection * 180.0) / Math.PI;
}

/** A deterministic toe-offset sweep standing in for the seeded draw. */
const TOE_SWEEP = [-12, -8, -4, -1.5, 0, 1.5, 4, 8, 12];

function startLinesForMoi(moi: number, stroke: PuttStroke): number[] {
  return TOE_SWEEP.map(
    (toeMm) =>
      evaluatePutt(
        scenario({
          stroke: { ...stroke, strikeOffsetToeMm: toeMm },
          headMoiKgM2: moi,
        }),
      ).startAzimuthDeg,
  );
}

describe("registry vocabulary", () => {
  it("names the five putting variables the Python registry owns", () => {
    expect([...PUTT_VARIABLE_KEYS].sort()).toEqual(
      [
        "swing_sim.putting.aim_deg",
        "swing_sim.putting.clubhead_speed_mps",
        "swing_sim.putting.face_angle_deg",
        "swing_sim.putting.path_angle_deg",
        "swing_sim.putting.strike_offset_toe_mm",
      ].sort(),
    );
  });

  it("maps a stroke onto the registry-keyed base a plan varies about", () => {
    const stroke = puttStroke(1.7, {
      aimDeg: 0.5,
      faceAngleDeg: -0.25,
      pathAngleDeg: 1.0,
      strikeOffsetToeMm: 3.0,
    });
    expect(strokeBaseValues(stroke)).toEqual({
      [PUTT_SPEED_KEY]: 1.7,
      [PUTT_AIM_KEY]: 0.5,
      [PUTT_FACE_KEY]: -0.25,
      [PUTT_PATH_KEY]: 1.0,
      [PUTT_STRIKE_TOE_KEY]: 3.0,
    });
  });
});

describe("deterministic evaluation", () => {
  it("is byte-identical on repeat", () => {
    const first = puttingResultToJson(evaluatePutt(scenario()));
    const second = puttingResultToJson(evaluatePutt(scenario()));
    expect(second).toBe(first);
  });

  it("returns the same document with or without the trajectory", () => {
    const withPath = evaluatePuttWithTrajectory(scenario());
    expect(puttingResultToJson(withPath.document)).toBe(
      puttingResultToJson(evaluatePutt(scenario())),
    );
    expect(withPath.result.pathXM.length).toBeGreaterThan(1);
    expect(withPath.result.pathXM.length).toBe(withPath.result.timesS.length);
    expect(withPath.launch.startAzimuthDeg).toBe(
      withPath.document.startAzimuthDeg,
    );
  });

  it("reports the retained samples the summary was measured from", () => {
    const { result, document } = evaluatePuttWithTrajectory(scenario());
    expect(document.timeS).toBe(result.timesS[result.timesS.length - 1]);
    expect(document.totalDistanceM).toBe(result.totalDistanceM);
    expect(document.finalBreakM).toBe(result.breakM);
  });
});

describe("effective-mass law", () => {
  it("matches the closed-form start line for every strike offset", () => {
    const moi = DEFAULT_PUTTER_MOI_KG_M2;
    const stroke = puttStroke(1.6, { faceAngleDeg: 0.0, pathAngleDeg: 1.5 });
    const observed = startLinesForMoi(moi, stroke);
    TOE_SWEEP.forEach((toeMm, index) => {
      const expected = closedFormStartDeg(0.0, 0.0, 1.5, toeMm, moi);
      expect(Math.abs(observed[index] - expected)).toBeLessThanOrEqual(
        1e-12 * Math.max(Math.abs(expected), 1e-9),
      );
    });
  });

  it("tightens the start line by the MOI ratio", () => {
    const stroke = puttStroke(1.6, { faceAngleDeg: 0.0, pathAngleDeg: 1.5 });
    const lowMoi = 4.5e-4;
    const highMoi = 9.0e-4;
    const bladeSigma = finiteSampleStandardDeviation(
      startLinesForMoi(lowMoi, stroke),
    );
    const malletSigma = finiteSampleStandardDeviation(
      startLinesForMoi(highMoi, stroke),
    );
    expect(malletSigma).toBeLessThan(bladeSigma);
    expect(Math.abs(malletSigma / bladeSigma - lowMoi / highMoi)).toBeLessThan(
      1e-3,
    );
  });

  it("makes the start line MOI-free for a square stroke", () => {
    // face == path: the tangential impulse has no lever, so the
    // effective-mass reduction cannot move the start line at all.
    const stroke = puttStroke(1.6, { faceAngleDeg: 1.0, pathAngleDeg: 1.0 });
    const lines = startLinesForMoi(4.5e-4, stroke);
    expect(finiteSampleStandardDeviation(lines)).toBe(0.0);
  });
});

describe("fails closed", () => {
  it("refuses an unnamed scenario", () => {
    expect(() => evaluatePutt(scenario({ scenarioId: "  " }))).toThrow(
      /scenarioId/,
    );
  });

  it("refuses a provenance whose capture model is not the scenario's", () => {
    expect(() =>
      evaluatePutt(scenario({ captureModel: "speed_threshold" })),
    ).toThrow(/captureModel/);
  });

  it("refuses a non-finite stroke value", () => {
    expect(() => puttStroke(Number.NaN)).toThrow();
    expect(() => puttStroke(1.6, { aimDeg: Number.POSITIVE_INFINITY })).toThrow();
  });

  it("refuses a surface that is not a green surface", () => {
    expect(() =>
      evaluatePutt(
        scenario({
          surface: { kind: "hillside" } as unknown as PuttScenario["surface"],
        }),
      ),
    ).toThrow(/GreenSurface/);
  });

  it("refuses a draw outside the impact model's envelope", () => {
    expect(() =>
      evaluatePutt(scenario({ stroke: puttStroke(1.6, { aimDeg: 90 }) })),
    ).toThrow();
    expect(() => evaluatePutt(scenario({ stroke: puttStroke(50) }))).toThrow();
  });
});

describe("outcome vocabulary", () => {
  it("leaves nothing when the putt holes", () => {
    const holed = evaluatePutt(scenario({ stroke: puttStroke(1.55) }));
    if (!holed.holed) throw new Error("gate needs a holed reference putt");
    expect(puttOutcome(holed).leaveDistanceM).toBe(0.0);
    expect(puttOutcome(holed).captureMarginM).toBe(holed.captureMarginM);
  });

  it("reports the comebacker when the putt misses", () => {
    const missed = evaluatePutt(scenario({ stroke: puttStroke(1.0) }));
    expect(missed.holed).toBe(false);
    expect(puttOutcome(missed).leaveDistanceM).toBe(missed.missDistanceM);
  });

  it("summarizes a cohort with make percent as the holed fraction", () => {
    const outcomes = [-2.0, -1.0, 0.0, 1.0, 2.0].map((aimDeg) =>
      puttOutcome(
        evaluatePutt(scenario({ stroke: puttStroke(1.55, { aimDeg }) })),
      ),
    );
    const holed = outcomes.filter((outcome) => outcome.holed).length;
    const summary = summarizePuttOutcomes(outcomes);
    expect(summary.holedCount).toBe(holed);
    expect(summary.makePercent).toBeCloseTo((100.0 * holed) / 5, 12);
    expect(summary.nRuns).toBe(5);
    for (const outcome of outcomes) {
      if (outcome.holed) expect(outcome.leaveDistanceM).toBe(0.0);
    }
  });
});
