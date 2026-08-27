/**
 * Parity pins for the putting vertical (epic #4125, H3).
 *
 * The pinned reference putts mirror
 * `tests/rate_of_closure/test_putting.py::TestReferencePuttPins`
 * value-for-value — both sides use the same closed forms and the same
 * fixed-step RK4 (dt = 2 ms), so tolerances are tight, not banded.
 */

import { describe, expect, it } from "vitest";

import {
  captureSpeedMps,
  clubheadSpeedFromBackstroke,
  DEFAULT_PUTTER_MOI_KG_M2,
  GOLF_BALL_MASS_KG,
  GOLF_BALL_RADIUS_M,
  simulatePutt,
  STIMP_RELEASE_SPEED_MPS,
  stimpToRollingMu,
  strike,
} from "./putting";

/** The H1 club-library putter (350 g, 3 deg, COR 0.78). */
const PUTTER = {
  name: "Putter",
  headMassKg: 0.35,
  loftDeg: 3.0,
  cor: 0.78,
};

describe("strike — Python parity", () => {
  it("pins the reference launch (1.8 m/s clubhead)", () => {
    const launch = strike(PUTTER, 1.8);
    expect(launch.ballSpeedMps).toBeCloseTo(2.828565312464848, 10);
    expect(launch.launchAngleDeg).toBeCloseTo(3.5452147542505257, 8);
    expect(launch.horizontalSpeedMps).toBeCloseTo(2.8231523192738344, 10);
    expect(launch.spinRadS).toBeCloseTo(-3.153929533539754, 10);
  });

  it("zero loft gives a pure 1-D COR impulse", () => {
    const launch = strike({ ...PUTTER, loftDeg: 0 }, 2.0);
    const expected = (2.0 * 1.78 * 0.35) / (0.35 + 0.04593);
    expect(launch.ballSpeedMps).toBeCloseTo(expected, 12);
    expect(launch.spinRadS).toBeCloseTo(0, 12);
  });

  it("rejects out-of-range speeds", () => {
    expect(() => strike(PUTTER, 0)).toThrow();
    expect(() => strike(PUTTER, 2, -30)).toThrow();
  });
});

describe("strike 2-D — analytic gates (#4800 P1, Python twin)", () => {
  /** 2/7 rolling cap (swing_sim.impact SPHERE_ROLLING_CAP_FACTOR). */
  const CAP = 2.0 / 7.0;
  const transfer =
    ((1.0 + PUTTER.cor) * PUTTER.headMassKg) /
    (PUTTER.headMassKg + GOLF_BALL_MASS_KG);
  const BALL_MOI =
    0.4 * GOLF_BALL_MASS_KG * GOLF_BALL_RADIUS_M * GOLF_BALL_RADIUS_M;

  it("square face + square path launches straight with zero sidespin", () => {
    const launch = strike(PUTTER, 1.8);
    expect(launch.startAzimuthDeg).toBe(0.0);
    expect(launch.sidespinRadS).toBe(0.0);
  });

  it("aim rotates the start line without touching the solve", () => {
    const neutral = strike(PUTTER, 1.8);
    const aimed = strike(PUTTER, 1.8, 0.0, { aimDeg: 4.0 });
    expect(aimed.startAzimuthDeg).toBe(4.0);
    expect(aimed.sidespinRadS).toBe(0.0);
    expect(aimed.ballSpeedMps).toBe(neutral.ballSpeedMps);
    expect(aimed.launchAngleDeg).toBe(neutral.launchAngleDeg);
    expect(aimed.spinRadS).toBe(neutral.spinRadS);
  });

  it("face-only vs path-only split matches the 2/7 rolling-cap ratio", () => {
    const k = CAP / transfer;
    const angle = 2.0;
    const faceOnly = strike(PUTTER, 1.8, 0.0, { faceAngleDeg: angle });
    const pathOnly = strike(PUTTER, 1.8, 0.0, { pathAngleDeg: angle });
    const rad = (angle * Math.PI) / 180.0;
    const expectedPath =
      (Math.atan2(CAP * Math.sin(rad), transfer * Math.cos(rad)) * 180.0) /
      Math.PI;
    expect(pathOnly.startAzimuthDeg).toBeCloseTo(expectedPath, 12);
    expect(faceOnly.startAzimuthDeg! / angle).toBeCloseTo(1.0 - k, 3);
    expect(pathOnly.startAzimuthDeg! / angle).toBeCloseTo(k, 3);
    expect(faceOnly.startAzimuthDeg!).toBeGreaterThan(
      pathOnly.startAzimuthDeg!,
    );
    expect(pathOnly.startAzimuthDeg!).toBeGreaterThan(0.0);
  });

  it("face and path deflections are exact complements", () => {
    for (const angle of [0.5, 2.0, 8.0]) {
      const faceOnly = strike(PUTTER, 1.8, 0.0, { faceAngleDeg: angle });
      const pathOnly = strike(PUTTER, 1.8, 0.0, { pathAngleDeg: angle });
      expect(faceOnly.startAzimuthDeg! + pathOnly.startAzimuthDeg!).toBeCloseTo(
        angle,
        12,
      );
    }
  });

  it("sidespin sign and magnitude follow face-to-path", () => {
    const v = 2.0;
    const draw = strike(PUTTER, v, 0.0, { pathAngleDeg: 3.0 });
    const fade = strike(PUTTER, v, 0.0, { faceAngleDeg: 3.0 });
    const expected =
      ((1.0 - CAP) * v * Math.sin((3.0 * Math.PI) / 180.0)) /
      GOLF_BALL_RADIUS_M;
    expect(draw.sidespinRadS!).toBeGreaterThan(0.0);
    expect(fade.sidespinRadS!).toBeLessThan(0.0);
    expect(draw.sidespinRadS).toBeCloseTo(expected, 10);
    expect(fade.sidespinRadS).toBeCloseTo(-expected, 10);
    const aimed = strike(PUTTER, v, 0.0, { aimDeg: 5.0, pathAngleDeg: 3.0 });
    expect(aimed.sidespinRadS).toBe(draw.sidespinRadS);
  });

  it("ball speed decreases monotonically with strike offset", () => {
    const toeSpeeds = [0.0, 4.0, 8.0, 16.0, 32.0].map(
      (r) => strike(PUTTER, 2.0, 0.0, { strikeOffsetToeMm: r }).ballSpeedMps,
    );
    for (let i = 1; i < toeSpeeds.length; i++) {
      expect(toeSpeeds[i]).toBeLessThan(toeSpeeds[i - 1]);
    }
    const highSpeeds = [0.0, 3.0, 6.0, 12.0].map(
      (r) => strike(PUTTER, 2.0, 0.0, { strikeOffsetHighMm: r }).ballSpeedMps,
    );
    for (let i = 1; i < highSpeeds.length; i++) {
      expect(highSpeeds[i]).toBeLessThan(highSpeeds[i - 1]);
    }
    const combined = strike(PUTTER, 2.0, 0.0, {
      strikeOffsetToeMm: 8.0,
      strikeOffsetHighMm: 6.0,
    });
    const toeOnly = strike(PUTTER, 2.0, 0.0, { strikeOffsetToeMm: 8.0 });
    expect(combined.ballSpeedMps).toBeLessThan(toeOnly.ballSpeedMps);
  });

  it("off-center effective mass matches the impact-package formula", () => {
    const flat = { ...PUTTER, loftDeg: 0 };
    const rM = 10.0e-3;
    const launch = strike(flat, 2.0, 0.0, { strikeOffsetToeMm: 10.0 });
    const mEff =
      1.0 / (1.0 / flat.headMassKg + (rM * rM) / DEFAULT_PUTTER_MOI_KG_M2);
    const expected =
      (2.0 * (1.0 + flat.cor) * mEff) / (mEff + GOLF_BALL_MASS_KG);
    expect(launch.ballSpeedMps).toBeCloseTo(expected, 12);
  });

  it("head MOI hook scales the off-center loss", () => {
    const base = strike(PUTTER, 2.0, 0.0, { strikeOffsetToeMm: 10.0 });
    const explicit = strike(PUTTER, 2.0, 0.0, {
      strikeOffsetToeMm: 10.0,
      headMoiKgM2: DEFAULT_PUTTER_MOI_KG_M2,
    });
    expect(explicit.ballSpeedMps).toBe(base.ballSpeedMps);
    const lowMoi = strike(PUTTER, 2.0, 0.0, {
      strikeOffsetToeMm: 10.0,
      headMoiKgM2: 2.0e-4,
    });
    const highMoi = strike(PUTTER, 2.0, 0.0, {
      strikeOffsetToeMm: 10.0,
      headMoiKgM2: 9.0e-4,
    });
    expect(lowMoi.ballSpeedMps).toBeLessThan(base.ballSpeedMps);
    expect(highMoi.ballSpeedMps).toBeGreaterThan(base.ballSpeedMps);
  });

  it("center strike ignores head MOI", () => {
    const neutral = strike(PUTTER, 2.0);
    const withMoi = strike(PUTTER, 2.0, 0.0, { headMoiKgM2: 2.0e-4 });
    expect(withMoi.ballSpeedMps).toBe(neutral.ballSpeedMps);
    expect(withMoi.spinRadS).toBe(neutral.spinRadS);
  });

  it("attack angle square to the face kills spin", () => {
    const launch = strike(PUTTER, 2.0, 0.0, {
      attackAngleDeg: PUTTER.loftDeg,
    });
    expect(launch.spinRadS).toBeCloseTo(0.0, 12);
    expect(launch.launchAngleDeg).toBeCloseTo(PUTTER.loftDeg, 10);
  });

  it("hitting down adds backspin", () => {
    const down = strike(PUTTER, 2.0, 0.0, { attackAngleDeg: -3.0 });
    const level = strike(PUTTER, 2.0);
    const up = strike(PUTTER, 2.0, 0.0, { attackAngleDeg: 2.0 });
    expect(down.spinRadS).toBeLessThan(level.spinRadS);
    expect(level.spinRadS).toBeLessThan(up.spinRadS);
    expect(up.spinRadS).toBeLessThan(0.0);
  });

  it("energy is never created", () => {
    for (const face of [-10.0, 0.0, 10.0]) {
      for (const path of [-10.0, 0.0, 10.0]) {
        for (const attack of [-5.0, 0.0, 5.0]) {
          for (const toe of [0.0, 15.0]) {
            for (const speed of [0.5, 3.0]) {
              const launch = strike(PUTTER, speed, 0.0, {
                faceAngleDeg: face,
                pathAngleDeg: path,
                attackAngleDeg: attack,
                strikeOffsetToeMm: toe,
              });
              const ballKe =
                0.5 * GOLF_BALL_MASS_KG * launch.ballSpeedMps ** 2 +
                0.5 *
                  BALL_MOI *
                  (launch.spinRadS ** 2 + launch.sidespinRadS! ** 2);
              const headKe = 0.5 * PUTTER.headMassKg * speed ** 2;
              expect(ballKe).toBeLessThan(headKe);
            }
          }
        }
      }
    }
  });

  it("defaults are bit-identical to the legacy 1-D solve", () => {
    for (const speed of [0.3, 1.0, 1.8, 3.2]) {
      for (const lean of [-2.0, 0.0, 1.5]) {
        const launch = strike(PUTTER, speed, lean);
        const delta = ((PUTTER.loftDeg + lean) * Math.PI) / 180.0;
        const massRatio =
          PUTTER.headMassKg / (PUTTER.headMassKg + GOLF_BALL_MASS_KG);
        const legacyTransfer = (1.0 + PUTTER.cor) * massRatio;
        const vNormal = legacyTransfer * speed * Math.cos(delta);
        const uTangential = speed * Math.sin(delta);
        const vTangential = CAP * uTangential;
        const spin = (-(1.0 - CAP) * uTangential) / GOLF_BALL_RADIUS_M;
        const horizontal =
          vNormal * Math.cos(delta) - vTangential * Math.sin(delta);
        const vertical =
          vNormal * Math.sin(delta) + vTangential * Math.cos(delta);
        expect(launch.ballSpeedMps).toBe(Math.hypot(horizontal, vertical));
        expect(launch.launchAngleDeg).toBe(
          (Math.atan2(vertical, horizontal) * 180.0) / Math.PI,
        );
        expect(launch.horizontalSpeedMps).toBe(horizontal);
        expect(launch.spinRadS).toBe(spin);
        expect(launch.startAzimuthDeg).toBe(0.0);
        expect(launch.sidespinRadS).toBe(0.0);
      }
    }
  });

  it("rejects out-of-range 2-D parameters", () => {
    expect(() => strike(PUTTER, 2, 0, { aimDeg: 60 })).toThrow(/aimDeg/);
    expect(() => strike(PUTTER, 2, 0, { faceAngleDeg: 25 })).toThrow(
      /faceAngleDeg/,
    );
    expect(() => strike(PUTTER, 2, 0, { pathAngleDeg: -25 })).toThrow(
      /pathAngleDeg/,
    );
    expect(() => strike(PUTTER, 2, 0, { attackAngleDeg: 15 })).toThrow(
      /attackAngleDeg/,
    );
    expect(() => strike(PUTTER, 2, 0, { strikeOffsetToeMm: 50 })).toThrow(
      /strikeOffsetToeMm/,
    );
    expect(() => strike(PUTTER, 2, 0, { strikeOffsetHighMm: -30 })).toThrow(
      /strikeOffsetHighMm/,
    );
    expect(() => strike(PUTTER, 2, 0, { headMoiKgM2: 0 })).toThrow(/MOI/);
    expect(() => strike(PUTTER, 2, 0, { headMoiKgM2: Number.NaN })).toThrow(
      /MOI/,
    );
  });
});

describe("stimpmeter — Python parity", () => {
  it("derives the quoted ~1.83 m/s release speed", () => {
    expect(STIMP_RELEASE_SPEED_MPS).toBeCloseTo(1.8287317526214812, 10);
  });

  it("round-trips stimp through the roll-out formula", () => {
    for (const stimp of [6, 10, 14]) {
      const mu = stimpToRollingMu(stimp);
      const rolloutFt =
        (STIMP_RELEASE_SPEED_MPS * STIMP_RELEASE_SPEED_MPS) /
        (2 * mu * 9.80665) /
        0.3048;
      expect(rolloutFt).toBeCloseTo(stimp, 10);
    }
  });

  it("pins mu at stimp 10", () => {
    expect(stimpToRollingMu(10)).toBeCloseTo(0.05594153480923128, 12);
  });
});

describe("capture bound", () => {
  it("pins R sqrt(g / 2r) ~= 0.82 m/s", () => {
    expect(captureSpeedMps()).toBeCloseTo(0.8186396513958939, 12);
    expect(GOLF_BALL_RADIUS_M).toBeCloseTo(0.021335, 6);
  });
});

describe("simulatePutt — Python parity", () => {
  const launch = strike(PUTTER, 1.8);

  it.each([
    ["stimpFt", Number.NaN],
    ["stimpFt", Number.POSITIVE_INFINITY],
    ["stimpFt", Number.NEGATIVE_INFINITY],
    ["stimpFt", 2.999999],
    ["stimpFt", 16.000001],
    ["gradePercent", Number.NaN],
    ["gradePercent", Number.POSITIVE_INFINITY],
    ["gradePercent", Number.NEGATIVE_INFINITY],
    ["gradePercent", -0.000001],
    ["gradePercent", 10.000001],
    ["aspectDeg", Number.NaN],
    ["aspectDeg", Number.POSITIVE_INFINITY],
    ["aspectDeg", Number.NEGATIVE_INFINITY],
    ["aspectDeg", -360.000001],
    ["aspectDeg", 360.000001],
    ["muSlide", Number.NaN],
    ["muSlide", Number.POSITIVE_INFINITY],
    ["muSlide", Number.NEGATIVE_INFINITY],
    ["muSlide", 0],
    ["muSlide", -0.000001],
    ["muSlide", 1.500001],
  ] as const)("rejects invalid GreenConditions %s=%s", (field, value) => {
    const green = {
      stimpFt: 10,
      gradePercent: 0,
      aspectDeg: 0,
      muSlide: 0.2,
      [field]: value,
    };

    expect(() => simulatePutt(launch, green, 3)).toThrow();
  });

  it.each([
    { stimpFt: 3, gradePercent: 0, aspectDeg: -360, muSlide: 1e-6 },
    { stimpFt: 16, gradePercent: 10, aspectDeg: 360, muSlide: 1.5 },
  ])("accepts legal GreenConditions boundaries: %o", (green) => {
    expect(() => simulatePutt(launch, green, 3)).not.toThrow();
  });

  it("pins the breaking reference putt (stimp 10, 2 %, aspect 90)", () => {
    const result = simulatePutt(
      launch,
      { stimpFt: 10, gradePercent: 2, aspectDeg: 90 },
      3.0,
    );
    expect(result.holed).toBe(false);
    expect(result.totalDistanceM).toBeCloseTo(4.417405938785078, 7);
    expect(result.skidDistanceM).toBeCloseTo(0.5103817275162047, 7);
    expect(result.timeS).toBeCloseTo(4.388, 2);
    expect(result.breakM).toBeCloseTo(0.8176068791755766, 7);
    expect(result.missDistanceM).toBeCloseTo(1.4994647284222105, 7);
  });

  it("pins the holed reference putt (1.6 m/s, flat stimp 10)", () => {
    const launch = strike(PUTTER, 1.6);
    const result = simulatePutt(
      launch,
      { stimpFt: 10, gradePercent: 0, aspectDeg: 0 },
      3.0,
    );
    expect(result.holed).toBe(true);
    expect(result.speedAtHoleMps).toBeCloseTo(0.5903262895096224, 7);
    expect(result.marginMps).toBeCloseTo(0.2283133618862715, 7);
  });

  it("mirror aspect mirrors the break", () => {
    const launch = strike(PUTTER, 2.0);
    const left = simulatePutt(
      launch,
      { stimpFt: 10, gradePercent: 2, aspectDeg: 90 },
      10,
    );
    const right = simulatePutt(
      launch,
      { stimpFt: 10, gradePercent: 2, aspectDeg: -90 },
      10,
    );
    expect(left.breakM).toBeCloseTo(-right.breakM, 9);
    expect(left.totalDistanceM).toBeCloseTo(right.totalDistanceM, 9);
  });

  it("speed is monotone non-increasing on a flat green", () => {
    const result = simulatePutt(
      strike(PUTTER, 2.0),
      { stimpFt: 10, gradePercent: 0, aspectDeg: 0 },
      10,
    );
    for (let i = 1; i < result.speedsMps.length; i++) {
      expect(result.speedsMps[i]).toBeLessThanOrEqual(
        result.speedsMps[i - 1] + 1e-12,
      );
    }
  });

  it("backstroke proxy matches the pendulum formula", () => {
    expect(clubheadSpeedFromBackstroke(0.3)).toBeCloseTo(
      0.3 * Math.sqrt(9.80665 / 0.889),
      12,
    );
  });

  it("rejects explicit null sliding friction instead of defaulting it", () => {
    expect(() =>
      simulatePutt(
        strike(PUTTER, 2),
        { stimpFt: 10, gradePercent: 0, aspectDeg: 0, muSlide: null } as never,
        3,
      ),
    ).toThrow(/muSlide/);
  });
});
