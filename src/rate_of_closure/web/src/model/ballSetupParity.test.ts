import { describe, expect, it } from "vitest";

import fixture from "./__fixtures__/ball_setup_golden_v1.json";
import { getClub } from "./club";
import {
  BALL_HEIGHT_REFERENCE,
  GOLF_BALL_RADIUS_M,
  ballCenterPosition,
  ballSetupFromJson,
  ballSetupToJson,
  defaultBallSetupForClub,
  resolveBallSetup,
  type BallSetup,
  type SupportMode,
} from "./ballSetup";
import { ballSetupFromSimulationDocument } from "./ballSetupPersistence";

type GoldenSetup = {
  support_mode: SupportMode;
  tee_height_m: number;
  height_reference: string;
  ball_center_m: [number, number, number];
};

const setupFromGolden = (setup: GoldenSetup): BallSetup => ({
  supportMode: setup.support_mode,
  teeHeightM: setup.tee_height_m,
});

const invalidHeight = (value: number | string): number => {
  if (value === "nan") return Number.NaN;
  if (value === "positive_infinity") return Number.POSITIVE_INFINITY;
  if (value === "negative_infinity") return Number.NEGATIVE_INFINITY;
  return Number(value);
};

const expectGoldenSetup = (actual: BallSetup, expected: GoldenSetup): void => {
  expect(Object.keys(expected).sort()).toEqual([
    "ball_center_m",
    "height_reference",
    "support_mode",
    "tee_height_m",
  ]);
  expect(actual.supportMode).toBe(expected.support_mode);
  expect(actual.teeHeightM).toBeCloseTo(expected.tee_height_m, 12);
  const serialized = ballSetupToJson(actual);
  expect(serialized.height_reference).toBe(expected.height_reference);
  expected.ball_center_m.forEach((coordinate, index) => {
    expect(serialized.ball_center_m[index]).toBeCloseTo(coordinate, 12);
  });
};

describe("ball support Python/React golden parity", () => {
  it("uses a strict versioned SI fixture contract", () => {
    expect(Object.keys(fixture).sort()).toEqual([
      "ball_radius_m",
      "default_cases",
      "geometry_cases",
      "height_reference",
      "invalid_cases",
      "legacy_cases",
      "override_cases",
      "schema",
      "schema_version",
      "units",
    ]);
    expect(fixture.schema).toBe("rate_of_closure.ball_setup_golden");
    expect(fixture.schema_version).toBe(1);
    expect(fixture.units).toEqual({ length: "m" });
    expect(fixture.height_reference).toBe(BALL_HEIGHT_REFERENCE);
    expect(fixture.ball_radius_m).toBeCloseTo(GOLF_BALL_RADIUS_M, 12);
  });

  it("matches driver/non-driver defaults and explicit overrides", () => {
    for (const testCase of fixture.default_cases) {
      const actual = defaultBallSetupForClub(getClub(testCase.club_name));
      expectGoldenSetup(actual, testCase.expected as GoldenSetup);
    }
    for (const testCase of fixture.override_cases) {
      const actual = resolveBallSetup(setupFromGolden(testCase.input as GoldenSetup));
      expectGoldenSetup(actual, testCase.expected as GoldenSetup);
    }
  });

  it("matches geometry, serialization, and Ground zero effective height", () => {
    for (const testCase of fixture.geometry_cases) {
      const setup = ballSetupFromJson(testCase.input);
      expect(setup.teeHeightM).toBe(testCase.effective_tee_height_m);
      ballCenterPosition(setup).forEach((coordinate, index) => {
        expect(coordinate).toBeCloseTo(testCase.ball_center_m[index], 12);
      });
      expectGoldenSetup(setup, testCase.serialized as GoldenSetup);
    }
  });

  it("rejects negative and non-finite heights", () => {
    for (const testCase of fixture.invalid_cases) {
      expect(() => resolveBallSetup({
        supportMode: testCase.support_mode as SupportMode,
        teeHeightM: invalidHeight(testCase.tee_height),
      })).toThrow(new RegExp(testCase.error_pattern, "i"));
    }
  });

  it("migrates legacy documents to Ground", () => {
    for (const testCase of fixture.legacy_cases) {
      const actual = ballSetupFromSimulationDocument(testCase.document);
      expectGoldenSetup(actual, testCase.expected as GoldenSetup);
    }
  });
});
