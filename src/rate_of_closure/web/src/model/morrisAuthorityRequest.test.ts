/** UI-neutral Morris authority request and factor-row tests. */

import { describe, expect, it } from "vitest";

import {
  buildMorrisFactorRows,
  serializeMorrisAuthorityRequest,
  suggestedMorrisFactorDrafts,
  validateMorrisFactorDrafts,
  type MorrisAuthorityRequest,
  type MorrisFactorDraft,
} from "./morrisAuthorityRequest";
import { defaultMorrisAuthorityBase } from "./morrisWorkflowDefaults";
import { getClub } from "./club";
import { DEFAULT_SCENARIO } from "./impact";

const draft = (overrides: Partial<MorrisFactorDraft> = {}): MorrisFactorDraft => ({
  variableKey: "swing_sim.swing.yaw_deg",
  enabled: true,
  lower: -2,
  upper: 2,
  ...overrides,
});

const base = (): MorrisAuthorityRequest["base"] => ({
  clubName: "Driver 10.5°", supportMode: "tee", teeHeightM: 0.0381,
  planeYawDeg: 0, planeSideTiltDeg: -45, planeForwardTiltDeg: 0,
  pendulumM1Kg: 4, pendulumL1M: 0.65, pendulumLc1M: 0.3, pendulumI1KgM2: 0.4,
  pendulumM2Kg: 0.5, pendulumL2M: 1.05, pendulumLc2M: 0.55, pendulumI2KgM2: 0.08,
  dampingShoulder: 0.4, dampingWrist: 0.25, swingDurationS: 1.0,
  flightModel: "waterloo_penner", impactOffsetToeMm: 0, impactOffsetHighMm: 0,
});

describe("Morris factor rows", () => {
  it("fails closed when the current scenario cannot round-trip through the pinned authority", () => {
    expect(() => defaultMorrisAuthorityBase(getClub("Driver 10.5°"), DEFAULT_SCENARIO)).toThrow(
      /clubheadSpeedMph=113.*current value is unsupported/,
    );
    expect(() => defaultMorrisAuthorityBase(getClub("Driver 10.5°"), {
      ...DEFAULT_SCENARIO, clubheadSpeedMph: 113,
    })).not.toThrow();
  });

  it("fails closed instead of discarding custom club geometry", () => {
    const canonical = getClub("Driver 10.5°");
    expect(() => defaultMorrisAuthorityBase(
      { ...canonical, loftDeg: 11.25 },
      { ...DEFAULT_SCENARIO, clubheadSpeedMph: 113 },
    )).toThrow(/cannot represent custom club field loftDeg.*canonical Driver 10\.5°/);
    expect(() => defaultMorrisAuthorityBase(
      { ...canonical, headStyle: undefined },
      { ...DEFAULT_SCENARIO, clubheadSpeedMph: 113 },
    )).not.toThrow();
  });

  it("matches Python R13.6 base-centered suggestions and ground applicability", () => {
    const suggested = suggestedMorrisFactorDrafts(base());
    expect(suggested.map(({ variableKey, enabled }) => ({ variableKey, enabled }))).toEqual([
      { variableKey: "swing_sim.swing.yaw_deg", enabled: true },
      { variableKey: "swing_sim.swing.side_tilt_deg", enabled: true },
      { variableKey: "swing_sim.swing.forward_tilt_deg", enabled: true },
      { variableKey: "swing_sim.swing.damping_shoulder", enabled: true },
      { variableKey: "swing_sim.swing.damping_wrist", enabled: true },
      { variableKey: "swing_sim.impact.delivery.impact_offset_toe_mm", enabled: true },
      { variableKey: "swing_sim.impact.delivery.impact_offset_high_mm", enabled: true },
      { variableKey: "swing_sim.club.head_mass_kg", enabled: true },
      { variableKey: "swing_sim.club.head_moi_kg_m2", enabled: true },
      { variableKey: "swing_sim.ball_setup.tee_height_m", enabled: true },
    ]);
    const expectedBounds = [
      { variableKey: "swing_sim.swing.yaw_deg", enabled: true, lower: -3, upper: 3 },
      { variableKey: "swing_sim.swing.side_tilt_deg", enabled: true, lower: -48, upper: -42 },
      { variableKey: "swing_sim.swing.forward_tilt_deg", enabled: true, lower: -3, upper: 3 },
      { variableKey: "swing_sim.swing.damping_shoulder", enabled: true, lower: 0.3, upper: 0.5 },
      { variableKey: "swing_sim.swing.damping_wrist", enabled: true, lower: 0.15, upper: 0.35 },
      { variableKey: "swing_sim.impact.delivery.impact_offset_toe_mm", enabled: true, lower: -8, upper: 8 },
      { variableKey: "swing_sim.impact.delivery.impact_offset_high_mm", enabled: true, lower: -6, upper: 6 },
      { variableKey: "swing_sim.club.head_mass_kg", enabled: true, lower: 0.196, upper: 0.204 },
      { variableKey: "swing_sim.club.head_moi_kg_m2", enabled: true, lower: 0.00048, upper: 0.00056 },
      { variableKey: "swing_sim.ball_setup.tee_height_m", enabled: true, lower: 0.0321, upper: 0.0441 },
    ];
    expectedBounds.forEach((expected, index) => {
      expect(suggested[index].lower).toBeCloseTo(expected.lower, 12);
      expect(suggested[index].upper).toBeCloseTo(expected.upper, 12);
    });
    const ground = suggestedMorrisFactorDrafts({
      ...base(), supportMode: "ground", teeHeightM: 0, clubName: "Pitching Wedge",
    });
    expect(ground).toHaveLength(9);
    expect(ground.map((item) => item.variableKey)).toEqual(
      expect.not.arrayContaining(["swing_sim.ball_setup.tee_height_m"]),
    );
  });

  it("uses registry metadata and makes tee height inapplicable on ground", () => {
    const rows = buildMorrisFactorRows([draft(), draft({
      variableKey: "swing_sim.ball_setup.tee_height_m",
      lower: 0.02,
      upper: 0.05,
    })], "ground");

    expect(rows[0]).toMatchObject({
      specId: "swing_sim.swing.yaw_deg",
      label: "Swing-Plane Yaw",
      unit: "deg",
      applicability: null,
      enabled: true,
      applicable: true,
      validationError: null,
    });
    expect(rows[1]).toMatchObject({
      label: "Tee Height",
      applicability: "tee_only",
      applicable: false,
      validationError: "Tee height requires tee support",
    });
    expect(rows.every(Object.isFrozen)).toBe(true);
  });

  it("returns deterministic row-local validation without throwing", () => {
    const drafts = [
      draft({ lower: 2, upper: 2 }),
      draft(),
      draft({ variableKey: "swing_sim.club.cor" }),
    ];

    const result = validateMorrisFactorDrafts(drafts, "tee");

    expect(result.valid).toBe(false);
    expect(result.errors).toEqual([
      "Lower bound must be less than upper bound",
      "Enabled factor variables must be unique.",
      "Unsupported Morris factor",
    ]);
    expect(Object.isFrozen(result)).toBe(true);
  });

  it.each([
    ["coercive enabled", { enabled: 0 }],
    ["coercive variable key", { variableKey: 7 }],
    ["coercive lower bound", { lower: "-2" }],
  ])("rejects %s primitives", (_name, override) => {
    expect(() => buildMorrisFactorRows([
      { ...draft(), ...override } as unknown as MorrisFactorDraft,
    ], "tee")).toThrow();
  });

  it.each([
    ["swing_sim.swing.damping_shoulder", -0.01, 0.2],
    ["swing_sim.impact.delivery.impact_offset_toe_mm", -81, 0],
    ["swing_sim.impact.delivery.impact_offset_high_mm", 0, 41],
    ["swing_sim.club.head_mass_kg", 0.09, 0.2],
    ["swing_sim.club.head_moi_kg_m2", 5e-5, 2.1e-3],
    ["swing_sim.ball_setup.tee_height_m", -0.001, 0.03],
  ])("rejects out-of-range %s endpoints", (variableKey, lower, upper) => {
    const result = validateMorrisFactorDrafts([
      draft({ variableKey, lower, upper }),
    ], "tee");
    expect(result.valid).toBe(false);
    expect(result.errors[0]).toMatch(/endpoint limits/);
  });
});

describe("Morris authority request serialization", () => {
  const requestWith = (
    baseOverrides: Partial<MorrisAuthorityRequest["base"]> = {},
    requestOverrides: Partial<MorrisAuthorityRequest> = {},
  ): MorrisAuthorityRequest => ({
    requestId: "study-1", base: { ...base(), ...baseOverrides }, factors: [draft()],
    trajectories: 12, levels: 4, seed: 73, minimumEffects: 2, workerCount: 2,
    ...requestOverrides,
  });

  it("serializes the typed request to the exact snake-case authority document", () => {
    const request: MorrisAuthorityRequest = {
      requestId: "study-1",
      base: base(),
      factors: [draft()], trajectories: 12, levels: 4, seed: 73,
      minimumEffects: 2, workerCount: 2,
    };

    const document = serializeMorrisAuthorityRequest(request);

    expect(document).toMatchObject({
      schema_id: "rate-of-closure/morris-request",
      schema_version: 1,
      request_id: "study-1",
      factors: [{
        spec_id: "swing_sim.swing.yaw_deg", variable_key: "swing_sim.swing.yaw_deg",
        lower: -2, upper: 2, unit: "deg",
      }],
      trajectories: 12, levels: 4, seed: 73, minimum_effects: 2, worker_count: 2,
    });
    expect(document.base).toMatchObject({
      club_name: "Driver 10.5°", support_mode: "tee", tee_height_m: 0.0381,
      damping_shoulder: 0.4, impact_offset_toe_mm: 0,
    });
  });

  it("fails closed before serializing an invalid or disabled factor", () => {
    const request = {
      requestId: "study-1", base: base(), factors: [draft({ enabled: false })],
      trajectories: 12, levels: 4, seed: 73, minimumEffects: 2, workerCount: 2,
    } as unknown as MorrisAuthorityRequest;

    expect(() => serializeMorrisAuthorityRequest(request)).toThrow(/enabled factor/);
  });

  it("rejects coercive base numeric fields at the serializer boundary", () => {
    const request = {
      requestId: "study-1",
      base: { ...base(), teeHeightM: "0.0381" },
      factors: [draft()], trajectories: 12, levels: 4, seed: 73,
      minimumEffects: 2, workerCount: 2,
    } as unknown as MorrisAuthorityRequest;

    expect(() => serializeMorrisAuthorityRequest(request)).toThrow(/teeHeightM/);
  });

  it.each([
    ["unknown club", { clubName: "Unknown Club" }, /club library/],
    ["unknown flight model", { flightModel: "unknown" }, /flightModel/],
    ["ground tee", { supportMode: "ground", teeHeightM: 0.01 }, /support mode/],
    ["negative tee", { teeHeightM: -0.01 }, /support mode/],
    ["first mass", { pendulumM1Kg: 0 }, /must be positive/],
    ["first length", { pendulumL1M: 0 }, /must be positive/],
    ["first center", { pendulumLc1M: 0 }, /must be positive/],
    ["first inertia", { pendulumI1KgM2: 0 }, /must be positive/],
    ["second mass", { pendulumM2Kg: 0 }, /must be positive/],
    ["second length", { pendulumL2M: 0 }, /must be positive/],
    ["second center", { pendulumLc2M: 0 }, /must be positive/],
    ["second inertia", { pendulumI2KgM2: 0 }, /must be positive/],
    ["first center beyond length", { pendulumLc1M: 0.7 }, /must not exceed/],
    ["second center beyond length", { pendulumLc2M: 1.1 }, /must not exceed/],
    ["shoulder damping", { dampingShoulder: -0.1 }, /nonnegative/],
    ["wrist damping", { dampingWrist: -0.1 }, /nonnegative/],
    ["zero duration", { swingDurationS: 0 }, /positive/],
    ["toe offset", { impactOffsetToeMm: 81 }, /ToeMm/],
    ["high offset", { impactOffsetHighMm: -41 }, /HighMm/],
  ] as const satisfies ReadonlyArray<readonly [
    string, Partial<MorrisAuthorityRequest["base"]>, RegExp,
  ]>)("rejects authority-incompatible base semantics: %s", (_name, overrides, message) => {
    expect(() => serializeMorrisAuthorityRequest(requestWith(overrides))).toThrow(message);
  });

  it("enforces the authority sample and observation-cell resource formula", () => {
    expect(() => serializeMorrisAuthorityRequest(requestWith({}, {
      trajectories: 29_411,
      minimumEffects: 2,
    }))).not.toThrow();
    expect(() => serializeMorrisAuthorityRequest(requestWith({}, {
      trajectories: 29_412,
      minimumEffects: 2,
    }))).toThrow(/resource limits/);
  });

  it("serializes reversed drafts in canonical factor order", () => {
    const side = draft({
      variableKey: "swing_sim.swing.side_tilt_deg", lower: -3, upper: 3,
    });
    const document = serializeMorrisAuthorityRequest(requestWith({}, {
      factors: [side, draft()],
    }));
    expect(document.factors.map((factor) => factor.variable_key)).toEqual([
      "swing_sim.swing.yaw_deg", "swing_sim.swing.side_tilt_deg",
    ]);
  });
});
