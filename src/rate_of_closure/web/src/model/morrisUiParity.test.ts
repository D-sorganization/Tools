/** Cross-runtime parity against the Python-generated Morris UI fixture. */

import { describe, expect, it } from "vitest";

import fixture from "./__fixtures__/morris_ui_parity_v1.json";
import { CLUB_LIBRARY } from "./club";
import { parseMorrisJobEnvelope } from "./morrisAuthorityContract";
import {
  buildMorrisFactorRows,
  AUTHORITY_FLIGHT_MODELS,
  serializeMorrisAuthorityRequest,
  type MorrisAuthorityBase,
  type MorrisAuthorityRequest,
  type MorrisFactorDraft,
} from "./morrisAuthorityRequest";
import { presentMorrisJob, presentMorrisReport } from "./morrisPresentation";

const snakeCase = (key: string): string => key.replace(/[A-Z]/g, (character) => `_${character.toLowerCase()}`);

const snakeDocument = (value: unknown): unknown => {
  if (Array.isArray(value)) return value.map(snakeDocument);
  if (value === null || typeof value !== "object") return value;
  return Object.fromEntries(Object.entries(value).map(([key, entry]) => [snakeCase(key), snakeDocument(entry)]));
};

const drafts = (): MorrisFactorDraft[] => fixture.factor_drafts.map((draft) => ({
  variableKey: draft.variable_key,
  enabled: draft.enabled,
  lower: draft.lower,
  upper: draft.upper,
}));

const base = (): MorrisAuthorityBase => {
  const source = fixture.submitted_request.base;
  return {
    clubName: source.club_name,
    supportMode: source.support_mode as MorrisAuthorityBase["supportMode"],
    teeHeightM: source.tee_height_m,
    planeYawDeg: source.plane_yaw_deg,
    planeSideTiltDeg: source.plane_side_tilt_deg,
    planeForwardTiltDeg: source.plane_forward_tilt_deg,
    pendulumM1Kg: source.pendulum_m1_kg,
    pendulumL1M: source.pendulum_l1_m,
    pendulumLc1M: source.pendulum_lc1_m,
    pendulumI1KgM2: source.pendulum_i1_kg_m2,
    pendulumM2Kg: source.pendulum_m2_kg,
    pendulumL2M: source.pendulum_l2_m,
    pendulumLc2M: source.pendulum_lc2_m,
    pendulumI2KgM2: source.pendulum_i2_kg_m2,
    dampingShoulder: source.damping_shoulder,
    dampingWrist: source.damping_wrist,
    swingDurationS: source.swing_duration_s,
    flightModel: source.flight_model,
    impactOffsetToeMm: source.impact_offset_toe_mm,
    impactOffsetHighMm: source.impact_offset_high_mm,
  };
};

const request = (): MorrisAuthorityRequest => ({
  requestId: fixture.submitted_request.request_id,
  base: base(),
  factors: drafts(),
  trajectories: fixture.submitted_request.trajectories,
  levels: fixture.submitted_request.levels,
  seed: fixture.submitted_request.seed,
  minimumEffects: fixture.submitted_request.minimum_effects,
  workerCount: fixture.submitted_request.worker_count,
});

describe("Python-generated Morris UI parity", () => {
  it("matches factor rows and the exact submitted request", () => {
    expect(fixture.schema_id).toBe("rate-of-closure/morris-ui-parity");
    expect(fixture.schema_version).toBe(1);
    expect(CLUB_LIBRARY.map((club) => club.name)).toEqual(fixture.authority_club_names);
    expect(AUTHORITY_FLIGHT_MODELS).toEqual(fixture.authority_flight_models);

    const rows = buildMorrisFactorRows(drafts(), "tee");

    expect(snakeDocument(rows)).toEqual(fixture.expected_factor_rows);
    expect(serializeMorrisAuthorityRequest(request())).toEqual(fixture.submitted_request);
  });

  it("matches completed lifecycle and every target-scoped table", () => {
    const job = parseMorrisJobEnvelope(structuredClone(fixture.completed_job));

    expect(snakeDocument(presentMorrisJob(job))).toEqual(fixture.expected_job_presentation);
    for (const [targetName, expected] of Object.entries(fixture.expected_tables)) {
      expect(snakeDocument(presentMorrisReport(job.report!, targetName))).toEqual(expected);
    }
  });
});
