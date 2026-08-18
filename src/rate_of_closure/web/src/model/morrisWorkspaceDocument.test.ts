import { describe, expect, it } from "vitest";

import parityWorkspace from "../../../../../tests/rate_of_closure/fixtures/morris_workspace_v1.json";

import { getClub } from "./club";
import { DEFAULT_SCENARIO } from "./impact";
import { parseMorrisJobEnvelope } from "./morrisAuthorityContract";
import {
  RATE_MORRIS_VARIABLE_KEYS,
  serializeMorrisAuthorityRequest,
  suggestedMorrisFactorDrafts,
} from "./morrisAuthorityRequest";
import { defaultMorrisAuthorityBase } from "./morrisWorkflowDefaults";
import {
  INVALID_MORRIS_BOUNDS_MESSAGE,
  MAX_MORRIS_WORKSPACE_BYTES,
  createMorrisWorkspaceDocument,
  morrisCsvCell,
  morrisWorkspaceReportToCsv,
  morrisWorkspaceToJson,
  parseMorrisWorkspaceJson,
} from "./morrisWorkspaceDocument";

const base = defaultMorrisAuthorityBase(
  getClub("Driver 10.5°"),
  Object.freeze({ ...DEFAULT_SCENARIO, clubheadSpeedMph: 113 }),
);
const drafts = suggestedMorrisFactorDrafts(base).map((draft, index) => ({
  ...draft, enabled: index === 0,
}));
const design = Object.freeze({
  trajectories: 12, levels: 4, seed: 73, minimumEffects: 2, workerCount: 1,
});
const request = serializeMorrisAuthorityRequest({
  requestId: "workspace-request", base, factors: drafts,
  ...design,
});
const source = request.factors[0];
const completedJobDocument = {
  schema_id: "rate-of-closure/morris-job", schema_version: 1,
  job_id: "workspace-job", request_id: request.request_id,
  status: "completed", completed_samples: 24, total_samples: 24,
  cancel_requested: false,
  report: {
    schema_id: "swing-sim/morris-global-sensitivity-report", schema_version: 1,
    method: "morris-elementary-effects",
    design: { trajectories: 12, levels: 4, seed: 73, total_samples: 24, normalized_step: 2 / 3 },
    assumptions: ["Deterministic test authority."],
    interaction_caveat: "Morris sigma does not separate interaction from nonlinearity.",
    estimates: [{
      source: { spec_id: source.spec_id, variable_key: source.variable_key, unit: source.unit,
        bounds: [source.lower, source.upper], time_window_s: null, point_ids: [] },
      target: { name: "carry_m", unit: "m", kind: "shot-outcome", time_s: null, point_id: null, coordinate_frame: "app_frame:x_target,y_up,z_right" },
      effects: { mu: 2, mu_star: 2, mu_star_standard_error: 0, sigma: 0 },
      availability: "available", sample_adequacy: "adequate",
      denominator: { total_pairs: 12, valid_pairs: 12, typed_no_impact_pairs: 0, no_impact_unavailable_pairs: 0, failed_pairs: 0, nonfinite_pairs: 0 },
    }],
  }, error: null,
};
const completed = parseMorrisJobEnvelope(completedJobDocument);

describe("Morris workspace v1", () => {
  it("round-trips all canonical drafts and completed aggregate evidence", () => {
    const workspace = createMorrisWorkspaceDocument(base, drafts, design, { request, job: completed });
    const restored = parseMorrisWorkspaceJson(morrisWorkspaceToJson(workspace));
    expect(restored.setup.factorDrafts).toHaveLength(RATE_MORRIS_VARIABLE_KEYS.length);
    expect(restored.setup.factorDrafts.filter((draft) => !draft.enabled)).toHaveLength(9);
    expect(restored.completedEvidence?.job.report?.estimates[0].effects.muStar).toBe(2);
  });

  it("rejects duplicate fields, non-completed evidence, and crossed identities", () => {
    const workspace = createMorrisWorkspaceDocument(base, drafts, design, { request, job: completed });
    const json = morrisWorkspaceToJson(workspace);
    expect(() => parseMorrisWorkspaceJson(json.replace('"schema_id":', '"schema_id":"duplicate","schema_id":')))
      .toThrow(/duplicate JSON field/);
    expect(() => parseMorrisWorkspaceJson(json.replace('"status": "completed"', '"status": "running"')))
      .toThrow(/completed/);
    expect(() => parseMorrisWorkspaceJson(json.replace('"request_id": "workspace-request"', '"request_id": "crossed-request"')))
      .toThrow(/request identity|request_id/);
  });

  it("exports deterministic aggregate CSV with full provenance and no sample claim", () => {
    const workspace = createMorrisWorkspaceDocument(base, drafts, design, { request, job: completed });
    const csv = morrisWorkspaceReportToCsv(workspace);
    expect(csv).toContain("request_id,job_id,evidence_state,export_scope,trajectories,levels,seed,total_samples");
    expect(csv).toContain("source_spec_id,source_variable_key,source_unit,source_lower,source_upper");
    expect(csv).toContain("mu,mu_star,mu_star_standard_error,sigma,availability,sample_adequacy");
    expect(csv).toContain("typed_no_impact_pairs,no_impact_unavailable_pairs,failed_pairs,nonfinite_pairs");
    expect(csv).not.toContain("raw_sample");
    expect(morrisWorkspaceReportToCsv(workspace)).toBe(csv);
  });

  it("retains a disabled tee row for ground workspaces", () => {
    const ground = Object.freeze({ ...base, supportMode: "ground" as const, teeHeightM: 0 });
    const groundDrafts = suggestedMorrisFactorDrafts(ground);
    const workspace = createMorrisWorkspaceDocument(ground, groundDrafts, design, null);
    expect(workspace.setup.factorDrafts).toHaveLength(10);
    expect(workspace.setup.factorDrafts[workspace.setup.factorDrafts.length - 1]).toMatchObject({
      variableKey: "swing_sim.ball_setup.tee_height_m", enabled: false,
    });
  });

  it("accepts and deterministically rewrites the shared Python/React parity fixture", () => {
    const restored = parseMorrisWorkspaceJson(JSON.stringify(parityWorkspace));
    expect(restored.completedEvidence?.job.jobId).toBe("workspace-job");
    expect(parseMorrisWorkspaceJson(morrisWorkspaceToJson(restored))).toEqual(restored);
  });

  it("losslessly retains invalid raw bounds for a disabled canonical draft", () => {
    const workspace = createMorrisWorkspaceDocument(base, drafts, design, null);
    const wire = JSON.parse(morrisWorkspaceToJson(workspace));
    wire.setup.factor_drafts[1] = {
      ...wire.setup.factor_drafts[1],
      enabled: false,
      lower: "draft pending",
      validation_error: INVALID_MORRIS_BOUNDS_MESSAGE,
    };
    const restored = parseMorrisWorkspaceJson(JSON.stringify(wire));
    expect(restored.setup.factorDrafts[1]).toMatchObject({
      enabled: false,
      lower: "draft pending",
      validationError: INVALID_MORRIS_BOUNDS_MESSAGE,
    });
    expect(JSON.parse(morrisWorkspaceToJson(restored)).setup.factor_drafts[1].lower)
      .toBe("draft pending");
  });

  it("exports current disabled incomplete drafts with an explicit validation record", () => {
    const incomplete = drafts.map((draft, index) => index === 1
      ? { ...draft, enabled: false, lower: null }
      : draft);
    const workspace = createMorrisWorkspaceDocument(base, incomplete, design, null);
    expect(workspace.setup.factorDrafts[1]).toMatchObject({
      enabled: false,
      lower: "",
      validationError: INVALID_MORRIS_BOUNDS_MESSAGE,
    });
  });

  it("preserves a setup-only workspace with no enabled factors", () => {
    const disabled = drafts.map((draft) => ({ ...draft, enabled: false }));
    const workspace = createMorrisWorkspaceDocument(base, disabled, design, null);
    expect(workspace.setup.factorDrafts.every((draft) => !draft.enabled)).toBe(true);
    expect(workspace.completedEvidence).toBeNull();
  });

  it("enforces the canonical byte, node, and raw-text budgets before schema use", () => {
    expect(() => parseMorrisWorkspaceJson(" ".repeat(MAX_MORRIS_WORKSPACE_BYTES + 1)))
      .toThrow(/byte limit/);
    expect(() => parseMorrisWorkspaceJson(JSON.stringify(new Array(25_000).fill(0))))
      .toThrow(/node limit/);
    const wire = JSON.parse(morrisWorkspaceToJson(
      createMorrisWorkspaceDocument(base, drafts, design, null),
    ));
    wire.setup.factor_drafts[1] = {
      ...wire.setup.factor_drafts[1], enabled: false,
      lower: "x".repeat(129), validation_error: INVALID_MORRIS_BOUNDS_MESSAGE,
    };
    expect(() => parseMorrisWorkspaceJson(JSON.stringify(wire))).toThrow(/bounded text/);
    wire.setup.factor_drafts[1].lower = "x".repeat(128);
    expect(parseMorrisWorkspaceJson(JSON.stringify(wire)).setup.factorDrafts[1].lower)
      .toHaveLength(128);
    wire.setup.factor_drafts[1].lower = "😀".repeat(128);
    expect(() => parseMorrisWorkspaceJson(JSON.stringify(wire))).not.toThrow();
    wire.setup.factor_drafts[1].lower += "😀";
    expect(() => parseMorrisWorkspaceJson(JSON.stringify(wire))).toThrow(/bounded text/);
    const depth32 = `${"[".repeat(32)}0${"]".repeat(32)}`;
    expect(() => parseMorrisWorkspaceJson(depth32)).toThrow(/plain object/);
    const depth33 = `${"[".repeat(33)}0${"]".repeat(33)}`;
    expect(() => parseMorrisWorkspaceJson(depth33)).toThrow(/depth limit/);
  });

  it("uses one strict decimal grammar and editor range for raw numeric bounds", () => {
    const makeWire = () => JSON.parse(morrisWorkspaceToJson(
      createMorrisWorkspaceDocument(base, drafts, design, null),
    ));
    for (const lower of ["-1.5", "+.5", "1.", "1e0", "-2.5E-4"]) {
      const wire = makeWire();
      wire.setup.factor_drafts[0].lower = lower;
      expect(() => parseMorrisWorkspaceJson(JSON.stringify(wire))).not.toThrow();
    }
    for (const lower of [" 1", "1 ", "0x1", "0b1", "0o1", "NaN", "Infinity", "1_000"]) {
      const wire = makeWire();
      wire.setup.factor_drafts[0].lower = lower;
      expect(() => parseMorrisWorkspaceJson(JSON.stringify(wire))).toThrow(/enabled factor/);
    }
    const bounded = makeWire();
    bounded.setup.factor_drafts[0].lower = "-1000000000";
    bounded.setup.factor_drafts[0].upper = "1000000000";
    expect(() => parseMorrisWorkspaceJson(JSON.stringify(bounded))).not.toThrow();
    bounded.setup.factor_drafts[0].lower = "-1000000001";
    expect(() => parseMorrisWorkspaceJson(JSON.stringify(bounded))).toThrow(/enabled factor/);
  });

  it("requires exact disabled-draft errors and a null error for valid ground tee bounds", () => {
    const wire = JSON.parse(morrisWorkspaceToJson(
      createMorrisWorkspaceDocument(base, drafts, design, null),
    ));
    wire.setup.factor_drafts[1] = {
      ...wire.setup.factor_drafts[1], enabled: false,
      lower: "draft pending", validation_error: "arbitrary explanation",
    };
    expect(() => parseMorrisWorkspaceJson(JSON.stringify(wire))).toThrow(/exactly reflect/);
    wire.setup.factor_drafts[1].validation_error = INVALID_MORRIS_BOUNDS_MESSAGE;
    expect(() => parseMorrisWorkspaceJson(JSON.stringify(wire))).not.toThrow();

    const ground = Object.freeze({ ...base, supportMode: "ground" as const, teeHeightM: 0 });
    const groundWorkspace = createMorrisWorkspaceDocument(
      ground, suggestedMorrisFactorDrafts(ground), design, null,
    );
    const tee = groundWorkspace.setup.factorDrafts[groundWorkspace.setup.factorDrafts.length - 1];
    expect(tee).toMatchObject({ enabled: false, validationError: null });
    const groundWire = JSON.parse(morrisWorkspaceToJson(groundWorkspace));
    groundWire.setup.factor_drafts[groundWire.setup.factor_drafts.length - 1].validation_error = INVALID_MORRIS_BOUNDS_MESSAGE;
    expect(() => parseMorrisWorkspaceJson(JSON.stringify(groundWire))).toThrow(/exactly reflect/);
  });

  it("enforces workspace design caps independently of the authority request schema", () => {
    const makeWire = () => JSON.parse(morrisWorkspaceToJson(
      createMorrisWorkspaceDocument(base, drafts, design, null),
    ));
    for (const trajectories of [1, 5_001]) {
      const wire = makeWire(); wire.setup.trajectories = trajectories;
      expect(() => parseMorrisWorkspaceJson(JSON.stringify(wire))).toThrow(/trajectories/);
    }
    const seed = makeWire(); seed.setup.seed = 2 ** 31;
    expect(() => parseMorrisWorkspaceJson(JSON.stringify(seed))).toThrow(/seed/);
  });

  it("deep-freezes imported setup and completed evidence", () => {
    const restored = parseMorrisWorkspaceJson(morrisWorkspaceToJson(
      createMorrisWorkspaceDocument(base, drafts, design, { request, job: completed }),
    ));
    const visit = (value: unknown): void => {
      if (value === null || typeof value !== "object") return;
      expect(Object.isFrozen(value)).toBe(true);
      Object.values(value).forEach(visit);
    };
    visit(restored);
    expect(Reflect.set(restored.completedEvidence!.request.base, "plane_yaw_deg", 99)).toBe(false);
    expect(Reflect.set(restored.completedEvidence!.job.report!.estimates[0].effects, "mu", 99)).toBe(false);
    expect(restored.completedEvidence!.request.base.plane_yaw_deg).toBe(base.planeYawDeg);
    expect(restored.completedEvidence!.job.report!.estimates[0].effects.mu).toBe(2);
  });

  it("neutralizes text spreadsheet formulas while leaving negative metrics numeric", () => {
    for (const text of ["=formula", "+formula", "-formula", "@formula", "\tformula", "\rformula"]) {
      const expected = text.startsWith("\r") ? `"'${text}"` : `'${text}`;
      expect(morrisCsvCell(text)).toBe(expected);
    }
    expect(morrisCsvCell(-2)).toBe("-2");
  });
});
