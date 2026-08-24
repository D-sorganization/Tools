import { describe, expect, it } from "vitest";

import { getClub } from "./club";
import { DRIVER_TEE_HEIGHT_M } from "./ballSetup";
import { DEFAULT_SCENARIO } from "./impact";
import { passiveDoublePendulumRun } from "./doublePendulum";
import { starterTorqueProfile } from "./torqueProfileEditor";
import { DEFAULT_PRIMARY_VIEW_STATE } from "./viewPreferences";
import { defaultViewWorkspace } from "./viewWorkspace";
import variationFixture from "./__fixtures__/workspace_variation_parity.json";
import { planFromJson } from "./variation";
import {
  buildCapabilityWorkflow,
  capabilityWorkflowFromJson,
  capabilityWorkflowToJson,
  defaultCapabilityWorkflowInputs,
} from "./capabilityWorkflow";
import {
  boxTolerance,
  createSpatialTarget,
  targetPointFromFrame,
} from "./spatialTarget";
import {
  createWorkspaceDocument,
  parseWorkspaceDocument,
  type WorkspaceSessionSnapshot,
} from "./workspaceSession";
import { validatedVariationWorkspace } from "./workspaceVariationSession";

const snapshot = (): WorkspaceSessionSnapshot => {
  const profile = starterTorqueProfile();
  const ballSetup = { supportMode: "tee" as const, teeHeightM: DRIVER_TEE_HEIGHT_M };
  return {
    scenario: { ...DEFAULT_SCENARIO, omegaShaftDps: -900 },
    club: getClub("Driver 10.5°"),
    units: { speed: "mph", rotation: "deg/s", length: "mm", distance: "yd" },
    simulation: {
      ballSetup,
      ballSetupUserOverridden: false,
      spatialTarget: createSpatialTarget({
        label: "Apex gate",
        kind: "aerial_waypoint",
        point: targetPointFromFrame([137.5, 3.25, 24.25], "flight"),
        tolerance: boxTolerance([4.5, 2.5, 3.5]),
        elevationSource: "absolute",
      }),
    },
    torque: {
      profiles: Object.freeze([profile]),
      activeProfileId: profile.profileId,
      runConfig: passiveDoublePendulumRun(),
    },
    variation: validatedVariationWorkspace({
      plan: planFromJson(JSON.stringify(variationFixture.plan)),
      analysisExecution: "both" as const,
      selectedOutputMetrics: ["carry_m", "lateral_m", "apex_m"],
    }, ballSetup),
    capability: customCapabilityWorkflow(),
    modules: DEFAULT_PRIMARY_VIEW_STATE,
    viewWorkspace: defaultViewWorkspace,
  };
};

const customCapabilityWorkflow = () => {
  const payload = JSON.parse(capabilityWorkflowToJson(buildCapabilityWorkflow({
    ...defaultCapabilityWorkflowInputs(), profileId: "workspace-profile",
    objective: "minimize_expected_miss", targetDistanceM: 241,
    targetLateralM: -4, spinAxisTiltDeg: -3.5,
  })));
  payload.profile.provenance = "measured/session-42";
  payload.profile.confidence = 0.71;
  payload.profile.clubs[0].provenance = "fit/driver-42";
  payload.profile.clubs[0].confidence = 0.63;
  payload.profile.clubs[0].matrix = [
    [1, 0.2, 0], [0.2, 1, 0.1], [0, 0.1, 1],
  ];
  payload.profile.clubs[0].parameters[0].bias = 0.4;
  payload.request.problem_id = "custom-problem-42";
  payload.request.cvar_alpha = 0.83;
  payload.request.minimum_success_fraction = 0.64;
  payload.request.target.kind = "fairway";
  payload.request.target.band_half_length_m = 21;
  payload.request.target.half_width_m = 8;
  payload.evaluator_config.spin_defaults[0].provenance = "measured/spin-42";
  return capabilityWorkflowFromJson(JSON.stringify(payload));
};

const metadata = {
  documentId: "workspace.web.test",
  title: "Web test",
  createdAtUtc: "2026-08-10T12:00:00Z",
  modifiedAtUtc: "2026-08-10T12:01:00Z",
  appVersion: "1.14.30",
};

describe("whole workspace session contract", () => {
  it("round trips the supported live explorer state", () => {
    const encoded = createWorkspaceDocument(snapshot(), metadata);
    expect(parseWorkspaceDocument(encoded)).toEqual(snapshot());
    expect(JSON.parse(encoded).schema_version).toBe(3);
    const session = JSON.parse(encoded).model_session;
    expect(session.schema_version).toBe(5);
    expect(session.data.simulation_setup.data.ball_setup.provenance).toEqual({
      kind: "club_default",
      club_name: "Driver 10.5°",
    });
    expect(session.data.simulation_setup.data.spatial_target).toMatchObject({
      source_frame: "flight",
      tolerance: {
        kind: "box",
        half_extents_m: { x: 4.5, elevation: 2.5, right: 3.5 },
      },
    });
    expect(session.data.torque_selection.data).toMatchObject({
      active_profile_id: "profile.web.starter_drive.v1",
      selection_provenance: {
        kind: "library_profile",
        profile_source: "direct",
      },
    });
    expect(session.data.variation_study).toEqual(variationFixture.selection);
    const binding = JSON.parse(encoded).variation_plan;
    expect(binding.state).toBe("canonical");
    expect(binding.document.plan).toEqual(variationFixture.plan);
    expect(binding.document.plan).not.toHaveProperty("ball_setup");
    expect(session.data.capability_request).toMatchObject({
      schema_version: "capability-optimization-workflow/v1",
      request: {
        objective: "minimize_expected_miss",
        target: { distance_m: 241, lateral_m: -4 },
      },
    });
    expect(session.data.capability_request).not.toHaveProperty("result");
    expect(capabilityWorkflowToJson(
      parseWorkspaceDocument(encoded).capability,
    )).toBe(capabilityWorkflowToJson(snapshot().capability));
  });

  it("requires an explicit capability fallback to migrate a v4 session", () => {
    const value = JSON.parse(createWorkspaceDocument(snapshot(), metadata));
    value.model_session.schema_version = 4;
    delete value.model_session.data.capability_request;
    const text = JSON.stringify(value);

    expect(() => parseWorkspaceDocument(text)).toThrow(/explicit capability/i);
    expect(parseWorkspaceDocument(text, {
      legacyCapabilityFallback: snapshot().capability,
    }).capability).toEqual(snapshot().capability);
  });

  it("rejects computed capability output before returning workspace state", () => {
    const value = JSON.parse(createWorkspaceDocument(snapshot(), metadata));
    value.model_session.data.capability_request.computed_result = {};

    expect(() => parseWorkspaceDocument(JSON.stringify(value))).toThrow(
      /capability workflow/i,
    );
  });

  it("rejects a variation plan that duplicates the simulation ball setup", () => {
    const value = JSON.parse(createWorkspaceDocument(snapshot(), metadata));
    value.variation_plan.document.plan.ball_setup = {
      support_mode: "tee",
      tee_height_m: DRIVER_TEE_HEIGHT_M,
    };
    expect(() => parseWorkspaceDocument(JSON.stringify(value))).toThrow(
      /must not duplicate simulation ball_setup/i,
    );
  });

  it("migrates a v2 raw plan as explicit legacy evidence", () => {
    const value = JSON.parse(createWorkspaceDocument(snapshot(), metadata));
    value.schema_version = 2;
    value.variation_plan = value.variation_plan.document.plan;

    const migrated = parseWorkspaceDocument(JSON.stringify(value));

    expect(migrated.variation.planEvidence?.metadata).toBeNull();
    expect(migrated.variation.planEvidence?.provenance).toBeNull();
    expect(migrated.variation.planEvidence?.warning).toMatch(
      /not evidence of historical reproducibility/i,
    );
    expect(JSON.parse(createWorkspaceDocument(migrated, metadata)).variation_plan.state)
      .toBe("legacy");
  });

  it("rejects a workspace plan-metadata substitution", () => {
    const value = JSON.parse(createWorkspaceDocument(snapshot(), metadata));
    value.variation_plan.document.plan.seed += 1;

    expect(() => parseWorkspaceDocument(JSON.stringify(value))).toThrow(/digest/i);
  });

  it("rejects club-default provenance that disagrees with saved geometry", () => {
    const value = JSON.parse(createWorkspaceDocument(snapshot(), metadata));
    const setup =
      value.model_session.data.simulation_setup.data.ball_setup.setup;
    setup.tee_height_m = 0.05;
    setup.ball_center_m[1] = 0.05 + 0.04267 / 2;
    expect(() => parseWorkspaceDocument(JSON.stringify(value))).toThrow(
      /club-default ball setup/i,
    );
  });

  it("requires an explicit fallback to migrate a v1 explorer session", () => {
    const value = JSON.parse(createWorkspaceDocument(snapshot(), metadata));
    value.model_session.schema_version = 1;
    value.model_session.data = {
      scenario: value.model_session.data.scenario,
      units: value.model_session.data.units,
    };
    const text = JSON.stringify(value);
    expect(() => parseWorkspaceDocument(text)).toThrow(/explicit.*migration/i);
    expect(
      parseWorkspaceDocument(text, {
        legacySimulationFallback: snapshot().simulation,
        legacyTorqueFallback: snapshot().torque,
        legacyVariationFallback: snapshot().variation,
        legacyCapabilityFallback: snapshot().capability,
      }).simulation,
    ).toEqual(snapshot().simulation);
  });

  it("preserves a cross-club v1 fallback as an explicit override", () => {
    const iron: WorkspaceSessionSnapshot = {
      ...snapshot(),
      club: getClub("7-Iron"),
      simulation: {
        ...snapshot().simulation,
        ballSetup: { supportMode: "ground", teeHeightM: 0 },
      },
    };
    const value = JSON.parse(createWorkspaceDocument(iron, metadata));
    value.model_session.schema_version = 1;
    value.model_session.data = {
      scenario: value.model_session.data.scenario,
      units: value.model_session.data.units,
    };
    const migrated = parseWorkspaceDocument(JSON.stringify(value), {
      legacySimulationFallback: snapshot().simulation,
      legacyTorqueFallback: snapshot().torque,
      legacyVariationFallback: snapshot().variation,
      legacyCapabilityFallback: snapshot().capability,
    });
    expect(migrated.simulation.ballSetup).toEqual(
      snapshot().simulation.ballSetup,
    );
    expect(migrated.simulation.ballSetupUserOverridden).toBe(true);
  });

  it("requires an explicit torque fallback to migrate a v2 session", () => {
    const value = JSON.parse(createWorkspaceDocument(snapshot(), metadata));
    value.model_session.schema_version = 2;
    delete value.model_session.data.torque_selection;
    delete value.model_session.data.variation_study;
    delete value.model_session.data.capability_request;
    const text = JSON.stringify(value);
    expect(() => parseWorkspaceDocument(text)).toThrow(/explicit torque/i);
    expect(
      parseWorkspaceDocument(text, {
        legacyTorqueFallback: snapshot().torque,
        legacyVariationFallback: snapshot().variation,
        legacyCapabilityFallback: snapshot().capability,
      }).torque,
    ).toEqual(snapshot().torque);
  });

  it.each([
    ["torque_unit", "lbf*ft"],
    ["coefficient_order", "descending"],
  ])("rejects noncanonical torque profile %s", (field, invalid) => {
    const value = JSON.parse(createWorkspaceDocument(snapshot(), metadata));
    value.prescribed_torque_profiles[0][field] = invalid;
    expect(() => parseWorkspaceDocument(JSON.stringify(value))).toThrow(
      new RegExp(field),
    );
  });

  it("rejects invalid variation selection before returning applicable values", () => {
    const value = JSON.parse(createWorkspaceDocument(snapshot(), metadata));
    value.model_session.data.variation_study.data.selected_output_metrics = [
      "unknown_metric",
    ];
    expect(() => parseWorkspaceDocument(JSON.stringify(value))).toThrow(
      /metric/i,
    );
  });

  it("requires a nonconflicting variation fallback for legacy v3", () => {
    const value = JSON.parse(createWorkspaceDocument(snapshot(), metadata));
    value.model_session.schema_version = 3;
    delete value.model_session.data.variation_study;
    delete value.model_session.data.capability_request;
    const text = JSON.stringify(value);

    expect(() => parseWorkspaceDocument(text)).toThrow(/explicit variation/i);
    expect(
      parseWorkspaceDocument(text, {
        legacyVariationFallback: snapshot().variation,
        legacyCapabilityFallback: snapshot().capability,
      }).variation,
    ).toEqual(snapshot().variation);

    const conflict = {
      ...snapshot().variation,
      plan: { ...snapshot().variation.plan, seed: 99 },
    };
    expect(() =>
      parseWorkspaceDocument(text, {
        legacyVariationFallback: conflict,
        legacyCapabilityFallback: snapshot().capability,
      }),
    ).toThrow(/conflicts/i);
  });

  it("rejects corrupt module and compositor documents", () => {
    const missingModule = JSON.parse(
      createWorkspaceDocument(snapshot(), metadata),
    );
    missingModule.layout.module_order = ["explorer"];
    expect(() => parseWorkspaceDocument(JSON.stringify(missingModule))).toThrow(
      /module/i,
    );

    const futureView = JSON.parse(
      createWorkspaceDocument(snapshot(), metadata),
    );
    futureView.layout.view_workspace.data.format =
      "rate_of_closure.view_workspace/9";
    expect(() => parseWorkspaceDocument(JSON.stringify(futureView))).toThrow(
      /format/i,
    );
  });

  it("matches the native stable identity and strict UTC metadata boundary", () => {
    const localTime = JSON.parse(createWorkspaceDocument(snapshot(), metadata));
    localTime.metadata.created_at_utc = "2026-08-10T12:00:00-07:00";
    expect(() => parseWorkspaceDocument(JSON.stringify(localTime))).toThrow(
      /UTC/i,
    );

    const unstableId = JSON.parse(
      createWorkspaceDocument(snapshot(), metadata),
    );
    unstableId.metadata.document_id = "workspace id with spaces";
    expect(() => parseWorkspaceDocument(JSON.stringify(unstableId))).toThrow(
      /identifier/i,
    );
  });
});
